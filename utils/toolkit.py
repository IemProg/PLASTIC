import os
import numpy as np
import torch
import io
import os
import time
import math
from collections import defaultdict, deque
import datetime

import torch.distributed as dist


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)



def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def compute_bwt(results_matrix):
    """
    Compute Backward Transfer (BWT).
    
    Parameters:
        results_matrix: A list where results_matrix[i] contains accuracies for tasks 0 to i
                      after training on task i.
                 
    Returns:
        bwt: The backward transfer value.
    """
    T = len(results_matrix)
    if T <= 1:
        return 0.0
    
    bwt = 0.0
    count = 0
    # For all tasks except the first one
    for i in range(T - 1):
        # Compare performance on task i after all training (last row) with 
        # performance right after training on task i
        bwt += results_matrix[-1][i] - results_matrix[i][i]
        count += 1
    
    return bwt / count if count > 0 else 0.0

def compute_fwt(results_matrix, baseline_performances):
    """
    Compute Forward Transfer (FWT).
    
    Parameters:
        results_matrix: A list where results_matrix[i] contains accuracies for tasks 0 to i
                      after training on task i.
        baseline_performances: A list of baseline performances for each task (before any training).
                 
    Returns:
        fwt: The forward transfer value.
    """
    T = len(results_matrix)
    if T <= 1 or not baseline_performances:
        return 0.0
    
    fwt = 0.0
    count = 0
    
    # For each task (except the first), compare its performance before any training
    # with its baseline performance
    for i in range(1, T):
        if i-1 < len(baseline_performances):
            try:
                # Get performance on current task from previous task's results
                # Note: this might not exist if the task wasn't evaluated yet
                before_performance = results_matrix[i-1][-1]  # Last element of previous task's results
                baseline = baseline_performances[i-1]
                fwt += before_performance - baseline
                count += 1
            except IndexError:
                # Skip if we don't have the performance measurement
                continue
    
    return fwt / count if count > 0 else 0.0
