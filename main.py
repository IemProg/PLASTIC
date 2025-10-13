import json
import argparse
from trainer import train

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb

import math
import numpy as np


def main():
    args = setup_parser().parse_args()
    name_dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    backbone_type = args.convnet_type
    increment = args.increment
    nb_classes = args.nb_classes
    adapter_size = args.adapter_size
    sam_rho = args.rho
    seed = args.seed
    optimizer = args.optimizer
    dset_variant = args.dset_variant
    layers = args.layers
    print_forget = args.print_forget

    # Test-time adaptation params
    niter = args.niter
    selection_p = args.selection_p
    e_margin = args.e_margin
    d_margin = args.d_margin

    N_augment = args.N_augment
    augment_train = args.augment_train
    reweight_entropy = args.reweight_entropy
    batch_tta = args.batch_tta
    random_aug = args.random_aug

    mahalanobis = args.mahalanobis
    start_TTA = args.start_TTA

    subset = args.subset
    subset_per = args.subset_per

    evaluate_flops = args.evaluate_flops

    # Optimizer params
    init_lr = args.init_lr
    weight_decay = args.weight_decay
    no_reset = args.no_reset

    adapter_bottleneck = args.adapter_bottleneck
    ffn_adapter_scalar = args.ffn_adapter_scalar
    type_tta = args.type_tta
    enable_ema = args.enable_ema
    ema_beta = args.ema_beta
    kl_weight = args.kl_weight

    shuffle = args.shuffle

    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # restore dataset name from arguments
    args["dataset"] = name_dataset
    args["tuned_epoch"] = int(epochs)
    args["batch_size"] = batch_size
    args["convnet_type"] = backbone_type
    args["increment"] = increment
    args["nb_classes"] = nb_classes

    args["ffn_num"] = adapter_size
    args["rho"] = sam_rho
    args["seed"] = [seed]
    args["optimizer"] = optimizer
    args["dset_variant"] = dset_variant
    args["layers"] = layers
    args["niter"] = niter
    args["reweight_entropy"] = reweight_entropy
    args["selection_p"] = selection_p

    args["e_margin"] = e_margin
    args["d_margin"] = d_margin
    args["N_augment"] = N_augment
    args["augment_train"] = augment_train
    args["batch_tta"] = batch_tta
    args["random_aug"] = random_aug

    args["mahalanobis"] = mahalanobis

    args["subset"] = subset
    args["subset_per"] = subset_per

    args["evaluate_flops"] = evaluate_flops

    args["start_TTA"] = start_TTA
    args["init_lr"] = init_lr
    args["weight_decay"] = weight_decay
    args["no_reset"] = no_reset
    args["enable_ema"] = enable_ema
    args["ema_beta"] = ema_beta
    args["kl_weight"] = kl_weight

    args["adapter_bottleneck"] = adapter_bottleneck
    args["ffn_adapter_scalar"] = ffn_adapter_scalar
    args["type_tta"] = type_tta
    args["shuffle"] = shuffle

    args["print_forget"] = print_forget

    exp_name = args["experiment_name"] + "_" + args["dataset"] + "-B"+str(args["init_cls"]) + "Inc" + str(args["increment"])
    wandb.init(
        # Set the project where this run will be logged
        project="SimpleCL",
        # Group experimnets
        group=args["exp_grp"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=exp_name,
        entity="imad-ma",
        # Track hyperparameters and run metadata
        config=args)

    np.random.seed(args["seed"][0])

    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--experiment_name', type=str, default="finetune linear adapters",
                        help='Name of the experiment.')
    parser.add_argument('--exp_grp', type=str, default="Linear Adapters",
                        help='Name of the experiment group.')
    parser.add_argument('--dataset', type=str, default="vtab",
                        help='Name of the dataset')
    parser.add_argument('--optimizer', type=str, default="sgd",
                        help='optimizer')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of epochs')
    parser.add_argument('--convnet_type', type=str, default="pretrained_vit_b16_224_in21k_adapter",
                        help='Type of the backbone model')
    parser.add_argument('--increment', type=int, default=10,
                        help='Increment of classes per task')
    parser.add_argument('--nb_classes', type=int, default=100,
                        help='Number of classes')

    parser.add_argument('--init_cls', type=int, default=10,
                        help='Initial classes')

    parser.add_argument('--adapter_size', type=int, default=64,
                        help='hidden size of the adapter')
    parser.add_argument('--outdir', type=str, default="outputs",
                        help='Save checkpoints')
    parser.add_argument('--rho', type=float, default=2,
                        help='Rho for SAM optimizer')
    parser.add_argument('--seed', type=int, default=10,
                        help='seed')
    parser.add_argument('--dset_variant', default='0', choices=['0', '1', '2', '3', '4'],
                            type=str, help='variant of the 5 dataset, start from different task')
    parser.add_argument('--layers', choices=['all', "adapter", "norm", "head"], default="norm",
                            type=str, help='Type of layers to use for test-time adaptation')
    parser.add_argument('--niter', type=int, default=1,
                        help='Number of iteration steps for adaptation')
    parser.add_argument('--selection_p', default=0.5, type=float,
                        help='confidence selection percentile')
    parser.add_argument('--reweight_entropy', default=False, action='store_true',
                        help='Reweight samples based on entropy score')
    parser.add_argument('--selected_idx', type=int, default=0,
                        help='Selected samples for adaptation with low entropy')
    parser.add_argument('--N_augment', type=int, default=8,
                        help='Number of augmented samples for adaptation')
    parser.add_argument('--augment_train', default=False, action='store_true')

    parser.add_argument('--batch_tta', default=False, action='store_true',
                        help='TTA on batch level, default=False')

    parser.add_argument('--random_aug', default=False, action='store_true',
                        help='Apply random augmentation at each iteration, default=False')

    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    parser.add_argument('--pretrained', action='store_true',
                        help="Use pretrained or not (default=False)")

    parser.add_argument('--start_TTA', type=int, default=1,
                        help='Test-time adaptation starting from this task')

    #learning params
    parser.add_argument('--init_lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay for the learning rate')


    parser.add_argument('--mahalanobis', default=False, action='store_true',
                        help='Use mahalanobis instead of the dot product for the cosine similarity')

    parser.add_argument('--subset', default=False, action='store_true',
                        help='Use mahalanobis instead of the dot product for the cosine similarity')

    parser.add_argument('--subset_per', type=int, default=40,
                        help='Test-time adaptation starting from this task')

    parser.add_argument('--evaluate_flops', default=False, action='store_true',
                        help='Evaluate inference time and FLOPS of the model')

    parser.add_argument('--no_reset', default=False, action='store_true',
                        help='Do no reset the model after TTA adaptation')

    parser.add_argument('--adapter_bottleneck', type=int, default=64,
                        help='Size of adapter bottleneck')

    parser.add_argument('--ffn_adapter_scalar', type=float, default=0.1,
                        help='Adapter scaling factor')

    parser.add_argument('--type_tta', type=str, default="memo", choices=['memo', 'sar'],
                        help='Type of TTA applied')

    parser.add_argument('--enable_ema', default=False, action='store_true',
                        help='Enable EMA for the model at phase 2')
    parser.add_argument('--ema_beta', type=float, default=0.90,
                        help='Beta for EMA')
    parser.add_argument('--kl_weight', type=float, default=0.5,
                        help='hyperparameter for the KL loss term')

    parser.add_argument('--shuffle', action='store_false',
                        help='Shuffle classes order, default=True')
    
    parser.add_argument('--print_forget', action='store_false', default=True,
                        help='Print forgetting, default=True')

    return parser

if __name__ == '__main__':
    main()