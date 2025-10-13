import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager, FiveDsets_DataManager
from utils.toolkit import count_parameters, compute_bwt, compute_fwt
import os

import wandb
import numpy as np
import sys
import time
from torch.utils.data import DataLoader

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    res_finals, res_avgs = [], []
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        res_final, res_avg = _train(args)
        res_finals.append(res_final)
        res_avgs.append(res_avg)

    logging.info('final accs: {}'.format(res_finals))
    logging.info('avg accs: {}'.format(res_avgs))
    wandb.log({"Final Acc": res_finals})


def _train(args):
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)

    dataset_lists = {"0": ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST'],
                    "1": ['MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST', 'SVHN'],
                    "2": ['CIFAR10', 'NotMNIST', 'FashionMNIST', 'SVHN', 'MNIST'],
                    "3": ['NotMNIST', 'FashionMNIST', 'SVHN', 'MNIST', 'CIFAR10'],
                    "4": ['FashionMNIST', 'SVHN', 'MNIST', 'CIFAR10', 'NotMNIST'],
                    }

    if args["dataset"] != "5datasets":
        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args=args
        )
    else:
        data_manager = FiveDsets_DataManager(
            "CIFAR10",
            args["shuffle"],
            args["seed"],
            args=args,
        )

        # default: ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        dataset_list = dataset_lists[args["dset_variant"]]  # Get the right variant of 5 datasets benchmark
        logging.info(
            "datasets are: {}".format(dataset_list)
        )


    print("args['model_name']: ", args["model_name"])
    model = factory.get_model(args["model_name"], args)
    print("model: ", model)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    cnn_baseline_performances = []  # Track baseline performances for CNN
    nme_baseline_performances = []  # Track baseline performances for NME
    
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        trainable_params = count_parameters(model._network, True)
        logging.info(
            "Trainable params: {}".format(trainable_params)
        )
        wandb.log({"Trainable params": trainable_params})

        if args["dataset"] == "5datasets":
            logging.info(
                "Current task: {}".format(dataset_list[task])
            )
            data_manager = FiveDsets_DataManager(
                dataset_list[task],
                args["shuffle"],
                args["seed"],
                args=args,
            )

        # Train the model on current task
        begin_task = time.time()
        model.incremental_train(data_manager)
        end_task = time.time()

        logging.info("Task {} training time: {}".format(task, end_task - begin_task))
        wandb.log({"Training Time": end_task - begin_task})

        if args["evaluate_flops"]:
            model.computer_FLOPS()
            model.profile()
            model.evaluate_adapt_inference()
            sys.exit(0)

        # Get baseline performance for next task (if exists)
        if task < data_manager.nb_tasks - 1:
            if args["dataset"] == "5datasets":
                next_data_manager = FiveDsets_DataManager(
                    dataset_list[task + 1],
                    args["shuffle"],
                    args["seed"],
                    args=args,
                )
            else:
                next_data_manager = copy.deepcopy(data_manager)
            
            # Store current data loaders
            current_train_loader = model.train_loader
            current_test_loader = model.test_loader
            
            # Get the range of classes for the next task
            start_class = model._known_classes
            end_class = start_class + next_data_manager.get_task_size(task + 1)
            
            # Set up data loaders for next task
            train_dataset = next_data_manager.get_dataset(
                np.arange(start_class, end_class),
                source="train",
                mode="train",
            )
            test_dataset = next_data_manager.get_dataset(
                np.arange(0, end_class),
                source="test",
                mode="test",
                subset=False
            )
            
            model.train_loader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, num_workers=8)
            model.test_loader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False, num_workers=8)
            
            # Get baseline performance for the next task
            baseline_cnn_accy, baseline_nme_accy = model.eval_task()
            
            # Restore original data loaders
            model.train_loader = current_train_loader
            model.test_loader = current_test_loader
            
            # Store baseline performances
            if baseline_cnn_accy is not None:
                cnn_baseline_performances.append(baseline_cnn_accy["top1"])
            if baseline_nme_accy is not None:
                nme_baseline_performances.append(baseline_nme_accy["top1"])

        # Evaluate current task
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            # Compute and log transfer metrics after each task
            # if len(cnn_matrix) > 1:  # Remove the baseline check since we handle it in the compute_fwt function
            #     cnn_bwt = compute_bwt(cnn_matrix)
            #     cnn_fwt = compute_fwt(cnn_matrix, cnn_baseline_performances)
            #     nme_bwt = compute_bwt(nme_matrix)
            #     nme_fwt = compute_fwt(nme_matrix, nme_baseline_performances)
                
            #     logging.info("CNN BWT: {}".format(cnn_bwt))
            #     logging.info("CNN FWT: {}".format(cnn_fwt))
            #     logging.info("NME BWT: {}".format(nme_bwt))
            #     logging.info("NME FWT: {}".format(nme_fwt))
                
            #     wandb.log({
            #         "CNN BWT": cnn_bwt,
            #         "CNN FWT": cnn_fwt,
            #         "NME BWT": nme_bwt,
            #         "NME FWT": nme_fwt
            #     })

            wandb.log({"top1": cnn_accy["top1"]})
            wandb.log({"top5": cnn_accy["top5"]})

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            wandb.log({"nme_top1": nme_curve["top1"]})
            wandb.log({"nme_top5": nme_curve["top5"]})

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            wandb.log({"Average Accuracy": sum(cnn_curve["top1"]) / len(cnn_curve["top1"])})
            wandb.log({"Average Accuracy(NME)": sum(nme_curve["top1"]) / len(nme_curve["top1"])})

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            # Compute and log CNN transfer metrics after each task
            # if len(cnn_matrix) > 1:  # Remove the baseline check since we handle it in the compute_fwt function
            #     cnn_bwt = compute_bwt(cnn_matrix)
            #     cnn_fwt = compute_fwt(cnn_matrix, cnn_baseline_performances)
                
            #     logging.info("CNN BWT: {}".format(cnn_bwt))
            #     logging.info("CNN FWT: {}".format(cnn_fwt))
                
            #     wandb.log({
            #         "CNN BWT": cnn_bwt,
            #         "CNN FWT": cnn_fwt
            #     })

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            wandb.log({"top1": cnn_accy["top1"]})
            wandb.log({"top5": cnn_accy["top5"]})

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            wandb.log({"Average Accuracy": sum(cnn_curve["top1"])/len(cnn_curve["top1"])})
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if args['print_forget']:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
            wandb.log({"Forgetting (CNN)": forgetting})

        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
            logging.info('Forgetting (NME): {}'.format(forgetting))
            wandb.log({"Forgetting (NME)": forgetting})

    return (cnn_curve['top1'][-1], np.array(cnn_curve['top1']).mean())
    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
