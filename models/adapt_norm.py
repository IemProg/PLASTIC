import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet, SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

from utils.third_party import aug
from PIL import Image
from sklearn.covariance import LedoitWolf

import wandb
import copy

num_workers = 4


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8

        self.set_trainable()

    def set_trainable(self):
        for param in self._network.parameters():
            param.requires_grad = False

        if self.args["layers"] == "adapter":
            for name, param in self._network.named_parameters():
                if 'adapt' in name:
                    param.requires_grad = True
                    #print(name)

        elif self.args["layers"] == "norm":
            for name, param in self._network.named_parameters():
                if 'norm' in name:
                    param.requires_grad = True
                    #print(name)

        elif self.args["layers"] == "head":
            if self._network.fc != None:
                for name, param in self._network.fc.named_parameters():
                    if 'fc' in name:
                        param.requires_grad = True
                        #print(name)

        elif self.args["layers"] == "all":
            for name, param in self._network.named_parameters():
                param.requires_grad = True
                #print(name)
        else:
            raise NotImplementedError("Unknown layers: {}".format(self.args["layers"]))

    def after_task(self):
        if self.args["dataset"] == "5datasets":
            self._known_classes = 0
        else:
            self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                # if self.args["augment_train"]:
                #     data = [aug(data) for _ in range(self.args["N_augment"])]
                #     data = torch.cat(data, dim=0)
                #     label = label.repeat(self.args["batch_size"])
                data = data.cuda()
                label = label.cuda()
                embedding = model.convnet(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"],
                                       shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.args["batch_size"],
                                                    shuffle=True,
                                                    num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)

        self.set_trainable()
        if self.args['optimizer'] == 'sgd':
            # self.optimizer = optim.SGD([{"params": self._network.convnet.parameters()},
            #                             {"params": self._network.fc.parameters(), 'lr':0.01}],
            #                             momentum=0.9, lr=0.0001,
            #                             weight_decay=self.weight_decay)
            self.optimizer = optim.SGD(self._network.convnet.parameters(),
                                        momentum=0.9, lr=0.0001,
                                        weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr,
                                    weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'asam':
            from sam import SAM

            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            self.optimizer = SAM(self._network.parameters(), base_optimizer, lr=self.init_lr, momentum=0.9,
                            weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args['tuned_epoch'],
                                                         eta_min=self.min_lr)

        if self._cur_task == 0:
            self._init_train(train_loader, test_loader, self.optimizer, scheduler)
        else:
            pass

        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))

        self.set_trainable()

        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                optimizer.zero_grad()

                if self.args['optimizer'] == 'asam' or self.args['optimizer'] == 'sam':
                    # first forward-backward pass
                    loss = F.cross_entropy(logits, targets)
                    loss.backward()
                    # optimizer.step()
                    optimizer.first_step(zero_grad=True)  # using SAM

                    # second forward-backward pass
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)  # make sure to do a full forward pass
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            wandb.log({"training loss": losses})

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            wandb.log({"Training Accuracy": train_acc})

            test_acc = self._compute_accuracy(self._network, test_loader)

            wandb.log({"Test Accuracy": test_acc})

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)