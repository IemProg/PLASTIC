import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleCosineIncrementalNet, MultiBranchCosineIncrementalNet, SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

import wandb
import copy
import timeit
import math
from copy import deepcopy

from utils.third_party import aug
from PIL import Image
#from sklearn.covariance import LedoitWolf

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')

        if 'resnet' in args['convnet_type']:
            self._network = SimpleCosineIncrementalNet(args, args["pretrained"])
            self.batch_size = 128
            self.init_lr = args["init_lr"] if args["init_lr"] is not None else 0.01
        else:
            self._network = SimpleVitNet(args, args["pretrained"])
            self.batch_size = args["batch_size"]
            self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self.episodic = False
        self.margin_e0 = 0.4*math.log(1000)             # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = 0.2      # threshold e_m for model recovery scheme

        # to record the moving average of model output entropy, as model recovery criteria
        self.ema = self.args["enable_ema"]

    def set_trainable(self):
        for param in self._network.parameters():
            param.requires_grad = False

        if self.args["layers"] == "adapter":
            for name, param in self._network.named_parameters():
                if 'adapt' in name:
                    param.requires_grad = True

        elif self.args["layers"] == "norm":
            for name, param in self._network.named_parameters():
                if 'norm' in name:
                    param.requires_grad = True

        elif self.args["layers"] == "head":
            if self._network.fc != None:
                for name, param in self._network.fc.named_parameters():
                    if 'fc' in name:
                        param.requires_grad = True
                        print(name)

        elif self.args["layers"] == "all":
            for name, param in self._network.named_parameters():
                param.requires_grad = True
                print(name)
        else:
            raise NotImplementedError("Unknown layers: {}".format(self.args["layers"]))

    def after_task(self):
        if self.args["dataset"] == "5datasets":
            self._known_classes = 0
        else:
            self._known_classes = self._total_classes

    def marginal_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

    def tta_batch(self, images, n_augmentations, aug):
        all = []
        selected_idx = None if self.args["selected_idx"] == 0 else self.args["selected_idx"]
        for image in images:
            image = np.clip(image.cpu().numpy() * 255., 0, 255).astype(np.uint8).transpose(1, 2, 0)
            inputs = torch.stack([aug(Image.fromarray(image)) for _ in range(n_augmentations)])
            all.append(inputs)

        # Reshape them to be (B*N, C, H, W)
        N, C, H, W = inputs.shape
        all = torch.stack(all).reshape(-1, C, H, W).cuda()

        return all

    def replace_fc(self, trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model.convnet(data)['features'] #model(data)['features'] #
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

        # Save the new head weights
        self.head_weights = copy.deepcopy(self._network.fc.state_dict())

        return model
    
    def _eval_cnn_with_adapt_batch(self, loader):
        y_pred, y_true = [], []
        losses, losses_before = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            if not self.args["no_reset"]:
                ### Reset the student model each time.
                self._network.convnet.load_state_dict(self.init_state_dict, strict=True)
                # Load the head weights
                self._network.fc.load_state_dict(self.head_weights, strict=True)

            inputs = inputs.to(self._device)
            # with torch.no_grad():
            #     outputs = self._network(inputs)["logits"]
            #     loss_before = F.cross_entropy(outputs.to(self._device), targets.to(self._device))

            #     losses_before += loss_before.item()

            #TODO: set the layers to be trainable
            self.set_trainable()

            if self.args["type_tta"] == "sar":
                marg_loss, ema, reset_flag = self.forward_and_adapt_sar(inputs, self._network, self.optimizer,
                                                                        self.margin_e0,
                                                                        self.reset_constant_em, self.ema)
                if reset_flag:
                    self.reset()
                self.ema = ema  # update moving average value of loss
            else:
                marg_loss = self.adapt_batch(inputs)

            self._network.to(self._device).eval()
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                predicts = torch.topk(
                    outputs, k=self.topk, dim=1, largest=True, sorted=True
                )[1]  # [bs, topk]

                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())

                loss = F.cross_entropy(outputs.to(self._device), targets.to(self._device))
                losses += loss.item()

            wandb.log({"Marginal loss per sample": marg_loss / inputs.shape[0]})

        #wandb.log({"Cross-Entropy loss Before Adaptation": loss_before / len(loader)})
        wandb.log({"Cross-Entropy loss W Adaptation": losses / len(loader)})

        #print("Cross-Entropy Before adaptation: ", loss_before / len(loader))
        print("Cross-Entropy W adaptation: ", losses / len(loader))


        ## Apply EMA for each batch
        if self.args["enable_ema"]:
            #Idea is: the teacher model should be EMA, and the student should be reset each time.

            network_params = self.update_ema(self._network.convnet, old_model=self.teacher_model.convnet.state_dict(), beta=self.args["ema_beta"])
            self.teacher_model.convnet.load_state_dict(network_params)
            #network_params = self.update_ema(self._network.fc, old_model=self.head_weights, beta=self.args["ema_beta"])
            #self.head_weights = copy.deepcopy(network_params)

        return np.concatenate(y_pred, axis=0), np.concatenate(y_true, axis=0)  # [N, topk]

    def load_model_and_optimizer(self, model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self._network, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    def eval_task(self):
        # We start always from the same initial weights after the adaptation on Task-1
        self._network.convnet.load_state_dict(self.init_state_dict, strict=True)
        self._network.fc.load_state_dict(self.head_weights, strict=True)

        # Evaluation with no adaptation
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        wandb.log({"Testing Accuracy Before adaptation: ": cnn_accy["top1"]})
        print("Testing Accuracy Before adaptation: ", cnn_accy["top1"])

        if  self._cur_task >= self.args["start_TTA"]:
            ### Evaluation with test-time adaptation
            if self.args["batch_tta"]:
                y_pred, y_true = self._eval_cnn_with_adapt_batch(self.test_loader)
                cnn_accy = self._evaluate(y_pred, y_true)
            else:
                raise NotImplementedError("Test-time adaptation is not implemented for single image")
            wandb.log({"Testing Accuracy After adaptation: ": cnn_accy["top1"]})
            print("Testing Accuracy After adaptation: ", cnn_accy["top1"])

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def adapt_batch(self, image):
        ## if random augmentation is disabled, we use the same augmentation for all the iterations
        self.teacher_model.eval()  # Ensure teacher model is in evaluation mode

        if not self.args["random_aug"]:
            if self.args["N_augment"] != 0:
                # inputs = [aug(image) for _ in range(self.args["N_augment"])]
                inputs = self.tta_batch(image, self.args["N_augment"], aug=aug)
            else:                inputs = image

        for iteration in range(self.args["niter"]):   #niter=1, self.args["niter"]
            if self.args["random_aug"]:
                if self.args["N_augment"] != 0:
                    # inputs = [aug(image) for _ in range(self.args["N_augment"])]
                    inputs = self.tta_batch(image, self.args["N_augment"], aug=aug)
                else:
                    inputs = image


            self.optimizer.zero_grad()
            teacher_outputs = self.teacher_model(inputs)["logits"]
            outputs = self._network.extract_vector(inputs)              #["logits"]
            #outputs = self._network(inputs)["features"]

            norms = torch.norm(outputs, p=2, dim=0, keepdim=True) + 1e-7
            outputs = torch.div(outputs, norms) / 1.0     #self.t

            norms = torch.norm(self._network.fc.weight, p=2, dim=0, keepdim=True) + 1e-7
            protos = torch.div(self._network.fc.weight, norms) / 1.0     #self.t

            cosine_similarities = outputs@protos.T
            loss, logits = self.marginal_entropy(cosine_similarities)
            kl_loss = self.distillation_with_soft_targets(logits, teacher_outputs)
            
            loss = loss + self.args["kl_weight"] * kl_loss

            loss.backward()
            self.optimizer.step()


        return loss.item()

    def softmax_entropy(self, x):
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def update_ema(self, model, old_model, beta=0.99):
        if self.args["layers"] == "adapter":
            name_layer = "adapt"
        elif self.args["layers"] == "norm":
            name_layer = "norm"
        else:
            raise NotImplementedError("layers not implemented")
        
        with torch.no_grad():  # Ensure no gradient is computed for this operation.
            # Iterate over parameters by name to check if they belong to a LayerNorm layer
            for ema_name, ema_param in model.named_parameters():
                if name_layer in ema_name.lower():  # Check if the parameter is part of a LayerNorm layer
                    # Find the corresponding parameter in the source model by name
                    model_param = old_model[ema_name]
                    # Update the EMA model's parameter for the LayerNorm layer
                    ema_param.data.mul_(beta).add_(model_param.data, alpha=1 - beta)

        return model.state_dict()
    
    def distillation_with_soft_targets(self, student_logits, teacher_logits, temperature=1.0):
        return F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                        F.softmax(teacher_logits / temperature, dim=-1),
                        reduction='batchmean') * (temperature ** 2)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_sar(self, x, model, optimizer, margin, reset_constant, ema):
        """
        Forward and adapt model input data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        optimizer.zero_grad()
        # forward
        outputs = self._network.extract_vector(x)
        norms = torch.norm(outputs, p=2, dim=0, keepdim=True) + 1e-7
        outputs = torch.div(outputs, norms) / 1.0  # self.t
        norms = torch.norm(self._network.fc.weight, p=2, dim=0, keepdim=True) + 1e-7
        protos = torch.div(self._network.fc.weight, norms) / 1.0  # self.t
        outputs = outputs @ protos.T

        # Adapt: Filtering reliable samples/gradients for further adaptation; first time forward
        entropys = self.softmax_entropy(outputs)
        filter_ids_1 = torch.where(entropys < margin)
        entropys = entropys[filter_ids_1]
        loss = entropys.mean(0)
        loss.backward()
        optimizer.first_step(zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)

        # Second forward
        outputs = self._network.extract_vector(x)
        norms = torch.norm(outputs, p=2, dim=0, keepdim=True) + 1e-7
        outputs = torch.div(outputs, norms) / 1.0  # self.t
        norms = torch.norm(self._network.fc.weight, p=2, dim=0, keepdim=True) + 1e-7
        protos = torch.div(self._network.fc.weight, norms) / 1.0  # self.t
        outputs = outputs @ protos.T
        entropys2 = self.softmax_entropy(outputs)

        entropys2 = entropys2[filter_ids_1]  # second time forward
        loss_second_value = entropys2.clone().detach().mean(0)
        filter_ids_2 = torch.where(
            entropys2 < margin)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
        loss_second = entropys2[filter_ids_2].mean(0)
        if not np.isnan(loss_second.item()):
            ema = self.update_ema(ema, loss_second.item())  # record moving average loss values for model recovery

        # Second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
        loss_second.backward()
        optimizer.second_step(zero_grad=True)

        # Perform model recovery
        reset_flag = False
        # if ema is not None:
        #     if ema < 0.2:
        #         print("ema < 0.2, now reset the model")
        #         reset_flag = True

        return loss_second, ema, reset_flag #outputs, ema, reset_flag


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="train",
                                                subset = self.args["subset"], ratio = self.args["subset_per"])
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes),
                                                            source="test", mode="test",
                                                            subset=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                        num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                              source="train", mode="test",
                                                               subset=self.args["subset"], ratio=self.args["subset_per"])
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self.args['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                       weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            self.optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'asam' or self.args['optimizer'] == 'sam':
            from sam import SAM

            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            self.optimizer = SAM(self._network.parameters(), base_optimizer, lr=self.init_lr, momentum=0.9,
                                 weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args['tuned_epoch'],
                                                         eta_min=self.min_lr)

        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self._network, self.optimizer)

        if self._cur_task == 0:
            # show total parameters and trainable parameters
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            if total_params != total_trainable_params:
                for name, param in self._network.named_parameters():
                    if param.requires_grad:
                        print(name, param.numel())
            #self._init_train(train_loader, test_loader, self.optimizer, scheduler)

            # save the model weights after adapting on the first task
            self.init_state_dict = copy.deepcopy(self._network.convnet.state_dict())
            self.teacher_model = copy.deepcopy(self._network)
        else:
            pass
        self.replace_fc(train_loader_for_protonet, self._network, None)
        self.head_weights = copy.deepcopy(self._network.fc.state_dict())
        self.teacher_model.fc = copy.deepcopy(self._network.fc)

    def construct_dual_branch_network(self):
        network = MultiBranchCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network = network.to(self._device)

    def computer_FLOPS(self):
        from ptflops import get_model_complexity_info
        import re

        with torch.cuda.device(0):
            # Model thats already available
            net = self._network.convnet
            macs, params_ = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                      print_per_layer_stat=False, verbose=False)
            # Extract the numerical value
            flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
            # Extract the unit
            flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

            print('Computational complexity: {:<8}'.format(macs))
            print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
            print('Number of parameters: {:<8}'.format(params_))
            wandb.log({"MACS": macs})
            wandb.log({"FLOPS": flops})

    def profile(self):
        inputs = torch.randn(self.batch_size, 3, 224, 224)
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                self._network.convnet.cuda()
                self._network.convnet(inputs.cuda())

        print(prof.key_averages().table(sort_by="cuda_time_total"))
        index_cuda = prof.key_averages().table(sort_by="cuda_time_total").find("Self CUDA time total:")
        cuda_time = prof.key_averages().table(sort_by="cuda_time_total")[index_cuda+22:-3]
        wandb.log({"Time(CUDA)": cuda_time})


    def evaluate_adapt_inference(self):
        loader = self.test_loader
        _, inputs, targets = next(iter(loader))

        times = []
        for i in range(5):
            start = timeit.default_timer()
            self.adapt_batch(inputs)
            stop = timeit.default_timer()
            t = stop - start
            times.append(t)
        std = np.array(times).std()
        t = np.array(times).mean()

        print("Execution time (Mean): ", t)
        print("Execution time (STD): ", std)
        wandb.log({"Execution time (STD)": std})
        return wandb.log({"Execution time (Mean)": t})

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
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