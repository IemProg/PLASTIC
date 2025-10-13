import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, \
    CosineLinear, CosineLinearCov, SimpleContinualLinear
import timm
from convs.vit import *


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    print("name: ", name)
    # Resnet
    if name == "pretrained_resnet18":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model = resnet18(pretrained=False, args=args)
        #model.load_state_dict(torch.load("./pretrained_models/resnet18-f37072fd.pth"), strict=False)
        return model.eval()
    elif name == "pretrained_resnet50":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model = resnet50(pretrained=False, args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet50-11ad3fa6.pth"), strict=False)
        return model.eval()
    elif name == "pretrained_resnet101":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model = resnet101(pretrained=False, args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet101-cd907fc2.pth"), strict=False)
        return model.eval()
    elif name == "pretrained_resnet152":
        from convs.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
        model = resnet152(pretrained=False, args=args)
        model.load_state_dict(torch.load("./pretrained_models/resnet152-f82ba261.pth"), strict=False)
        return model.eval()
    elif name == 'vit-b-p16':
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        # del model.head
        # del model.norm
        model.norm = nn.LayerNorm(768)
        model.out_dim = 768
        return model.eval()

    elif '_adapter' in name:
        from convs import vision_transformer_adapter
        from convs import vits_adapter
        from easydict import EasyDict
        ffn_num = args["ffn_num"]
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )

        if args["model_name"] == "adam_adapter" or args["model_name"] == "ranpac" or args["model_name"] == "adapt_tta" or args["model_name"] == "fecam":
            if name == "pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(num_classes=0,
                                                                                global_pool=False, drop_path_rate=0.0,
                                                                                tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                                                                                      global_pool=False,
                                                                                      drop_path_rate=0.0,
                                                                                      tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "mocov3_adapter":
                model = vits_adapter.vit_base_patch16_224_mocov3_adapter(pretrained=True, global_pool=False,
                                                                    drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "dinov2_adapter":
                model = vits_adapter.vit_base_patch14_dinov2_adapter(pretrained=True, global_pool=False,
                                                                   drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "clip_adapter":
                model = vits_adapter.vit_base_patch16_clip_224_adapter(pretrained=True, global_pool=False,
                                                                   drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "sam_adapter":
                model = vits_adapter.vit_base_patch16_224_sam_adapter(pretrained=True, global_pool=False,
                                                               drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            elif name == "mae_adapter":
                model = vits_adapter.vit_base_patch16_224_mae_adapter(pretrained=True, global_pool=False,
                                                              drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    # SimpleCIL or SimpleCIL w/ Finetune
    elif name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif name == "mocov3":
        model = vit_base_patch16_224_mocov3(pretrained=True)
        model.out_dim = 768
        return model.eval()
    elif name == "dinov2":
        model = vit_base_patch14_dinov2(pretrained=True)
        model.out_dim = 768
        return model.eval()
    elif name == "clip":
        model = vit_base_patch16_clip_224(pretrained=True)
        model.out_dim = 768
        return model.eval()
    elif name == "sam":
        model = vit_base_patch16_224_sam(pretrained=True)
        model.out_dim = 768
        return model.eval()
    elif name== "mae":
        model = vit_base_patch16_224_mae(pretrained=True)
        model.out_dim = 768
        return model.eval()
    # Coda_Prompt
    elif '_coda_prompt' in name:
        from convs import vit_coda_promtpt
        model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"])
        # model = vision_transformer_coda_prompt.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
        #                 num_heads=12, ckpt_layer=0,
        #                 drop_path_rate=0)
        # from timm.models import vit_base_patch16_224
        # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
        # del load_dict['head.weight']; del load_dict['head.bias']
        # model.load_state_dict(load_dict)
        return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))


def load_state_vision_model(model, ckpt_path):
    ckpt_state = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt_state:
        # our upstream converted checkpoint
        ckpt_state = ckpt_state['state_dict']
        prefix = ''
    elif 'model' in ckpt_state:
        # prototype checkpoint
        ckpt_state = ckpt_state['model']
        prefix = 'module.'
    else:
        # official checkpoint
        prefix = ''

    logger = logging.getLogger('global')
    if ckpt_state:
        logger.info('==> Loading model state "{}XXX" from pre-trained model..'.format(prefix))

        own_state = model.state_dict()
        state = {}
        for name, param in ckpt_state.items():
            if name.startswith(prefix):
                state[name[len(prefix):]] = param
        success_cnt = 0
        for name, param in state.items():
            if name in own_state:
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    if isinstance(param, bool):
                        own_state[name] = param
                    else:
                        # normal version
                        own_state[name].copy_(param)
                    success_cnt += 1
                except Exception as err:
                    logger.warn(err)
                    logger.warn('while copying the parameter named {}, '
                                'whose dimensions in the model are {} and '
                                'whose dimensions in the checkpoint are {}.'
                                .format(name, own_state[name].size(), param.size()))
                    logger.warn("But don't worry about it. Continue pretraining.")
        ckpt_keys = set(state.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        logger.info('Successfully loaded {} key(s) from {}'.format(success_cnt, ckpt_path))
        for k in missing_keys:
            logger.warn('Caution: missing key from checkpoint: {}'.format(k))
        redundancy_keys = ckpt_keys - own_keys
        for k in redundancy_keys:
            logger.warn('Caution: redundant key from checkpoint: {}'.format(k))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.args = args
        print('This is for the BaseNet initialization.')
        self.convnet = get_convnet(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc

class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class SimpleVitRanPAC(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        # for RanPAC
        self.W_rand = None
        self.RP_dim = None

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = x.cuda()
        x = self.convnet(x)["features"]
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = self.fc(x)
        out.update({"features": x})
        return out


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data

            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])

            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        # out.update(x)
        return out


class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

        # no need the convnet.

        print(
            'Clear the convnet in MultiBranchCosineIncrementalNet, since we are using self.convnets with dual branches')
        self.convnet = torch.nn.Identity()

        for param in self.convnet.parameters():
            param.requires_grad = False

        self.convnets = nn.ModuleList()
        self.args = args

        if 'resnet' in args['convnet_type']:
            self.modeltype = 'cnn'
        else:
            self.modeltype = 'vit'

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.modeltype == 'cnn':
            features = [convnet(x)["features"] for convnet in self.convnets]
            features = torch.cat(features, 1)
            # import pdb; pdb.set_trace()
            out = self.fc(features)
            out.update({"features": features})
            return out
        else:
            features = [convnet(x)["features"] for convnet in self.convnets]
            # We concatenate the features of adapters space with the features of the original space
            #print("features", [item.shape for item in features])
            features = torch.cat(features, 1)
            # import pdb; pdb.set_trace()
            out = self.fc(features)
            out.update({"features": features})
            return out

    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_ssf', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))  # pretrained model without scale
        elif 'vpt' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_vpt', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))  # pretrained model without vpt
        elif 'adapter' in self.args['convnet_type']:
            if "linear" in self.args['convnet_type']:
                newargs = copy.deepcopy(self.args)
                newargs['convnet_type'] = newargs['convnet_type'].replace('_linearadapter', '')
                print(newargs['convnet_type'])
                self.convnets.append(get_convnet(newargs))  # pretrained model without adapter
            else:
                newargs = copy.deepcopy(self.args)
                newargs['convnet_type'] = newargs['convnet_type'].replace('_adapter', '')
                print(newargs['convnet_type'])
                self.convnets.append(get_convnet(newargs))  # pretrained model without adapter
        else:
            self.convnets.append(get_convnet(self.args))  # the pretrained model itself

        self.convnets.append(tuned_model.convnet)  # adapted tuned model

        self._feature_dim = self.convnets[0].out_dim * len(self.convnets)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])

    def construct_multi_branch_network(self, all_tuned_models):
        if 'ssf' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_ssf', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))  # pretrained model without scale
        elif 'vpt' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_vpt', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))  # pretrained model without vpt
        elif 'adapter' in self.args['convnet_type']:
            newargs = copy.deepcopy(self.args)
            newargs['convnet_type'] = newargs['convnet_type'].replace('_adapter', '')
            print(newargs['convnet_type'])
            self.convnets.append(get_convnet(newargs))  # pretrained model without adapter
        else:
            self.convnets.append(get_convnet(self.args))  # the pretrained model itself

        for tuned_model in all_tuned_models:
            self.convnets.append(tuned_model.convnet)  # adapted tuned model

        self._feature_dim = self.convnets[0].out_dim * len(self.convnets)
        self.fc = self.generate_fc(self._feature_dim, self.args['init_cls'])

class FinetuneIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, fc_with_ln=False):
        super().__init__(convnet_type, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln


    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.convnet(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy()
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy()
        return features


    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.convnet(x)
        else:
            x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)

        return out