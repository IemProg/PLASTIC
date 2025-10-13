import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels

from typing import Any, Tuple

from .dataset_utils import read_image_file, read_label_file
import torch

import os, sys

from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args=None):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t

class iMNIST(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        #transforms.Lambda(lambda x: x.convert('RGB')),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.MNIST("./data", train=True, download=True)
        test_dataset = datasets.MNIST("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iSVHN(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
                    #toTensor
                    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.SVHN("./data", split="train", download=True)
        test_dataset = datasets.SVHN("./data", split="test", download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.labels
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.labels
        )

class iFashionMNIST(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [transforms.Lambda(lambda x: x.repeat(3, 1, 1))]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.FashionMNIST("./data", train=True, download=True)
        test_dataset = datasets.FashionMNIST("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iNotMNIST(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
    ]

    fpath_test = os.path.join("./data/NotMNIST", 'test')

    class_order = np.arange(10).tolist()

    def get_data(self, fpath):
        folders = os.listdir(fpath)
        X, Y = [], []

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        return np.array(X), np.array(Y)

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/NotMNIST/train/"
        test_dir = "./data/NotMNIST/test/"

        #train_dset = datasets.ImageFolder(train_dir)
        #self.train_data, self.train_targets = split_images_labels(train_dset.imgs)

        fpath_train = os.path.join("./data/NotMNIST", 'train')
        self.train_data, self.train_targets = self.get_data(fpath_train)

        #test_dset = datasets.ImageFolder(test_dir)
        #self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        fpath_test = os.path.join("./data/NotMNIST", 'test')
        self.test_data, self.test_targets = self.get_data(fpath_test)


class iCIFAR10(iData):
    use_path = False
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iCIFAR224(iData):
    use_path = False

    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [
        # transforms.ToTensor(),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]


    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCIFAR10_C(iData):
    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(10).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)
        #NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # test_trsf = transforms.Compose([transforms.RandomCrop(32, padding=4),
        #                                     transforms.RandomHorizontalFlip(),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(*NORM)])

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

        tesize = 10000
        corruption = 'gaussian_noise'
        level = 5


        import torchvision

        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR-10-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_brightness(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'brightness'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_fog(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'fog'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_frost(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'frost'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_snow(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'snow'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_impulse(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'impulse_noise'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_zoom_blur(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'zoom_blur'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )


class iCIFAR100_C_motion_blur(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(100).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'motion_blur'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_glass_blur(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(10).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'glass_blur'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )


class iCIFAR100_C_defocus_blur(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(10).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'defocus_blur'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class iCIFAR100_C_ShotNoise(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(10).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'shot_noise'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )


class iCIFAR100_C(iData):

    use_path = False

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(10).tolist()
    def download_data(self):
        train_trsf = build_transform(True, None)
        test_trsf = build_transform(False, None)

        common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                                   'snow', 'frost', 'fog', 'brightness',
                                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption = 'gaussian_noise'
        level = 5
        tesize = 10000

        import torchvision

        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)

        if corruption == 'original':
            print('Test on the original test set')
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)
        elif corruption in common_corruptions:
            print(f'Test on {corruption} level {level}')
            teset_raw = np.load("./data" + f'/CIFAR100-C/{corruption}.npy')
            teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
            test_dset = torchvision.datasets.CIFAR100(root="./data", train=False,
                                                 download=True, transform=test_trsf)

            #test_dset = datasets.ImageFolder(teset_raw)
            test_dset.data = torch.from_numpy(teset_raw)
        else:
            raise Exception('Corruption not found!')

        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        #self.test_data, self.test_targets = split_images_labels(test_dset.data)

        self.test_data, self.test_targets = test_dset.data, np.array(
            test_dset.targets
        )

class vtab(iData):
    use_path = True
    
    train_trsf=build_transform(True, None)
    test_trsf=build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class resisc(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/seq_vtab/resisc/train/"
        test_dir = "./data/seq_vtab/resisc/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class dtd(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/seq_vtab/dtd/train/"
        test_dir = "./data/seq_vtab/dtd/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class pets(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/seq_vtab/pets/train/"
        test_dir = "./data/seq_vtab/pets/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class eurosat(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/seq_vtab/eurosat/train/"
        test_dir = "./data/seq_vtab/eurosat/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class flowers(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/seq_vtab/flowers/train/"
        test_dir = "./data/seq_vtab/flowers/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class ivtab_C_gaussian(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(50).tolist()
    def download_data(self):
        corruption = 'gaussian_noise'
        level = 5

        train_dir = "./data/vtab-cil/vtab/train/"
        suffix_name = corruption + "/Level"+str(level)
        test_dir = "./data/vtab-C/test/"+suffix_name

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        print("we are here")

class ivtab_C_shot(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(50).tolist()

    def download_data(self):
        corruption = 'shot_noise'
        level = 5

        train_dir = "./data/vtab-cil/vtab/train/"
        suffix_name = corruption + "/Level" + str(level)
        test_dir = "./data/vtab-C/test/" + suffix_name

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class ivtab_C_impulse(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(50).tolist()

    def download_data(self):
        corruption = 'impulse_noise'
        level = 5

        train_dir = "./data/vtab-cil/vtab/train/"
        suffix_name = corruption + "/Level" + str(level)
        test_dir = "./data/vtab-C/test/" + suffix_name

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iOmnibenchmark_gaussian(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(50).tolist()
    def download_data(self):
        corruption = 'gaussian_noise'
        level = 5

        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"

        suffix_name = corruption + "/Level"+str(level)
        test_dir = "./data/omnibenchmark-C/test/"+suffix_name

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iOmnibenchmark_shot(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(50).tolist()
    def download_data(self):
        corruption = 'shot_noise'
        level = 5

        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"

        suffix_name = corruption + "/Level"+str(level)
        test_dir = "./data/omnibenchmark-C/test/"+suffix_name

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iOmnibenchmark_impulse(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)

    common_trsf = []
    class_order = np.arange(50).tolist()
    def download_data(self):
        corruption = 'impulse_noise'
        level = 5

        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"

        suffix_name = corruption + "/Level"+str(level)
        test_dir = "./data/omnibenchmark-C/test/"+suffix_name

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)