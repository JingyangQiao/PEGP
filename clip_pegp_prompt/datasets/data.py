import os.path
import torch
import numpy as np
from torchvision import datasets, transforms


class iData():
    train_trans = []
    test_trans = []
    common_trans = []
    class_order = None


class iCifar100(iData):
    scale = (0.05, 1.)
    ratio = (3./4., 4./3.)
    train_trans = [
        transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    test_trans = [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
    common_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]

    def __init__(self):
        class_order = np.arange(100).tolist()
        self.class_order = class_order


class ImageNet100(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expandvars(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.filename = 'imagenet-100'
        self.fpath = os.path.join(root, self.filename)

        if not os.path.isfile(self.fpath):
            if not download:
                raise RuntimeError('Dataset not found. You can use download=True to download it')

        if not os.path.exists(os.path.join(root, 'imagenet-100')):
            import zipfile
            zip_ref = zipfile.ZipFile(os.path.join(root, 'imagenet-100.rar'), 'r')
            zip_ref.extractall(os.path.join(root))
            zip_ref.close()

        if self.train:
            fpath = self.fpath + '/train'

        else:
            fpath = self.fpath + '/val'

        self.dataset = datasets.ImageFolder(fpath, transform=transform)
        self.data_path, self.targets = [], []
        for i in range(len(self.dataset.imgs)):
            self.data_path.append(self.dataset.imgs[i][0])
            self.targets.append(self.dataset.imgs[i][1])


class iImageNet100(iData):
    scale = (0.05, 1.)
    ratio = (3./4., 4./3.)
    train_trans = [
        transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    test_trans = [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
    common_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    def __init__(self):
        class_order = np.arange(100).tolist()
        self.class_order = class_order

