import os
import numpy as np
from PIL import Image, ImageFilter
import random
import logging

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from PIL import ImageFilter, ImageOps

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, STL10

from .data_utils import GaussianBlur, Solarization, ImageFolderWithPaths, random_split_image_folder, class_random_split

class STL_DataModule(LightningDataModule):
    name = 'stl10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            subset: str = None,
            val_split: int = 50000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            strategy: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.size = (3, 96, 96)
        self.DATASET = STL10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.strategy = strategy
        self.subset = subset

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 10


    def prepare_data(self):
        self.DATASET(self.data_dir, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)

    def train_dataloader(self):
        
        # Get transformations
        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        # Dataset loader
        dataset_train = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'unlabelled'), transform=transf)

        # Train / Val split  
        # self.data, self.labels = utils.random_split_image_folder(data=self.train_dataset.samples,
        #                                                          labels=self.train_dataset.targets,
        #                                                          n_classes=1000,
        #                                                          n_samples_per_class=np.repeat(10, 1000).reshape(-1))

        # Loader
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )
        return loader

    def val_dataloader(self):

        # Get transformations
        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms
        transf_ssl = self.default_transforms() if self.val_transforms_ssl is None else self.val_transforms_ssl

        # Dataset loader
        dataset_train = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'train'), transform=transf)
        dataset_valid = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'test'), transform=transf)
        dataset_ssl = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'test'), transform=transf_ssl)

        # Train / Val split  
        # self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
        #                                                          labels=self.train_imagenet.targets,
        #                                                          n_classes=1000,
        #                                                          n_samples_per_class=np.repeat(10, 1000).reshape(-1))

        # Loader
        loader = DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
        loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
        loader_ssl = DataLoader(
            dataset_ssl,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
        
        loaders = [loader_ssl, loader_train, loader]
        
        return loaders

    def test_dataloader(self):

        # Get transformations
        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        # Dataset loader
        dataset_train = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'train'), transform=transf)
        dataset_test = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'test'), transform=transf)

        # Loader
        loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )
        loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )
        return [loader_train, loader]

    def default_transforms(self):
        default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return default_transforms

class STLTrainDataTransform(object):
    def __init__(self, args):

        color_jitter = transforms.ColorJitter(
            0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
        
        global_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, args.global_scale),                   
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])

        local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.local_dim, args.local_scale),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        
        self.global_transforms = global_transforms
        self.local_transforms = local_transforms 
        
        self.global_views = args.global_views
        self.local_views = args.local_views

    def __call__(self, sample):
        img_views = []

        if self.global_views > 0:
            img_views += [self.global_transforms(sample) for i in range(self.global_views)]

        if self.local_views > 0:
            img_views += [self.local_transforms(sample) for i in range(self.local_views)]
        
        return img_views

class STL_ft_DataModule(LightningDataModule):
    name = 'stl10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            subset: str = None,
            val_split: int = 50000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            strategy: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.size = (3, 96, 96)
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.strategy = strategy
        self.subset = subset

    @property
    def num_classes(self):
        return 10

    def train_dataloader(self):
        
        # Get transformations
        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        # Dataset loader
        dataset_train = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'train'), transform=transf)

        # Train / Val split  
        # self.data, self.labels = utils.random_split_image_folder(data=self.train_dataset.samples,
        #                                                          labels=self.train_dataset.targets,
        #                                                          n_classes=1000,
        #                                                          n_samples_per_class=np.repeat(10, 1000).reshape(-1))

        # Loader
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )
        return loader

    def val_dataloader(self):

        # Get transformations
        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms
        transf_ssl = self.default_transforms() if self.val_transforms_ssl is None else self.val_transforms_ssl

        # Dataset loader
        dataset_train = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'train'), transform=transf)
        dataset_valid = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'test'), transform=transf)
        dataset_ssl = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'test'), transform=transf_ssl)

        # Train / Val split  
        # self.data, self.labels = utils.random_split_image_folder(data=self.train_imagenet.samples,
        #                                                          labels=self.train_imagenet.targets,
        #                                                          n_classes=1000,
        #                                                          n_samples_per_class=np.repeat(10, 1000).reshape(-1))

        # Loader
        loader = DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
        loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
        loader_ssl = DataLoader(
            dataset_ssl,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
        
        loaders = [loader_ssl, loader_train, loader]
        
        return loaders

    def test_dataloader(self):

        # Get transformations
        transf = self.default_transforms() if self.test_transforms is None else self.test_transforms

        # Dataset loader
        dataset_train = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'train'), transform=transf)
        dataset_test = ImageFolderWithPaths(root=os.path.join(self.data_dir, 'test'), transform=transf)

        # Loader
        loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )
        loader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            sampler=None
        )
        return [loader_train, loader]

    def default_transforms(self):
        default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return default_transforms

class STLTrainDataTransform(object):
    def __init__(self, args):

        color_jitter = transforms.ColorJitter(
            0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
        
        global_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, args.global_scale),                   
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])

        local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.local_dim, args.local_scale),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        
        self.global_transforms = global_transforms
        self.local_transforms = local_transforms 
        
        self.global_views = args.global_views
        self.local_views = args.local_views

    def __call__(self, sample):
        img_views = []

        if self.global_views > 0:
            img_views += [self.global_transforms(sample) for i in range(self.global_views)]

        if self.local_views > 0:
            img_views += [self.local_transforms(sample) for i in range(self.local_views)]
        
        return img_views


class STLEvalDataTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((int(args.img_dim//0.8), int(args.img_dim//0.8)), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((args.img_dim, args.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.test_transform = test_transform
        self.global_views = args.global_views
        self.local_views = args.local_views

    def __call__(self, sample):
        img_views = []
        
        if self.global_views > 0:
            img_views += [self.test_transform(sample) for i in range(self.global_views)]
        if self.local_views > 0:
            img_views += [self.test_transform(sample) for i in range(self.local_views)]

        return img_views


class STLTrainLinTransform(object):
    def __init__(self, args):

        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((args.img_dim, args.img_dim), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class STLEvalLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.Resize((int(args.img_dim//0.8), int(args.img_dim//0.8)), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((args.img_dim, args.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class STLTestLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([
            transforms.Resize((int(args.img_dim//0.8), int(args.img_dim//0.8)), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((args.img_dim, args.img_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43, 0.42, 0.39],
                                 std=[0.27, 0.26, 0.27])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x