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
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, STL10

from .data_utils import GaussianBlur, Solarization, random_split_image_folder, class_random_split

class Cifar10_DataModule(LightningDataModule):
    name = 'cifar10'
    extra_args = {}

    def __init__(
            self,
            data_dir: str = None,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            strategy: str = 'ddp',
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.size = (3, 32, 32)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_samples = 60000 - val_split
        self.strategy = strategy

        print("\n\n batch_size in dataloader:{}".format(self.batch_size))

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        self.DATASET(self.data_dir, train=True, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True,
                     transform=transforms.ToTensor(), **self.extra_args)

    def train_dataloader(self):
        
        # Get transformations
        transf = self.default_transforms() if self.train_transforms is None else self.train_transforms

        # Dataset loader
        dataset = self.DATASET(self.data_dir, train=True, download=True,
                               transform=transf, **self.extra_args)
        
        # Train / Val split                        
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Loader
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):

        # Get transformations
        transf = self.default_transforms() if self.val_transforms is None else self.val_transforms
        transf_ssl = self.default_transforms() if self.val_transforms_ssl is None else self.val_transforms_ssl
        
        # Dataset loader
        dataset_train = self.DATASET(self.data_dir, train=True, download=True,
                               transform=transf, **self.extra_args)
        
        dataset = self.DATASET(self.data_dir, train=False, download=True,
                               transform=transf, **self.extra_args)
        
        dataset_ssl = self.DATASET(self.data_dir, train=False, download=True,
                               transform=transf_ssl, **self.extra_args)
        
        # Train / Val split    
        train_length = len(dataset_train)
        _, dataset_val = random_split(
            dataset_train,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        loader = DataLoader(
            dataset_val,
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
        dataset = self.DATASET(self.data_dir, train=False, download=True,
                               transform=transf, **self.extra_args)

        dataset_train = self.DATASET(self.data_dir, train=True, download=True,
                               transform=transf, **self.extra_args)

        loader = DataLoader(
            dataset,
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
        cf10_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        return cf10_transforms

# Transforms

class CifarTrainDataTransform(object):
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
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])

        local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.local_dim, args.local_scale),
            transforms.RandomApply([color_jitter], p=args.jitter_p),
            transforms.RandomGrayscale(p=args.grey_p),
            # transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
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


class CifarEvalDataTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
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


class CifarTrainLinTransform(object):
    def __init__(self, args):

        data_transforms = transforms.Compose([  # transforms.ToPILImage(),
            transforms.RandomResizedCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.train_transform = data_transforms

    def __call__(self, sample):
        transform = self.train_transform
        x = transform(sample)
        return x


class CifarEvalLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([  # transforms.ToPILImage(),
            transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x


class CifarTestLinTransform(object):
    def __init__(self, args):

        test_transform = transforms.Compose([
            transforms.CenterCrop((32 * 0.875, 32 * 0.875)),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                 std=[0.24703223, 0.24348513, 0.26158784])
        ])
        self.test_transform = test_transform

    def __call__(self, sample):
        transform = self.test_transform
        x = transform(sample)
        return x