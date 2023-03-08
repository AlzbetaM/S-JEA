import os
import logging
import numpy as np
import math

import h5py

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from torchvision import transforms

from dataloaders.cifar10 import *
from dataloaders.imagenet import *
from dataloaders.imagenette import *
from dataloaders.tinyimagenet import *
from dataloaders.stl import *


def get_dm(args):

    # init default datamodule
    if args.dataset == 'cifar10':
        args.img_dim = 32
        args.local_dim = 28

        dm = Cifar10_DataModule.from_argparse_args(args)
        dm.train_transforms = CifarTrainDataTransform(args)
        dm.val_transforms = CifarEvalLinTransform(args)
        dm.val_transforms_ssl = CifarEvalDataTransform(args)
        
        args.num_classes = dm.num_classes
        args.num_batches = len(dm.train_dataloader())

        dm_ft = Cifar10_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = CifarTrainLinTransform(args)
        dm_ft.val_transforms = CifarEvalLinTransform(args)
        dm_ft.test_transforms = CifarTestLinTransform(args)
        dm_ft.val_transforms_ssl = CifarEvalDataTransform(args)

    elif args.dataset == 'imagenet':
        args.img_dim = 224
        args.local_dim = 96
        
        dm = Imagenet_DataModule.from_argparse_args(args)
        dm.train_transforms = INTrainDataTransform(args)
        dm.val_transforms = INEvalLinTransform(args)
        dm.val_transforms_ssl = INEvalDataTransform(args)

        args.num_classes = dm.num_classes
        args.num_batches = len(dm.train_dataloader())

        dm_ft = Imagenet_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = INTrainLinTransform(args)
        dm_ft.val_transforms = INEvalLinTransform(args)
        dm_ft.test_transforms = INTestLinTransform(args)
        dm_ft.val_transforms_ssl = INEvalDataTransform(args)

    elif args.dataset == 'imagenette':
        args.img_dim = 128
        args.local_dim = 48
        
        dm = Imagenette_DataModule.from_argparse_args(args)
        dm.train_transforms = INTrainDataTransform(args)
        dm.val_transforms = INEvalLinTransform(args)
        dm.val_transforms_ssl = INEvalDataTransform(args)

        args.num_classes = dm.num_classes
        args.num_batches = len(dm.train_dataloader())

        dm_ft = Imagenette_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = INTrainLinTransform(args)
        dm_ft.val_transforms = INEvalLinTransform(args)
        dm_ft.test_transforms = INTestLinTransform(args)
        dm_ft.val_transforms_ssl = INEvalDataTransform(args)

    elif args.dataset == 'tinyimagenet':
        args.img_dim = 64
        args.local_dim = 32
        
        dm = TinyImagenet_DataModule.from_argparse_args(args)
        dm.train_transforms = INTrainDataTransform(args)
        dm.val_transforms = INEvalLinTransform(args)
        dm.val_transforms_ssl = INEvalDataTransform(args)

        args.num_classes = dm.num_classes
        args.num_batches = len(dm.train_dataloader())

        dm_ft = TinyImagenet_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = INTrainLinTransform(args)
        dm_ft.val_transforms = INEvalLinTransform(args)
        dm_ft.test_transforms = INTestLinTransform(args)
        dm_ft.val_transforms_ssl = INEvalDataTransform(args)

    elif args.dataset == 'stl10':
        args.img_dim = 64
        args.local_dim = 32
        
        dm = STL_DataModule.from_argparse_args(args)
        dm.train_transforms = STLTrainDataTransform(args)
        dm.val_transforms = STLEvalLinTransform(args)
        dm.val_transforms_ssl = STLEvalDataTransform(args)

        args.num_classes = dm.num_classes
        args.num_batches = len(dm.train_dataloader())

        dm_ft = STL_ft_DataModule.from_argparse_args(args)
        dm_ft.train_transforms = STLTrainLinTransform(args)
        dm_ft.val_transforms = STLEvalLinTransform(args)
        dm_ft.test_transforms = STLTestLinTransform(args)
        dm_ft.val_transforms_ssl = STLEvalDataTransform(args)
        
    return dm, dm_ft, args