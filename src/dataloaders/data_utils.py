import os
import logging
import numpy as np
import math
import random

import h5py

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from torchvision import transforms, datasets
from PIL import Image, ImageFilter, ImageOps

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class HDF5_Dataset(Dataset):
    def __init__(self, root):

        self.root = root
        self.num_imgs = len(h5py.File(root, 'r')['labels'])

        with h5py.File(root, 'r') as f:
            self.targets = f['labels'][:]

        self.samples = np.arange(self.num_imgs)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):

        with h5py.File(self.root, 'r') as f:
            img = f['imgs'][index]
            target = f['labels'][index]

        return img, int(target)


class CustomDatasetHDF5(Dataset):
    """ Creates a custom pytorch dataset.
        - Creates two views of the same input used for unsupervised visual
        representational learning. (SimCLR, Moco, MocoV2)
    Args:
        data (array): Array / List of datasamples
        labels (array): Array / List of labels corresponding to the datasamples
        transforms (Dictionary, optional): The torchvision transformations
            to make to the datasamples. (Default: None)
        target_transform (Dictionary, optional): The torchvision transformations
            to make to the labels. (Default: None)
        two_crop (bool, optional): Whether to perform and return two views
            of the data input. (Default: False)
    Returns:
        img (Tensor): Datasamples to feed to the model.
        labels (Tensor): Corresponding lables to the datasamples.
    """

    def __init__(self, root, data, transform=None, target_transform=None):

        # shuffle the dataset
        idx = np.random.permutation(data.shape[0])

        self.data = data[idx]

        self.transform = transform
        self.target_transform = target_transform
        self.root = root

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        hdf_idx = self.data[index]

        with h5py.File(self.root, 'r') as f:
            image = f['imgs'][index]
            target = f['labels'][index]

        image = np.transpose(image, (1, 2, 0))

        image = Image.fromarray(image)

        if self.transform is not None:

            # Data augmentation and normalisation
            img = self.transform(image)

        if self.target_transform is not None:

            # Transforms the target, i.e. object detection, segmentation
            target = self.target_transform(target)

        target = torch.LongTensor(np.asarray(target, dtype=float))

        return img, target


def class_random_split(data, labels, n_classes, n_samples_per_class):
    """
    Creates a class-balanced validation set from a training set.
    args:
        data (Array / List): Array of data values or list of paths to data.
        labels (Array, int): Array of each data samples semantic label.
        n_classes (int): Number of Classes.
        n_samples_per_class (int): Quantity of data samples to be placed
                                    per class into the validation set.
    return:
        train / valid (dict): New Train and Valid splits of the dataset.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    if isinstance(data, torch.Tensor):
        # torch.stack transforms list of tensors to tensor
        return {'train': torch.stack(train_x), 'valid': torch.stack(valid_x)}, \
            {'train': torch.stack(train_y), 'valid': torch.stack(valid_y)}
    # transforms list of np arrays to tensor
    return {'train': torch.from_numpy(np.stack(train_x)),
            'valid': torch.from_numpy(np.stack(valid_x))}, \
        {'train': torch.from_numpy(np.stack(train_y)),
         'valid': torch.from_numpy(np.stack(valid_y))}


def random_split_image_folder(data, labels, n_classes, n_samples_per_class):
    """
    Creates a class-balanced validation set from a training set.
    Specifically for the image folder class.
    args:
        data (Array / List): Array of data values or list of paths to data.
        labels (Array, int): Array of each data samples semantic label.
        n_classes (int): Number of Classes.
        n_samples_per_class (int): Quantity of data samples to be placed
                                    per class into the validation set.
    return:
        train / valid (dict): New Train and Valid splits of the dataset.
    """

    train_x, train_y, valid_x, valid_y = [], [], [], []

    if isinstance(labels, list):
        labels = np.array(labels)

    for i in range(n_classes):
        # get indices of all class 'c' samples
        c_idx = (np.array(labels) == i).nonzero()[0]
        # get n unique class 'c' samples
        valid_samples = np.random.choice(c_idx, n_samples_per_class[i], replace=False)
        # get remaining samples of class 'c'
        train_samples = np.setdiff1d(c_idx, valid_samples)
        # assign class c samples to validation, and remaining to training
        train_x.extend(data[train_samples])
        train_y.extend(labels[train_samples])
        valid_x.extend(data[valid_samples])
        valid_y.extend(labels[valid_samples])

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    return {'train': train_x, 'valid': valid_x}, \
        {'train': torch.from_numpy(np.stack(train_y)), 'valid': torch.from_numpy(np.stack(valid_y))}

class GaussianBlur(object):
    """
    Gaussian blur augmentation:
        https://github.com/facebookresearch/moco/
    """

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img