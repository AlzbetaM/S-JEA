#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import warnings
from typing import List

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Torch
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import neptune


class PTPrintingCallback(pl.Callback):
    ''' Callback to handle all metric printing and visualisations during self-supervised pretraining '''

    def __init__(self, path, args):
        self.args = args

        if self.args.num_nodes > 1:
            self.num_devices = self.args.num_nodes * self.args.devices
        else:
            self.num_devices = 1

        self.cur_loss = 10000000.0
        self.path = path

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # loss = np.mean(pl_module.train_loss) # /self.num_devices
        epoch = trainer.current_epoch
        loss = trainer.callback_metrics['loss_epoch']

        if self.cur_loss > loss:
            self.cur_loss = loss

            if self.path != None:
                save_path = os.path.join(self.path, ('best_epoch.ckpt'))

                trainer.save_checkpoint(save_path)

        pl_module.train_loss = []

        print("\n [Train] Avg Loss: {:.4f}".format(loss))

        for param_group in trainer.optimizers[0].param_groups:
            lr = param_group['lr']

        pl_module.logger.experiment['train/lr_epoch'].log(lr)
        pl_module.logger.experiment['train/loss_epoch'].log(loss)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # loss = np.mean(pl_module.valid_loss)#/self.num_devices
        epoch = trainer.current_epoch

        loss = trainer.callback_metrics['valid_loss/dataloader_idx_0']

        pl_module.valid_loss = []

        print("\n Epoch: {}".format(epoch))
        print("\n [Valid] Avg Loss: {:.4f}, Avg KNN Acc: {:.4f}".format(loss, pl_module.val_knn))

        pl_module.logger.experiment['valid/loss_epoch'].log(loss)
        pl_module.logger.experiment['valid/knn_acc'].log(pl_module.val_knn)
        pl_module.logger.experiment['valid/knn_acc1'].log(pl_module.val_knn1)
        pl_module.logger.experiment['valid/knn_acc2'].log(pl_module.val_knn2)
        pl_module.logger.experiment['valid/knn_acc3'].log(pl_module.val_knn3)
        pl_module.logger.experiment['valid/knn_acc4'].log(pl_module.val_knn4)
        if pl_module.hparams.stacked == 2:
            pl_module.logger.experiment['valid/knn_acc_s'].log(pl_module.val_knn_stacked)
            pl_module.logger.experiment['valid/knn_acc_s1'].log(pl_module.val_knn_stacked1)
            pl_module.logger.experiment['valid/knn_acc_s2'].log(pl_module.val_knn_stacked2)
            pl_module.logger.experiment['valid/knn_acc_s3'].log(pl_module.val_knn_stacked3)
            pl_module.logger.experiment['valid/knn_acc_s4'].log(pl_module.val_knn_stacked4)


class FTPrintingCallback(pl.Callback):
    ''' Callback to handle all metric printing and visualisations during self-supervised linear evaluation '''

    def __init__(self, path, args):
        self.args = args
        if self.args.num_nodes > 1:
            self.num_devices = self.args.num_nodes * self.args.devices
        else:
            self.num_devices = 1

        self.cur_loss = 1E+24
        self.path = path

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        loss = np.mean(pl_module.train_loss)  # /self.num_devices
        acc = np.mean(pl_module.train_acc)  # /self.num_devices
        t5 = np.mean(pl_module.train_t5)  # /self.num_devices

        pl_module.train_loss = []
        pl_module.train_acc = []
        pl_module.train_t5 = []

        print("\n [Train] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg top5: {:.4f}".format(
            loss, acc, t5))

        pl_module.logger.experiment['ft_train/t5_epoch'].log(t5)
        pl_module.logger.experiment['ft_train/loss_epoch'].log(loss)
        pl_module.logger.experiment['ft_train/acc_epoch'].log(acc)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        loss = np.mean(pl_module.valid_loss)  # /self.num_devices
        acc = np.mean(pl_module.valid_acc)  # /self.num_devices
        t5 = np.mean(pl_module.valid_t5)  # /self.num_devices

        epoch = trainer.current_epoch

        if self.cur_loss > loss:
            self.cur_loss = loss

            if self.path != None:
                save_path = os.path.join(self.path, ('best_epoch.ckpt'))

                print("!! Saving to !! :{} ".format(save_path))

                trainer.save_checkpoint(save_path)

        pl_module.valid_loss = []
        pl_module.valid_acc = []
        pl_module.valid_t5 = []

        print("\n Epoch: {}".format(epoch))
        print("\n [Valid] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg tckptop5: {:.4f}".format(
            loss, acc, t5))

        pl_module.logger.experiment['ft_valid/t5_epoch'].log(t5)
        pl_module.logger.experiment['ft_valid/loss_epoch'].log(loss)
        pl_module.logger.experiment['ft_valid/acc_epoch'].log(acc)

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        loss = np.mean(pl_module.test_loss)  # /self.num_devices
        acc = np.mean(pl_module.test_acc)  # /self.num_devices
        t5 = np.mean(pl_module.test_t5)  # /self.num_devices

        epoch = trainer.current_epoch

        print("\n [Test] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg top5: {:.4f}, Avg KNN Acc: {:.4f}"
              .format(loss, acc, t5, pl_module.test_knn))

        pl_module.logger.experiment['ft_test/t5_epoch'].log(t5)
        pl_module.logger.experiment['ft_test/loss_epoch'].log(loss)
        pl_module.logger.experiment['ft_test/acc_epoch'].log(acc)
        pl_module.logger.experiment['ft_test/knn_acc_epoch'].log(pl_module.test_knn)


class TestNeptuneCallback(pl.Callback):
    def __init__(self, experiment):
        self.experiment = experiment

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        loss = np.array(pl_module.test_loss).mean()
        acc = np.array(pl_module.test_acc).mean()
        t5 = np.array(pl_module.test_t5).mean()

        epoch = trainer.current_epoch

        pl_module.logger.experiment['ft_test/test_top5'].log(t5)
        pl_module.logger.experiment['ft_test/test_loss'].log(loss)
        pl_module.logger.experiment['ft_test/test_acc'].log(acc)

        print("\n [Test] Avg Loss: {:.4f}, Avg Acc: {:.4f}, Avg top5: {:.4f}".format(loss, acc, t5))


class CheckpointSave(pl.Callback):
    ''' Callback to handle training checkpointing '''

    def __init__(self, path):
        if path != None:
            self.path = os.path.join(path, 'latest_checkpoint.ckpt')

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        print('... Saving checkpoint! ...')

        trainer.save_checkpoint(self.path)


# Helper Functions

def sample_weights(labels):
    """ Calculates per sample weights. """
    class_sample_count = np.unique(labels, return_counts=True)[1]
    class_weights = 1. / torch.Tensor(class_sample_count)
    return class_weights[list(map(int, labels))]


def rank_zero_check():
    if not int(os.environ.get('SLURM_PROCID', 0)) > 0 and not int(os.environ.get('LOCAL_RANK', 0)) > 0:
        return True
    else:
        return False


class CosineWD_LR_Schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 start_lr,
                 ref_lr,
                 ref_wd,
                 T_max,
                 last_epoch=-1,
                 final_lr=0.,
                 final_wd=0.):

        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max_lr = T_max - warmup_steps
        super().__init__(optimizer)

    def step(self):
        self._step += 1

        ## LR
        if self._step < self.warmup_steps:
            progress_lr = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress_lr * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress_lr = float(self._step - self.warmup_steps) / float(max(1, self.T_max_lr))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress_lr)))

        ## WD
        progress_wd = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress_wd))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd


class LinearWarmupCosineAnnealingLR_FixLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
            start_lr: float = 0.001,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.start_lr = start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                self.start_lr
                if 'fix_lr' in group and group['fix_lr'] else
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                self.start_lr
                if 'fix_lr' in group and group['fix_lr'] else group["lr"]
                                                              + (base_lr - self.eta_min) * (1 - math.cos(
                    math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            self.start_lr
            if 'fix_lr' in group and group['fix_lr'] else
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                    1
                    + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
            )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


# warmup + decay as a function
def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """Linear warmup for warmup_steps, optionally with cosine annealing or linear decay to 0 at total_steps."""
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn


class SteppedLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            lrs
    ) -> None:

        self.milestones = milestones
        self.lrs = lrs

        self.update_steps = 0

        super().__init__(optimizer)

    def set_lr(self, optimizer, lr):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        print("Updated to: {}".format(lr))

    def get_lr(self, optimizer):
        for pg in optimizer.param_groups:
            lr = pg["lr"]

        return lr

    def step(self, val_loss=None):

        if self.update_steps in self.milestones:
            lr_ind = self.milestones.index(self.update_steps)
            lr = self.lrs[lr_ind]

            self.set_lr(self.optimizer, lr)

        self.update_steps += 1


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
