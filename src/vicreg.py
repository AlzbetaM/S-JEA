##!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from configargparse import ArgumentParser
import collections
from collections import OrderedDict
import copy
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import neptune

# Torch
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import pytorch_lightning as pl

# My Methods and Classes
import network as models
from optimiser import LARSSGD
from utils import rank_zero_check, CosineWD_LR_Schedule


class VICReg(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 num_batches,
                 num_nodes,
                 devices,
                 learning_rate: float = 0.2,
                 weight_decay: float = 1.5e-6,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 warmup_epochs: int = 0,
                 max_epochs: int = 1,
                 o_units: int = 256,
                 h_units: int = 4096,
                 model: str = 'resnet18',
                 tau: float = 0.996,
                 optimiser: str = 'sgd',
                 effective_bsz: int = 256,
                 **kwargs):
        super().__init__()
        # Command line args
        self.save_hyperparameters()

        # Compute resources for distributed operations
        self.world_size = self.hparams.num_nodes * self.hparams.devices

        # Get Encoder Model
        self.encoder_online = models.__dict__[self.hparams.model](dataset=self.hparams.dataset, norm_layer='bn2d')
        # Remove standard fc layer
        self.encoder_online.fc = Identity()

        # Get the embedding dimensions of the encoders
        emb_dim = 512 if '18' in self.hparams.model else 512 if '34' in self.hparams.model else \
            2048 if '50' in self.hparams.model else 2048 if '50' in self.hparams.model else \
            2048 if '101' in self.hparams.model else 96

        # Define the projection head
        fc = OrderedDict([])
        fc['fc1'] = torch.nn.Linear(emb_dim, self.hparams.h_units)
        if self.hparams.use_bn:
            fc['bn1'] = torch.nn.BatchNorm1d(self.hparams.h_units)
        fc['relu1'] = torch.nn.ReLU()
        fc['fc2'] = torch.nn.Linear(self.hparams.h_units, self.hparams.h_units)
        if self.hparams.use_bn:
            fc['bn2'] = torch.nn.BatchNorm1d(self.hparams.h_units)
        fc['relu2'] = torch.nn.ReLU()
        fc['fc3'] = torch.nn.Linear(self.hparams.h_units, self.hparams.o_units)

        if self.hparams.stacked == 2:
            self.encoder_stacked = models.__dict__[self.hparams.model]\
                (dataset=self.hparams.dataset, norm_layer='bn2d', dim=512)
        # projection head settings for multiple encoders
        self.x = 16  # dimension for second encoder input
        self.y = 16  # dimension for second encoder input
        if self.hparams.stacked != 2:
            self.encoder_online.fc = torch.nn.Sequential(fc)
        elif self.hparams.projection == "both":
            self.encoder_online.fc = torch.nn.Sequential(fc)
            self.encoder_stacked.fc = torch.nn.Sequential(fc)
        elif self.hparams.projection == "simple":
            self.encoder_online.fc = torch.nn.Sequential(fc)
        elif self.hparams.projection == "stacked":
            self.encoder_stacked.fc = torch.nn.Sequential(fc)

        self.num_batches = num_batches
        self.effective_bsz = effective_bsz
        self.initial_tau = self.hparams.tau
        self.current_tau = self.hparams.tau
        self.initial_sharpen = self.hparams.initial_sharpen
        self.current_sharpen = self.hparams.initial_sharpen

        print("\n\n\n effective_bsz:{} \n\n\n".format(self.effective_bsz))

        self.rep = np.empty((0, self.hparams.o_units + 1), dtype=float)
        self.train_feature_bank = []
        self.train_label_bank = []
        self.test_feature_bank = []
        self.test_label_bank = []
        self.plot_test_path_bank = []

        self.val_knn = 0.0
        self.val_knn1 = 0.0
        self.val_knn2 = 0.0
        self.val_knn3 = 0.0
        self.val_knn4 = 0.0

        if self.hparams.stacked == 2:
            self.train_feature_bank_stacked = []
            self.test_feature_bank_stacked = []
            self.val_knn_stacked = 0.0
            self.val_knn_stacked1 = 0.0
            self.val_knn_stacked2 = 0.0
            self.val_knn_stacked3 = 0.0
            self.val_knn_stacked4 = 0.0

    def shared_step(self, batch, batch_idx, mode):
        # This statement is for plotting visualisation purposes
        if len(batch) > 2:
            img_batch, _, img_path = batch
        else:
            img_batch, _ = batch

        imgs = [u for u in img_batch]  # Multiple image views not implemented

        # Pass each view to the encoder
        z_i, _, y_i = self.encoder_online(imgs[0])
        z_j, _, y_j = self.encoder_online(imgs[1])

        # Ensure float32
        z_i, z_j = z_i.float(), z_j.float()

        # Compute loss
        loss_inv = self.invariance_loss(z_i, z_j)

        loss_var, _ = self.variance_loss(z_i, z_j)
        loss_cov = self.covariance_loss(z_i, z_j)

        loss = ((self.hparams.inv * loss_inv) + (self.hparams.var * loss_var) + (self.hparams.covar * loss_cov))
        all_loss = loss

        # for stacked VICReg
        if self.hparams.stacked == 1 or self.hparams.stacked == 2:
            # torch.Size([128, 512, 4, 4]))
            # stacked encoder
            if self.hparams.stacked == 2:
                stack_i, _, _ = self.encoder_stacked(y_i)
                stack_j, _, _ = self.encoder_stacked(y_j)
            else:
                stack_i, _, _ = self.encoder_online(y_i)
                stack_j, _, _ = self.encoder_online(y_j)

            # Stacked loss
            s_loss_inv = self.invariance_loss(stack_i, stack_j)
            s_loss_var, _ = self.variance_loss(stack_i, stack_j)
            s_loss_cov = self.covariance_loss(stack_i, stack_j)

            s_loss = ((self.hparams.inv * s_loss_inv) + (self.hparams.var * s_loss_var) + (
                    self.hparams.covar * s_loss_cov))
            all_loss = loss + s_loss

        # Logging
        if rank_zero_check() and mode == 'train':
            self.logger.experiment["train/loss_inv"].log(loss_inv.item())
            self.logger.experiment["train/loss_var"].log(loss_var.item())
            self.logger.experiment["train/loss_cov"].log(loss_cov.item())
            self.logger.experiment["train/loss"].log(loss)

            if self.hparams.stacked == 1 or self.hparams.stacked == 2:
                self.logger.experiment["train/s_loss_inv"].log(s_loss_inv.item())
                self.logger.experiment["train/s_loss_var"].log(s_loss_var.item())
                self.logger.experiment["train/s_loss_cov"].log(s_loss_cov.item())
                self.logger.experiment["train/s_loss"].log(s_loss)

        return all_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        # Progress Bar
        self.log_dict({'loss': loss}, prog_bar=True, on_epoch=True, sync_dist=True)
        # Logging
        if rank_zero_check():
            self.logger.experiment["train/loss_step"].log(loss.item())

        return loss

    def val_shared_step(self, batch, batch_idx, idx):
        # This statement is for plotting visualisation purposes
        if len(batch) > 2:
            img, y, img_path = batch
        else:
            img, y = batch

        # no_grad ensures we don't train
        with torch.no_grad():
            projection, embedding, s = self.encoder_online(img)
            if self.hparams.stacked == 2:
                s_projection, s_embedding, _ = self.encoder_stacked(s)

        if idx == 1:
            self.train_feature_bank.append(F.normalize(embedding, dim=1))
            self.train_label_bank.append(y)
            if self.hparams.stacked == 2:
                self.train_feature_bank_stacked.append(F.normalize(s_embedding, dim=1))

        elif idx == 2:
            self.test_feature_bank.append(F.normalize(embedding, dim=1))
            self.test_label_bank.append(y)
            if self.hparams.stacked == 2:
                self.test_feature_bank_stacked.append(F.normalize(s_embedding, dim=1))

            if len(batch) > 2 and self.hparams.dataset == 'stl10':
                for i in range(len(img_path)):
                    img_name = img_path[i].split('/')[-1][:-4]
                    class_id = img_path[i].split('/')[-2]
                    img_path[i] = [int(class_id), int(img_name)]
                self.plot_test_path_bank.append(torch.Tensor(img_path))

    def validation_step(self, batch, batch_idx, dataloader_idx):

        if dataloader_idx == 0:
            loss = self.shared_step(batch, batch_idx, 'val')
            # Progress Bar
            self.log_dict({'valid_loss': loss}, prog_bar=True, on_epoch=True, sync_dist=True)
            # Logging
            if rank_zero_check():
                self.logger.experiment["valid/loss_step"].log(loss.item())

        elif dataloader_idx == 1:
            self.val_shared_step(batch, batch_idx, dataloader_idx)

        elif dataloader_idx == 2:
            self.val_shared_step(batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, output) -> None:
        # Compute final global metrics and plot visualisation of embedding space
        self.train_feature_bank = torch.cat(self.train_feature_bank, dim=0).t().contiguous()
        self.test_feature_bank = torch.cat(self.test_feature_bank, dim=0).contiguous()
        self.train_label_bank = torch.cat(self.train_label_bank, dim=0).contiguous()
        self.test_label_bank = torch.cat(self.test_label_bank, dim=0).contiguous()

        total_top1, total_num = 0.0, 0
        total_top11 = 0.0
        total_top12 = 0.0
        total_top13 = 0.0
        total_top14 = 0.0

        for feat, label in zip(self.test_feature_bank, self.test_label_bank):
            feat = torch.unsqueeze(feat.cuda(non_blocking=True), 0)

            pred_label = self.knn_predict(feat, self.train_feature_bank, self.train_label_bank,
                                          self.hparams.num_classes, 80, 0.1)
            pred_label1 = self.knn_predict(feat, self.train_feature_bank, self.train_label_bank,
                                           self.hparams.num_classes, 100, 0.1)
            pred_label2 = self.knn_predict(feat, self.train_feature_bank, self.train_label_bank,
                                           self.hparams.num_classes, 130, 0.1)
            pred_label3 = self.knn_predict(feat, self.train_feature_bank, self.train_label_bank,
                                           self.hparams.num_classes, 160, 0.1)
            pred_label4 = self.knn_predict(feat, self.train_feature_bank, self.train_label_bank,
                                           self.hparams.num_classes, 200, 0.1)

            total_num += feat.size(0)
            total_top1 += (pred_label[:, 0].cpu() == label.cpu()).float().sum().item()
            total_top11 += (pred_label1[:, 0].cpu() == label.cpu()).float().sum().item()
            total_top12 += (pred_label2[:, 0].cpu() == label.cpu()).float().sum().item()
            total_top13 += (pred_label3[:, 0].cpu() == label.cpu()).float().sum().item()
            total_top14 += (pred_label4[:, 0].cpu() == label.cpu()).float().sum().item()

        self.val_knn = total_top1 / total_num * 100
        self.val_knn1 = total_top11 / total_num * 100
        self.val_knn2 = total_top12 / total_num * 100
        self.val_knn3 = total_top13 / total_num * 100
        self.val_knn4 = total_top14 / total_num * 100

        if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'stl10':
            if self.plot_test_path_bank:
                self.plot_test_path_bank = torch.cat(self.plot_test_path_bank, dim=0).contiguous()
            self.tsne_plot("pretrain", self.test_feature_bank)

        if self.hparams.stacked == 2:
            self.train_feature_bank_stacked = torch.cat(self.train_feature_bank_stacked, dim=0).t().contiguous()
            self.test_feature_bank_stacked = torch.cat(self.test_feature_bank_stacked, dim=0).contiguous()
            total_top1, total_num = 0.0, 0
            total_top11 = 0.0
            total_top12 = 0.0
            total_top13 = 0.0
            total_top14 = 0.0
            for feat, label in zip(self.test_feature_bank_stacked, self.test_label_bank):
                feat = torch.unsqueeze(feat.cuda(non_blocking=True), 0)

                pred_label = self.knn_predict(feat, self.train_feature_bank_stacked, self.train_label_bank,
                                              self.hparams.num_classes, 80, 0.1)
                pred_label1 = self.knn_predict(feat, self.train_feature_bank_stacked, self.train_label_bank,
                                               self.hparams.num_classes, 100, 0.1)
                pred_label2 = self.knn_predict(feat, self.train_feature_bank_stacked, self.train_label_bank,
                                               self.hparams.num_classes, 130, 0.1)
                pred_label3 = self.knn_predict(feat, self.train_feature_bank_stacked, self.train_label_bank,
                                               self.hparams.num_classes, 160, 0.1)
                pred_label4 = self.knn_predict(feat, self.train_feature_bank_stacked, self.train_label_bank,
                                               self.hparams.num_classes, 200, 0.1)

                total_num += feat.size(0)
                total_top1 += (pred_label[:, 0].cpu() == label.cpu()).float().sum().item()
                total_top11 += (pred_label1[:, 0].cpu() == label.cpu()).float().sum().item()
                total_top12 += (pred_label2[:, 0].cpu() == label.cpu()).float().sum().item()
                total_top13 += (pred_label3[:, 0].cpu() == label.cpu()).float().sum().item()
                total_top14 += (pred_label4[:, 0].cpu() == label.cpu()).float().sum().item()

            self.val_knn_stacked = total_top1 / total_num * 100
            self.val_knn_stacked1 = total_top11 / total_num * 100
            self.val_knn_stacked2 = total_top12 / total_num * 100
            self.val_knn_stacked3 = total_top13 / total_num * 100
            self.val_knn_stacked4 = total_top14 / total_num * 100

            if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'stl10':
                self.tsne_plot("s_pretrain", self.test_feature_bank_stacked)

            self.train_feature_bank_stacked = []
            self.test_feature_bank_stacked = []

        self.train_feature_bank = []
        self.train_label_bank = []
        self.test_feature_bank = []
        self.test_label_bank = []
        self.plot_test_path_bank = []

    def tsne_plot(self, name, features):
        dest = str(self.hparams.stacked) + "_" + self.hparams.projection + "_"
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(features.cpu().detach().numpy())
        tx = z[:, 0]
        ty = z[:, 1]

        # scale and move the 'x' coordinates so they fit [0; 1] range
        tx_range = (np.max(tx) - np.min(tx))
        tx_from_zero = tx - np.min(tx)
        tx = tx_from_zero / tx_range

        # scale and move the 'y' coordinates so they fit [0; 1] range
        ty_range = (np.max(ty) - np.min(ty))
        ty_from_zero = ty - np.min(ty)
        ty = ty_from_zero / ty_range

        if self.hparams.dataset == 'stl10':
            classes = ["truck", "airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship"]
            np.savez(dest + name, path_bank=self.plot_test_path_bank.cpu().detach().numpy(),
                     label_bank=self.test_label_bank.cpu().detach().numpy(),
                     ty=ty, tx=tx)
        else:
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Define the figure size
        fig = plt.figure(figsize=(15, 15))
        scatter = plt.scatter(tx, ty, c=self.test_label_bank.cpu().detach().numpy(), cmap='tab10')
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)

        if rank_zero_check():
            self.logger.experiment['tsne/' + name].upload(neptune.types.File.as_image(fig))
        plt.clf()
        plt.close()

    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]

        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)

        return pred_labels

    def configure_optimizers(self):

        # This is for large batch sizes
        if self.effective_bsz > 1024:
            lr = (self.hparams.learning_rate * (self.effective_bsz / 256))
        else:
            lr = self.hparams.learning_rate

        # Get parameters of the model we want to train
        # The splitting of parameters into groups is for LARS optim
        param_groups = [
            {'params': (p for n, p in self.encoder_online.named_parameters()
                        if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
            {'params': (p for n, p in self.encoder_online.named_parameters()
                        if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
             'WD_exclude': True,
             'weight_decay': 0}]
        if self.hparams.stacked == 2:
            param_groups += [{'params': (p for n, p in self.encoder_stacked.named_parameters()
                                         if ('bias' not in n) and ('bn' not in n) and len(p.shape) != 1)},
                             {'params': (p for n, p in self.encoder_stacked.named_parameters()
                                         if ('bias' in n) or ('bn' in n) or (len(p.shape) == 1)),
                              'WD_exclude': True,
                              'weight_decay': 0}]

        if self.hparams.optimiser == 'lars':
            optimizer_euc = LARSSGD(
                param_groups, lr=lr, weight_decay=self.hparams.weight_decay, eta=0.001, nesterov=False)

        elif self.hparams.optimiser == 'adam':
            optimizer_euc = Adam(param_groups, lr=lr, weight_decay=self.hparams.weight_decay)

        elif self.hparams.optimiser == 'adamw':
            optimizer_euc = AdamW(param_groups, lr=lr, weight_decay=self.hparams.weight_decay)

        elif self.hparams.optimiser == 'sgd':
            optimizer_euc = SGD(param_groups, lr=lr,
                                weight_decay=self.hparams.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError('{} not setup.'.format(self.hparams.optimiser))

        # Learning rate scheduling (decrease the learning rate during training)
        if self.hparams.warmup_epochs == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_euc, (self.num_batches / self.world_size) * self.trainer.max_epochs,
                last_epoch=(self.num_batches / self.world_size) * self.hparams.max_epochs,
                eta_min=0.002)
        else:
            scheduler = CosineWD_LR_Schedule(
                optimizer_euc,
                warmup_steps=(self.num_batches / self.world_size) * self.hparams.warmup_epochs,
                start_lr=0.0002,
                ref_lr=lr,
                last_epoch=(self.num_batches / self.world_size) * self.hparams.max_epochs,
                final_lr=1.0e-06,
                ref_wd=self.hparams.weight_decay,
                final_wd=self.hparams.final_weight_decay,
                T_max=int(1.25 * self.trainer.max_epochs * (self.num_batches / self.world_size)))

        return [optimizer_euc], [{'scheduler': scheduler,
                                  'interval': 'step'}]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        (args, _) = parser.parse_known_args()

        # Data
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='cifar10, imagenet')
        parser.add_argument('--data_dir', type=str, default=None)
        parser.add_argument('--subset', type=str, default=None, help='path to subset file.')
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--jitter_d', type=float, default=0.5)
        parser.add_argument('--jitter_p', type=float, default=0.8)
        parser.add_argument('--blur_p', type=float, default=0.5)
        parser.add_argument('--grey_p', type=float, default=0.2)
        parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0])
        parser.add_argument('--global_scale', nargs=2, type=float, default=[0.3, 1.0])
        parser.add_argument('--local_scale', nargs=2, type=float, default=[0.05, 0.3])
        parser.add_argument('--global_views', type=int, default=2)
        parser.add_argument('--local_views', type=int, default=2)

        # optim
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=0.02)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--final_weight_decay', type=float, default=0.4)
        parser.add_argument('--warmup_epochs', type=float, default=10)
        parser.add_argument('--optimiser', default='sgd',
                            help='Optimiser, (Options: sgd, adam, lars).')

        # Model
        parser.add_argument('--model', default='tiny',
                            help='Model, (Options: tiny, small, base, large).')

        parser.add_argument('--h_units', type=int, default=256)
        parser.add_argument('--o_units', type=int, default=256)
        parser.add_argument('--gain', type=float, default=1.0)

        parser.add_argument('--use_bn', dest='use_bn', action='store_false',
                            help='To use batch norm (Default: True)')
        parser.set_defaults(use_bn=True)

        parser.add_argument('--inv', type=float, default=25.)
        parser.add_argument('--var', type=float, default=25.)
        parser.add_argument('--covar', type=float, default=1.)

        parser.add_argument('--tau', type=float, default=0.996)
        parser.add_argument('--final_tau', type=float, default=1.0)
        parser.add_argument('--initial_sharpen', type=float, default=0.25)
        parser.add_argument('--final_sharpen', type=float, default=0.25)

        # Misc
        parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true',
                            help='Save the checkpoints to Neptune (Default: False)')
        parser.set_defaults(save_checkpoint=False)
        parser.add_argument('--print_freq', type=int, default=1)

        # newly added to help switching between code versions
        parser.add_argument('--stacked', type=int, default=0)
        parser.add_argument('--projection', type=str, default='both')

        return parser

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """
        return F.mse_loss(z1, z2)

    def variance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """
        # Gather operation for distributed computation
        z1 = torch.cat(tuple(self.all_gather(z1, sync_grads=True)), dim=0)
        z2 = torch.cat(tuple(self.all_gather(z2, sync_grads=True)), dim=0)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)

        std_loss = (torch.mean(F.relu(1.0 - std_z1)) / 2) + (torch.mean(F.relu(1.0 - std_z2)) / 2)

        return std_loss, std_z1

    def covariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """
        # Gather operation for distributed computation
        z1 = torch.cat(tuple(self.all_gather(z1, sync_grads=True)), dim=0)
        z2 = torch.cat(tuple(self.all_gather(z2, sync_grads=True)), dim=0)

        N, D = z1.size()

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)

        cov_loss = (self.off_diagonal(cov_z1).pow_(2).sum().div(D)) + (self.off_diagonal(cov_z2).pow_(2).sum().div(D))
        return cov_loss


class Identity(torch.nn.Module):
    """
    An identity class to replace arbitrary layers in pretrained models
    Example::
        from pl_bolts.utils import Identity
        model = resnet18()
        model.fc = Identity()
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
