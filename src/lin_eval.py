#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from configargparse import ArgumentParser

# Torch
import torch
import torchmetrics
import torch.nn.functional as F
from torch.optim import Adam, SGD, LBFGS
import pytorch_lightning as pl

# My methods and Classes
import network as models
from optimiser import LARSSGD
import neptune
from utils import rank_zero_check
import os


class SSLLinearEval(pl.LightningModule):
    ''' Linear Evaluation training '''

    def __init__(self,
                 encoder,
                 num_classes,
                 model,
                 batch_size,
                 stack=0,
                 ft_learning_rate: float = 0.2,
                 ft_weight_decay: float = 1.5e-6,
                 ft_epochs: int = 1,
                 ft_optimiser: str = 'sgd',
                 effective_bsz: int = 256,
                 **kwargs):
        super().__init__()
        # Command line args
        self.save_hyperparameters()
        self.stacked = stack

        # Define the model and remove the projection head

        # Freeze encoder
        if self.stacked >= 1:
            self.enc1 = encoder[0]
            self.enc2 = encoder[1]
            if self.stacked == 2:
                self.enc3 = encoder[2]
                self.enc3.fc = Identity
                for param in self.enc3.parameters():
                    param.requires_grad = False
            else:
                self.enc2.fc = Identity()
            for param in self.enc1.parameters():
                param.requires_grad = False
            for param in self.enc2.parameters():
                param.requires_grad = False
        else:
            self.enc = encoder
            self.enc.fc = Identity()
            for param in self.enc.parameters():
                param.requires_grad = False

        emb_dim = 512 if '18' in self.hparams.model else 512 if '34' in self.hparams.model else 2048 if '50' in self.hparams.model else 2048 if '50' in self.hparams.model else 2048 if '101' in self.hparams.model else 96
        print("\n Num Classes: {}".format(num_classes))

        # Define the linear evaluation head and train it
        self.lin_head = models.Sup_Head(emb_dim, num_classes)
        for param in self.lin_head.parameters():
            param.requires_grad = True

        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        self.batch_size = batch_size
        self.ft_learning_rate = ft_learning_rate
        self.ft_weight_decay = ft_weight_decay
        self.ft_optimiser = ft_optimiser
        self.ft_epochs = ft_epochs
        self.num_classes = num_classes
        self.effective_bsz = effective_bsz

        print("\n\n\n effective_bsz:{} \n\n\n".format(self.effective_bsz))

        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []

        self.train_t5 = []
        self.valid_t5 = []
        self.test_t5 = []

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.train_feature_bank = []
        self.train_label_bank = []
        self.test_feature_bank = []
        self.test_label_bank = []
        self.test_path_bank = []
        self.test_knn = 0.0

    def encode(self, x):
        with torch.no_grad():
            j = 16
            if self.hparams.projection == "stacked" or self.hparams.projection == "none":
                j = 32
            if self.stacked >= 1:
                s, _ = self.enc1(x)
                s = s.repeat(1, 3).reshape(self.hparams.ft_batch_size, 3, 16, j)
                if self.stacked == 2:
                    s, _ = self.enc2(s)
                    s = s.repeat(1, 3).reshape(self.hparams.ft_batch_size, 3, 16, j)
                    return self.enc3(s)
                return self.enc2(s)
            else:
                return self.enc(x)

    def training_step(self, batch, batch_idx):
        # This statement is for plotting visualisation purposes
        if len(batch) > 2:
            x, y, img_path = batch
        else:
            x, y = batch

        # no_grad ensures we don't train encoder
        with torch.no_grad():
            _, feats = self.encode(x)

        # Linear eval head
        feats = feats.view(feats.size(0), -1)
        logits = self.lin_head(feats)

        # Compute loss and metrics
        loss = self.criterion(logits, y)
        acc = self.accuracy(F.softmax(logits), y)
        t5 = self.top5(logits, y)

        # Progress bar
        self.log_dict({'train_acc': acc, 'train_loss': loss},
                      prog_bar=True, on_epoch=True, sync_dist=True)

        # Logging
        if rank_zero_check():
            self.logger.experiment["ft_train/loss_step"].log(loss.item())
            self.logger.experiment["ft_train/acc_step"].log(acc.item())
            self.logger.experiment["ft_train/t5_step"].log(t5)

        # Global metrics   
        self.train_loss.append(loss.item())
        self.train_acc.append(acc.item())
        self.train_t5.append(t5)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):

        if dataloader_idx == 2:
            # This statement is for plotting visualisation purposes
            if len(batch) > 2:
                x, y, img_path = batch
            else:
                x, y = batch

            # no_grad ensures we don't train
            with torch.no_grad():
                _, feats = self.encode(x)
                feats = feats.view(feats.size(0), -1)
                logits = self.lin_head(feats)

            # Compute loss and metrics
            loss = self.criterion(logits, y)
            acc = self.accuracy(F.softmax(logits), y)
            t5 = self.top5(logits, y)

            # Progress Bar
            self.log_dict({'val_acc': acc, 'val_loss': loss},
                          prog_bar=True, on_epoch=True, sync_dist=True)

            # Logging
            if rank_zero_check():
                self.logger.experiment["ft_valid/loss_step"].log(loss.item())
                self.logger.experiment["ft_valid/acc_step"].log(acc.item())
                self.logger.experiment["ft_valid/t5_step"].log(t5)

            # Global metrics
            self.valid_loss.append(loss.item())
            self.valid_acc.append(acc.item())
            self.valid_t5.append(t5)

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

    def knn_shared_step(self, batch, batch_idx, mode):
        # This statement is for plotting visualisation purposes
        if len(batch) > 2:
            img, y, img_path = batch
        else:
            img, y = batch

        with torch.no_grad():
            _, feature = self.encode(img)
            feature = feature.view(feature.size(0), -1)

        if mode == 'train':
            self.train_feature_bank.append(F.normalize(feature, dim=1))
            self.train_label_bank.append(y)

        elif mode == 'test':
            self.test_feature_bank.append(F.normalize(feature, dim=1))
            self.test_label_bank.append(y)
            if len(batch) > 2:
                for i in range(len(img_path)):
                    img_name = img_path[i].split('/')[-1][:-4]
                    class_id = img_path[i].split('/')[-2]
                    img_path[i] = [int(class_id), int(img_name)]
                self.test_path_bank.append(torch.Tensor(img_path))

    def test_step(self, batch, batch_idx, dataloader_idx):

        if dataloader_idx == 0:
            self.knn_shared_step(batch, batch_idx, 'train')

        elif dataloader_idx == 1:
            self.knn_shared_step(batch, batch_idx, 'test')
            # This statement is for plotting visualisation purposes
            if len(batch) > 2:
                x, y, img_path = batch
            else:
                x, y = batch

            with torch.no_grad():
                _, feats = self.encode(x)
                feats = feats.view(feats.size(0), -1)
                logits = self.lin_head(feats)

            # Compute loss and metrics
            loss = self.criterion(logits, y)
            acc = self.accuracy(F.softmax(logits), y)
            t5 = self.top5(logits, y)

            # Progress Bar
            self.log_dict({'test_acc': acc, 'test_loss': loss, 'test_t5': t5}, sync_dist=True)

            # Logging
            if rank_zero_check():
                self.logger.experiment["ft_test/loss_step"].log(loss.item())
                self.logger.experiment["ft_test/acc_step"].log(acc.item())
                self.logger.experiment["ft_test/t5_step"].log(t5)

            # Global Metrics
            self.test_loss.append(loss.item())
            self.test_acc.append(acc.item())
            self.test_t5.append(t5)

    def test_epoch_end(self, output) -> None:
        # Compute final global metrics and plot visualisation of classification space
        self.train_feature_bank = torch.cat(self.train_feature_bank, dim=0).t().contiguous()
        self.test_feature_bank = torch.cat(self.test_feature_bank, dim=0).contiguous()
        self.train_label_bank = torch.cat(self.train_label_bank, dim=0).contiguous()
        self.test_label_bank = torch.cat(self.test_label_bank, dim=0).contiguous()
        self.test_path_bank = torch.cat(self.test_path_bank, dim=0).contiguous()

        self.test_feature_bank = self.all_gather(self.test_feature_bank)
        self.test_label_bank = self.all_gather(self.test_label_bank)
        self.test_path_bank = self.all_gather(self.test_path_bank)

        self.test_feature_bank = torch.flatten(self.test_feature_bank, end_dim=1)
        self.test_label_bank = torch.flatten(self.test_label_bank, end_dim=1)
        self.test_path_bank = torch.flatten(self.test_path_bank, end_dim=1)

        total_top1, total_num = 0.0, 0

        for feat, label in zip(self.test_feature_bank, self.test_label_bank):
            feat = torch.unsqueeze(feat.cuda(non_blocking=True), 0)

            pred_label = self.knn_predict(feat, self.train_feature_bank, self.train_label_bank,
                                          self.hparams.num_classes, 200, 0.1)

            total_num += feat.size(0)

            total_top1 += (pred_label[:, 0].cpu() == label.cpu()).float().sum().item()

        self.test_knn = total_top1 / total_num * 100

        self.log_dict({'test_knn': self.test_knn}, sync_dist=True)

        if self.hparams.dataset == 'cifar10' or self.hparams.dataset == 'stl10':
            # TSNE
            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(self.test_feature_bank.cpu().detach().numpy())
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
            else:
                classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            # Define the figure size
            fig = plt.figure(figsize=(15, 15))
            scatter = plt.scatter(tx, ty, c=self.test_label_bank.cpu().detach().numpy(), cmap='tab10')
            plt.legend(handles=scatter.legend_elements()[0], labels=classes)
            if self.hparams.dataset == 'stl10':
                if self.stacked == 1:
                    nm = 's_plot_data.npz'
                elif self.stacked == 2:
                    nm = 's2_plot_data.npz'
                else:
                    nm = 'plot_data.npz'
                np.savez(nm, path_bank=self.test_path_bank.cpu().detach().numpy(),
                         label_bank=self.test_label_bank.cpu().detach().numpy(),
                         ty=ty, tx=tx)

            if rank_zero_check():
                self.logger.experiment['tsne/test_tsne'].upload(neptune.types.File.as_image(fig))
            plt.clf()
            plt.close()

        self.train_feature_bank = []
        self.train_label_bank = []

        self.test_feature_bank = []
        self.test_label_bank = []
        self.test_path_bank = []

    def configure_optimizers(self):

        lr = self.hparams.learning_rate

        # Get parameters of the model we want to train
        params = self.lin_head.parameters()

        print("\n OPTIM :{} \n".format(self.ft_optimiser))

        if self.ft_optimiser == 'lars':

            param_list = self.lin_head.parameters()

            optimizer = LARSSGD(
                param_list, lr=lr, weight_decay=self.hparams.weight_decay, eta=0.001, nesterov=False)

        elif self.ft_optimiser == 'adam':
            optimizer = Adam(params, lr=lr,
                             weight_decay=self.ft_weight_decay)
        elif self.ft_optimiser == 'sgd':
            optimizer = SGD(params, lr=lr, weight_decay=self.ft_weight_decay, momentum=0.9, nesterov=False)

        elif self.ft_optimiser == 'lbfgs':
            optimizer = LBFGS(params, lr=lr)
        else:
            raise NotImplementedError('{} not setup.'.format(self.ft_optimiser))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.ft_epochs, last_epoch=-1)

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        (args, _) = parser.parse_known_args()

        # optim
        parser.add_argument('--ft_epochs', type=int, default=2)
        parser.add_argument('--ft_batch_size', type=int, default=128)
        parser.add_argument('--ft_learning_rate', type=float, default=0.02)
        parser.add_argument('--ft_weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--ft_optimiser', default='sgd',
                            help='Optimiser, (Options: sgd, adam, lars).')
        parser.add_argument('--ft_accumulate_grad_batches', type=int, default=1)

        parser.add_argument('--ft_hyperbolic', dest='ft_hyperbolic', action='store_true',
                            help='hyperbolic (Default: False)')
        parser.set_defaults(ft_hyperbolic=False)
        return parser

    def top5(self, x, y):
        _, output_topk = x.topk(5, 1, True, True)

        acc_top5 = (output_topk == y.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / y.size(0)

        return acc_top5


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
