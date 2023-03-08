#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required

def fix_lr_params(model):
    return {'params': model.parameters(), 'fix_lr': True}

def collect_params(model_list):
    reg = []
    non_reg = []
    
    for model in model_list:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                non_reg.append(param)
            else:
                reg.append(param)

            #reg.append(param)
            
    return [{'params': reg}, {'params': non_reg, 'weight_decay': 0.}]

def collect_params_lars(model_list, exclude_bias_and_bn=True):
    """
    exclude_bias_and bn: exclude bias and bn from both weight decay and LARS adaptation
        in the PyTorch implementation of ResNet, `downsample.1` are bn layers
    """
    param_list = []
    for model in model_list:
        for name, param in model.named_parameters():
            # print(name)
            if exclude_bias_and_bn and ('bn' in name or 'downsample.1' in name or 'bias' in name or 'norm' in name):
                param_dict = {'params': param, 'weight_decay': 0., 'lars_exclude': True}
                    
            else:
                param_dict = {'params': param}

            param_list.append(param_dict)
    return param_list


class LARSSGD(Optimizer):
    """
    Layer-wise adaptive rate scaling
    https://github.com/yaox12/BYOL-PyTorch
    - Based on:
    https://github.com/noahgolmant/pytorch-lars
    params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        lr (int): Length / Number of layers we want to apply weight decay, else do not compute
        momentum (float, optional): momentum factor (default: 0.9)
        nesterov (bool, optional): flag to use nesterov momentum (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
            ("\beta")
        eta (float, optional): LARS coefficient (default: 0.001)
    - Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    - Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    """

    def __init__(self,
                 params,
                 lr,
                 momentum=0.9,
                 dampening=0.0,
                 weight_decay=0.0,
                 eta=0.001,
                 nesterov=False):

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov, eta=eta)

        super(LARSSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARSSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                p_grad = p.grad

                if lars_exclude:
                    learning_rate = 1.0
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(p_grad).item()

                    learning_rate = eta * weight_norm / (grad_norm + weight_decay * weight_norm + 1e-12)

                scaled_learning_rate = learning_rate * lr

                p_grad = p_grad.add(p, alpha=weight_decay).mul(scaled_learning_rate)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(p_grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(p_grad, alpha=1 - dampening)
                    if nesterov:
                        p_grad = p_grad.add(buf, alpha=momentum)
                    else:
                        p_grad = buf
                p.add_(-p_grad)

        return loss


class LARSWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        eta: float = 1e-3,
        clip: bool = False,
        eps: float = 1e-8,
        exclude_bias_n_norm: bool = False,
    ):
        """Wrapper that adds LARS scheduling to any optimizer.
        This helps stability with huge batch sizes.
        Args:
            optimizer (Optimizer): torch optimizer.
            eta (float, optional): trust coefficient. Defaults to 1e-3.
            clip (bool, optional): clip gradient values. Defaults to False.
            eps (float, optional): adaptive_lr stability coefficient. Defaults to 1e-8.
            exclude_bias_n_norm (bool, optional): exclude bias and normalization layers from lars.
                Defaults to False.
        """

        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group

        self.__setstate__ = self.optim.__setstate__  # type: ignore
        self.__getstate__ = self.optim.__getstate__  # type: ignore
        self.__repr__ = self.optim.__repr__  # type: ignore

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @property  # type: ignore
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, state):
        self.optim.state = state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            for p in group["params"]:
                if p.grad is not None and (p.ndim != 1 or not self.exclude_bias_n_norm):
                    self.update_p(p, group, weight_decay)

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                print(group["lr"])
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr
