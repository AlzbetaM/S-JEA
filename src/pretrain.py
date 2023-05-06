#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import json
import numpy as np
from configargparse import ArgumentParser

# Torch
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

# My Methods and Classes
from vicreg import VICReg
from dataloaders.handlers import get_dm
from utils import PTPrintingCallback, CheckpointSave, rank_zero_check
from lin_eval import SSLLinearEval


def cli_main():

    # Arguments
    default_config = os.path.join(os.getcwd(), 'config.conf')
    parser = ArgumentParser(
        description='Pytorch Lightning VICReg', default_config_files=[default_config])
    parser.add_argument('-c', '--my-config', required=False,
                        is_config_file=True, help='config file path', default=default_config)
    parser.add_argument('--val_every_n', type=int, default=1)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--mode', type=str, default="async")
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--log_file', type=str, default="log_files.txt")
    parser.add_argument('--ft_val_freq', type=int, default=10)
    parser.add_argument('--pt2', dest='pt2', action='store_true',
                        help='Using Pytorch 2 compile (Default: False)')
    parser.set_defaults(pt2=False)

    # Getting the commandline args from each module
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VICReg.add_model_specific_args(parser)
    parser = SSLLinearEval.add_model_specific_args(parser)
    args = parser.parse_args()

    # Some init
    seed_everything(args.seed)
    args.status = 'Pretrain'
    args.devices = int(args.devices)
    
    # Logging and Checkpointing Setup
    if rank_zero_check():
        run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Create Directory for runs
        save_dir = os.path.join(args.default_root_dir, 'checkpoints')
        run_dir = os.path.join(save_dir, ("VICReg_" + run_name))
        pt_model_dir = os.path.join(run_dir, 'pretrain')
        ft_model_dir = os.path.join(run_dir, 'finetune')
        reps_model_dir = os.path.join(run_dir, 'reps')

        args.pt_model_dir, args.ft_model_dir, args.reps_model_dir = pt_model_dir, ft_model_dir, reps_model_dir

        os.makedirs(pt_model_dir, exist_ok=True)
        os.makedirs(os.path.join(pt_model_dir, 'umap/'), exist_ok=True)
        os.makedirs(ft_model_dir, exist_ok=True)
        os.makedirs(reps_model_dir, exist_ok=True)
        
        with open(os.path.join(run_dir, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
    else:
        pt_model_dir = None
        ft_model_dir = None
        reps_model_dir = None
        
    # Neptune Logger www.neptune.ai    
    neptune_logger = NeptuneLogger(
            mode=args.mode,
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsI"
                    "joiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYmNlN2NlYS05Y2"
                    "E1LTQyZjktOWMzYS04MDIyNmYyNTIxMGQifQ==",
            project=args.project_name,
            tags=[args.tag, args.projection, str(args.stacked), args.dataset],
            source_files=['**/*.py']
        )

    # Get DataModule / Dataloaders
    dm, ft_dm, args = get_dm(args)

    # Setup batch size for parallel computation
    if args.strategy == 'ddp':
        args.effective_bsz = args.batch_size * args.num_nodes * args.devices
    elif args.strategy == 'ddp2':
        args.effective_bsz = args.batch_size * args.num_nodes
    else:
        args.effective_bsz = args.batch_size

    # Initialise trainer 
    trainer = pl.Trainer.from_argparse_args(
        args, max_epochs=args.max_epochs,
        logger=neptune_logger,
        callbacks=[PTPrintingCallback(pt_model_dir, args), CheckpointSave(pt_model_dir)],
        deterministic=False,
        fast_dev_run=False,
        sync_batchnorm=True,
        accelerator='gpu',
        enable_checkpointing=False,
        replace_sampler_ddp=True,
        resume_from_checkpoint=args.resume_ckpt,
        check_val_every_n_epoch=args.val_every_n,
        multiple_trainloader_mode='max_size_cycle'
        )

    # Define the model (here I turn on pytorch 2 or not, this is experimental version of pytorch)
    if args.pt2:
        model = torch.compile(VICReg(**args.__dict__))
    else:
        model = VICReg(**args.__dict__)
    
    # Fit the model with the data
    trainer.fit(model, dm)

    # Save and define path to trained model for finetuning
    if rank_zero_check():
        print("os.listdir(pt_model_dir) :{}".format(os.listdir(pt_model_dir)))

        checkpoint_path = os.path.join(pt_model_dir,
                                        os.listdir(pt_model_dir)[-1])

        # send trained model to the cloud experiment logger
        if args.save_checkpoint:
            neptune_logger.experiment['checkpoints/trained.ckpt'].upload(checkpoint_path)

        log_files = [pt_model_dir, ft_model_dir, checkpoint_path]

        save_log_file = os.path.join(args.default_root_dir, args.log_file)

        np.savetxt(save_log_file, log_files, delimiter=" ", fmt="%s")
        log_pt_path = os.path.join(os.path.dirname(os.path.normpath(pt_model_dir)), "pretrain_" + args.log_file)
        np.savetxt(log_pt_path, log_files, delimiter=" ", fmt="%s")

    neptune_logger.experiment.stop()


if __name__ == '__main__':
    cli_main()
