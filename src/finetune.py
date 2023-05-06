#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
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
from utils import FTPrintingCallback, rank_zero_check
from lin_eval import SSLLinearEval


def cli_main(stacked=False):

    # Arguments
    default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')
    parser = ArgumentParser(
        description='Pytorch Lightning VICReg', default_config_files=[default_config])
    parser.add_argument('-c', '--my-config', required=False,
                        is_config_file=True, help='config file path')
    parser.add_argument('--pt_checkpoint', type=str, default=None)
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
    args.status = 'Finetune'
    args.devices = int(args.devices)
    args.batch_size = args.ft_batch_size
    
    # Logging
    x = "Finetune_S" if stacked else "Finetune"
    neptune_logger = NeptuneLogger(
            mode=args.mode,
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsI"
                    "joiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYmNlN2NlYS05Y2"
                    "E1LTQyZjktOWMzYS04MDIyNmYyNTIxMGQifQ==",
            project=args.project_name,
            tags=[x, args.tag, args.projection, str(args.stacked), args.dataset],
            source_files=['**/*.py']
        )
        
    # Get DataModule / Data loaders
    dm, ft_dm, args = get_dm(args)

    # Load the self-supervised pretrained model
    load_log_file = os.path.join(args.default_root_dir, args.log_file)

    log_dirs = np.genfromtxt(load_log_file, delimiter=" ", dtype='str')

    print("\n\n Log Dir: {}\n\n".format(log_dirs))

    ft_model_dir = log_dirs[1]
    args.ft_model_dir = ft_model_dir
    checkpoint_path = log_dirs[0] + '/latest_checkpoint.ckpt'

    print("Loading checkpoint: {}".format(checkpoint_path))

    # Populate the self-supervised encoder with the pretrained weights
    encoder = VICReg.load_from_checkpoint(checkpoint_path, strict=False)
    
    # Setup batch size for parallel computation

    if args.strategy == 'ddp':
        args.effective_bsz = args.ft_batch_size * args.num_nodes * args.devices
    elif args.strategy == 'ddp2':
        args.effective_bsz = args.ft_batch_size * args.num_nodes
    else:
        args.effective_bsz = args.ft_batch_size

    # Initialise trainer 
    trainer_ft = pl.Trainer.from_argparse_args(
        args, max_epochs=args.ft_epochs,
        logger=neptune_logger,
        callbacks=[FTPrintingCallback(ft_model_dir, args)],
        deterministic=True,
        enable_checkpointing=False,
        fast_dev_run=False,
        sync_batchnorm=True,
        accelerator='gpu',
        replace_sampler_ddp=True,
        check_val_every_n_epoch=args.ft_val_freq,
        accumulate_grad_batches=args.ft_accumulate_grad_batches
        )

    if stacked:
        if args.pt2:
            ft_model = torch.compile(SSLLinearEval([encoder.encoder_online, encoder.encoder_stacked],
                                                   stack=True, **args.__dict__))
        else:
            ft_model = SSLLinearEval([encoder.encoder_online, encoder.encoder_stacked], stack=True, **args.__dict__)
    else:
        # Define the model (here I turn on pytorch 2 or not, this is experimental version of pytorch)
        if args.pt2:
            ft_model = torch.compile(SSLLinearEval(encoder.encoder_online, **args.__dict__))
        else:
            ft_model = SSLLinearEval(encoder.encoder_online, **args.__dict__)

    # Update logging files
    if rank_zero_check():
        if args.mode != 'offline':
            print("Experiment: {}".format(str(neptune_logger.experiment['sys/id'].fetch())))

            log_dirs = np.append(log_dirs, str(neptune_logger.experiment['sys/id'].fetch()))

        save_log_file = os.path.join(args.default_root_dir, args.log_file)

        np.savetxt(save_log_file, log_dirs, delimiter=" ", fmt="%s")
        log_ft_path = os.path.join(os.path.dirname(os.path.normpath(ft_model_dir)), "finetune_" + args.log_file)
        np.savetxt(log_ft_path, log_dirs, delimiter=" ", fmt="%s")

    # Fit the model with the data
    trainer_ft.fit(ft_model, ft_dm)

    checkpoint_path = os.path.join(ft_model_dir, os.listdir(ft_model_dir+'/')[-1])
    
    # send trained model to the cloud experiment logger
    if rank_zero_check() and args.save_checkpoint:
        neptune_logger.experiment['checkpoints/finetune.ckpt'].upload(checkpoint_path)

    log_file = os.path.join(args.default_root_dir, args.log_file)
    log_dirs = np.genfromtxt(log_file, delimiter=" ", dtype='str')
    
    # Test time
    print("checkpoint_path: {}".format(checkpoint_path))
    
    # Perform inference
    trainer_ft.test(ckpt_path=checkpoint_path, datamodule=ft_dm)  
    
    neptune_logger.experiment.stop()

    if args.stacked == 1 and not stacked:
        cli_main(True)


if __name__ == '__main__':
    cli_main()
