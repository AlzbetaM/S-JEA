import pytorch_lightning as pl
import torch
from configargparse import ArgumentParser
import json
from neptune.integrations.pytorch_lightning import NeptuneLogger

from VICReg.src.utils import FTPrintingCallback
from dataloaders.handlers import get_dm
from lin_eval import SSLLinearEval


def get_args(d):
    parser = ArgumentParser()
    for k, v in d.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()
    return args


def infer():
    f = open("C:/Users/alzbe/Documents/VICReg2/VICReg/model/commandline_args.txt", 'r').read()
    arg_dictionary = json.loads(f)
    args = get_args(arg_dictionary)

    args.ft_epochs = 0
    args.status = 'Infer'
    args.devices = int(args.devices)
    args.batch_size = args.ft_batch_size
    args.effective_bsz = args.ft_batch_size
    args.data_dir = "../" + args.data_dir
    args.strategy = None

    trainer = pl.Trainer.from_argparse_args \
        (args, max_epochs=args.ft_epochs,
         deterministic=True,
         enable_checkpointing=False,
         fast_dev_run=False,
         sync_batchnorm=False,
         accelerator='cpu',
         check_val_every_n_epoch=args.ft_val_freq,
         accumulate_grad_batches=args.ft_accumulate_grad_batches)

    _, ft_dm, args = get_dm(args)
    checkpoint_path = "C:/Users/alzbe/Documents/VICReg2/VICReg/model/best_epoch.ckpt"

    model = SSLLinearEval.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'))
    model.freeze()
    # Perform inference
    pred = model(ft_dm)
    print(pred)


if __name__ == '__main__':
    infer()
