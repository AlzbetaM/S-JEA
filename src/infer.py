import pytorch_lightning as pl
from configargparse import ArgumentParser
import json

from dataloaders.handlers import get_dm
from lin_eval import SSLLinearEval
from dataloaders.cifar10 import Cifar10_DataModule


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

    trainer = pl.Trainer()
    dm, ft_dm, args = get_dm(args)
    checkpoint_path = "C:/Users/alzbe/Documents/VICReg2/VICReg/model/best_epoch.ckpt"

    model = SSLLinearEval.load_from_checkpoint(checkpoint_path)
    # Perform inference
    trainer.predict(model, ft_dm)


if __name__ == '__main__':
    infer()
