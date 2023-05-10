import pytorch_lightning as pl
from configargparse import ArgumentParser, Action
from dataloaders.handlers import get_dm
from vicreg import VICReg
from lin_eval import SSLLinearEval
import json


def get_args(d):
    parser = ArgumentParser()
    for k, v in d.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()
    return args

def infer():
    f = open("../Data/model/commandline_args.txt", 'r').read()
    arg_dictionary = json.loads(f)
    args = get_args(arg_dictionary)

    args.ft_epochs = 0
    args.status = 'Infer'
    args.devices = int(args.devices)
    args.batch_size = args.ft_batch_size
    args.effective_bsz = args.ft_batch_size
    args.data_dir = "../" + args.data_dir
    args.strategy = None

    trainer_ft = pl.Trainer.from_argparse_args(
        args, max_epochs=args.ft_epochs,
        deterministic=True,
        enable_checkpointing=False,
        fast_dev_run=False,
        sync_batchnorm=True,
        accelerator='cpu',
        check_val_every_n_epoch=args.ft_val_freq,
        accumulate_grad_batches=args.ft_accumulate_grad_batches
    )

    _, ft_dm, args = get_dm(args)
    checkpoint_path = '../model/latest_checkpoint.ckpt'

    encoder = VICReg.load_from_checkpoint(checkpoint_path)
    model = SSLLinearEval([encoder.encoder_online, encoder.encoder_stacked], stack=True, **args.__dict__)

    trainer_ft.fit(model, ft_dm)
    checkpoint_path = "../model/best_epoch.ckpt"

    # Perform inference
    trainer_ft.test(ckpt_path=checkpoint_path, datamodule=ft_dm)



if __name__ == '__main__':
    infer()