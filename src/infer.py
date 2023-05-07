import pytorch_lightning as pl
from configargparse import ArgumentParser, Action
from dataloaders.handlers import get_dm
from vicreg import VICReg
from lin_eval import SSLLinearEval


def infer():
    parser = ArgumentParser(
        description='Pytorch Lightning VICReg', default_config_files=["Data/model/commandline_args.txt"])
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

    args.ft_epochs = 0
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
    dm, ft_dm, args = get_dm(args)
    checkpoint_path = '../Data/latest_checkpoint.ckpt'

    encoder = VICReg.load_from_checkpoint(checkpoint_path)
    model = SSLLinearEval([encoder.encoder_online, encoder.encoder_stacked], stack=True, **args.__dict__)
    trainer_ft.fit(model, ft_dm)

    checkpoint_path = "../Data/model/finetune/best_epoch.ckpt"
    # Perform inference
    trainer_ft.test(ckpt_path=checkpoint_path, datamodule=ft_dm)


if __name__ == '__main__':
    infer()