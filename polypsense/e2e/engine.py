import os

import torch
import lightning as L

from polypsense.e2e.dm import End2EndDataModule
from polypsense.e2e.factory import get_model


def train(args):
    torch.set_float32_matmul_precision("high")

    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = get_model(args)

    trainer = L.Trainer(
        logger=get_logger(args, model),
        callbacks=get_callbacks(args),
        max_epochs=args.max_epochs,
        accelerator="cuda",
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.test(model, dm)
    trainer.fit(model, dm, ckpt_path=args.resume and args.ckpt_path)
    trainer.test(model, dm, ckpt_path="best")


def eval(args):
    torch.set_float32_matmul_precision("highest")

    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = get_model(args)

    trainer = L.Trainer(
        logger=get_logger(args, model),
        accelerator="cuda",
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.test(model, dm)


def get_datamodule(args):
    default_eval_n_views = 2
    default_eval_batch_size = 16
    default_eval_fragment_length = 8

    return End2EndDataModule(
        dataset_root=args.dataset_root,
        im_size=args.im_size,
        fragment_length=args.fragment_length,
        fragment_stride=args.fragment_stride,
        fragment_drop_last=args.fragment_drop_last,
        fragment_padding_mode=args.fragment_padding_mode,
        bbox_scale_factor=args.bbox_scale_factor,
        min_tracklet_length=args.min_tracklet_length,
        sampler=args.sampler,
        sampler_ttb_tmin=args.sampler_ttb_tmin,
        sampler_ttb_tmax=args.sampler_ttb_tmax,
        aug_resize=args.aug_resize,
        aug_anchorcrop=args.aug_anchorcrop,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_views=args.n_views,
        seed=args.seed,
        eval_n_views=args.eval_n_views or default_eval_n_views,
        eval_batch_size=args.eval_batch_size or default_eval_batch_size,
        eval_fragment_length=args.eval_fragment_length or default_eval_fragment_length,
        eval_aug_resize=args.eval_aug_resize,
        eval_aug_anchorcrop=args.eval_aug_anchorcrop,
    )


def get_logger(args, model):
    logger = L.pytorch.loggers.WandbLogger(
        entity="your-wandb-entity",
        project=args.wandb_project,
        id=args.exp_id,
        name=args.exp_name,
        save_dir=os.path.join(os.getcwd(), "wandb_logs"),
        config=vars(args),
        allow_val_change=True,
        resume="allow",
    )
    # wandb.watch breaks the ability to save nn.Modules as hparams. This happens
    # because the watch method attach a non-pickble object to the module and
    # Pytorch Lightning removes non pickable objects from the hparams before
    # saving it. See https://github.com/wandb/wandb/issues/2588.
    return logger


def get_callbacks(args):
    ckpt_dirpath = (
        os.path.join(
            os.getcwd(),
            "wandb_logs",
            args.wandb_project,
            args.exp_name,
            "checkpoints",
        )
        if args.exp_name
        else None
    )

    # check notebook `videomae_weights_only.ipynb` for details on how to use
    # checkpoints saved with weights_only=True

    best_model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=ckpt_dirpath,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{step}-{val_loss:.2f}-best",
        save_last="link",
        save_weights_only=True,
        enable_version_counter=False,
    )
    best_model_checkpoint.CHECKPOINT_NAME_LAST = "best"

    last_model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=ckpt_dirpath,
        monitor="step",
        mode="max",
        filename="{epoch}-{step}-last",
        save_last="link",
        enable_version_counter=False,
    )

    return [
        best_model_checkpoint,  # must be first in callback list
        last_model_checkpoint,
        L.pytorch.callbacks.LearningRateMonitor(),
    ]
