import logging

import lightning as L
import torch

from polypsense.downstream.engine_util import get_callbacks, get_logger
from polypsense.downstream.histology.dm import HistologyDataModule
from polypsense.downstream.histology.evaluator import HistologyEvaluator
from polypsense.downstream.zoo import get_encoder


def train(args):
    logging.basicConfig(level=logging.INFO)

    torch.set_float32_matmul_precision("high")

    L.seed_everything(args.seed, workers=True)

    dm = get_dm(args)
    encoder = get_encoder(args)
    evaluator = get_evaluator(encoder, args)

    engine = L.Trainer(
        accelerator="cuda",
        devices=1,
        logger=get_logger(args),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=args.max_epochs,
        callbacks=get_callbacks(args),
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    stages = get_stages(args)

    if "fit" in stages:
        engine.validate(evaluator, dm)
        engine.fit(evaluator, dm)

    if "validate" in stages:
        engine.validate(evaluator, dm)

    if "test" in stages:
        engine.test(evaluator, dm)


def get_dm(args):
    return HistologyDataModule(
        train_images=args.train_images,
        train_annotations=args.train_annotations,
        val_images=args.val_images,
        val_annotations=args.val_annotations,
        batch_size=args.batch_size,
        im_size=args.data_im_size,
        fragment_length=args.data_fragment_length,
        fragment_stride=args.data_fragment_stride,
        fragment_drop_last=args.data_fragment_drop_last,
        fragment_padding_mode=args.data_fragment_padding_mode,
        aug_resize=args.data_aug_resize,
        aug_anchorcrop=args.data_aug_anchorcrop,
        aug_normalize=args.data_aug_normalize,
        bbox_scale_factor=args.data_bbox_scale_factor,
        class_mapping=args.class_mapping,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def get_evaluator(encoder, args):
    return HistologyEvaluator(
        encoder,
        encoder_pooling=args.encoder_pooling,
        encoder_out_dim=args.encoder_out_dim,
        n_classes=args.n_classes,
        opt_type=args.opt_type,
        opt_lr=args.opt_lr,
        opt_weight_decay=args.opt_weight_decay,
        lr_sched_type=args.lr_sched_type,
        cls_type=args.cls_type,
        cls_hidden_dim=args.cls_hidden_dim,
        cls_init=args.cls_init,
        cls_use_layer_norm=args.cls_use_layer_norm,
        pos_weight=args.pos_weight,
        exp_name=args.exp_name,
        train_encoder=args.train_encoder,
    )


def get_stages(args):
    return ["fit", "test"]
