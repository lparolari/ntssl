import lightning as L


def get_logger(args):
    import os

    logger = L.pytorch.loggers.WandbLogger(
        entity=args.wandb_entity,
        project=args.wandb_project,
        id=args.exp_id,
        name=args.exp_name,
        save_dir=os.path.join(os.getcwd(), "wandb_logs"),
        allow_val_change=True,
        resume="allow",
        config=vars(args),
    )
    return logger


def get_callbacks(args):
    return [
        L.pytorch.callbacks.LearningRateMonitor(),
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            filename="epoch={epoch:02d}-step={step}-val_loss={val/loss:.2f}",
            auto_insert_metric_name=False,
        ),
    ]
