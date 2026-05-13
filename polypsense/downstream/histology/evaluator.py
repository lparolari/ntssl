import logging
import pathlib
from typing import Optional

import lightning as L
import torch

from polypsense.downstream.classifier import get_classifier
from polypsense.downstream.evaluator_util import OutputAccumulator
from polypsense.downstream.metric import accuracy_identity, accuracy_instance
from polypsense.downstream.pooler import get_pooler

logger = logging.getLogger(__name__)


class HistologyEvaluator(L.LightningModule):
    TRAIN_LOSS = "train/loss"
    TRAIN_ACC_INSTANCE = "train/acc_instance"
    TRAIN_ACC_IDENTITY = "train/acc_identity"
    VAL_ACC_IDENTITY = "val/acc_identity"
    VAL_ACC_INSTANCE = "val/acc_instance"
    VAL_LOSS = "val/loss"
    TEST_ACC_IDENTITY = "test/acc_identity"
    TEST_ACC_INSTANCE = "test/acc_instance"
    TEST_LOSS = "test/loss"

    OPT_TYPE_LIST = ["adamw", "sgd"]
    LR_SCHED_TYPE_LIST = ["none", "onecycle"]

    def __init__(
        self,
        encoder,
        *,
        train_encoder: bool,
        encoder_pooling: str,
        encoder_out_dim: int,
        n_classes: int,
        opt_type: str,
        opt_lr: float,
        opt_weight_decay: float,
        lr_sched_type: str,
        cls_type: str,
        cls_init: str,
        cls_hidden_dim: int,
        cls_use_layer_norm: bool,
        pos_weight: Optional[float | torch.Tensor],
        exp_name: str,
    ):
        super().__init__()

        if n_classes != 2:
            raise NotImplementedError("Only binary classification is supported.")

        if isinstance(pos_weight, float):
            pos_weight = torch.tensor(pos_weight)

        self.encoder = encoder
        self.train_encoder = train_encoder
        self.opt_type = opt_type
        self.opt_lr = opt_lr
        self.opt_weight_decay = opt_weight_decay
        self.lr_sched_type = lr_sched_type
        self.output_dir = pathlib.Path("output/histologycls")
        self.exp_name = exp_name

        self._setup_encoder()

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.pooler = get_pooler(encoder_pooling, in_dim=encoder_out_dim)
        self.classifier = get_classifier(
            # fmt: off
            cls_type, **{
                "in_dim": self.pooler.out_dim, "out_dim": 1, # shared
                "use_layer_norm": cls_use_layer_norm, "weights_init": cls_init, # linear
                "hidden_dim": cls_hidden_dim, # mlp
            },
            # fmt: on
        )

        self.train_accumulator = OutputAccumulator()
        self.val_accumulator = OutputAccumulator()
        self.test_accumulator = OutputAccumulator()

        self.save_hyperparameters(ignore=["encoder"])

    def forward(self, x):
        # x: [b, t, c, h, w]
        x = self.encoder(x)  # [b, p, d]
        x = self.pooler(x)  # [b, d]
        logits = self.classifier(x)  # [b, c]
        logits = logits.squeeze(-1)  # [b]
        return logits  # [b]

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = target["histology"]
        identity = target["identity"]
        logits = self.forward(x)  # [b]
        loss = self.loss(logits, y.float())  # scalar
        self.log(self.TRAIN_LOSS, loss, prog_bar=True)
        self.train_accumulator.accumulate(logits, y, identity)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        y = target["histology"]
        identity = target["identity"]
        logits = self.forward(x)  # [b]
        loss = self.loss(logits, y.float())  # scalar
        self.log(self.VAL_LOSS, loss, prog_bar=True)
        self.val_accumulator.accumulate(logits, y, identity)
        return loss

    def test_step(self, batch, batch_idx):
        x, target = batch
        y = target["histology"]
        identity = target["identity"]
        logits = self.forward(x)  # [b]
        loss = self.loss(logits, y.float())  # scalar
        self.log(self.TEST_LOSS, loss, prog_bar=True)
        self.test_accumulator.accumulate(logits, y, identity)
        return loss

    def on_train_epoch_end(self):
        preds, targets, identity = self.train_accumulator.gather()

        instance_acc = accuracy_instance(preds, targets)
        self.log(self.TRAIN_ACC_INSTANCE, instance_acc, prog_bar=True)

        identity_acc = accuracy_identity(preds, targets, identity)
        self.log(self.TRAIN_ACC_IDENTITY, identity_acc, prog_bar=True)

        self._save_outputs(
            self.train_accumulator.gather(),
            subdir=f"train_epoch={self.current_epoch:02d}",
        )
        self.train_accumulator.clear()

    def on_validation_epoch_end(self):
        preds, targets, identity = self.val_accumulator.gather()

        instance_acc = accuracy_instance(preds, targets)
        self.log(self.VAL_ACC_INSTANCE, instance_acc, prog_bar=True)

        identity_acc = accuracy_identity(preds, targets, identity)
        self.log(self.VAL_ACC_IDENTITY, identity_acc, prog_bar=True)

        self._save_outputs(
            self.val_accumulator.gather(), subdir=f"val_epoch={self.current_epoch:02d}"
        )
        self.val_accumulator.clear()

    def on_test_epoch_end(self):
        preds, targets, identity = self.test_accumulator.gather()

        instance_acc = accuracy_instance(preds, targets)
        self.log(self.TEST_ACC_INSTANCE, instance_acc, prog_bar=True)

        identity_acc = accuracy_identity(preds, targets, identity)
        self.log(self.TEST_ACC_IDENTITY, identity_acc, prog_bar=True)

        self._save_outputs(self.test_accumulator.gather(), subdir="test")
        self.test_accumulator.clear()

    def configure_optimizers(self):
        if self.opt_type not in self.OPT_TYPE_LIST:
            raise ValueError(
                f"Optimizer type {self.opt_type} not supported, please choose from {self.OPT_TYPE_LIST}."
            )

        if self.opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.opt_lr,
                momentum=0.9,
                weight_decay=self.opt_weight_decay,
            )

        if self.opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.opt_lr,
                weight_decay=self.opt_weight_decay,
            )

        if self.lr_sched_type not in self.LR_SCHED_TYPE_LIST:
            raise ValueError(
                f"LR scheduler type {self.lr_sched_type} not supported, please choose from {self.LR_SCHED_TYPE_LIST}."
            )

        lr_schedulers = []

        if self.lr_sched_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.opt_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
            )
            lr_schedulers.append({"scheduler": scheduler, "interval": "step"})

        return [optimizer], lr_schedulers

    def _setup_encoder(self):
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def _save_outputs(self, outputs, *, subdir):
        out_file = self.output_dir / self.exp_name / subdir / "outputs.pth"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        torch.save(outputs, out_file)
