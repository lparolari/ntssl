import lightning as L
import torch
import torch.nn as nn

from polypsense.e2e.loss import get_criterion
from polypsense.e2e.metric import accuracy, get_map_score


class SFEv2(nn.Module):
    def __init__(self, backbone_arch, backbone_weights):
        super().__init__()

        self.backbone_arch = backbone_arch
        self.backbone_weights = backbone_weights

        self._build_backbone()

    def forward(self, x):
        if not x.dim() in [4, 5]:
            raise ValueError(
                f"Input tensor must have 4 or 5 dimensions [b, s, c, h, w] "
                f"(s can be omitted), got input with shape {x.shape}"
            )

        if x.dim() == 4:
            x = x.unsqueeze(1)  # [b, 1, c, h, w]

        b, s, c, h, w = x.shape

        x = x.view(b * s, c, h, w)
        x = self.backbone(x)  # [b*s, d]
        x = x.view(b, s, -1)  # [b, s, d]
        x = x.mean(dim=1)  # [b, d]

        return x  # [b, d]

    def _build_backbone(self):
        import torchvision.models as models

        resnet_dict = {
            "resnet18": lambda: models.resnet18(weights=self.backbone_weights),
            "resnet50": lambda: models.resnet50(weights=self.backbone_weights),
        }

        if self.backbone_arch not in resnet_dict:
            raise ValueError(
                f"Invalid '{self.backbone_arch}' backbone. Use one of: {list(resnet_dict.keys())}"
            )

        self.backbone = resnet_dict[self.backbone_arch]()
        self.backbone.fc = nn.Identity()  # remove fc


class SFEv2Module(L.LightningModule):
    def __init__(
        self,
        *,
        backbone_arch,
        backbone_weights,
        d_model,
        d_proj,
        lr=None,
        loss_type=None,
        loss_temperature=None,
        loss_lambda=None,
        loss_gamma=None,
    ):
        super().__init__()

        self.lr = lr

        self.model = SFEv2(backbone_arch, backbone_weights)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self.criterion = get_criterion(
            loss_type,
            loss_temperature,
            loss_lambda,
            loss_gamma,
            return_logits=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)  # [b, d]

    def step(self, batch, stage=None):
        x, y = batch["clip"], batch["label"]
        position_identity = batch["position_identity"]

        h = self(x)  # [b, d]
        z = self.head(h)  # [b, p]

        loss, logits = self.criterion(z, y, position_identity)  # [1], [b, b]

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            top1, top5 = accuracy(logits, y, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)
            preds = y[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(y, preds, y)
            self.log(f"{stage}_map", map)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        return optimizer


class MVEv2(nn.Module):
    def __init__(
        self,
        sfe,
        d_in,
        d_model,
        d_feedforward,
        dropout,
        n_heads,
        n_layers,
        pooling_strategy,
        max_seq_len,
        use_positional_encoding,
    ):
        super().__init__()

        self.sfe = sfe

        self.d_in = d_in
        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.dropout = dropout
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len
        self.use_positional_encoding = use_positional_encoding

        self._build_model()

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)  # [b*s, c, h, w]
        x = self.sfe(x)  # [b*s, d']
        x = self.adapter(x)  # [b*s, d]
        x = x.view(b, s, -1)  # [b, s, d]
        if self.use_positional_encoding:
            x = x + self.pos_embedding[:, :s, :]  # [b, s, d]
        x = self._prepend_cls_token(x)  # [b, 1+s, d]
        x = self.transformer_encoder(x)  # [b, 1+s, d]
        return self._pooling(x)  # [b, d]

    def _prepend_cls_token(self, x):
        if self.pooling_strategy in ["mean", "max"]:
            return x  # x: [b, s, d]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [b, 1, d]
        return torch.cat((cls_tokens, x), dim=1)  # [b, 1+s, d]

    def _pooling(self, x):
        if self.pooling_strategy == "mean":
            return x.mean(dim=1)
        if self.pooling_strategy == "max":
            return x.max(dim=1)[0]
        return x[:, 0]

    def _build_model(self):
        self.adapter = nn.Linear(self.d_in, self.d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_feedforward,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.n_layers,
        )

        if self.pooling_strategy not in ["mean", "max"]:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))  # [1, 1, d]

        if self.use_positional_encoding:
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.max_seq_len, self.d_model)
            )  # [1, max_seq_len, d]
            # Initialize with small random values
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)


class MVEv2Module(L.LightningModule):
    def __init__(
        self,
        *,
        sfe_backbone_arch=None,
        sfe_backbone_weights=None,
        sfe_ckpt=None,
        sfe_freeze=None,
        d_in=None,
        d_proj=None,
        d_model=None,
        d_feedforward=2048,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        pooling_strategy=None,
        lr=None,
        loss_type=None,
        loss_temperature=None,
        loss_lambda=None,
        loss_gamma=None,
        loss_epsilon=None,
        loss_supervision=None,
        max_seq_len=None,
        use_positional_encoding=None,
        batch_size=None,
        n_views=None,
    ):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.n_views = n_views

        sfe = self._get_sfe(
            sfe_backbone_arch,
            sfe_backbone_weights,
            sfe_ckpt,
            sfe_freeze,
        )

        self.model = MVEv2(
            sfe=sfe,
            d_in=d_in,
            d_model=d_model,
            d_feedforward=d_feedforward,
            dropout=dropout,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling_strategy=pooling_strategy,
            max_seq_len=max_seq_len,
            use_positional_encoding=use_positional_encoding,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self.criterion = get_criterion(
            loss_type,
            loss_temperature,
            loss_lambda,
            loss_gamma,
            loss_epsilon,
            supervision=loss_supervision,
            return_logits=True,
        )

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage=None):
        x = batch["clip"]
        identity = batch["identity"]
        position_identity = batch["position_identity"]

        h = self(x)  # [b, d]
        z = self.head(h)  # [b, p]

        loss, logits = self.criterion(z, identity, position_identity)  # [1], [b, b]

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)
            preds = identity[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(identity, preds, identity)
            self.log(f"{stage}_map", map)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        return optimizer

    def _get_sfe(
        self,
        sfe_backbone_arch,
        sfe_backbone_weights,
        sfe_ckpt,
        sfe_freeze,
    ):
        if sfe_ckpt:
            sfe = SFEv2Module.load_from_checkpoint(
                sfe_ckpt,
                map_location="cpu",
            ).model

            if sfe_freeze:
                sfe.eval()
                for param in sfe.parameters():
                    param.requires_grad = False

            return sfe

        return SFEv2(sfe_backbone_arch, sfe_backbone_weights)
