import lightning as L
import torch
import torch.nn as nn
import torchvision.models as models

from polypsense.e2e.loss import MILNCELoss
from polypsense.e2e.metric import accuracy, get_map_score


class SFEv3(nn.Module):
    def __init__(self, backbone_arch, backbone_weights):
        super().__init__()

        self.backbone_arch = backbone_arch
        self.backbone_weights = backbone_weights

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

    def forward(self, x):
        if not x.dim() == 4:
            raise ValueError(
                f"Input tensor must have 4 dimensions [b, c, h, w], got input with shape {x.shape}"
            )

        x = self.backbone(x)  # [b, d]

        return x  # [b, d]


class MVEv3(nn.Module):
    def __init__(
        self,
        sfe,
        d_in,
        d_model,
        d_feedforward,
        dropout,
        n_heads,
        n_layers,
        use_feature_norm,
        use_encoder_norm,
    ):
        super().__init__()

        self.sfe = sfe

        self.adapter = nn.Linear(d_in, d_model)
        self.feature_norm = nn.LayerNorm(d_model) if use_feature_norm else nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [1, 1, d]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        encoder_norm = nn.LayerNorm(d_model) if use_encoder_norm else None
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=encoder_norm,
        )

    def forward(self, x):
        b, s, c, h, w = x.shape

        x = x.reshape(b * s, c, h, w)  # [b*s, c, h, w]
        x = self.sfe(x)  # [b*s, d_in]

        x = x.reshape(b, s, -1)  # [b, s, d_in]
        x = self.adapter(x)  # [b, s, d_model]
        x = self.feature_norm(x)  # [b, s, d_model]

        x = self._prepend_cls_token(x)  # [b, 1+s, d_model]
        x = self.encoder(x)  # [b, 1+s, d_model]

        return x  # [b, 1+s, d_model]

    def _prepend_cls_token(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [b, 1, d]
        return torch.cat((cls_tokens, x), dim=1)  # [b, 1+s, d]


class HierarchicalMVEEncoder(L.LightningModule):
    def __init__(
        self,
        *,
        backbone_arch=None,
        backbone_weights=None,
        d_in=None,
        d_proj=None,
        d_model=None,
        d_feedforward=2048,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        use_feature_norm=None,
        use_encoder_norm=None,
        loss_lr=None,
        loss_temperature=None,
        loss_weights=None,
        n_views=None,
        mask_intra_fragment_frames=None,
    ):
        super().__init__()

        if loss_weights is None:
            raise ValueError("loss_weights must be provided for HierarchicalMVEEncoder")

        self.lr = loss_lr
        self.k = n_views
        self.weights = loss_weights
        self.mask_intra_fragment_frames = mask_intra_fragment_frames

        sfe = SFEv3(
            backbone_arch=backbone_arch,
            backbone_weights=backbone_weights,
        )

        self.mve = MVEv3(
            sfe=sfe,
            d_in=d_in,
            d_model=d_model,
            d_feedforward=d_feedforward,
            dropout=dropout,
            n_heads=n_heads,
            n_layers=n_layers,
            use_feature_norm=use_feature_norm,
            use_encoder_norm=use_encoder_norm,
        )

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_proj),
        )

        self._milnce_criterion = MILNCELoss(loss_temperature, return_logits=True)

        self.save_hyperparameters()

    def forward(self, x):
        return self.mve(x)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["clip"]  # [b, s, c, h, w]
        identity = batch["identity"]  # [b]

        # forward the model
        h = self(x)  # [b, 1+s, d]

        # get projections
        z = self.projector(h)  # [b, 1+s, p]

        # separe fragment (global) and frame (local) representations
        z_fragment = z[:, 0, :].unsqueeze(1)  # [b, 1, d]
        z_frame = z[:, 1:, :]  # [b, s, d]

        # compute losses
        # - inter-fragment fragment bag
        # - inter-fragment frame bag
        # - intra-fragment frame bag

        b = z_frame.size(0)
        s = z_frame.size(1)
        device = z_frame.device

        # inter-fragment fragment bag
        y_xffragment = torch.arange(b // self.k).repeat_interleave(self.k)  # [b]
        y_xffragment = y_xffragment.to(device)
        loss_xffragment, logits_xffragment = self._milnce_criterion(
            z_fragment.reshape(b, -1), y_xffragment
        )

        # inter-fragment frame bag
        ignore = (
            get_intra_fragment_frames_mask(b, self.k, s, device=device)
            if self.mask_intra_fragment_frames
            else None
        )

        y_xfframe = torch.arange(b // self.k).repeat_interleave(self.k * s)  # [b * s]
        y_xfframe = y_xfframe.to(device)
        loss_xfframe, logits_xfframe = self._milnce_criterion(
            z_frame.reshape(b * s, -1),
            y_xfframe,
            ignore=ignore,
        )

        # intra-fragment frame bag
        y_ifframe = torch.arange(b).repeat_interleave(s)  # [b * s]
        y_ifframe = y_ifframe.to(device)
        loss_ifframe, logits_ifframe = self._milnce_criterion(
            z_frame.reshape(b * s, -1), y_ifframe
        )

        w1, w2, w3 = self.weights

        # total loss
        loss = w1 * loss_xffragment + w2 * loss_xfframe + w3 * loss_ifframe

        self.log(f"train_loss", loss, prog_bar=True)

        self.log(f"train/xffragment/loss", loss_xffragment)
        self.log(f"train/xfframe/loss", loss_xfframe)
        self.log(f"train/ifframe/loss", loss_ifframe)

        top1_xffragment, top5_xffragment = accuracy(
            logits_xffragment, identity, topk=(1, 5)
        )
        self.log(f"train/xffragment/top1", top1_xffragment)
        self.log(f"train/xffragment/top5", top5_xffragment)
        preds = identity[
            logits_xffragment.topk(logits_xffragment.size(0), dim=1)[1]
        ]  # [b, b]
        map = get_map_score(identity, preds, identity)
        self.log(f"train/xffragment/map", map)

        return loss

    def eval_step(self, batch, stage=None):
        x = batch["clip"]  # [b, s, c, h, w]
        identity = batch["identity"]  # [b]

        # forward the model
        h = self(x)  # [b, 1+s, d]

        # get projections
        z = self.projector(h)  # [b, 1+s, p]

        z_pool = z[:, 0, :]  # [b, d]

        loss, logits = self._milnce_criterion(z_pool, identity)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            top1, top5 = accuracy(logits, identity, topk=(1, 5))
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)
            preds = identity[logits.topk(logits.size(0), dim=1)[1]]  # [b, b]
            map = get_map_score(identity, preds, identity)
            self.log(f"{stage}_map", map)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        return optimizer


def get_intra_fragment_frames_mask(n, k, s, device=None):
    """
    Returns a mask with ones on the diagonal blocks corresponding to frames from
    the same fragment, and zeros elsewhere.

    Let
        F(i) = i // s be the fragment index of frame i
    then M[i, j] =
     - 1 (to mask)   if F(i) = F(j)
     - 0 (to keep)   otherwise

    Args:
        n: number of fragments
        k: number of views (size of the bag)
        s: number of frames per fragment
        device: device where to create the mask tensor
    """
    # with n=60, k=5, s=8
    b = n // k  # 12 (number of bags)
    N = b * k * s  # 480 (total frames)

    idx = torch.arange(N, device=device)
    fragment_idx = idx // s  # (480,)

    return fragment_idx[:, None] == fragment_idx[None, :]
