import torch
from transformers import Dinov2Model


class DinoV2Pooler(torch.nn.Module):
    def __init__(self, spatial_pooling, temporal_pooling):
        """
        Args:
            spatial_pooling: str, one of ["cls", "mean", "patchmean", "cat"]
                - "cls": use the class token
                - "mean": use the mean of all patches
                - "patchmean": use the mean of all patches except the class token
                - "cat": concatenate the class token and the mean of all patches

            temporal_pooling: str, one of ["mean", "cat"]
                - "mean": use the mean across time steps
                - "cat": concatenate across all time steps
        """
        super().__init__()
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

    def forward(self, x):
        b, t, p, d = x.shape

        if self.spatial_pooling == "cls":
            x = x[:, :, 0, :]  # [b, t, d]

        if self.spatial_pooling == "mean":
            x = x.mean(dim=-2)  # [b, t, d]

        if self.spatial_pooling == "patchmean":
            x = x[:, 1:, :].mean(dim=-2)  # [b, t, d]

        if self.spatial_pooling == "cat":
            cls = x[:, :, 0, :]  # [b, t, d]
            patchmean = x[:, :, 1:, :].mean(dim=-2)  # [b, t, d]
            x = torch.cat([cls, patchmean], dim=-1)  # [b, t, 2 * d]

        if self.temporal_pooling == "mean":
            x = x.mean(dim=-2)  # [b, d]

        if self.temporal_pooling == "cat":
            x = x.reshape(b, -1)  # [b, t * d]

        return x


class DinoV2(torch.nn.Module):
    def __init__(
        self,
        pretrained_weights,
    ):
        super().__init__()

        self.model = Dinov2Model.from_pretrained(pretrained_weights)
        self.model.train()

    def forward(self, x):
        # x [b, t, c, h, w]
        b, t, c, h, w = x.shape
        x = x.flatten(0, 1)  # [b * t, c, h, w]
        x = self.model(x).last_hidden_state  # [b * t, 1 + p, d]
        x = x.view(b, t, -1, x.shape[-1])  # [b, t, 1 + p, d]
        return x  # [b, t, 1 + p, d]
