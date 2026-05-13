import torch
from transformers import DINOv3ViTModel


class DinoV3(torch.nn.Module):
    """
    Implement DinoV3 model.

    Shapes for ViT small are as follows:
    - 1 cls token
    - 4 register tokens
    - 196 patch tokens (14x14)
     => total 201 tokens
    - d = 384
    """

    def __init__(
        self,
        pretrained_weights,
    ):
        super().__init__()

        self.model = DINOv3ViTModel.from_pretrained(pretrained_weights)
        self.model.train()

    def forward(self, x):
        # x [b, t, c, h, w]
        b, t, c, h, w = x.shape
        x = x.flatten(0, 1)  # [b * t, c, h, w]
        x = self.model(x).last_hidden_state  # [b * t, 1 + r + p, d]
        x = x.view(b, t, -1, x.shape[-1])  # [b, t, 1 + r + p, d]
        return x  # [b, t, 1 + r + p, d]


class DinoV3Pooler(torch.nn.Module):
    def __init__(self, spatial_pooling, temporal_pooling):
        super().__init__()
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

    def forward(self, x):

        if self.spatial_pooling == "cls":
            x = x[:, :, 0, :]  # [b, t, d]

        if self.temporal_pooling == "mean":
            x = x.mean(dim=-2)  # [b, d]

        return x
