from abc import ABC, abstractmethod

import torch

POOLING_CHOICES = [
    "none",
    "cls",
    "mean",
    "patchmean",
    "cat",
    "cls-mean",
    "mean-mean",
    "patchmean-mean",
    "cls-cat",
    "mean-cat",
    "patchmean-cat",
    "cat-cat",
]


class Pooler(torch.nn.Module, ABC):
    def __init__(self, in_dim):
        super().__init__()
        self._in_dim = in_dim

    @property
    def in_dim(self):
        return self._in_dim

    @property
    @abstractmethod
    def out_dim(self):
        raise NotImplementedError()

    def forward(self, x):
        """
        Args:
            x: A tensor of size [b, p, d]  where p is the number of tokens

        Returns:
            A tensor of size [b, d]
        """
        raise NotImplementedError()


class IdentityPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x  # [b, p, d]


class MeanPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x.mean(dim=-2)  # [b, d]


class ClsPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x[:, 0, :]  # [b, d]


class PatchMeanPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim

    def forward(self, x):
        return x[:, 1:, :].mean(dim=-2)  # [b, d]


class CatPooler(Pooler):
    @property
    def out_dim(self):
        return self.in_dim * 2

    def forward(self, x):
        cls = x[:, 0, :]  # [b, d]
        patchmean = x[:, 1:, :].mean(dim=-2)  # [b, d]
        return torch.cat([cls, patchmean], dim=-1)  # [b, 2 * d]


class SpaceTimePooler(Pooler):
    def __init__(self, in_dim, spatial_pooling, temporal_pooling):
        super().__init__(in_dim)
        self.spatial_pooler = get_pooler(spatial_pooling, in_dim)
        self.temporal_pooler = get_pooler(temporal_pooling, self.spatial_pooler.out_dim)

    @property
    def out_dim(self):
        return self.temporal_pooler.out_dim

    def forward(self, x):
        b, t, p, d = x.shape
        x = x.view(b * t, p, d)  # [b * t, p, d]
        x = self.spatial_pooler(x)  # [b * t, d']
        x = x.view(b, t, -1)  # [b, t, d']
        x = self.temporal_pooler(x)  # [b, d'']
        return x


def get_pooler(pooling: str, in_dim: int) -> Pooler:
    if pooling not in POOLING_CHOICES:
        raise ValueError(
            f"Invalid pooling method. Choose from {POOLING_CHOICES} or do not set."
        )

    if pooling == "mean":
        return MeanPooler(in_dim)
    elif pooling == "cls":
        return ClsPooler(in_dim)
    elif pooling == "patchmean":
        return PatchMeanPooler(in_dim)
    elif pooling == "cat":
        return CatPooler(in_dim)
    elif _is_space_time(pooling):
        spatial_pooling, temporal_pooling = pooling.split("-")
        return SpaceTimePooler(in_dim, spatial_pooling, temporal_pooling)
    else:
        return IdentityPooler(in_dim)


def _is_space_time(pooling):
    return "-" in pooling
