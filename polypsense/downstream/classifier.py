import torch.nn as nn


def get_classifier(cls_type, **kwargs):
    if cls_type == "linear":
        return LinearClassifier(
            out_dim=kwargs.get("in_dim"),
            use_layer_norm=kwargs.get("use_layer_norm"),
            weights_init=kwargs.get("weights_init"),
            num_classes=kwargs.get("out_dim"),
        )
    elif cls_type == "mlp":
        return MLPClassifier(
            in_dim=kwargs.get("in_dim"),
            hidden_dim=kwargs.get("hidden_dim"),
            out_dim=kwargs.get("out_dim"),
        )
    else:
        raise ValueError(f"Classifier type {cls_type} not supported.")


class LinearClassifier(nn.Module):
    WEIGHTS_INIT_LIST = ["default", "normal"]

    def __init__(self, *, out_dim, use_layer_norm, weights_init, num_classes):
        super().__init__()

        if weights_init not in self.WEIGHTS_INIT_LIST:
            raise ValueError(
                f"Weights init {weights_init} not supported, please choose from {self.WEIGHTS_INIT_LIST}."
            )

        linear_layer = nn.Linear(out_dim, num_classes)

        if weights_init == "normal":
            nn.init.normal_(linear_layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(linear_layer.bias)

        if use_layer_norm:
            self.linear = nn.Sequential(
                nn.LayerNorm(out_dim),
                linear_layer,
            )
        else:
            self.linear = linear_layer

    def forward(self, x):
        return self.linear(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)
