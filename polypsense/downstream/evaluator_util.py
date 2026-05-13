import torch


class OutputAccumulator:
    def __init__(self):
        self.outputs = []

    def gather(self):
        preds = torch.cat([out["pred"] for out in self.outputs], dim=0)
        targets = torch.cat([out["target"] for out in self.outputs], dim=0)
        identities = torch.cat([out["identity"] for out in self.outputs], dim=0)
        return preds, targets, identities

    def clear(self):
        self.outputs = []

    def accumulate(self, pred, target, identity):
        self.outputs.append(
            {
                "pred": pred.detach().cpu(),
                "target": target.detach().cpu(),
                "identity": identity.detach().cpu(),
            }
        )
