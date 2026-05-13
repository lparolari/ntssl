import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cross_entropy(p, q):
    """
    See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    for details.

    NOTE: this should return the same value as `F.cross_entropy` but for some
    reason original authors reimplemented it. The signature is different though:
    p comes first.
    """
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    """
    Stabilizes the input logits by subtracting the maximum value along the last
    dimension.

    This technique is often used to improve numerical stability when computing
    functions involving exponentials, such as the softmax function. By shifting
    the logits, it prevents potential overflow or underflow issues.
    """
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.return_logits = return_logits

    def forward(self, feats, labels, *_, **__):
        # feats: [b, d]
        # labels: [b]

        b = feats.size(0)
        device = feats.device

        feats = F.normalize(feats, dim=-1, p=2)  # [b, d]

        targets = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()  # [b, b]

        # get the self mask, note that original implementation uses
        # `torch.scatter` to mask portion of the matrix interesed by current
        # node (rank) computations:
        # https://github.com/google-research/syn-rep-learn/blob/45f451b0d53d25eecdb4d7b9e5a852e1c43e7f5b/StableRep/models/losses.py#L78-L91
        mask = 1 - torch.eye(b).to(device)  # [b] (float)

        # apply self-masking on targets
        targets = targets * mask  # [b, b]

        # compute logits and apply self masking on logits, -1e9 is used to get 0
        # probability in cross-entry
        logits = torch.matmul(feats, feats.T) / self.temperature  # [b, b]

        # apply self-masking on logits: logits' maximum value is 1 on the
        # diagonal (from matmul of normalized feat), so
        # - on the diagonal remove (1 - mask) * 1e9, where mask is 0, i.e. -1e9
        # - off the diag remove (1 - mask) * 1e9, where mask is 1, i.e. 0
        logits = logits - (1 - mask) * 1e9  # [b, b]

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)  # [b, b]

        # compute ground-truth distribution
        p = targets / targets.sum(1, keepdim=True).clamp(min=1.0)  # [b, b]

        # compute cross-entropy loss
        loss = compute_cross_entropy(p, logits)

        if self.return_logits:
            return loss, logits

        return loss


class DistanceAwareMultiPosConLoss(nn.Module):
    """
    Adaptation of Multi-Positive Contrastive Loss
    (https://arxiv.org/pdf/2306.00984.pdf) to support soft targets originated
    from distances between samples.
    """

    def __init__(self, temperature=0.1, lam=1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.lam = lam
        self.return_logits = return_logits

    def forward(self, feats, labels, positions, *_, **__):
        # feats: [b, d]
        # labels: [b]

        b = feats.size(0)
        device = feats.device

        feats = F.normalize(feats, dim=-1, p=2)  # [b, d]

        targets = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()  # [b, b]

        # get the self mask, note that original implementation uses
        # `torch.scatter` to mask portion of the matrix interesed by current
        # node (rank) computations:
        # https://github.com/google-research/syn-rep-learn/blob/45f451b0d53d25eecdb4d7b9e5a852e1c43e7f5b/StableRep/models/losses.py#L78-L91
        mask = 1 - torch.eye(b).to(device)  # [b] (float)

        # apply self-masking on targets
        targets = targets * mask  # [b, b]

        # compute logits and apply self masking on logits, -1e9 is used to get 0
        # probability in cross-entry
        logits = torch.matmul(feats, feats.T) / self.temperature  # [b, b]

        # apply self-masking on logits: logits' maximum value is 1 on the
        # diagonal (from matmul of normalized feat), so
        # - on the diagonal remove (1 - mask) * 1e9, where mask is 0, i.e. -1e9
        # - off the diag remove (1 - mask) * 1e9, where mask is 1, i.e. 0
        logits = logits - (1 - mask) * 1e9  # [b, b]

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)  # [b, b]

        # compute supervised ground-truth distribution
        p = targets / targets.sum(1, keepdim=True).clamp(min=1.0)  # [b, b]

        # compute soft targets with distance awareness
        distances = torch.abs(positions.view(-1, 1) - positions.view(1, -1))  # [b, b]
        distances = torch.exp(-self.lam * distances)  # [b, b]
        distances = distances * targets  # [b, b]
        distances = distances / distances.sum(1, keepdim=True).clamp(
            min=1e-8
        )  # to distribution

        # apply distance-awareness on supervised targets
        p = p * distances
        p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # compute cross-entropy loss
        loss = compute_cross_entropy(p, logits)

        if self.return_logits:
            return loss, logits

        return loss


class MILNCELoss(torch.nn.Module):
    """
    Implement the MIL-NCE loss.
    """

    def __init__(self, temperature=0.07, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.return_logits = return_logits

    def forward(self, z, y, ignore=None):
        """
        z: A tensor of shape [b, d] with features
        y: A tensor of shape [b] with labels
        ignore: A tensor of shape [b, b] with optional mask (0 to keep, 1 to
              ignore). Diagonal masking is applied indipendently.
        """
        if ignore is None:
            b = y.size(0)
            ignore = torch.zeros((b, b)).bool().to(y.device)

        # make sure features are normalized
        z = F.normalize(z, dim=-1, p=2)  # [b, d]

        # compute temperature-scaled similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T), self.temperature
        )  # [b, b]

        # stabilize logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=-1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # [b, b]

        # mask out self-similarities
        diagonal = torch.eye(logits.size(0)).bool().to(logits.device)  # [b, b]

        # mask out ignored samples
        logits = logits.masked_fill(ignore, float("-inf"))  # [b, b]
        logits = logits.masked_fill(diagonal, float("-inf"))  # [b, b]

        # get positive mask
        pos_mask = (y[:, None] == y[None, :]) & (~ignore) & (~diagonal)  # [b, b]

        assert pos_mask.any(
            dim=1
        ).all(), "No positive pairs found in at least one batch."

        # numerator: logsumexp over positives only
        pos_logits = logits.masked_fill(~pos_mask, float("-inf"))  # [b, b]
        numerator = torch.logsumexp(pos_logits, dim=1)  # [b]

        # denominator: logsumexp over all non-self samples (i.e. both positives
        # and negatives)
        denominator = torch.logsumexp(logits, dim=1)  # [b]

        loss = (denominator - numerator).mean()  # [1]

        if self.return_logits:
            return loss, logits

        return loss


def get_criterion(
    loss_type,
    temperature=None,
    lam=None,
    return_logits=False,
):
    if not loss_type or loss_type == "supervised":
        return MultiPosConLoss(
            temperature=temperature,
            return_logits=return_logits,
        )

    if loss_type == "temporally-aware":
        return DistanceAwareMultiPosConLoss(
            temperature=temperature,
            lam=lam,
            return_logits=return_logits,
        )

    if loss_type == "mil-nce":
        return MILNCELoss(
            temperature=temperature,
            return_logits=return_logits,
        )

    raise ValueError(f"Unknown loss type: {loss_type}")
