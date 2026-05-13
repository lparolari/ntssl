import torch


def accuracy_instance(preds, targets, threshold=0.5):
    """
    Compute accuracy.

    Note: preds can be either class indices (int) or logits/probabilities
    (float). If float, they will be converted to class indices via argmax
    (multiclass) or sigmoid + thresholding (binary).
    """

    # If preds are floating point → logits or probabilities
    if preds.is_floating_point():
        if preds.ndim == targets.ndim + 1:
            # Multiclass: apply argmax along class dimension
            preds = preds.argmax(dim=1)
        else:
            # Binary: apply sigmoid + threshold
            preds = torch.sigmoid(preds)
            preds = (preds > threshold).long()

    acc = (preds == targets).float().mean().item()
    return acc


def accuracy_identity(preds, targets, identities, threshold=0.5):
    """
    Compute identity-wise accuracy, i.e. the accuracy computed on the
    predictions obtained via majority vote from the fragment predictions for
    each identity.

    Note: preds can be either class indices (int) or logits/probabilities
    (float). If float, they will be converted to class indices via argmax
    (multiclass) or sigmoid + thresholding (binary).

    Args:
        preds (torch.Tensor): fragment predictions, shape [n] (int) or [n, c] (float)
        targets (torch.Tensor): fragment targets, shape [n]
        identities (torch.Tensor): fragment identities, shape [n]

    Returns:
        acc (float): identity-wise accuracy
    """
    # If preds are floating point → logits or probabilities
    if preds.is_floating_point():
        if preds.ndim == targets.ndim + 1:
            # Multiclass: apply argmax along class dimension
            preds = preds.argmax(dim=1)
        else:
            # Binary: apply sigmoid + threshold
            preds = torch.sigmoid(preds)
            preds = (preds > threshold).long()

    id_targets = []
    id_preds = []

    # Construct the two tensors with prediction and targets for each identity. A
    # prediction for an identity is obtained via majority vote from the
    # predictions of its fragments.
    for id_val in identities.unique():
        mask = identities == id_val

        # assert that all targets for this identity are the same
        assert len(targets[mask].unique()) == 1

        # identity target: since all targets are the same, we can safely take
        # the first one
        id_target = targets[mask][0].item()

        # majority vote from predictions
        id_pred = torch.mode(preds[mask]).values.item()

        id_targets.append(id_target)
        id_preds.append(id_pred)

    id_targets = torch.tensor(id_targets)
    id_preds = torch.tensor(id_preds)

    acc = (id_preds == id_targets).float().mean().item()

    return acc
