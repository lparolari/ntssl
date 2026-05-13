from polypsense.e2e.transforms import anchor_crop, compose, normalize, resize, to_tensor


def get_fragment_transforms(
    *,
    aug_resize: bool,
    aug_anchorcrop: bool,
    aug_normalize: bool,
    im_size: int,
    bbox_scale_factor: float,
):

    return compose(
        to_tensor()
        + resize(aug_resize, im_size)
        + anchor_crop(aug_anchorcrop, bbox_scale_factor, im_size)
        + normalize(aug_normalize)
    )
