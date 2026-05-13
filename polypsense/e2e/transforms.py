import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.tv_tensors import BoundingBoxes, Image, Video, wrap

# -------------------------------------------------------------------
# FACADE
# -------------------------------------------------------------------


def compose(transforms=[]):
    if not transforms:
        return None
    return T.Compose(transforms)


def resize(aug_resize=True, im_size=232):
    return [T.Resize(size=[im_size] * 2)] if aug_resize else []


def to_tensor():
    return [
        T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
    ]


def anchor_crop(aug_anchorcrop=True, scale=1.0, im_size=232):
    """
    Crops the image to the bounding boxes scaled by a factor.
    """
    return [AnchorCrop(scale, im_size)] if aug_anchorcrop else []


def normalize(aug_normalize=True):
    return (
        [T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])]
        if aug_normalize
        else []
    )


def inv_normalize(aug_invnormalize=True):
    return (
        [
            T.Normalize(
                mean=[-0.45 / 0.225, -0.45 / 0.225, -0.45 / 0.225],
                std=[1 / 0.225, 1 / 0.225, 1 / 0.225],
            )
        ]
        if aug_invnormalize
        else []
    )


def prnt():
    class Prnt(T.Transform):
        def transform(self, inpt, params):
            if isinstance(inpt, BoundingBoxes):
                print(inpt)
            return inpt

    return [Prnt()]


# -------------------------------------------------------------------
# FUNCTIONAL
# -------------------------------------------------------------------


def scale_bounding_boxes(inpt: BoundingBoxes, factor: float = 1.0) -> BoundingBoxes:
    """
    Expands a bounding box such that the diagonal of the new bounding box is
    factor times the original diagonal.

    This implementation works on the width and height instead of diagonal
    because from Pitagora theorem we know that the diagonal of a rectangle is

        d = sqrt(width^2+height^2)

    and if we scale the width and height by a factor k, we get:

        d' = sqrt((kw)^2+(kh)^2)
           = sqrt(k^2 ⋅ (w^2 + h^2))
           = k ⋅ sqrt(w^2 + h^2)
           = k ⋅ d

    thus the diagonal is scaled by the same factor k.
    """
    convert = F.convert_bounding_box_format
    inpt_dtype = inpt.dtype

    _, _, width, height = convert(inpt, new_format="xywh").unbind(dim=-1)
    x1, y1, x2, y2 = convert(inpt, new_format="xyxy").unbind(dim=-1)

    dx = (width * (factor - 1)) / 2
    dy = (height * (factor - 1)) / 2

    new_x1 = x1 - dx
    new_y1 = y1 - dy
    new_x2 = x2 + dx
    new_y2 = y2 + dy

    # Create the expanded bounding box
    scaled_data = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

    # Convert the bounding box back to the original format
    scaled_bbox = BoundingBoxes(
        scaled_data, format="xyxy", canvas_size=inpt.canvas_size
    )
    output = convert(scaled_bbox, new_format=inpt.format)

    return output.to(inpt_dtype)


def anchor_crop_image(inpt: Image, anchors: BoundingBoxes, size: int) -> Image:
    anchors = F.convert_bounding_box_format(anchors, new_format="xywh")
    x, y, w, h = anchors.int().unbind(dim=-1)
    out = F.crop(
        inpt,
        top=y.item(),
        left=x.item(),
        height=h.item(),
        width=w.item(),
    )
    out = F.resize(out, size=[size] * 2)
    return wrap(
        out,
        like=inpt,
    )


def anchor_crop_video(inpt: Video, anchors: BoundingBoxes, size: int) -> Video:
    anchors = F.convert_bounding_box_format(anchors, new_format="xywh")

    out = []
    for frame, bbox in zip(inpt, anchors):
        x, y, w, h = bbox.int().unbind(dim=-1)
        cropped = F.crop(
            frame,
            top=y.item(),
            left=x.item(),
            height=h.item(),
            width=w.item(),
        )
        resized = F.resize(cropped, size=[size] * 2)
        out.append(resized)

    out = torch.stack(out)

    return wrap(out, like=inpt)


def anchor_crop_bboxes(
    inpt: BoundingBoxes, anchors: BoundingBoxes, size: int
) -> BoundingBoxes:
    anchors = F.convert_bounding_box_format(anchors, new_format="xywh")

    out = []
    for bbox, anchor in zip(inpt, anchors):
        x, y, w, h = anchor.int().unbind(dim=-1)
        cropped = F.crop(
            wrap(bbox, like=inpt),
            top=y.item(),
            left=x.item(),
            height=h.item(),
            width=w.item(),
        )
        resized = F.resize(cropped, size=[size] * 2)
        out.append(resized)

    out = torch.cat(out)

    return BoundingBoxes(out, format=inpt.format, canvas_size=(size, size))


def _anchor_crop(inpt, anchors, size):
    if isinstance(inpt, Image):
        return anchor_crop_image(inpt, anchors, size=size)
    elif isinstance(inpt, Video):
        return anchor_crop_video(inpt, anchors, size=size)
    elif isinstance(inpt, BoundingBoxes):
        return anchor_crop_bboxes(inpt, anchors, size=size)
    else:
        return inpt


# -------------------------------------------------------------------
# CLASS
# -------------------------------------------------------------------


class AnchorCrop(T.Transform):
    """
    Crop an image or a video to given anchor bounding boxes. Can crop to single
    or multiple bounding boxes.
    """

    def __init__(self, scale, size):
        super().__init__()
        self.scale = scale
        self.size = size

    def make_params(self, flat_inputs):
        # Assuming one instnace of BoundingBoxes per sample as reported here:
        # https://pytorch.org/vision/main/generated/torchvision.tv_tensors.BoundingBoxes.html#torchvision.tv_tensors.BoundingBoxes
        bboxes = [inp for inp in flat_inputs if isinstance(inp, BoundingBoxes)]
        bboxes = bboxes and bboxes[0]
        return {"bboxes": bboxes}

    def transform(self, inpt, params):
        bboxes = params["bboxes"]
        anchors = scale_bounding_boxes(bboxes, self.scale)
        anchors = F.clamp_bounding_boxes(anchors)
        return _anchor_crop(inpt, anchors, size=self.size)
