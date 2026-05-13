import torch
from torchvision.transforms.v2.functional import to_image
from torchvision.tv_tensors import BoundingBoxes, Video


class MultiTaskDataset:
    def __init__(self, ds, fragment_transforms=None):
        self.ds = ds
        self.frame_transform = None
        self.fragment_transforms = fragment_transforms

    # TODO: we should provide id(), img(), img_ann(), tgt_ann() but note that
    # wrapping `self.ds` is not enough. For example, `self.img()` should not
    # just return `self.ds.img()` because it returns raw images without
    # transforms. Same things for other methods, e.g. tgt_ann() that does not
    # return properties that are in the actual MultiTaskDataset targets.

    def id(self, index):
        return self.ds.id(index)

    def img(self, index):
        return Video(
            torch.stack([to_image(frame) for frame in self.ds.img(self.id(index))])
        )

    def img_ann(self, index):
        return self.ds.img_ann(self.id(index))

    def tgt_ann(self, index):
        id = self.id(index)
        img_ann = self.img_ann(id)
        tgt_ann = self.ds.tgt_ann(id)

        bboxes = BoundingBoxes(
            tgt_ann["bbox"],
            format="xywh",
            canvas_size=(img_ann["height"], img_ann["width"]),
        )

        return {
            **tgt_ann,
            "bboxes": bboxes,
            "identity": tgt_ann["identity_id"],  # renaming for convenience
        }

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x, y = self.img(index), self.tgt_ann(index)

        if self.fragment_transforms:
            x, bboxes = self.fragment_transforms(x, y["bboxes"])

        target = {**y, "bboxes": bboxes}

        return x, target
