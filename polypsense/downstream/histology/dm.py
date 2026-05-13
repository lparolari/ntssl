import logging
import random

import lightning as L
import torch

from polypsense.dataset.fragment import FragmentDataset
from polypsense.dataset.instance import InstanceDataset
from polypsense.dataset.mapping import MappingDataset
from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.trackleter import Trackleter
from polypsense.dataset.subset import SubsetDataset
from polypsense.downstream.data_util import get_fragment_transforms
from polypsense.downstream.histology.data import rspp_ad_vs_nonad_mapping


class HistologyDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_images: str,
        train_annotations: str,
        val_images: str,
        val_annotations: str,
        *,
        batch_size: int,
        im_size: int,
        fragment_length: int,
        fragment_stride: int,
        fragment_drop_last: bool,
        fragment_padding_mode: str,
        aug_resize: bool,
        aug_anchorcrop: bool,
        aug_normalize: bool,
        bbox_scale_factor: float,
        class_mapping: str,
        num_workers: int,
        seed: int,
    ):
        super().__init__()
        self.train_images = train_images
        self.train_annotations = train_annotations
        self.val_images = val_images
        self.val_annotations = val_annotations
        self.batch_size = batch_size
        self.im_size = im_size
        self.fragment_length = fragment_length
        self.fragment_stride = fragment_stride
        self.fragment_drop_last = fragment_drop_last
        self.fragment_padding_mode = fragment_padding_mode
        self.aug_resize = aug_resize
        self.aug_anchorcrop = aug_anchorcrop
        self.aug_normalize = aug_normalize
        self.bbox_scale_factor = bbox_scale_factor
        self.class_mapping = class_mapping
        self.num_workers = num_workers
        self.seed = seed

        self.rng = random.Random(seed)

    def setup(self, stage=None):
        # Create the dataset following the same procedure as in the
        # end-to-end data module, but stopping at the multitask dataset

        if stage == "fit" or stage is None:
            self.train_dataset = self._make_ds("train")
            self.val_dataset = self._make_ds("val")

        if stage == "validate" or stage is None:
            self.val_dataset = self._make_ds("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("val")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _make_ds(self, split):
        def _get_images_path(split):
            return {
                "train": self.train_images,
                "val": self.val_images,
            }[split]

        def _get_annotations_path(split):
            return {
                "train": self.train_annotations,
                "val": self.val_annotations,
            }[split]

        def _get_fragment_transforms(split):
            return {
                "train": self._get_train_fragment_transforms(),
                "val": self._get_eval_fragment_transforms(),
            }[split]

        instance_ds = InstanceDataset.from_instances(
            _get_images_path(split), _get_annotations_path(split)
        )

        trackleter = Trackleter(
            instance_ds,
            separation_thresh=1,
            iou_thresh=0.1,
            identity_key="identity_id",
            pbar=True,
        )

        fragmenter = Fragmenter(
            trackleter,
            min_tracklet_length=30,
            fragment_length=self.fragment_length,
            stride=self.fragment_stride,
            drop_last=self.fragment_drop_last,
            padding_mode=self.fragment_padding_mode,
        )

        fragment_ds = FragmentDataset(
            instance_ds,
            fragments=fragmenter.fragments(),
            reduce_img_ann_keys=[
                ("id", "all"),
                ("file_name", "all"),
                ("width", "one"),
                ("height", "one"),
                ("frame_id", "all"),
                ("sequence_id", "one"),
                ("dataset_id", "one"),
            ],
            reduce_tgt_ann_keys=[
                ("id", "all"),
                ("image_id", "all"),
                ("bbox", "all"),
                ("identity_id", "one"),
                ("histology", "one"),
            ],
        )

        mapping_ds = MappingDataset(
            fragment_ds,
            mapping={
                "identity_id": "auto",
                "histology": self._get_histology_mapping(),
            },
        )

        subset_indices = [
            i
            for i in range(len(mapping_ds))
            if mapping_ds.tgt_ann(i)["histology"] != -1
        ]
        subset_ds = SubsetDataset(mapping_ds, subset_indices)
        logging.info(
            f"Removed {len(mapping_ds) - len(subset_ds)} samples with unknown histology, {len(subset_ds)} samples remain."
        )

        multitask_ds = MultiTaskDataset(subset_ds, _get_fragment_transforms(split))

        return multitask_ds

    def _get_train_fragment_transforms(self):
        return get_fragment_transforms(
            aug_resize=self.aug_resize,
            aug_anchorcrop=self.aug_anchorcrop,
            aug_normalize=self.aug_normalize,
            im_size=self.im_size,
            bbox_scale_factor=self.bbox_scale_factor,
        )

    def _get_eval_fragment_transforms(self):
        return get_fragment_transforms(
            aug_resize=self.aug_resize,
            aug_anchorcrop=self.aug_anchorcrop,
            aug_normalize=self.aug_normalize,
            im_size=self.im_size,
            bbox_scale_factor=self.bbox_scale_factor,
        )

    def _get_histology_mapping(self):
        if self.class_mapping == "ad_vs_nonad":
            return rspp_ad_vs_nonad_mapping
        else:
            return None


import torch
from torchvision.transforms.v2.functional import to_image
from torchvision.tv_tensors import BoundingBoxes, Video


class MultiTaskDataset:
    def __init__(self, ds, fragment_transforms=None):
        self.ds = ds
        self.frame_transform = None
        self.fragment_transforms = fragment_transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        id = self.ds.id(index)
        img_ann = self.ds.img_ann(id)
        tgt_ann = self.ds.tgt_ann(id)

        bboxes = BoundingBoxes(
            tgt_ann["bbox"],
            format="xywh",
            canvas_size=(img_ann["height"], img_ann["width"]),
        )
        clip = Video(torch.stack([to_image(image) for image in self.ds.img(id)]))

        if self.fragment_transforms:
            clip, bboxes = self.fragment_transforms(clip, bboxes)

        target = {
            "bboxes": bboxes,
            "identity": tgt_ann["identity_id"],
            "histology": tgt_ann["histology"],
        }

        return clip, target
