import random

import lightning as L
import torch

from polypsense.dataset.fragment import FragmentDataset
from polypsense.dataset.instance import InstanceDataset
from polypsense.dataset.mapping import MappingDataset
from polypsense.dataset.multitask import MultiTaskDataset
from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.trackleter import Trackleter
from polypsense.downstream.data_util import get_fragment_transforms


class SizeDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_images: str,
        train_annotations: str,
        val_images: str,
        val_annotations: str,
        *,
        dataset_type: str,
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
        num_workers: int,
        seed: int,
    ):
        super().__init__()
        self.train_images = train_images
        self.train_annotations = train_annotations
        self.val_images = val_images
        self.val_annotations = val_annotations
        self.dataset_type = dataset_type
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
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
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
        if self.dataset_type == "coco":
            return self._make_ds_from_coco(split)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _make_ds_from_coco(self, split):
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
            ],
            reduce_tgt_ann_keys=[
                ("id", "all"),
                ("image_id", "all"),
                ("category_id", "one"),
                ("bbox", "all"),
                ("area", "all"),
                ("iscrowd", "all"),
                ("identity_id", "one"),
                ("histology", "one"),
                ("morphology", "one"),
                ("size", "one"),
                ("location", "one"),
            ],
        )

        mapping_ds = MappingDataset(
            fragment_ds,
            mapping={
                "identity_id": "auto",
                "size": lambda x: 1 if x > 5.0 else 0,
            },
        )

        multitask_ds = MultiTaskDataset(mapping_ds, _get_fragment_transforms(split))

        return SizeDataset(multitask_ds)

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


class SizeDataset:
    def __init__(self, ds):
        self.ds = ds

        self.identity = torch.tensor(
            [ds.tgt_ann(ds.id(i))["identity_id"] for i in range(len(ds))]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]
