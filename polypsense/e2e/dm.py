import pathlib
import random
import logging

import lightning as L
import torch
import torch.utils

from polypsense.dataset import InstanceDataset
from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.identifier import Identifier
from polypsense.dataset.repo.trackleter import Trackleter
from polypsense.dataset.repo.videoer import Videoer
from polypsense.e2e.data import FragmentIdentityDataset
from polypsense.e2e.sampler import (
    MultiPosConBatchSampler,
    TemporalKNNBagBatchSampler,
    TemporalTemperatureBagBatchSampler,
)
from polypsense.e2e.transforms import anchor_crop, compose, normalize, resize, to_tensor


class End2EndDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root,
        *,
        im_size,
        fragment_length,
        fragment_stride,
        fragment_drop_last,
        fragment_padding_mode,
        bbox_scale_factor,
        min_tracklet_length,
        sampler,
        sampler_ttb_tmin,
        sampler_ttb_tmax,
        aug_resize,
        aug_anchorcrop,
        aug_normalize,
        batch_size,
        num_workers,
        n_views,
        seed,
        eval_n_views,
        eval_batch_size,
        eval_fragment_length,
        eval_aug_resize,
        eval_aug_anchorcrop,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.im_size = im_size
        self.fragment_length = fragment_length
        self.fragment_stride = fragment_stride
        self.fragment_drop_last = fragment_drop_last
        self.fragment_padding_mode = fragment_padding_mode
        self.bbox_scale_factor = bbox_scale_factor
        self.min_tracklet_length = min_tracklet_length
        self.sampler = sampler
        self.sampler_ttb_tmin = sampler_ttb_tmin
        self.sampler_ttb_tmax = sampler_ttb_tmax
        self.aug_resize = aug_resize
        self.aug_anchorcrop = aug_anchorcrop
        self.aug_normalize = aug_normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_views = n_views
        self.seed = seed
        self.rng = random.Random(seed)
        self.eval_n_views = eval_n_views
        self.eval_batch_size = eval_batch_size
        self.eval_fragment_length = eval_fragment_length
        self.eval_aug_resize = eval_aug_resize
        self.eval_aug_anchorcrop = eval_aug_anchorcrop

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_ds("train")
            self.val_dataset = self._make_ds("val")

        if stage == "validate" or stage is None:
            self.val_dataset = self._make_ds("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("test")

    def train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._get_eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def _get_train_dataloader(self, ds):
        batch_sampler = self._get_train_sampler(ds)
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def _get_eval_dataloader(self, ds):
        batch_sampler = self._get_eval_sampler(ds)
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
        )

    def _get_train_sampler(self, ds):
        if self.sampler is None or self.sampler == "multipos":
            return MultiPosConBatchSampler(
                ds.y,
                k=self.n_views,
                batch_size=self.batch_size,
                seed=self.rng.randint(0, 2**32 - 1),
            )
        if self.sampler == "temporalknnbag":
            return TemporalKNNBagBatchSampler(
                ds.v,
                ds.pv,
                k=self.n_views,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                seed=self.rng.randint(0, 2**32 - 1),
            )
        if self.sampler == "temporaltemperaturebag":
            return TemporalTemperatureBagBatchSampler(
                ds.v,
                ds.pv,
                k=self.n_views,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                c=self.trainer.current_epoch / self.trainer.max_epochs,
                t_min=self.sampler_ttb_tmin,
                t_max=self.sampler_ttb_tmax,
                seed=self.rng.randint(0, 2**32 - 1),
            )
        raise ValueError(f"Unknown sampler: {self.sampler}")

    def _get_eval_sampler(self, ds):
        return MultiPosConBatchSampler(
            ds.y,
            k=self.eval_n_views,
            batch_size=self.eval_batch_size,
            seed=self.seed,
        )

    def _make_ds(self, split):
        instance_ds = self._get_instance_ds(split)

        trackleter = Trackleter(instance_ds)
        identifier = Identifier(instance_ds)
        videoer = Videoer(instance_ds)
        fragmenter = Fragmenter(
            trackleter,
            min_tracklet_length=self.min_tracklet_length,
            fragment_length=(
                self.fragment_length if split == "train" else self.eval_fragment_length
            ),
            stride=self.fragment_stride,
            drop_last=self.fragment_drop_last,
            padding_mode=self.fragment_padding_mode,
        )

        ds = FragmentIdentityDataset(
            instance_ds,
            fragmenter,
            identifier,
            videoer,
            fragment_transforms=self._get_fragment_transforms(split),
            return_dict=True,
        )

        return ds

    def _get_instance_ds(self, split):
        dataset_root = pathlib.Path(self.dataset_root)
        img_folder = dataset_root / "images"
        ann_file = dataset_root / "annotations" / f"instances_{split}.json"

        instance_ds = InstanceDataset.from_instances(img_folder, ann_file)

        return instance_ds

    def _get_fragment_transforms(self, split):
        return {
            "train": compose(
                to_tensor()
                + resize(self.aug_resize, self.im_size)
                + anchor_crop(self.aug_anchorcrop, self.bbox_scale_factor, self.im_size)
                + normalize(self.aug_normalize)
            ),
            "val": compose(
                to_tensor()
                + resize(self.eval_aug_resize, self.im_size)
                + anchor_crop(
                    self.eval_aug_anchorcrop, self.bbox_scale_factor, self.im_size
                )
                + normalize(self.aug_normalize)
            ),
            "test": compose(
                to_tensor()
                + resize(self.eval_aug_resize, self.im_size)
                + anchor_crop(
                    self.eval_aug_anchorcrop, self.bbox_scale_factor, self.im_size
                )
                + normalize(self.aug_normalize)
            ),
        }[split]


def collate_fn(batch):
    return torch.utils.data.default_collate(batch)
