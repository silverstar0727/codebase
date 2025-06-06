from typing import Optional
from transforms.base import BaseTransforms

import os
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import lightning as L


class HFCarDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: str,
        batch_size: int,
        transforms: BaseTransforms,
        num_workers: int = 4,
        train_split_name: str = "train",
        val_split_name: str = "validation",
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name

        self.transforms = transforms
        self.train_transform = transforms.train_transform()
        self.val_transform = transforms.val_transform()
        self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Check if dataset exists
        load_dataset(
            self.dataset_name,
            self.dataset_config_name,
        )

    def setup(self, stage: Optional[str] = None) -> None:

        def _default_batch_transforms(example_batch, transforms):
            if "label" in example_batch.keys() and len(example_batch.keys()) == 1:
                return example_batch
            example_batch["pixel_values"] = [transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
            example_batch["labels"] = example_batch["label"]
            example_batch["labels"] = [torch.tensor(label) for label in example_batch["label"]]
            return example_batch

        if stage == "fit" or stage is None:
            self.train_dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=self.train_split_name,
                # token=os.getenv("HF_TOKEN", None),
            )
            self.val_dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=self.val_split_name,
                # token=os.getenv("HF_TOKEN", None),
            )

            self.train_dataset.set_transform(partial(_default_batch_transforms, transforms=self.train_transform))
            self.val_dataset.set_transform(partial(_default_batch_transforms, transforms=self.val_transform))

            self.num_classes = len(self.train_dataset.features["label"].names)

        if stage == "validate" or stage is None:
            self.val_dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=self.val_split_name,
            )
            self.val_dataset = self.val_dataset.set_transform(_default_batch_transforms, transforms=self.val_transform)

        if stage == "test" or stage is None:
            try:
                self.test_dataset = load_dataset(
                    self.dataset_name,
                    self.dataset_config_name,
                    split="test",
                )
            except:
                raise ValueError("Test split not found in the dataset. Please ensure the dataset has a 'test' split.")

        if stage == "predict" or stage is None:
            raise ValueError("Predict stage is not supported in this data module. Use 'test' stage instead.")
            self.predict_dataset = CarPredictDataset("test", transform=self.test_transform)
            print(f"Predict dataset loaded with {len(self.predict_dataset)} images")

    def _collate_fn(self, batch):
        # Custom collate function to handle variable-length sequences
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])
        # return {"pixel_values": pixel_values, "labels": labels}
        return (pixel_values, labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self):
        raise ValueError("Predict stage is not supported in this data module. Use 'test' stage instead.")
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
