import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Optional, Union
from PIL import Image

from transforms.base import BaseTransforms


class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        # 학습/검증 폴더 결정
        data_folder = os.path.join(root_dir, 'train' if train else 'val')
        
        # 클래스 폴더 목록 가져오기
        self.classes = sorted([d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 이미지 파일 경로와 라벨 수집
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_folder, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class CarPredictDataset(Dataset):
    """예측용 데이터셋 - 라벨 없이 이미지만 로드"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 학습/검증 폴더 결정
        data_folder = "datasets/train"
        # 클래스 폴더 목록 가져오기
        self.classes = sorted([d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))])
        
        # 이미지 파일 경로 수집 (라벨 없음)
        self.image_paths = []
        for img_name in os.listdir(root_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(root_dir, img_name))
        
        # 파일명으로 정렬하여 일관된 순서 보장
        self.image_paths.sort()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 이미지와 파일명을 함께 반환 (파일명은 나중에 결과 매칭할 때 유용)
        return image, os.path.basename(img_path)


class CarDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        transforms: BaseTransforms,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.train_transform = transforms.train_transform()
        self.val_transform = transforms.val_transform()
        self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Check if dataset exists
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset directory {self.root} not found!")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CarDataset(
                self.root, transform=self.train_transform, train=True
            )
            self.val_dataset = CarDataset(
                self.root, transform=self.val_transform, train=False
            )
            self.num_classes = len(self.train_dataset.classes)
            print(f"Dataset loaded with {self.num_classes} classes")
        
        if stage == "validate" or stage is None:
            self.val_dataset = CarDataset(
                self.root, transform=self.val_transform, train=False
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = CarDataset(
                self.root, transform=self.test_transform, train=False
            )
        
        if stage == "predict" or stage is None:
            self.predict_dataset = CarPredictDataset(
                "test", transform=self.test_transform
            )
            print(f"Predict dataset loaded with {len(self.predict_dataset)} images")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )