from torchvision import transforms
from .base import BaseTransforms
import torch
import torchvision.transforms.functional as TF
import random
import math


class LowAngleTransform:
    """밑에서 위로 촬영한 각도를 시뮬레이션하는 커스텀 변환"""
    def __init__(self, max_angle=30, p=0.5):
        self.max_angle = max_angle
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(10, self.max_angle)  # 올려다보는 각도 시뮬레이션
            return TF.perspective(img, 
                                 startpoints=[[0, 0], [img.size[0]-1, 0], [0, img.size[1]-1], [img.size[0]-1, img.size[1]-1]],
                                 endpoints=[[0, angle], [img.size[0]-1, angle], 
                                          [0, img.size[1]-1], [img.size[0]-1, img.size[1]-1]])
        return img


class HoodOpenSimulation:
    """본넷이 열린 상태를 시뮬레이션하는 커스텀 변환"""
    def __init__(self, p=0.4):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size
            # 이미지 상단 1/3 부분(본넷 위치)에 랜덤한 변형 적용
            top_third = int(height * 0.4)
            
            # 본넷 부분만 자르기
            hood = img.crop((0, 0, width, top_third))
            
            # 본넷 부분에 변형 적용 (약간 열린 느낌)
            hood = TF.affine(hood, 
                            angle=0, 
                            translate=[0, int(top_third * 0.1)], 
                            scale=1.0, 
                            shear=[random.uniform(-10, 10), 0])
            
            # 변형된 본넷 부분을 원본에 붙이기
            result = img.copy()
            result.paste(hood, (0, 0))
            return result
        return img


class CarTransforms(BaseTransforms):
    def __init__(self, hood_open_prob=0.4, low_angle_prob=0.5):
        super().__init__()
        self.hood_open_prob = hood_open_prob
        self.low_angle_prob = low_angle_prob

    def train_transform(self):
        return transforms.Compose([
            # PIL 이미지에 적용되는 변환
            transforms.Resize((320, 320)),  # 더 큰 이미지로 시작
            LowAngleTransform(max_angle=40, p=self.low_angle_prob),  # 밑에서 위로 촬영 시뮬레이션
            HoodOpenSimulation(p=self.hood_open_prob),  # 본넷 열림 시뮬레이션
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-15, 25)),  # 위쪽으로 더 많이 회전 가능하도록 비대칭 설정
            transforms.RandomAffine(0, translate=(0.2, 0.3), scale=(0.8, 1.2), shear=15),  # 수직 이동 증가
            transforms.RandomPerspective(distortion_scale=0.3, p=0.6),  # 원근감 변화 증가
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # 그림자 효과 시뮬레이션을 위한 부분적 밝기 감소
            transforms.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor=0.7) 
                            if random.random() < 0.3 else img),
            transforms.RandomGrayscale(p=0.05),
            
            # PIL 이미지를 텐서로 변환
            transforms.ToTensor(),
            
            # 텐서에만 적용되는 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.1),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # ToTensor 이후에 위치
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transform(self):
        # 테스트셋에서의 특수한 조건을 고려한 변환
        return transforms.Compose([
            transforms.Resize((300, 300)),
            # TTA(Test-Time Augmentation) 적용을 위한 기반 작업
            transforms.FiveCrop(256),  # 상하좌우중앙 5개 크롭
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(crop) for crop in crops
            ]))
        ])