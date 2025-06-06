from torchvision import transforms
from .base import BaseTransforms


class CarTransforms(BaseTransforms):
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size

    def train_transform(self):
        return transforms.Compose([
            # 기본 리사이즈
            transforms.Resize((self.img_size, self.img_size)),
            
            # 부분 촬영 및 다양한 종횡비 대응
            transforms.RandomResizedCrop(self.img_size, scale=(0.5, 1.0), ratio=(0.5, 2.0)),
            
            # 좌우 반전 (차량은 대칭성이 있음)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 약간의 회전 (촬영 각도 변화)
            transforms.RandomRotation(15),
            
            # 원근 변환 (언더뷰, 사이드뷰 등)
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            
            # 조명 변화 대응
            transforms.ColorJitter(
                brightness=0.3,  # 작업등, 자연광 변화
                contrast=0.2,    # 흐린 날씨, 실내외 차이
                saturation=0.2,  # 카메라 설정 차이
                hue=0.1         # 조명 색온도 변화
            ),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # 가려짐 시뮬레이션 (장비, 다른 차량에 의한)
            transforms.RandomErasing(
                p=0.3, 
                scale=(0.02, 0.15), 
                ratio=(0.3, 3.0)
            ),
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transform(self):
        return self.val_transform()
