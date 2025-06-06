# src/transforms/small_scale_transforms.py
from torchvision import transforms
from .base import BaseTransforms
import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

class GaussianBlur:
    """가우시안 블러 변환"""
    def __init__(self, radius_range=(0.1, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img):
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class RandomLighting:
    """조명 변화 시뮬레이션"""
    def __init__(self, alpha_range=(0.5, 1.5)):
        self.alpha_range = alpha_range
    
    def __call__(self, img):
        alpha = random.uniform(*self.alpha_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(alpha)

class CutMix:
    """CutMix 데이터 증강 (소규모 데이터에 효과적)"""
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch):
        if random.random() > self.prob:
            return batch
            
        images, labels = batch
        batch_size = images.size(0)
        
        # 랜덤하게 쌍 선택
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Beta 분포에서 λ 샘플링
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Bounding box 계산
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 이미지 패치 교체
        images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
        
        # 라벨 비율 조정
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images, labels, shuffled_labels, lam

class MixUp:
    """MixUp 데이터 증강"""
    def __init__(self, alpha=0.2, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch):
        if random.random() > self.prob:
            return batch
            
        images, labels = batch
        batch_size = images.size(0)
        
        # 랜덤하게 섞기
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Beta 분포에서 λ 샘플링
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 이미지 믹싱
        mixed_images = lam * images + (1 - lam) * shuffled_images
        
        return mixed_images, labels, shuffled_labels, lam

class SmallScaleCarTransforms(BaseTransforms):
    """소규모 차량 데이터셋에 최적화된 변환"""
    
    def __init__(self, img_size: int = 384, use_advanced_aug: bool = True):
        super().__init__()
        self.img_size = img_size
        self.use_advanced_aug = use_advanced_aug

    def train_transform(self):
        """강력한 데이터 증강으로 overfitting 방지"""
        
        base_transforms = [
            # 기본 리사이즈
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
        ]
        
        if self.use_advanced_aug:
            # 고급 증강 기법들
            advanced_transforms = [
                # 1. 강한 기하학적 변환
                transforms.RandomResizedCrop(
                    self.img_size, 
                    scale=(0.4, 1.0),  # 더 강한 크롭
                    ratio=(0.5, 2.0),
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-20, 20)),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=(-10, 10)
                ),
                
                # 2. 강한 색상 변환
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2
                ),
                transforms.RandomGrayscale(p=0.1),
                
                # 3. 커스텀 변환들
                transforms.RandomApply([GaussianBlur()], p=0.3),
                transforms.RandomApply([RandomLighting()], p=0.3),
                
                # 4. 원근 변환 (차량 각도 다양화)
                transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
            ]
        else:
            # 기본 증강
            advanced_transforms = [
                transforms.RandomResizedCrop(self.img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        
        # 최종 변환
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # 랜덤 소거 (부분 가려짐)
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.25),
                ratio=(0.3, 3.0),
                value=0,
                inplace=False
            ),
        ]
        
        return transforms.Compose(base_transforms + advanced_transforms + final_transforms)

    def val_transform(self):
        """검증용 - TTA 적용 가능한 구조"""
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def test_transform(self):
        """테스트용 - TTA (Test Time Augmentation) 적용"""
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            
            # TTA를 위한 5-crop (중앙 + 4모서리)
            transforms.FiveCrop(self.img_size),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(crop) for crop in crops
            ]))
        ])

class PolicyBasedTransforms(BaseTransforms):
    """AutoAugment 스타일의 정책 기반 변환"""
    
    def __init__(self, img_size: int = 384):
        super().__init__()
        self.img_size = img_size
        
        # 차량 분류에 효과적인 정책들
        self.policies = [
            # Policy 1: 색상 + 기하학
            [("ColorJitter", 0.8, (0.4, 0.4, 0.4, 0.2)), ("Rotate", 0.7, 15)],
            # Policy 2: 크롭 + 플립
            [("RandomResizedCrop", 1.0, (0.5, 1.0)), ("HorizontalFlip", 0.5, None)],
            # Policy 3: 원근 + 밝기
            [("Perspective", 0.6, 0.3), ("Brightness", 0.8, 0.3)],
            # Policy 4: 어파인 + 대비
            [("Affine", 0.7, (0.1, 0.1, 0.9, 1.1)), ("Contrast", 0.8, 0.3)],
            # Policy 5: 블러 + 채도
            [("GaussianBlur", 0.4, (0.1, 1.5)), ("Saturation", 0.8, 0.3)],
        ]
    
    def _apply_policy(self, img, policy):
        """정책 적용"""
        for transform_name, prob, magnitude in policy:
            if random.random() < prob:
                if transform_name == "ColorJitter":
                    brightness, contrast, saturation, hue = magnitude
                    transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
                    img = transform(img)
                elif transform_name == "Rotate":
                    angle = random.uniform(-magnitude, magnitude)
                    img = TF.rotate(img, angle)
                elif transform_name == "RandomResizedCrop":
                    scale_min, scale_max = magnitude
                    scale = random.uniform(scale_min, scale_max)
                    size = int(min(img.size) * scale)
                    img = transforms.RandomResizedCrop(size)(img)
                elif transform_name == "HorizontalFlip":
                    img = TF.hflip(img)
                elif transform_name == "Perspective":
                    distortion = magnitude
                    width, height = img.size
                    # 원근 변환 포인트 계산
                    startpoints = [[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]
                    endpoints = [[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]
                    
                    # 랜덤 왜곡 적용
                    for i in range(4):
                        dx = random.uniform(-distortion * width, distortion * width)
                        dy = random.uniform(-distortion * height, distortion * height)
                        endpoints[i][0] = max(0, min(width-1, endpoints[i][0] + dx))
                        endpoints[i][1] = max(0, min(height-1, endpoints[i][1] + dy))
                    
                    img = TF.perspective(img, startpoints, endpoints)
                elif transform_name == "Brightness":
                    factor = 1 + random.uniform(-magnitude, magnitude)
                    img = TF.adjust_brightness(img, factor)
                elif transform_name == "Contrast":
                    factor = 1 + random.uniform(-magnitude, magnitude)
                    img = TF.adjust_contrast(img, factor)
                elif transform_name == "Saturation":
                    factor = 1 + random.uniform(-magnitude, magnitude)
                    img = TF.adjust_saturation(img, factor)
                elif transform_name == "Affine":
                    translate_x, translate_y, scale_min, scale_max = magnitude
                    translate = (random.uniform(-translate_x, translate_x) * img.size[0],
                               random.uniform(-translate_y, translate_y) * img.size[1])
                    scale = random.uniform(scale_min, scale_max)
                    img = TF.affine(img, angle=0, translate=translate, scale=scale, shear=0)
                elif transform_name == "GaussianBlur":
                    radius_min, radius_max = magnitude
                    radius = random.uniform(radius_min, radius_max)
                    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return img
    
    def train_transform(self):
        """정책 기반 훈련 변환"""
        def apply_random_policy(img):
            policy = random.choice(self.policies)
            return self._apply_policy(img, policy)
        
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.Lambda(apply_random_policy),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.0)),
        ])
    
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def test_transform(self):
        return self.val_transform()


class AugmentationMix:
    """CutMix + MixUp + 일반 증강을 동적으로 선택"""
    def __init__(self, cutmix_prob=0.3, mixup_prob=0.3, normal_prob=0.4):
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.normal_prob = normal_prob
        
        self.cutmix = CutMix(alpha=1.0, prob=1.0)
        self.mixup = MixUp(alpha=0.2, prob=1.0)
    
    def __call__(self, batch):
        """배치에 랜덤 증강 적용"""
        rand = random.random()
        
        if rand < self.cutmix_prob:
            return self.cutmix(batch)
        elif rand < self.cutmix_prob + self.mixup_prob:
            return self.mixup(batch)
        else:
            # 일반 배치 반환
            return batch


class TrivialAugmentTransforms(BaseTransforms):
    """TrivialAugment - 단순하지만 효과적인 증강"""
    
    def __init__(self, img_size: int = 384):
        super().__init__()
        self.img_size = img_size
        
        # 가능한 증강 연산들
        self.augment_list = [
            'identity', 'autocontrast', 'equalize', 'rotate', 'solarize',
            'color', 'posterize', 'contrast', 'brightness', 'sharpness',
            'shear_x', 'shear_y', 'translate_x', 'translate_y'
        ]
    
    def _apply_augment(self, img, aug_name, magnitude):
        """개별 증강 적용"""
        if aug_name == 'identity':
            return img
        elif aug_name == 'autocontrast':
            return TF.autocontrast(img)
        elif aug_name == 'equalize':
            return TF.equalize(img)
        elif aug_name == 'rotate':
            angle = magnitude * 30  # -30 to 30 degrees
            return TF.rotate(img, angle)
        elif aug_name == 'solarize':
            threshold = int(magnitude * 256)
            return TF.solarize(img, threshold)
        elif aug_name == 'color':
            factor = 1 + magnitude * 0.9  # 0.1 to 1.9
            return TF.adjust_saturation(img, factor)
        elif aug_name == 'contrast':
            factor = 1 + magnitude * 0.9
            return TF.adjust_contrast(img, factor)
        elif aug_name == 'brightness':
            factor = 1 + magnitude * 0.9
            return TF.adjust_brightness(img, factor)
        elif aug_name == 'sharpness':
            factor = 1 + magnitude * 0.9
            return TF.adjust_sharpness(img, factor)
        elif aug_name in ['shear_x', 'shear_y']:
            shear = magnitude * 0.3 * 180 / np.pi  # Convert to degrees
            if aug_name == 'shear_x':
                return TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[shear, 0])
            else:
                return TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, shear])
        elif aug_name in ['translate_x', 'translate_y']:
            pixels = int(magnitude * min(img.size) * 0.45)
            if aug_name == 'translate_x':
                return TF.affine(img, angle=0, translate=[pixels, 0], scale=1, shear=[0, 0])
            else:
                return TF.affine(img, angle=0, translate=[0, pixels], scale=1, shear=[0, 0])
        else:
            return img
    
    def train_transform(self):
        """TrivialAugment 적용"""
        def trivial_augment(img):
            # 랜덤하게 하나의 증강 선택
            aug_name = random.choice(self.augment_list)
            # 강도는 0-1 사이 균등 분포
            magnitude = random.uniform(-1, 1)
            return self._apply_augment(img, aug_name, magnitude)
        
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.RandomResizedCrop(self.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(trivial_augment),  # TrivialAugment 적용
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def test_transform(self):
        return self.val_transform()


class MultiScaleTransforms(BaseTransforms):
    """다중 스케일 훈련을 위한 변환"""
    
    def __init__(self, base_size: int = 384, scale_range: tuple = (0.7, 1.3)):
        super().__init__()
        self.base_size = base_size
        self.scale_range = scale_range
        
        # 가능한 스케일들
        self.scales = [
            int(base_size * 0.75),  # 288
            int(base_size * 0.875), # 336  
            int(base_size * 1.0),   # 384
            int(base_size * 1.125), # 432
            int(base_size * 1.25),  # 480
        ]
    
    def train_transform(self):
        """다중 스케일 훈련 변환"""
        def multi_scale_resize(img):
            # 랜덤 스케일 선택
            target_size = random.choice(self.scales)
            return transforms.Resize((target_size, target_size))(img)
        
        return transforms.Compose([
            transforms.Lambda(multi_scale_resize),  # 다중 스케일
            transforms.RandomResizedCrop(
                self.base_size,  # 최종 크기는 고정
                scale=(0.6, 1.0),
                ratio=(0.7, 1.4)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.base_size, self.base_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def test_transform(self):
        """멀티스케일 테스트 - 여러 스케일 평균"""
        def multi_scale_test(img):
            # 여러 스케일로 변환
            scales_transforms = []
            for scale in self.scales:
                transformed = transforms.Compose([
                    transforms.Resize((scale, scale)),
                    transforms.CenterCrop(self.base_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(img)
                scales_transforms.append(transformed)
            
            return torch.stack(scales_transforms)
        
        return transforms.Lambda(multi_scale_test)


# 사용 예시 및 권장 조합
class OptimalCarTransforms(BaseTransforms):
    """소규모 차량 데이터셋을 위한 최적 조합"""
    
    def __init__(self, img_size: int = 384, augment_strength: str = "medium"):
        super().__init__()
        self.img_size = img_size
        self.augment_strength = augment_strength
    
    def train_transform(self):
        if self.augment_strength == "light":
            # 가벼운 증강 (빠른 실험용)
            return SmallScaleCarTransforms(self.img_size, use_advanced_aug=False).train_transform()
        elif self.augment_strength == "medium":
            # 중간 증강 (권장)
            return TrivialAugmentTransforms(self.img_size).train_transform()
        elif self.augment_strength == "heavy":
            # 강한 증강 (매우 작은 데이터셋용)
            return SmallScaleCarTransforms(self.img_size, use_advanced_aug=True).train_transform()
        elif self.augment_strength == "policy":
            # 정책 기반 증강 (고급)
            return PolicyBasedTransforms(self.img_size).train_transform()
        elif self.augment_strength == "multiscale":
            # 다중 스케일 증강
            return MultiScaleTransforms(self.img_size).train_transform()
        else:
            raise ValueError(f"Unknown augment_strength: {self.augment_strength}")
    
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def test_transform(self):
        # TTA 적용
        return transforms.Compose([
            transforms.Resize((int(self.img_size * 1.15), int(self.img_size * 1.15))),
            transforms.FiveCrop(self.img_size),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(crop) for crop in crops
            ]))
        ])