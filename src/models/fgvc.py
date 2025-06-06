# src/models/fgvc.py (TResNet 구조에 맞게 완전히 재작성)
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from typing import Optional, Dict, Any
import timm
try:
    import wandb
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
except ImportError:
    wandb = None

# 기본 컨볼루션 블록
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# 안티 노이즈 디코더 (채널 수 적응형)
class AntiNoiseDecoder(nn.Module):
    def __init__(self, scale, in_channel):
        super(AntiNoiseDecoder, self).__init__()
        self.scale = scale
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        
        # Process path는 forward에서 동적으로 처리
        self.process = None
        
    def _build_process_path(self, input_channels):
        """입력 채널에 맞게 process path 동적 생성"""
        # 간단한 업샘플링 + 채널 감소
        layers = []
        
        # 먼저 적당한 채널로 감소
        current_channels = input_channels
        target_channels = min(256, current_channels)
        
        layers.append(nn.Conv2d(current_channels, target_channels, 3, 1, 1, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        # 점진적 업샘플링
        for _ in range(3):  # 8x 업샘플링
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            next_channels = target_channels // 2
            layers.append(nn.Conv2d(target_channels, next_channels, 3, 1, 1, bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            target_channels = next_channels
        
        # 마지막에 3채널로
        layers.append(nn.Conv2d(target_channels, 3, 3, 1, 1, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x, feature_map):
        # Process path 동적 생성
        if self.process is None:
            self.process = self._build_process_path(feature_map.size(1))
            if x.is_cuda:
                self.process = self.process.cuda()
        
        try:
            processed = self.process(feature_map)
            # 크기 맞춤
            if processed.size() != x.size():
                processed = F.interpolate(processed, size=x.shape[2:], mode='bilinear', align_corners=False)
        except Exception as e:
            print(f"Decoder error: {e}, using identity")
            processed = torch.zeros_like(x)
        
        return self.skip(x) + processed

# TResNet 특징 추출기 (올바른 구조)
class TResNetFeatures(nn.Module):
    def __init__(self, backbone):
        super(TResNetFeatures, self).__init__()
        self.backbone = backbone
        
        # TResNet의 body 구조: s2d -> conv1 -> layer1 -> layer2 -> layer3 -> layer4
        # 우리는 layer2, layer3, layer4의 출력을 사용
        self.features = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """TResNet의 layer2, layer3, layer4에 훅 등록"""
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # TResNet body에서 특징 추출
        if hasattr(self.backbone, 'body'):
            # layer2, layer3, layer4에 훅 등록
            self.backbone.body.layer2.register_forward_hook(get_activation('layer2'))
            self.backbone.body.layer3.register_forward_hook(get_activation('layer3'))
            self.backbone.body.layer4.register_forward_hook(get_activation('layer4'))
        else:
            print("Warning: TResNet body not found, using fallback")
    
    def forward(self, x):
        # 전체 백본 실행
        _ = self.backbone(x)
        
        # 훅으로 캡처된 특징들 반환
        feat1 = self.features.get('layer2', x)  # 중간 해상도
        feat2 = self.features.get('layer3', x)  # 고해상도  
        feat3 = self.features.get('layer4', x)  # 최고해상도
        
        return feat1, feat2, feat3

# 메인 네트워크
class TResNetAntiNoiseNetwork(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 백본 분류기 제거
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()
        elif hasattr(backbone, 'fc'):
            backbone.fc = nn.Identity()
        
        self.features = TResNetFeatures(backbone)
        
        # 실제 채널 수 확인
        self._get_feature_dims()
        
        # 동적으로 레이어 생성
        self._build_classifiers()
    
    def _get_feature_dims(self):
        """더미 입력으로 실제 특징 차원 확인"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 448, 448)
            if next(self.features.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            feat1, feat2, feat3 = self.features(dummy_input)
            self.feat1_dim = feat1.size(1)
            self.feat2_dim = feat2.size(1) 
            self.feat3_dim = feat3.size(1)
            
            print(f"TResNet feature dimensions: {self.feat1_dim}, {self.feat2_dim}, {self.feat3_dim}")
    
    def _build_classifiers(self):
        """특징 차원에 맞는 분류기들 생성"""
        # 각 레벨별 분류기
        self.conv_block1 = nn.Sequential(
            BasicConv(self.feat1_dim, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        self.conv_block2 = nn.Sequential(
            BasicConv(self.feat2_dim, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        self.conv_block3 = nn.Sequential(
            BasicConv(self.feat3_dim, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes)
        )
        
        # 메인 분류기
        self.main_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat3_dim, self.num_classes)
        )

    def forward(self, x):
        # 특징 추출
        feat1, feat2, feat3 = self.features(x)
        
        # 특징 맵 복사 (디코더용)
        map1 = feat1.clone()
        map2 = feat2.clone()
        map3 = feat3.clone()
        
        # 다중 스케일 분류
        # Level 1
        x1_ = self.conv_block1(feat1)
        x1_c = self.classifier1(x1_)
        
        # Level 2
        x2_ = self.conv_block2(feat2)
        x2_c = self.classifier2(x2_)
        
        # Level 3
        x3_ = self.conv_block3(feat3)
        x3_c = self.classifier3(x3_)
        
        # 메인 분류기
        main_output = self.main_classifier(feat3)
        
        return x1_c, x2_c, x3_c, main_output, map1, map2, map3

class TResNetAntiNoiseCarClassifier(L.LightningModule):
    def __init__(
        self,
        backbone_name: str = "tresnet_xl",
        pretrained: bool = True,
        img_size: int = 448,
        num_classes: int = 400,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        alpha: float = 1.0,
        use_sam: bool = False,
        teacher_path: Optional[str] = None,
        distill_phase_ratio: float = 0.3,
        vis_per_batch: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 하이퍼파라미터 저장
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.distill_phase_ratio = distill_phase_ratio
        self.vis_per_batch = vis_per_batch
        
        # TResNet 백본 로드
        self.backbone = self._load_tresnet_backbone(backbone_name, pretrained, img_size)
        
        # 네트워크 초기화
        self.network = TResNetAntiNoiseNetwork(self.backbone, num_classes)
        
        # 디코더들 (실제 차원에 맞춤)
        self.decoder1 = AntiNoiseDecoder(1, self.network.feat1_dim)
        self.decoder2 = AntiNoiseDecoder(2, self.network.feat2_dim)
        self.decoder3 = AntiNoiseDecoder(4, self.network.feat3_dim)
        
        # Teacher 모델
        self.teacher = None
        if teacher_path:
            self.teacher = self._load_teacher(teacher_path)
        
        # Loss 함수들
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # 메트릭
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Validation 결과 저장을 위한 리스트
        self.val_predictions = []
        self.val_references = []
        
        # 노이즈 변환 시퀀스
        self._setup_augmentations()
    
    def _load_tresnet_backbone(self, backbone_name: str, pretrained: bool, img_size: int):
        """TResNet 백본 로드"""
        try:
            # 정확한 모델명 확인
            if backbone_name == "tresnet_xl" and img_size == 448:
                model_name = "tresnet_xl.miil_in1k_448"
            elif backbone_name == "tresnet_l" and img_size == 448:
                model_name = "tresnet_l.miil_in1k_448"
            elif backbone_name == "tresnet_m" and img_size == 448:
                model_name = "tresnet_m.miil_in1k_448"
            else:
                model_name = backbone_name
            
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # head 제거
                global_pool='',  # pooling 제거
            )
            print(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            print(f"Failed to load {backbone_name}: {e}")
            # fallback
            model = timm.create_model(
                "tresnet_xl",
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
            )
            return model
    
    def _load_teacher(self, teacher_path: str):
        """Teacher 모델 로드"""
        try:
            teacher = torch.load(teacher_path, map_location='cpu')
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            return teacher
        except Exception as e:
            print(f"Warning: Could not load teacher model from {teacher_path}: {e}")
            return None
    
    def _setup_augmentations(self):
        """이미지 증강 시퀀스 설정"""
        sometimes_1 = lambda aug: iaa.Sometimes(0.2, aug)
        sometimes_2 = lambda aug: iaa.Sometimes(0.5, aug)
        
        self.transform_seq_aug = iaa.Sequential([
            sometimes_1(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-15, 15),
                shear=(-15, 15),
                order=[0, 1],
                cval=(0, 1),
                mode=ia.ALL
            )),
            sometimes_2(iaa.GaussianBlur((0, 3.0)))
        ], random_order=True)
        
        self.transform_seq_noise = iaa.Sequential([
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05), per_channel=0.5
            )
        ], random_order=True)
    
    def _add_noise(self, x, transformation_seq):
        """이미지에 노이즈 추가"""
        x = x.permute(0, 2, 3, 1)
        x = x.cpu().numpy()
        x = transformation_seq(images=x)
        x = torch.from_numpy(x.astype(np.float32))
        x = x.permute(0, 3, 1, 2)
        return x.to(self.device)
    
    def on_fit_start(self):
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0
        
        # matplotlib 한글 폰트 설정 (한국어 클래스명 지원)
        try:
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
        except:
            # 폰트 설정 실패 시 기본 설정 유지
            pass
        
        # Distillation 페이즈 에포크 계산
        self.distill_epochs = int(self.trainer.max_epochs * self.distill_phase_ratio)
    
    def forward(self, x):
        return self.network(x)
    
    def _compute_simple_loss(self, inputs, targets):
        """단순한 분류 손실 (안정성 우선)"""
        try:
            output_1, output_2, output_3, output_main, _, _, _ = self(inputs)
            
            # 모든 출력의 분류 손실
            loss = (self.ce_loss(output_1, targets) + 
                   self.ce_loss(output_2, targets) + 
                   self.ce_loss(output_3, targets) + 
                   self.ce_loss(output_main, targets) * 2) / 5
            
            return loss, output_main
            
        except Exception as e:
            print(f"Error in simple loss: {e}")
            # 매우 단순한 fallback
            _, _, _, output_main, _, _, _ = self(inputs)
            return self.ce_loss(output_main, targets), output_main
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # 안정성을 위해 단순한 손실만 사용
        loss, logits = self._compute_simple_loss(inputs, targets)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, targets)
        
        # 로깅
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_start(self) -> None:
        # Validation 결과 초기화
        self.val_predictions = []
        self.val_references = []
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # 검증에서는 메인 출력만 사용
        _, _, _, logits, _, _, _ = self(inputs)
        loss = self.ce_loss(logits, targets)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, targets)
        
        # 예측 결과 저장 (confusion matrix를 위해)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_references.extend(targets.cpu().numpy())
        
        self.log_dict({
            "val/loss": loss,
            "val/acc": acc,
        }, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def on_validation_epoch_end(self) -> None:
        # Confusion Matrix 생성 및 로깅
        if self.is_wandb and len(self.val_predictions) > 0:
            self.log_confusion_matrix()
    
    def log_confusion_matrix(self):
        """Confusion Matrix를 생성하고 W&B에 로깅"""
        try:
            # numpy 배열로 변환
            preds = np.array(self.val_predictions)
            refs = np.array(self.val_references)
            
            # 클래스 이름 가져오기
            if hasattr(self.trainer.datamodule.train_dataset, "classes"):
                labels = self.trainer.datamodule.train_dataset.classes
            else:
                labels = [f"Class_{i}" for i in range(self.num_classes)]
            
            # Confusion Matrix 생성
            cm = confusion_matrix(refs, preds)
            np.fill_diagonal(cm, 0)  # 정답 예측은 제거하여 혼동만 표시
            
            # Top-N 가장 혼동이 많은 클래스들 찾기
            top_n = min(20, len(labels))  # 최대 20개 또는 전체 클래스 수 중 작은 값
            misclassified_counts = cm.sum(axis=1)
            top_true_classes = np.argsort(misclassified_counts)[::-1][:top_n]
            
            # 각 혼동 클래스에 대해 가장 많이 혼동되는 예측 클래스 찾기
            top_confused_classes = set(top_true_classes)
            for cls in top_true_classes:
                most_confused_pred = np.argmax(cm[cls])
                top_confused_classes.add(most_confused_pred)
            
            # 서브 매트릭스 추출
            top_confused_classes = sorted(top_confused_classes)
            reduced_cm = cm[np.ix_(top_confused_classes, top_confused_classes)]
            reduced_labels = [labels[i] for i in top_confused_classes]
            
            # Confusion Matrix 플롯 생성
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                reduced_cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=reduced_labels,
                yticklabels=reduced_labels,
            )
            plt.title(f"Top-{top_n} Confused Classes (Validation) - epoch {self.current_epoch}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # W&B에 로깅
            self.logger.experiment.log({"val/confusion_matrix": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")
    
    def configure_optimizers(self):
        # 백본과 분류기에 다른 학습률 적용
        backbone_params = list(self.network.features.parameters())
        classifier_params = (
            list(self.network.conv_block1.parameters()) +
            list(self.network.conv_block2.parameters()) +
            list(self.network.conv_block3.parameters()) +
            list(self.network.classifier1.parameters()) +
            list(self.network.classifier2.parameters()) +
            list(self.network.classifier3.parameters()) +
            list(self.network.main_classifier.parameters()) +
            list(self.decoder1.parameters()) +
            list(self.decoder2.parameters()) +
            list(self.decoder3.parameters())
        )
        
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.learning_rate * 0.1},  # 백본은 낮은 학습률
            {"params": classifier_params, "lr": self.learning_rate}        # 분류기는 높은 학습률
        ], weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }