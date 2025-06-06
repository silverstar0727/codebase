import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import Accuracy
import numpy as np
from lightning.pytorch.utilities.rank_zero import rank_zero_only
try:
    import wandb
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
finally:
    pass

class CarClassifier(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        max_wrong_samples_per_epoch: int = 150,  # 매 에폭당 틀린 샘플 최대 개수
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        
        self.net = net
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_wrong_samples_per_epoch = max_wrong_samples_per_epoch
        
        # 사용자 정의 손실 함수 초기화
        self.criterion = loss_fn
        
        # 메트릭 초기화
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.net.num_classes)
        
        # Validation 결과 저장을 위한 리스트
        self.val_predictions = []
        self.val_references = []
        
    def forward(self, x):
        return self.net(x)
    
    def on_fit_start(self) -> None:
        # 손실 함수의 num_epochs 업데이트
        self.criterion.num_epochs = self.trainer.max_epochs
        
        # WandbLogger 확인 및 시각화 설정
        self.is_wandb = isinstance(self.logger, WandbLogger)
        
        # matplotlib 한글 폰트 설정 (한국어 클래스명 지원)
        try:
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
        except:
            # 폰트 설정 실패 시 기본 설정 유지
            pass

    def training_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 손실 계산
        loss = self.criterion(logits, labels)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        
        # 로깅
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        # 에폭이 끝날 때마다 마진 업데이트
        self.criterion.update_margin()
        self.log("pmd/margin", self.criterion.current_margin)
    
    def on_validation_epoch_start(self) -> None:
        # Validation 결과 초기화
        self.val_predictions = []
        self.val_references = []
        self.val_images = []  # 이미지 저장용
        self.val_indices = []  # 인덱스 저장용 (선택사항)

    
    def validation_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 기본 cross entropy 손실 (검증에서는 마진 없이)
        loss = F.cross_entropy(logits, labels)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)
        
        # 예측 결과 및 이미지 저장 (confusion matrix와 틀린 샘플 로깅을 위해)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_references.extend(labels.cpu().numpy())
        
        # 이미지를 CPU로 이동하여 저장 (메모리 효율을 위해 필요한 만큼만)
        if self.is_wandb:
            self.val_images.extend([img_tensor.cpu().clone() for img_tensor in img])
        
        # 로깅
        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": acc,
            },
            on_epoch=True,
            prog_bar=True,
        )
        
        return {"loss": loss, "preds": preds, "labels": labels}
    
    def log_wrong_predictions_to_wandb(self):
        """허깅페이스 스타일로 틀린 예측들을 W&B에 로깅"""
        if not self.is_wandb or not hasattr(self, 'val_images'):
            return
            
        # 데이터로더에서 클래스 이름 가져오기
        if hasattr(self.trainer.datamodule.val_dataset, "classes"):
            class_names = self.trainer.datamodule.val_dataset.classes
        else:
            class_names = [f"Class_{i}" for i in range(self.net.num_classes)]
        
        # 틀린 예측 찾기
        preds = np.array(self.val_predictions)
        refs = np.array(self.val_references)
        wrong_mask = preds != refs
        wrong_indices = np.where(wrong_mask)[0]
        
        if len(wrong_indices) == 0:
            print("No wrong predictions found!")
            return
        
        # W&B 테이블 생성
        table = wandb.Table(columns=["image", "true_label", "pred_label"])
        
        # ImageNet 정규화 값 (CPU에서)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 최대 150개로 제한
        max_samples = min(self.max_wrong_samples_per_epoch, len(wrong_indices))
        selected_indices = wrong_indices[:max_samples]
        
        for idx in selected_indices:
            # 이미지 처리
            img = self.val_images[idx].clone()
            
            # 역정규화 수행
            img = img * std + mean
            
            # [0, 1] 범위로 클리핑 후 [0, 255] 범위로 변환
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            img = (img * 255).astype("uint8")
            
            true_label = class_names[refs[idx]]
            pred_label = class_names[preds[idx]]
            
            table.add_data(
                wandb.Image(img),
                true_label,
                pred_label
            )
        
        # W&B에 로깅
        self.logger.experiment.log({
            f"val/wrong_predictions": table
        })

    
    def log_wrong_predictions_hf_style(self, preds, refs):
        """HuggingFace 스타일로 틀린 예측 로깅"""
        import wandb
        
        table = wandb.Table(columns=["image", "true_label", "pred_label"])
        
        wrong_idx = preds != refs
        val_dataset = self.trainer.datamodule.val_dataset
        
        # 클래스 이름
        if hasattr(val_dataset, "classes"):
            class_names = val_dataset.classes
        else:
            class_names = [f"Class_{i}" for i in range(self.net.num_classes)]
        
        cnt = 0
        for idx in np.where(wrong_idx)[0].tolist():
            if cnt >= self.max_wrong_samples_per_epoch:
                break
                
            # 원본 데이터셋에서 이미지 가져오기
            try:
                # val_dataset[idx]에서 이미지 추출
                sample = val_dataset[idx]
                if isinstance(sample, dict) and 'image' in sample:
                    image = sample['image']
                elif isinstance(sample, tuple):
                    image = sample[0]  # (image, label) 형태
                else:
                    continue
                    
                # PIL 이미지면 바로 사용, 텐서면 변환
                if hasattr(image, 'save'):  # PIL 이미지
                    wandb_image = wandb.Image(image)
                else:  # 텐서
                    from torchvision.transforms import ToPILImage
                    to_pil = ToPILImage()
                    wandb_image = wandb.Image(to_pil(image))
                
                table.add_data(
                    wandb_image,
                    class_names[refs[idx]],
                    class_names[preds[idx]]
                )
                cnt += 1
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        if cnt > 0:
            self.logger.experiment.log({
                f"val/wrong_predictions": table
            })
        
    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        # 틀린 예측 W&B 로깅
        if self.is_wandb and len(self.val_predictions) > 0:
            self.log_wrong_predictions_to_wandb()
        
        # Confusion Matrix 생성 및 로깅
        if len(self.val_predictions) > 0:
            self.log_confusion_matrix()
        
        # 메모리 정리
        if hasattr(self, 'val_images'):
            del self.val_images
        self.val_predictions.clear()
        self.val_references.clear()
    
    def log_confusion_matrix(self):
        """상위 20개 혼동 클래스에 대한 Confusion Matrix를 생성하고 W&B에 로깅"""
        try:
            import wandb
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix
            
            # matplotlib 한글 폰트 설정
            plt.rcParams["font.family"] = "NanumGothic"
            plt.rcParams["axes.unicode_minus"] = False
            
            # numpy 배열로 변환
            preds = np.array(self.val_predictions)
            refs = np.array(self.val_references)
            
            # 클래스 이름 가져오기
            if hasattr(self.trainer.datamodule.val_dataset, "classes"):
                labels = self.trainer.datamodule.val_dataset.classes
            else:
                labels = [f"Class_{i}" for i in range(self.net.num_classes)]
            
            # Confusion Matrix 생성
            cm = confusion_matrix(refs, preds)
            np.fill_diagonal(cm, 0)  # 정답 예측은 제거하여 혼동만 표시
            
            # Step 1: 가장 많이 틀린 상위 40개 클래스 찾기 (행 기준)
            top_n = 40
            misclassified_counts = cm.sum(axis=1)
            top_true_classes = np.argsort(misclassified_counts)[::-1][:top_n]
            
            # Step 2: 각 클래스마다 가장 많이 혼동되는 예측 클래스 추가 (열 기준)
            top_confused_classes = set(top_true_classes)
            for cls in top_true_classes:
                most_confused_pred = np.argmax(cm[cls])
                top_confused_classes.add(most_confused_pred)
            
            # Step 3: 서브매트릭스 추출
            top_confused_classes = sorted(top_confused_classes)
            reduced_cm = cm[np.ix_(top_confused_classes, top_confused_classes)]
            reduced_labels = [labels[i] for i in top_confused_classes]
            
            # 클래스 이름 축약 (40개 클래스에 적합하게)
            def truncate_label(label, max_len=15):
                if len(label) <= max_len:
                    return label
                return label[:max_len-3] + "..."
            
            truncated_labels = [truncate_label(label) for label in reduced_labels]
            
            # Confusion Matrix 플롯 생성 (40개 클래스에 적합한 크기)
            plt.figure(figsize=(24, 20))
            
            # Heatmap 생성 (40개 클래스용 설정)
            ax = sns.heatmap(
                reduced_cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=truncated_labels,
                yticklabels=truncated_labels,
                cbar_kws={'shrink': 0.6},
                annot_kws={'size': 6}  # 텍스트 크기 줄임
            )
            
            # 제목과 축 레이블 설정
            plt.title(f"Top-{top_n} Confused Classes - Epoch {self.current_epoch}", 
                    fontsize=16, pad=20)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("True Label", fontsize=12)
            
            # x축 레이블 회전 및 정렬로 겹침 방지 (40개용)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # W&B에 로깅
            if self.is_wandb:
                self.logger.experiment.log({
                    "val/confusion_matrix": wandb.Image(plt)
                })
            
            # 로컬 저장
            save_path = f"confusion_matrix_epoch_{self.current_epoch}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            print(f"Confusion matrix saved: {save_path}")
            print(f"Matrix shape: {reduced_cm.shape}")
            print(f"Selected classes: {len(top_confused_classes)}")
            print(f"Most confused classes: {[labels[i] for i in top_true_classes[:5]]}")
                    
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")
            import traceback
            traceback.print_exc()

    def test_step(self, batch, batch_idx):
        img, labels = batch
        logits = self(img)
        
        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, labels)
        
        # 로깅
        self.log("test/acc", acc, on_epoch=True)
        
        return {"preds": preds, "labels": labels}
    
    def on_predict_start(self) -> None:
        """예측 시작 시 결과를 저장할 리스트 초기화"""
        self.predict_results = []
        
    def predict_step(self, batch, batch_idx):
        """예측 단계 - TTA(Test Time Augmentation) 적용"""        
        img, filenames = batch  # predict_dataset에서 (이미지, 파일명) 반환
        
        all_predictions = []
        
        # 원본 이미지 예측
        with torch.no_grad():
            logits = self(img)
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs)
        
            img_flipped = torch.flip(img, dims=[-1])  # 가로 뒤집기
            logits_flipped = self(img_flipped)
            probs_flipped = F.softmax(logits_flipped, dim=1)
            all_predictions.append(probs_flipped)
        
        # 모든 예측 결과의 평균 계산
        ensemble_probs = torch.stack(all_predictions).mean(dim=0)
        
        # 배치 결과를 저장
        for i, filename in enumerate(filenames):
            # 파일명에서 확장자 제거하여 ID 생성
            file_id = filename.rsplit('.', 1)[0]  # .jpg, .png 등 확장자 제거
            
            # 각 클래스별 확률값과 함께 저장
            result = {
                'ID': file_id,
                'probabilities': ensemble_probs[i].cpu().numpy()
            }
            self.predict_results.append(result)
        
        return {
            "probabilities": ensemble_probs,
            "filenames": filenames
        }

    def on_predict_end(self) -> None:
        """예측 종료 시 CSV 파일로 저장"""
        import pandas as pd
        import numpy as np
        
        if not hasattr(self, 'predict_results') or not self.predict_results:
            print("No prediction results to save.")
            return
            
        # 기존 클래스 이름 가져오기
        original_class_names = self.trainer.datamodule.predict_dataset.classes
        
        # 추가할 클래스들
        additional_classes = [
            'K5_3세대_하이브리드_2020_2022',
            '디_올뉴니로_2022_2025',
            '718_박스터_2017_2024',
            'RAV4_2016_2018',
            'RAV4_5세대_2019_2024',
        ]
        
        # 모든 클래스 이름 합치기
        all_class_names = original_class_names + additional_classes
        
        # DataFrame 생성을 위한 데이터 준비
        data = {'ID': []}
        for class_name in all_class_names:
            data[class_name] = []
        
        # 결과 데이터를 DataFrame 형식으로 변환
        for result in self.predict_results:
            data['ID'].append(result['ID'])
            probs = result['probabilities']
            
            # 기존 클래스들의 확률값 추가
            for i, class_name in enumerate(original_class_names):
                data[class_name].append(probs[i])
            
            # 추가 클래스들의 확률값을 0.0으로 설정
            for class_name in additional_classes:
                data[class_name].append(0.0)
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # ID 기준으로 정렬 (파일명 순서)
        df = df.sort_values('ID').reset_index(drop=True)
        
        # 컬럼 순서 정렬 (ID는 첫 번째, 나머지는 알파벳 순)
        class_columns = sorted([col for col in df.columns if col != 'ID'])
        df = df[['ID'] + class_columns]
        
        # CSV 파일로 저장
        csv_path = 'predictions.csv'
        df.to_csv(csv_path, index=False)
        
        # 메모리 정리
        del self.predict_results

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
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
