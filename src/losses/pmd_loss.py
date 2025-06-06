import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveMarginDecayingLoss(nn.Module):
    def __init__(self, margin_init=0.5, margin_final=0.1, num_epochs=30):
        super(ProgressiveMarginDecayingLoss, self).__init__()
        self.margin_init = margin_init
        self.margin_final = margin_final
        self.num_epochs = num_epochs
        self.current_margin = margin_init
        self.current_epoch = 0
        
    def update_margin(self):
        # 에폭이 증가함에 따라 마진을 점진적으로 감소
        self.current_epoch += 1
        progress = min(self.current_epoch / self.num_epochs, 1.0)
        self.current_margin = self.margin_init - progress * (self.margin_init - self.margin_final)
        
    def forward(self, logits, targets):
        # Cross Entropy Loss 계산
        ce_loss = F.cross_entropy(logits, targets)
        
        # 여기서 마진을 적용한 추가 손실을 계산
        # 예시: 자신의 클래스 로짓과 가장 큰 오분류 클래스 로짓 간의 마진 강화
        batch_size = logits.size(0)
        mask = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        
        # 타겟 클래스의 로짓
        target_logits = torch.sum(logits * mask, dim=1)
        
        # 마스크를 반전시켜 타겟이 아닌 클래스의 로짓을 구함
        other_logits = logits * (1 - mask)
        
        # 타겟이 아닌 클래스 중 최대 로짓
        other_max_logits = other_logits.max(dim=1)[0]
        
        # 마진 손실 계산: 타겟 클래스의 로짓은 다른 최대 로짓보다 최소한 마진 이상 커야 함
        margin_loss = F.relu(other_max_logits - target_logits + self.current_margin).mean()
        
        # 최종 손실: 교차 엔트로피 + 마진 손실
        total_loss = ce_loss + margin_loss
        
        return total_loss
