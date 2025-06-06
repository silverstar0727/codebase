import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes: int, img_size: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        # TResNet backbone (feature extractor)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:  # [B, C, H, W]
                feature_dim = features.shape[1]
            else:  # Already flattened
                feature_dim = features.shape[1]
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Bottleneck FC layer (Feature Vector 1x2048 -> Bottleneck Feature 1x512)
        self.bottleneck_fc = nn.Linear(feature_dim, 512)
        
        # Final classification layer (FC 512x32K -> 32k logits)
        self.classifier = nn.Linear(512, num_classes)
        
        # L2 Normalization
        self.l2_norm = nn.functional.normalize
        
    def forward(self, x):
        # Extract features using TResNet backbone
        features = self.backbone(x)
        
        # Apply Global Average Pooling if features are still spatial
        if len(features.shape) == 4:  # [B, C, H, W]
            features = self.gap(features)
            features = features.flatten(1)  # [B, C]
        
        # Bottleneck FC layer
        bottleneck_features = self.bottleneck_fc(features)  # [B, 512]
        
        # L2 Normalization
        bottleneck_features = self.l2_norm(bottleneck_features, p=2, dim=1)
        
        # Final classification logits
        logits = self.classifier(bottleneck_features)  # [B, num_classes]
        
        return logits


# Usage example
if __name__ == "__main__":
    # Create model
    model = TimmModel(
        model_name='tresnet_m',  # or 'tresnet_l', 'tresnet_xl'
        num_classes=32000,  # 32k classes as shown in the diagram
        pretrained=True
    )
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    dummy_targets = torch.randint(0, 2, (batch_size, 32000))  # Multi-label targets
    
    # Forward pass
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Compute loss
    loss_dict = model.compute_loss(logits, dummy_targets)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Cross Entropy loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"Soft-Triplet loss: {loss_dict['soft_triplet_loss'].item():.4f}")