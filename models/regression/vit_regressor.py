import torch.nn as nn
from torchvision import models

DEFAULT_MODEL_NAME = "vit_b_16"
DEFAULT_IMAGE_SIZE = 224


class ViTRegressor(nn.Module):
    """Vision Transformer (ViT) regressor for image regression."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, pretrained=False):
        super(ViTRegressor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        if pretrained:
            self.vit_model = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1
            )
        else:
            self.vit_model = models.vit_b_16(weights=None)

        # Replace the classification head with a regression head (single output)
        in_features = self.vit_model.heads.head.in_features
        self.vit_model.heads.head = nn.Linear(in_features, 1)

    def forward(self, images):
        """Forward pass through ViT model for regression."""
        return self.vit_model(images)


def create_vit_regressor(model_name=DEFAULT_MODEL_NAME, pretrained=False):
    """Factory function to create a ViT regressor."""
    return ViTRegressor(model_name=model_name, pretrained=pretrained)
