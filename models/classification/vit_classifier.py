import torch.nn as nn
from torchvision import models

# Default configuration constants
DEFAULT_MODEL_NAME = "vit_b_16"
DEFAULT_IMAGE_SIZE = 224


class ViTClassifier(nn.Module):
    """Vision Transformer (ViT) classifier for image classification."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, num_classes=3, pretrained=False):
        super(ViTClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load ViT model
        if model_name == "vit_b_16":
            if pretrained:
                self.vit_model = models.vit_b_16(
                    weights=models.ViT_B_16_Weights.IMAGENET1K_V1
                )
            else:
                self.vit_model = models.vit_b_16(weights=None)
        elif model_name == "vit_b_32":
            if pretrained:
                self.vit_model = models.vit_b_32(
                    weights=models.ViT_B_32_Weights.IMAGENET1K_V1
                )
            else:
                self.vit_model = models.vit_b_32(weights=None)
        elif model_name == "vit_l_16":
            if pretrained:
                self.vit_model = models.vit_l_16(
                    weights=models.ViT_L_16_Weights.IMAGENET1K_V1
                )
            else:
                self.vit_model = models.vit_l_16(weights=None)
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")

        # Replace the classification head
        in_features = self.vit_model.heads.head.in_features
        self.vit_model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, images):
        """Forward pass through ViT model.

        Args:
            images: Input images tensor

        Returns:
            logits: Classification logits
        """
        return self.vit_model(images)


def create_vit_classifier(
    model_name=DEFAULT_MODEL_NAME, num_classes=3, pretrained=False
):
    """Factory function to create a ViT classifier.

    Args:
        model_name: Name of the ViT model to use ('vit_b_16', 'vit_b_32', 'vit_l_16')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        ViTClassifier: The configured ViT classifier
    """
    return ViTClassifier(
        model_name=model_name, num_classes=num_classes, pretrained=pretrained
    )
