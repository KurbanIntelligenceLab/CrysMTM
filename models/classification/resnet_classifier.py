import torch.nn as nn
from torchvision import models

# Default configuration constants
DEFAULT_MODEL_NAME = "resnet50"
DEFAULT_IMAGE_SIZE = 224


class ResNetClassifier(nn.Module):
    """ResNet classifier for image classification."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, num_classes=3, pretrained=False):
        super(ResNetClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load ResNet model
        if model_name == "resnet18":
            if pretrained:
                self.resnet_model = models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1
                )
            else:
                self.resnet_model = models.resnet18(weights=None)
        elif model_name == "resnet34":
            if pretrained:
                self.resnet_model = models.resnet34(
                    weights=models.ResNet34_Weights.IMAGENET1K_V1
                )
            else:
                self.resnet_model = models.resnet34(weights=None)
        elif model_name == "resnet50":
            if pretrained:
                self.resnet_model = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V1
                )
            else:
                self.resnet_model = models.resnet50(weights=None)
        elif model_name == "resnet101":
            if pretrained:
                self.resnet_model = models.resnet101(
                    weights=models.ResNet101_Weights.IMAGENET1K_V1
                )
            else:
                self.resnet_model = models.resnet101(weights=None)
        elif model_name == "resnet152":
            if pretrained:
                self.resnet_model = models.resnet152(
                    weights=models.ResNet152_Weights.IMAGENET1K_V1
                )
            else:
                self.resnet_model = models.resnet152(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Replace the final classification layer
        in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, images):
        """Forward pass through ResNet model.

        Args:
            images: Input images tensor

        Returns:
            logits: Classification logits
        """
        return self.resnet_model(images)


def create_resnet_classifier(
    model_name=DEFAULT_MODEL_NAME, num_classes=3, pretrained=False
):
    """Factory function to create a ResNet classifier.

    Args:
        model_name: Name of the ResNet model to use ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        ResNetClassifier: The configured ResNet classifier
    """
    return ResNetClassifier(
        model_name=model_name, num_classes=num_classes, pretrained=pretrained
    )
