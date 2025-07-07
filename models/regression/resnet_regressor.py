import torch.nn as nn
from torchvision import models

DEFAULT_MODEL_NAME = "resnet50"
DEFAULT_IMAGE_SIZE = 224


class ResNetRegressor(nn.Module):
    """ResNet regressor for image regression."""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, pretrained=False):
        super(ResNetRegressor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained

        if pretrained:
            self.resnet_model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        else:
            self.resnet_model = models.resnet50(weights=None)

        # Replace the final classification layer with a regression head (single output)
        in_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(in_features, 1)

    def forward(self, images):
        """Forward pass through ResNet model for regression."""
        return self.resnet_model(images)


def create_resnet_regressor(model_name=DEFAULT_MODEL_NAME, pretrained=False):
    """Factory function to create a ResNet regressor."""
    return ResNetRegressor(model_name=model_name, pretrained=pretrained)
