import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

DEFAULT_IMAGE_SIZE = 224
DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch16"


class CLIPClassifier(nn.Module):
    """CLIP-based classifier for multimodal (image + text) classification."""

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch16",
        num_classes=3,
        freeze_backbone=True,
    ):
        super(CLIPClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Freeze CLIP backbone if specified
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Classification head
        # CLIP concatenates image and text embeddings, so we double the projection dim
        embed_dim = self.clip_model.config.projection_dim * 2
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, batch):
        """Forward pass through CLIP model and classifier.

        Args:
            batch: Dictionary containing 'pixel_values' and 'input_ids' from processor

        Returns:
            logits: Classification logits
        """
        # Get image and text embeddings from CLIP
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            image_embeds = self.clip_model.get_image_features(batch["pixel_values"])
            text_embeds = self.clip_model.get_text_features(batch["input_ids"])

        # Concatenate embeddings
        features = torch.cat([image_embeds, text_embeds], dim=1)

        # Classification
        logits = self.classifier(features)
        return logits

    def get_processor(self):
        """Get the CLIP processor for data preprocessing."""
        return self.processor


def create_clip_classifier(
    model_name="openai/clip-vit-base-patch16", num_classes=3, freeze_backbone=True
):
    """Factory function to create a CLIP classifier.

    Args:
        model_name: Name of the CLIP model to use
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze the CLIP backbone

    Returns:
        CLIPClassifier: The configured CLIP classifier
    """
    return CLIPClassifier(
        model_name=model_name, num_classes=num_classes, freeze_backbone=freeze_backbone
    )
