import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch16"

class CLIPRegressor(nn.Module):
    """CLIP-based regressor for multimodal (image + text) regression."""
    def __init__(self, model_name=DEFAULT_MODEL_NAME, freeze_backbone=True):
        super(CLIPRegressor, self).__init__()
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        embed_dim = self.clip_model.config.projection_dim * 2
        self.regressor = nn.Linear(embed_dim, 1)
    def forward(self, batch):
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            image_embeds = self.clip_model.get_image_features(batch["pixel_values"])
            text_embeds = self.clip_model.get_text_features(batch["input_ids"])
        features = torch.cat([image_embeds, text_embeds], dim=1)
        return self.regressor(features)
    def get_processor(self):
        return self.processor

def create_clip_regressor(model_name=DEFAULT_MODEL_NAME, freeze_backbone=True):
    return CLIPRegressor(model_name=model_name, freeze_backbone=freeze_backbone) 