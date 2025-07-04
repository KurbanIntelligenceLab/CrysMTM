import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer

DEFAULT_MODEL_NAME = "prajjwal1/bert-tiny"
DEFAULT_MAX_LENGTH = 512


class BERTClassifier(nn.Module):
    """BERT-based classifier for text classification."""

    def __init__(self, model_name="prajjwal1/bert-tiny", num_classes=3, max_length=512):
        super(BERTClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length

        # Load BERT model and tokenizer
        self.bert_model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, batch):
        """Forward pass through BERT model.

        Args:
            batch: Dictionary containing tokenized inputs from tokenizer

        Returns:
            outputs: BERT model outputs with loss and logits
        """
        outputs = self.bert_model(**batch)
        return outputs

    def get_tokenizer(self):
        """Get the BERT tokenizer for data preprocessing."""
        return self.tokenizer


def create_bert_classifier(
    model_name="prajjwal1/bert-tiny", num_classes=3, max_length=512
):
    """Factory function to create a BERT classifier.

    Args:
        model_name: Name of the BERT model to use
        num_classes: Number of output classes
        max_length: Maximum sequence length for tokenization

    Returns:
        BERTClassifier: The configured BERT classifier
    """
    return BERTClassifier(
        model_name=model_name, num_classes=num_classes, max_length=max_length
    )
