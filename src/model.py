import torch
from torch import nn
from transformers import BertForSequenceClassification

def create_model(num_labels=2):
    """Create a BERT sequence classification model."""
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
    return model
