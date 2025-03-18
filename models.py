import torch
import torch.nn as nn
import torch.nn.functional as F

SEQUENCE_LENGTH = 100
N_FEATURES = 6
HIDDEN_SIZE = 64
NUM_LAYERS = 3
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Grok
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,  # Giảm chi phí tính toán
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = lambda x: torch.mean(x, dim=1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x