import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F

SEQUENCE_LENGTH = 50  # 1 second of data at 50Hz
N_FEATURES = 6  # AccelX,Y,Z and GyroX,Y,Z
HIDDEN_SIZE = 64
NUM_LAYERS = 3
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Original
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes):
        super().__init__()

        self.embedding = nn.Linear(input_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.fc(x)
        return x
    
# Deepseek
class TransformerModelDeepseek(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=3, 
                 num_heads=4, num_classes=5, dropout=0.1):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Conv1d(input_size, hidden_size//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.position_embed = nn.Parameter(torch.randn(1, hidden_size//2, 1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size//2,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.preprocess(x)
        x = x + self.position_embed
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim//2)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim//2, seq_len)
        x = self.adaptive_pool(x).squeeze(-1)
        return self.classifier(x)

# ChatGPT
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerModelChatGPT(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, num_heads=4, num_classes=5, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
# Grok
class TransformerModelGrok(nn.Module):
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
    
# Grok optimal
class PositionalEncodingGrok(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerModelGrokOptimal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes, dropout=0.1, max_len=100):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.pos_encoding = PositionalEncodingGrok(hidden_size, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = lambda x: torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_projection(x)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x
    
# Cursor optimal
class TransformerModelCursorOptimal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes, dropout=0.1, max_len=100):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.pos_encoding = PositionalEncodingGrok(hidden_size, max_len)
        self.dropout = nn.Dropout(dropout + 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout + 0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = lambda x: torch.cat([
            torch.mean(x, dim=1),
            torch.max(x, dim=1)[0]
        ], dim=1)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# Claude optimal
class TransformerModelClaudeOptimal(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, num_heads=4, 
                 num_classes=2, seq_length=100, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.accel_projection = nn.Linear(3, hidden_size // 2)
        self.gyro_projection = nn.Linear(3, hidden_size // 2)
        self.freq_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, hidden_size))
        nn.init.xavier_uniform_(self.pos_encoder)
        self.feature_weights = nn.Parameter(torch.ones(2))
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pooling_mean = nn.AdaptiveAvgPool1d(1)
        self.pooling_max = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        if self.input_size >= 6:  # Đảm bảo có đủ kênh cho accel và gyro
            accel_data = x[:, :, :3]  # 3 trục accelerometer
            gyro_data = x[:, :, 3:6]  # 3 trục gyroscope
            accel_features = self.accel_projection(accel_data)
            gyro_features = self.gyro_projection(gyro_data)
            x = torch.cat([accel_features, gyro_features], dim=2)
        else:
            x = nn.Linear(self.input_size, self.hidden_size)(x)
        
        x_time = x  # Đặc trưng thời gian là đầu vào ban đầu
        x_freq = self.freq_extractor(x)  # Đặc trưng tần số
        weights = F.softmax(self.feature_weights, dim=0)
        x = weights[0] * x_time + weights[1] * x_freq
        x = x + self.pos_encoder[:, :seq_length, :]
        x = self.layer_norm1(x)
        x = self.dropout(x)
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.layer_norm2(x)
        x_t = x.transpose(1, 2)  # [batch_size, hidden_size, seq_length]
        mean_pool = self.pooling_mean(x_t).squeeze(-1)  # [batch_size, hidden_size]
        max_pool = self.pooling_max(x_t).squeeze(-1)    # [batch_size, hidden_size]
        x = torch.cat([mean_pool, max_pool], dim=1)  # [batch_size, hidden_size*2]
        x = self.fc1(x)
        x = self.act(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Deepseek optimal
class TransformerModelDeepseekOptimal(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, num_heads=4, num_classes=2, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation=nn.GELU()  # Tăng tốc hội tụ
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Tập trung vào đặc trưng quan trọng
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)  # (batch_size, seq_len, hidden_size)
        x = x.permute(0, 2, 1)   # (batch_size, hidden_size, seq_len)
        x = self.pooling(x).squeeze(-1)  # (batch_size, hidden_size)
        x = self.fc(x)  # (batch_size, num_classes)
        return x
    
# Copilot optimal
class TransformerModelCopilotOptimal(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_heads=4, num_classes=3, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncodingCopilot(hidden_size, dropout)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pooling = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.attention_pooling(x)
        x = self.classifier(x)
        return x

class PositionalEncodingCopilot(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(0).unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)
    
