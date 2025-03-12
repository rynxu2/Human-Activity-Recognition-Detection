import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import math

# Hyperparameters
SEQUENCE_LENGTH = 50  # 1 second of data at 50Hz
N_FEATURES = 6  # AccelX,Y,Z and GyroX,Y,Z
HIDDEN_SIZE = 64
NUM_LAYERS = 3
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 50
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
    
# ChatGPT optimal
class TransformerModelChatGPTOptimal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes, dropout=0.1, pooling='mean'):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if pooling == 'mean':
            self.pooling = lambda x: torch.mean(x, dim=1)
        elif pooling == 'max':
            self.pooling = lambda x: torch.max(x, dim=1).values
        else:
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

class ActivityDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def prepare_sequences(data, sequence_length):
    sequences = []
    labels = []
    
    for i in range(0, len(data) - sequence_length, sequence_length//2):  # 50% overlap
        seq = data[i:i + sequence_length]
        if len(seq) == sequence_length:
            sequences.append(seq)
            labels.append(seq['ActivityLabel'].mode()[0])  # Most common label in sequence
    
    return np.array(sequences), np.array(labels)

def train_model():
    # Load data
    df = pd.read_csv('activity_data.csv')
    
    # Prepare features and labels
    features = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
    X = df[features]
    y = df['ActivityLabel']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create sequences
    X_scaled['ActivityLabel'] = y_encoded
    sequences, labels = prepare_sequences(X_scaled, SEQUENCE_LENGTH)
    
    # Remove ActivityLabel column from sequences
    sequences = sequences[:, :, :-1].astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = ActivityDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = ActivityDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = TransformerModel(
        input_size=N_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_classes=num_classes
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
    }, 'activity_model.pth')

if __name__ == '__main__':
    train_model() 