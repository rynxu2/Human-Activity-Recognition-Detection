import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from models import TransformerModel, SEQUENCE_LENGTH, N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, LEARNING_RATE

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
    
    for i in range(0, len(data) - sequence_length, sequence_length//2):
        seq = data[i:i + sequence_length]
        if len(seq) == sequence_length:
            sequences.append(seq)
            labels.append(seq['ActivityLabel'].mode()[0])
    
    return np.array(sequences), np.array(labels)

BATCH_SIZE = 32
EPOCHS = 100

df = pd.read_csv('data\\merged_data.csv')

features = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
X = df[features]
y = df['ActivityLabel']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_scaled['ActivityLabel'] = y_encoded
sequences, labels = prepare_sequences(X_scaled, SEQUENCE_LENGTH)

sequences = sequences[:, :, :-1].astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)

train_dataset = ActivityDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(label_encoder.classes_)
model = TransformerModel(
    input_size=N_FEATURES,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    num_classes=num_classes
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_accuracy = 0
best_model_state = None

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
    
    current_accuracy = 100. * correct / total
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {current_accuracy:.2f}%')
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model_state = model.state_dict()
        print(f'New best accuracy: {best_accuracy:.2f}%')

print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
torch.save({
    'model_state_dict': best_model_state,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'best_accuracy': best_accuracy,
}, 'results\\TransformerModelCopilotOptimal_best.pth')