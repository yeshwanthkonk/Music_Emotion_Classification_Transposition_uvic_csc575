import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to convert space-separated feature strings into numerical arrays
def parse_feature(feature_str):
    return np.array([float(x) for x in feature_str.strip("[]").split()])


# Load dataset (replace with actual file path)
df = pd.read_csv("../midi_features_with_label.csv")  # Replace with actual dataset file path

# Convert feature columns from string to numerical arrays
df['mfcc_mean'] = df['mfcc_mean'].apply(parse_feature)
df['chroma_mean'] = df['chroma_mean'].apply(parse_feature)
df['spectral_contrast_mean'] = df['spectral_contrast_mean'].apply(parse_feature)

# Flatten the features
mfcc_features = np.vstack(df['mfcc_mean'].values)
chroma_features = np.vstack(df['chroma_mean'].values)
spectral_features = np.vstack(df['spectral_contrast_mean'].values)

# Combine all features into a single feature matrix
X = np.hstack([mfcc_features, chroma_features, spectral_features, df[['tempo', 'loudness_variation']].values])

# Create feature names
feature_names = [f'mfcc_{i}' for i in range(mfcc_features.shape[1])] + \
                [f'chroma_{i}' for i in range(chroma_features.shape[1])] + \
                [f'spectral_{i}' for i in range(spectral_features.shape[1])] + \
                ['tempo', 'loudness_variation']

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=feature_names)

# Add target variable (Emotion Description)
y = df['Emotion_Description'].values

# Ensure labels are 0-indexed (important for CrossEntropyLoss)
y = y - np.min(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for LSTM (batch_size, sequence_length, feature_dim)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # (batch_size, seq_length=1, feature_dim)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# Create a PyTorch Dataset class
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create DataLoader
batch_size = 32
train_dataset = EmotionDataset(X_train_tensor, y_train_tensor)
test_dataset = EmotionDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the LSTM Model
class LSTM_EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM_EmotionClassifier, self).__init__()
        self.hidden_size = hidden_size

        # Bi-directional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the last output of LSTM
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model
input_size = X_train.shape[2]  # Feature dimension
hidden_size = 128
num_classes = len(set(y_train))
model = LSTM_EmotionClassifier(input_size, hidden_size, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

# Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
