import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn, save, load
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
df = pd.read_csv('cardio_train.csv', sep=';')

# Preprocess the data
df.drop('id', axis=1, inplace=True)
df['bmi'] = df['weight'] / (df['height'] / 100)**2
cholesterol_levels = {
    1: "Normal",
    2: "Borderline High",
    3: "High"
}
glucose_levels = {
    1: "Normal",
    2: "Borderline",
    3: "High"
}
df['cholesterol'] = df['cholesterol'].map(cholesterol_levels)
df['gluc'] = df['gluc'].map(glucose_levels)
df['age'] = (df['age'] / 365).astype(int)
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])
df = pd.get_dummies(df, columns=['age_group', 'cholesterol', 'gluc'], drop_first=True)
df.drop(columns=['height', 'weight', 'age'], inplace=True)
scaler = StandardScaler()
num_cols = ['ap_hi', 'ap_lo', 'bmi']
df[num_cols] = scaler.fit_transform(df[num_cols])
df = df[['ap_hi', 'alco', 'active', 'bmi', 'age_group_Senior', 
         'cholesterol_High', 'cholesterol_Normal', 'gluc_High', 'cardio']]
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# Split the dataset into features and targets
X = df.drop(columns=['cardio'])
y = df['cardio']

# Split the features and targets into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoaders for mini-batch training
batch_size = 64  # Experiment with 16, 32, 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification (2 classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model
model = Net(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lowered LR
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # Reduce LR every 30 epochs

# Train the Model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()  # Adjust learning rate

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

# Evaluate the Model
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.tolist())
        y_pred.extend(predicted.tolist())

# Print Evaluation Metrics
print(classification_report(y_true, y_pred, target_names=['Stroke', 'NO_Stroke']))
print(f'Test Accuracy: {accuracy_score(y_true, y_pred):.4f}')
