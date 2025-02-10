import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('mushrooms.csv')

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split the dataset into features and target
X = df.drop(columns=['class'])
y = df['class']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(x_train.values).float()
y_train_tensor = torch.from_numpy(y_train.values).long()
X_test_tensor = torch.from_numpy(x_test.values).float()
y_test_tensor = torch.from_numpy(y_test.values).long()

# Define the neural network model
class MushroomClassifier(nn.Module):
    def __init__(self, input_size):
        super(MushroomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = MushroomClassifier(input_size=x_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 90
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    print(classification_report(y_test_tensor, predicted, target_names=['Edible', 'Poisonous']))

# Test accuracy
test_accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
print(f'Test Accuracy: {test_accuracy:.4f}')

# Train accuracy
train_accuracy = accuracy_score(y_train_tensor.numpy(), model(X_train_tensor).argmax(-1).numpy())
print(f'Train Accuracy: {train_accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'model_state.pt')

# Load the model
model.load_state_dict(torch.load('model_state.pt'))

# Test a single sample
data = X.iloc[0]
data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    output = model(data_tensor)
    _, predicted_class = torch.max(output, 1)
    print(f'Predicted class: {"Edible" if predicted_class.item() == 0 else "Poisonous"}')