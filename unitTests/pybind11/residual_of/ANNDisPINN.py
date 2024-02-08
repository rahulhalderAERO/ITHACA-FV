import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network architecture
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size) # Fully connected layer 2
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Example data
X_train = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = torch.FloatTensor([[0], [1], [1], [0]])

# Define hyperparameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
num_epochs = 1000

# Initialize the model
model = SimpleANN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predicted = model(X_train)
    predicted = predicted.round() # Round predictions to 0 or 1
    print(f'Predicted: {predicted}')
