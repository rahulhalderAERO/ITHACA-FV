import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from of_pybind11_system import of_pybind11_system
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix


# Form the input in torch tensor : 


x = torch.linspace(0.1,0.8,5)

y = torch.linspace(0.1,0.8,5)

Input_Tensor = torch.zeros(x.numel()*y.numel(),2)

for i in range(5):
   for j in range(5):
       
       Input_Tensor[i*5+j,0] = x[i]
       Input_Tensor[i*5+j,1] = y[j]

#print("The value of Input_Tensor is ===", Input_Tensor)



a = of_pybind11_system(["."])
T = a.getT()

Output_Tensor = torch.from_numpy(T).float()

#print("The value of Output_Tensor is ===", Output_Tensor)

       




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
for epoch in range(num_epochs): #(num_epochs):
    # Forward pass
    outputs = model(Input_Tensor)
    outputs_copy = outputs.clone()
    outputs_numpy = outputs_copy.detach().numpy()
    a.setT(outputs_numpy)
    a.solve()
    #Get Temperature (T) Field from OF (the memory is shared with OF)
    T = a.getT()
    #Get Temperature (T) Field from OF (the memory is shared with OF)
    S = a.getS()
    A = a.get_system_matrix(T,S)
    A_array = A.toarray()
    b = a.get_rhs(T,S)
    A_tensor = torch.from_numpy(A_array).float()
    b_tensor = torch.from_numpy(b).float().reshape(-1,1)
    residual = torch.matmul(A_tensor,outputs) - b_tensor
    #print( "The size of A", residual.size())
    #print( "The size of b", b_tensor.size())
    residual_sq = residual**2 
    #print("The size of residual_sq", residual_sq)  
    loss = torch.mean(residual_sq)
    optimizer.zero_grad()
    loss.backward()


