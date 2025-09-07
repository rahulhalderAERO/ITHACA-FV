"""Module for FeedForward model"""
import torch
import torch.nn as nn
from pina.label_tensor import LabelTensor
from torch.autograd import Variable



class LSTM(torch.nn.Module):
    
    def __init__(self, input_variables, output_variables, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        if isinstance(input_variables, int):
            self.input_variables = None
            self.input_dimension = input_variables
        elif isinstance(input_variables, (tuple, list)):
            self.input_variables = input_variables
            self.input_dimension = len(input_variables)

        if isinstance(output_variables, int):
            self.output_variables = None
            self.output_dimension = output_variables
        elif isinstance(output_variables, (tuple, list)):
            self.output_variables = output_variables
            self.output_dimension = len(output_variables)
        
        self.num_classes = self.output_dimension
        self.num_layers = num_layers
        self.input_size = self.input_dimension
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    
    def forward(self, x):
    
        # Propagate input through LSTM : Add Sliding Windows
        
        tensors_x = torch.stack([x[i:(i+self.seq_length)] for i in range(len(x)-self.seq_length-1)])
        
        # print("The size of the tensor_x is ====",tensors_x.size())
        h_0 = Variable(torch.zeros(
            self.num_layers, tensors_x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, tensors_x.size(0), self.hidden_size))
        
        ula, (h_out, _) = self.lstm(tensors_x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        output = self.fc(h_out).as_subclass(LabelTensor)
        
        if self.output_variables:
            output.labels = self.output_variables
        
        return output