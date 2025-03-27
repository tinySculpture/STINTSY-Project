import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, list_hidden, activation='relu'):
        """Class constructor for NeuralNetwork

        Arguments:
            input_size {int} -- Number of features in the dataset
            num_classes {int} -- Number of classes in the dataset
            list_hidden {list} -- List of integers representing the number of
            units per hidden layer in the network
            activation {str, optional} -- Type of activation function. Choices
            include 'sigmoid', 'tanh', and 'relu'.
        """
        super(NeuralNetwork, self).__init__()
        self.activation = activation
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, list_hidden[0]))
        layers.append(self.get_activation_layer())
        
        # Hidden layers
        for i in range(len(list_hidden) - 1):
            layers.append(nn.Linear(list_hidden[i], list_hidden[i+1]))
            layers.append(self.get_activation_layer())
        
        # Output layer
        layers.append(nn.Linear(list_hidden[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        self.init_weights()
    
    def get_activation_layer(self):
        """Returns the activation function based on the user selection."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function. Choose 'relu', 'sigmoid', or 'tanh'.")
    
    def init_weights(self):
        """Initializes network weights with normal distribution."""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Performs forward propagation."""
        return self.model(x)
    
    def predict(self, x):
        """Returns class predictions."""
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)
    
    def predict_proba(self, x):
        """Returns class probabilities using softmax."""
        with torch.no_grad():
            outputs = self.forward(x)
            return F.softmax(outputs, dim=1)
