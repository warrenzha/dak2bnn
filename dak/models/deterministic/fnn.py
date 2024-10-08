import torch.nn as nn

__all__ = ['FNN', 'FNNRegression']


class FNN(nn.Module):
    def __init__(self, num_features=16):
        super(FNN, self).__init__()
        self.out_features = num_features

        # Define the layers
        self.fc1 = nn.LazyLinear(512)  # Automatically infers input size at runtime
        self.relu1 = nn.ReLU()  # Activation after first hidden layer

        self.fc2 = nn.Linear(512, 128)  # Second hidden layer
        self.relu2 = nn.ReLU()  # Activation after first hidden layer

        self.fc3 = nn.Linear(128, num_features)  # Output layer

    def forward(self, x):
        # Pass through the first hidden layer
        x = self.fc1(x)
        x = self.relu1(x)

        # Pass through the second hidden layer
        x = self.fc2(x)
        x = self.relu2(x)

        # Pass through the output layer
        x = self.fc3(x)

        return x


class FNNRegression(nn.Module):
    def __init__(self, num_features=16, hidden_features=[128, 64]):
        super(FNNRegression, self).__init__()
        self.out_features = num_features

        # Define the layers
        self.fc0 = nn.LazyLinear(hidden_features[0])  # Automatically infers input size at runtime
        self.relu0 = nn.ReLU()  # Activation after first hidden layer

        self.fc1 = nn.Linear(hidden_features[0], hidden_features[1])  # Second hidden layer
        self.relu1 = nn.ReLU()  # Activation after first hidden layer

        self.fc2 = nn.Linear(hidden_features[1], num_features)  # Output layer

    def forward(self, x):
        # Pass through the first hidden layer
        x = self.fc0(x)
        x = self.relu0(x)

        # Pass through the second hidden layer
        x = self.fc1(x)
        x = self.relu1(x)

        # Pass through the output layer
        x = self.fc2(x)

        return x
