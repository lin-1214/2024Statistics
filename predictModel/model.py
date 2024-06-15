import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=10, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Third convolutional layer
        self.conv3 = nn.Conv1d(
            in_channels=10, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Flatten layer to prepare for the fully connected layer
        self.flatten = nn.Flatten()

        # Adjust the Linear layer input features to reflect the output dimensions from conv3
        # Assuming each sequence is 10 steps long and convolutions do not reduce size due to padding
        self.fc1 = nn.Linear(5 * 6, 90)  # 5 output channels, 10 steps

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.output = nn.Linear(90, 5)

    def forward(self, x):
        # x shape expected to be [batch_size, num_features, sequence_length]
        # No need to unsqueeze since already in proper shape

        # Applying first conv layer
        x = self.relu1(self.conv1(x))

        # Applying second conv layer
        x = self.relu2(self.conv2(x))

        # Applying third conv layer (no pooling)
        x = self.relu3(self.conv3(x))

        # Flatten the data for the fully connected layer
        x = self.flatten(x)

        # Fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.output(x)
        return F.softmax(x, dim=1)


class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=6, hidden_size=10, batch_first=True)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=10, hidden_size=60, batch_first=True)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(60, 5)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)  # x shape: (batch_size, seq_length, 10)
        x = self.relu1(x)
        # Second LSTM layer
        x, _ = self.lstm2(x)  # x shape: (batch_size, seq_length, 60)
        # Take only the last output for the second LSTM
        x = self.relu2(x[:, -1, :])
        # Dropout
        x = self.dropout(x)
        # Output layer
        x = self.fc(x)
        return F.softmax(x, dim=1)


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.tcn1 = TCNBlock(in_channels=6, out_channels=10, dilations=[
                             1, 2, 4])  # num_features as channels
        self.tcn2 = TCNBlock(in_channels=10, out_channels=60, dilations=[
                             1])  # output of tcn1 as input channels here
        self.dropout = nn.Dropout(0.25)
        # Adjust based on the output size after all layers
        self.fc = nn.Linear(60, 5)

    def forward(self, x):
        # Transpose to (batch_size, num_features, sequence_length)
        x = x.transpose(1, 2)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x.mean(dim=2)  # Pooling over the sequence dimension
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(TCNBlock, self).__init__()
        # Assuming a simple stack of convolutions for demonstration
        layers = []
        for dilation in dilations:
            layers.append(nn.Conv1d(in_channels, out_channels,
                          kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            in_channels = out_channels  # Output of the current layer is the input to the next
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StackedModel(nn.Module):
    def __init__(self):
        super(StackedModel, self).__init__()
        self.cnn = SimpleCNN()
        self.lstm = SimpleLSTM()
        self.tcn = TCN()
        self.fc = nn.Linear(15, 5)  # 5 (output) * 3 (models)
        self.cnn.load_state_dict(torch.load("checkpoint/CNN_best.pth"))
        self.lstm.load_state_dict(torch.load("checkpoint/LSTM_best.pth"))
        self.tcn.load_state_dict(torch.load("checkpoint/TCN_best.pth"))

    def forward(self, x):
        # Input preprocessing
        # (batch_size, channels, seq_length) for CNN and TCN

        x_cnn = x
        x_lstm = x  # (batch_size, seq_length, channels) for LSTM

        # Forward through individual models
        out_cnn = self.cnn(x_cnn)
        out_lstm = self.lstm(x_lstm)
        out_tcn = self.tcn(x_cnn)

        # Concatenate outputs
        out = torch.cat((out_cnn, out_lstm, out_tcn), dim=1)

        # Final classification
        out = self.fc(out)
        return F.softmax(out, dim=1)
