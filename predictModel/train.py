import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleCNN, SimpleLSTM, TCN, StackedModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "../outputData"
OUTPUT_PATH = "./outputData"

def to_one_hot(labels):
    encoder = OneHotEncoder(sparse=False, categories='auto')
    labels = labels.reshape(-1, 1)
    return encoder.fit_transform(labels)

# Custom Dataset Class


class CustomTimeSeriesDataset(Dataset):
    def __init__(self, features, labels, window_size=10):
        # Assuming features and labels are already tensors
        self.features = features
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        # Using tensor indexing to create a window of features
        feature_window = self.features[idx:idx + self.window_size]
        # Fetch the label for the last day in the window
        label = self.labels[idx + self.window_size - 1]

        # There's no need to convert to tensors here since they are already tensors
        return feature_window, label

predicted_headers = np.array(['LSTM', 'CNN', 'TCN', 'Stacked'])
predicted_labels = np.array([])

# Model parameters
input_dim = 5  # Number of features
hidden_dim = 50  # Number of features in hidden state
num_layers = 1  # Number of stacked lstm layers
output_dim = 2  # Binary classification

# Dummy data preparation
# 100 samples, 10 time steps per sample, 5 features per time step
batch_size = 32
seq_length = 1
num_features = 6
num_classes = 5  # Output size from your models

# Create random data and labels
x_train = torch.randn(batch_size, seq_length, num_features)
x_train = x_train.transpose(1, 2)
y_train = torch.randint(0, num_classes, (batch_size,))

data_in = pd.read_csv(f'{DATA_PATH}/BTC_feature.csv')


data_in['time'] = pd.to_datetime(data_in['time'])
train_data = data_in[data_in['time'].dt.year.isin([2020, 2021, 2022, 2023])]
test_data = data_in[data_in['time'].dt.year.isin([2020, 2021, 2022, 2023])]
features_train = train_data[['Open', 'OYO',
                             'Sigma', 'HL_in', 'CO_in', 'Sigma_in']]
features_test = test_data[['Open', 'OYO',
                           'Sigma', 'HL_in', 'CO_in', 'Sigma_in']]


data_out = pd.read_csv(f'{DATA_PATH}/BTC_label.csv', parse_dates=['time'])
class_columns = ['class1', 'class2', 'class3', 'class4', 'class5']
data_out['class'] = data_out[class_columns].idxmax(
    axis=1).str.replace('class', '').astype(int) - 1

class_labels = to_one_hot(data_out['class'].values)
y_train = class_labels[train_data.index]
y_test = class_labels[test_data.index]

X_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(features_test.values, dtype=torch.float32)
# breakpoint()
# DataLoader setup
train_dataset = CustomTimeSeriesDataset(X_train_tensor, y_train)
test_dataset = CustomTimeSeriesDataset(X_test_tensor, y_test)

# train_dataset = TensorDataset(X_train_tensor, y_train)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=False)
# test_dataset = TensorDataset(X_test_tensor, y_test)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)
model = SimpleCNN()  # need transpose
model_lstm = SimpleLSTM()
model_TCN = TCN()  # need transpose
model_stacked = StackedModel()
loss_function = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model_TCN.parameters(), lr=0.001)


def train_model(model, train_loader, optimizer, loss_function, epochs, name):
    predicts = np.array([])
    model.train()  # Set the model to training mode

    best_val_loss = float('inf')
    # breakpoint()

    for epoch in range(epochs):
        total_loss = 0
        total = 0
        correct = 0
        # ten_day_data =
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(data)  # Forward pass
            loss = loss_function(outputs, target)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights
            total_loss += loss.item()  # Sum up batch loss

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            target = target.argmax(-1)
            correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        # if batch_idx % 10 == 0:
        #     print(
        #         f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')
        avg_val_loss = total_loss/len(train_loader)
        print(
            f'Epoch {epoch + 1}, Average Loss: {avg_val_loss}, Accuracy: {accuracy}%')
        # print(f'Epoch {epoch+1}, Average Loss: {total_loss/len(train_loader)}')
        if avg_val_loss < best_val_loss:
            checkpoint_path = f'checkpoint/{name}_best.pth'
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f'Saved best model at epoch {epoch+1} with validation loss: {best_val_loss}')

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(data)  # Forward pass

            total_loss += loss.item()  # Sum up batch loss
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            target = target.argmax(-1)
            correct += (predicted == target).sum().item()
            predicts = np.append(predicts, predicted.numpy())

        accuracy = 100 * correct / total
        # if batch_idx % 10 == 0:
        #     print(
        #         f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}')
        avg_val_loss = total_loss/len(train_loader)
        print(
            f'Epoch {epoch + 1},Valid Accuracy: {accuracy}%')
    
    return predicts

f = pd.DataFrame([])
date = []
dt = datetime.datetime(2017, 1, 10, 14, 0, 0)
end = datetime.datetime(2024, 6, 10, 13, 0, 0)
step = datetime.timedelta(days=1)

while dt < end:
    date.append(dt.strftime('%Y-%m-%d'))
    dt += step

f.insert(0, 'time', date)

# Execute training
predicted_labels = train_model(model_lstm, train_loader, optimizer, loss_function, epochs=50, name="LSTM")
f.insert(1, predicted_headers[0], predicted_labels[:len(date)])

predicted_labels = train_model(model, train_loader, optimizer, loss_function, epochs=50, name="CNN")
f.insert(2, predicted_headers[1], predicted_labels[:len(date)])

predicted_labels = train_model(model_TCN, train_loader, optimizer, loss_function, epochs=10, name="TCN")
f.insert(3, predicted_headers[2], predicted_labels[:len(date)])

predicted_labels = train_model(model_stacked, train_loader, optimizer, loss_function, epochs=25, name="Stack")
f.insert(4, predicted_headers[3], predicted_labels[:len(date)])
# train

f.to_csv(f'{OUTPUT_PATH}/BTC_predict.csv', index=False)


