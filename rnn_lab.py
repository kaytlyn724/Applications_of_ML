# Kaytlyn Daffern
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

torch.manual_seed(42)

# move to device as in cnn section
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
data = pd.read_csv('./coin_Bitcoin.csv')

# take a look at the csv file yourself first
# columns High, Low, Open are input features and column Close is target value
x = data[['High', 'Low', 'Open']]
y = data[['Close']]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# use StandardScaler from sklearn to standardize
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y)


# split into train and evaluation (8 : 2) using train_test_split from sklearn
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# now make x and y tensors, think about the shape of train_x, it should be (total_examples, sequence_lenth, feature_size)
# we wlll make sequence_length just 1 for simplicity, and you could use unsqueeze at dimension 1 to do this
# also when you create tensor, it needs to be float type since pytorch training does not take default type read using pandas
train_x = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)
test_y = torch.tensor(test_y, dtype=torch.float32)

# different from CNN which uses ImageFolder method, we don't have such method for RNN, so we need to write the dataset class ourselves, reference tutorial is in the main documentation
class BitCoinDataSet(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# now prepare the dataloader for training set and evaluation set, and hyperparameters
hidden_size = 64
num_layers = 5
learning_rate = 0.001
batch_size = 64
epoch_size = 10

train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check inputs in the evaluation section
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()



# model design goes here
class RNN(nn.Module):

    # there is no "correct" RNN model architecture for this lab either, you can start with a naive model as follows:
    # lstm with 5 layers (or rnn, or gru) -> linear -> relu -> linear
    # lstm: nn.LSTM (https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperparameters and setup
input_size = 3          # Number of features (High, Low, Open)
hidden_size = 64        # Hidden layer size
num_layers = 5          # Number of LSTM layers
learning_rate = 0.001
num_epochs = 10

# instantiate your rnn model
rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
# loss function is nn.MSELoss since it is regression task
criteria = nn.MSELoss()
# you can start with using Adam as optimizer as well
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

# start training
rnn.train()
for epoch in range(num_epochs): # start with 10 epochs

    loss = 0.0 # you can print out average loss per batch every certain batches

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # get inputs and target values from dataloaders and move to device
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()

        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)
        outputs = rnn(inputs)
        loss_batch = criteria(outputs, targets)
        loss_batch.backward()
        optimizer.step()

        loss += loss_batch.item() # add loss for current batch
        if batch_idx % 100 == 99:    # print average loss per batch every 100 batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')

prediction = []
ground_truth = []

# evaluation
rnn.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = rnn(inputs)
        prediction.extend(outputs.cpu().numpy().flatten())
        ground_truth.extend(targets.cpu().numpy().flatten())


# remember we standardized the y value before, so we must reverse the normalization before we compute r2score
prediction = scaler_y.inverse_transform(np.array(prediction).reshape(-1, 1))
ground_truth = scaler_y.inverse_transform(np.array(ground_truth).reshape(-1, 1))

# use r2_score from sklearn
r2score = r2_score(prediction,ground_truth)
print(f'R2 Score: {r2score:.4f}')
