import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv('wine_red.csv',sep = ';')# Get data

# copy the data and normalize
df_scaled = data.copy()
for column in df_scaled.columns:
    df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()

df_scaled_shape = df_scaled.shape
df_scaled.head()

# split the data into train and test set
train, test = train_test_split(df_scaled, test_size=0.2, random_state=42, shuffle=True)
print('Train dataset shape:'+str(train.shape))
print('Test dataset shape:'+str(test.shape))
train.head()

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import numpy as np
from numpy import random

#Define model hyperparameters
batch_size = 100
x_dim  = train.shape[1]
hidden_dim = 10
latent_dim = 6
lr = 1e-3
epochs = 50


# Define custom test Train and test datasets
class train_dataset(Dataset):

    def __init__(self, train):
        # data loading
        state1 = np.float32(train)
        self.st1 = torch.from_numpy(state1)
        self.n_samples1 = state1.shape[0]

    def __getitem__(self, index):
        return self.st1[index]

    def __len__(self):
        return self.n_samples1


class test_dataset(Dataset):

    def __init__(self, test):
        # data loading
        state2 = np.float32(test)
        self.st2 = torch.from_numpy(state2)
        self.n_samples2 = state2.shape[0]

    def __getitem__(self, index):
        return self.st2[index]

    def __len__(self):
        return self.n_samples2


# Dataloader loads data for training
train_loader = DataLoader(dataset = train_dataset(train),batch_size = 100, shuffle = True)
test_loader = DataLoader(dataset = test_dataset(test),batch_size = 100, shuffle = True)


# Define Encoder architecture

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC1     = nn.Linear(input_dim, hidden_dim)
        self.FC2     = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mu   = nn.Linear(hidden_dim, latent_dim)
        self.FC_var  = nn.Linear (hidden_dim, latent_dim)

        self.training = True

    def forward(self, x):
        h_       = F.relu(self.FC1(x))
        h_       = F.relu(self.FC2(h_))
        mean     = self.FC_mu(h_) # mu of simple tractable distribution  Q
        log_var  = self.FC_var(h_)  # sigma of Q

        return mean, log_var
# Define Decoder architecture

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.FC1    = nn.Linear(latent_dim, hidden_dim)
        self.FC2    = nn.Linear(hidden_dim, hidden_dim)
        self.FC_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h     = F.relu(self.FC1(x))
        h     = F.relu(self.FC2(h))

        #x_hat = torch.sigmoid(self.FC_output(h))
        x_hat = self.FC_out(h)
        return x_hat

# Define the complete model
encoder = Encoder(input_dim= x_dim, hidden_dim= hidden_dim, latent_dim= latent_dim)
decoder = Decoder(latent_dim= latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

class Model(nn.Module):
    def __init__(self, Encoder=encoder, Decoder=decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparam(self, mean, var):
        epsilon = torch.randn_like(var)# sampling epsilon
        z       = mean + var*epsilon # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var    = self.Encoder(x)
        z                = self.reparam(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)

        return x_hat, mean, log_var


#model = Model(Encoder=encoder, Decoder=decoder)
model = Model()

# Define Loss function and optimizer

from torch.optim import Adam
loss_fun = torch.nn.MSELoss()
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = loss_fun(x_hat, x)
    KLDiv             = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLDiv

optimizer = Adam(model.parameters(), lr=lr)

# Training pipeline

print("Start training VAE...")
#model.train()
def train(dataloader, model, loss_function, optimizer, batch_size, overall_train_loss, epoch):
    model.train()
    for batch_idx, (x) in enumerate(dataloader):
        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_train_loss / (batch_idx*batch_size))
    return overall_train_loss

def test(dataloader, model, loss_function, overall_test_loss):
    model.eval()
    with torch.no_grad():
        for batch_idx, (y) in enumerate(dataloader):
            x_hat, mean, log_var = model(y)
            overall_test_loss += loss_function(y, x_hat, mean, log_var)
    return overall_test_loss

epochs = 50
otrl = []
otel = []
for epoch in range(epochs):
    overall_train_loss = 0
    overall_test_loss = 0
    otrl.append(train(train_loader, model, loss_function, optimizer, batch_size, overall_train_loss,epoch))
    otel.append(test(test_loader, model, loss_function, overall_test_loss))

xc = np.arange(len(otrl))
plt.plot(xc, otrl, '-b', label = 'Overall Train loss')
plt.plot(xc, otel, '-g', label = 'Overall Test loss')
plt.legend()
plt.show()
