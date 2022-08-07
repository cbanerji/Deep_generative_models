import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import random


def train_AE(df_scaled, df_scaled_shape, epch = 400):
    #--------------------Define dataset-------------------------
    class StateDataset(Dataset):

        def __init__(self):
            # data loading
            state = np.float32(df_scaled)
            self.st = torch.from_numpy(state)
            self.n_samples = state.shape[0]

        def __getitem__(self, index):
            return self.st[index]

        def __len__(self):
            return self.n_samples

    dataset = StateDataset()

    # Dataloader loads data for training
    loader = DataLoader(dataset = dataset,batch_size = 64, shuffle = True)

    # Define Autoencoder architecture
    class AE(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.vec_shape = df_scaled_shape
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(self.vec_shape[1], 8),
                torch.nn.ELU(),
                torch.nn.Linear(8,4)
            )

            # Building an linear decoder with Linear
            # layer followed by ELU activation function
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(4,8),
                torch.nn.ELU(),
                torch.nn.Linear(8, self.vec_shape[1]),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    model = AE() # Model Initialization
    print(model)

    loss_function = torch.nn.MSELoss()  # Validation using MSE Loss function

    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-5) # Using an Adam Optimizer

    #------------Training loop-------------------------------
    epochs = epch
    outputs = []
    losses = []
    for epoch in range(epochs):
        for i,(a) in enumerate(loader):
          reconstructed = model(a) # Output of Autoencoder

          # Calculating the loss function
          loss = loss_function(reconstructed,a)

          # The gradients are set to zero,
          # the the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # Storing the losses in a list for plotting
          if i%50 == 0:
              losses.append(loss)
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epochs, a, reconstructed))
    mod = model.state_dict()
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses)
    plt.show()

    return mod
