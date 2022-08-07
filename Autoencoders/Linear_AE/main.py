import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import random
import pandas as pd
import numpy as np
import linearAE as tae

if __name__=="__main__":
    seed = 17
    epoch = 200

    data = pd.read_csv('wine_red.csv',sep = ';')
    # normalize the data
    df_scaled = data.copy()
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column]  / df_scaled[column].abs().max()

    df_scaled_shape = df_scaled.shape
    mod = tae.train_AE(df_scaled, df_scaled_shape) #Train autoencoder, the trained model is returned.
