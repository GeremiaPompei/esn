from random import Random

import numpy as np
import pandas as pd


def split_dataset(data, val_perc=0.1, seed=10):
    Random(seed).shuffle(data)
    X, Y = data[:-1], data[1:]
    test_size = int(len(data) * val_perc)
    return X[:-test_size], X[test_size:], Y[:-test_size], Y[test_size:]


def noisy_sin_dataset_initialization(cols=100, rows=200, val_perc=0.2):
    data = np.array([[[
        np.sin(y) + np.random.uniform(high=0.1)
    ] for y in range(0, cols)] for _ in range(rows)])
    return split_dataset(data, val_perc)


def cardano_dataset_initialization(val_perc=0.2):
    df = pd.read_csv("cardano_dataset.csv")
    df.drop(['Date', 'Vol.'], axis=1, inplace=True)
    df = df.iloc[::-1]  # reverse
    data = np.array([df.to_numpy()]).reshape(15, -1, 5)
    return split_dataset(data, val_perc)
