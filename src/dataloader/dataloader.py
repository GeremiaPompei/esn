from random import Random

import numpy as np
import pandas as pd


def split_dataset(data, val_perc=0.2):
    val_size = int(len(data) * val_perc)
    TR, TS = data[:-val_size], data[-val_size:]
    features = 1 if len(data.shape) == 1 else data.shape[-1]
    return \
        TR[:-1].reshape(-1, features), \
        TS[:-1].reshape(-1, features), \
        TR[1:].reshape(-1, features), \
        TS[1:].reshape(-1, features)


def noisy_sin(rows=2000, val_perc=0.2):
    data = np.array([[np.sin(y) + np.random.uniform(high=0.01)] for y in range(rows)])
    return split_dataset(data, val_perc)


def cardano(val_perc=0.2):
    df = pd.read_csv("./dataset/cardano_dataset.csv")
    df.drop(['Date', 'Vol.'], axis=1, inplace=True)
    df = df.iloc[::-1]  # reverse
    data = df.to_numpy()
    return split_dataset(data, val_perc)


def mg17(val_perc=0.2):
    with open(f'./dataset/MG17.csv') as file:
        data = file.read().split('\n')[:-1][0]
        data = np.array([float(r) for r in data.split(',')])
    return split_dataset(data, val_perc)


def sincode(val_perc=0.2):
    df = pd.read_csv("./dataset/sincode.csv")
    df.drop(['timestamp', 'id', 'sub_id', 'numero_seriale'], axis=1, inplace=True)
    df = df.iloc[::-1]  # reverse
    data = df.to_numpy()[:, :1]
    return split_dataset(data, val_perc)
