import time
import numpy as np
from matplotlib import pyplot as plt

from dataset import cardano_dataset_initialization, noisy_sin_dataset_initialization
from esn import ESN


def predict_loss_plot(model, X, Y):
    # predictions
    start = time.time()
    predictions = np.array([p for p in [model.predict(x[0], times=len(x)) for x in X]])
    end = time.time()
    print('prediction', end - start, 'seconds')

    # print
    loss = ((predictions - Y) ** 2).mean()
    print('loss', loss)

    # plot
    p = predictions[0].reshape(-1)
    y = Y[0].reshape(-1)
    plt.plot(p, label='prediction')
    plt.plot(y, label='label')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    for dataset_initialization in [noisy_sin_dataset_initialization, cardano_dataset_initialization]:
        print(dataset_initialization.__name__)
        # dataset construction
        start = time.time()
        X_TR, X_TS, Y_TR, Y_TS = dataset_initialization()
        end = time.time()
        print('dataset construction', end - start, 'seconds')

        # model initialization
        start = time.time()
        model = ESN(X_TR.shape[-1], Y_TR.shape[-1], n_reservoir=1000)
        end = time.time()
        print('model initialization', end - start, 'seconds')

        predict_loss_plot(model, X_TS, Y_TS)

        # training
        start = time.time()
        model.train(X_TR, Y_TR)
        end = time.time()
        print('training', end - start, 'seconds')

        predict_loss_plot(model, X_TS, Y_TS)
        print()


if __name__ == "__main__":
    main()
