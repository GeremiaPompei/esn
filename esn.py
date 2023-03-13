import numpy as np


class ESN:

    def __init__(self, n_inputs: int, n_outputs: int, n_reservoir: int = 500, seed: int = 0):
        self.random_state_ = np.random.RandomState(seed)

        # init weights
        self.W_in = self.random_state_.rand(n_reservoir, n_inputs) * 2 - 1
        W = self.random_state_.rand(n_reservoir, n_reservoir) - 0.5
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * radius
        self.W_out = self.random_state_.rand(n_outputs, n_reservoir) * 2 - 1

    def train(self, X: np.array, Y: np.array, _lambda: float = 0.0001):
        H = self.teacher_forcing(X)
        H = H.reshape(-1, H.shape[-1])
        Y = Y.reshape(H.shape[0], -1)
        pseudo_inverse = np.linalg.pinv(H.T @ H + _lambda * np.eye(H.T.shape[0]))
        self.W_out = Y.T @ H @ pseudo_inverse

    def predict_one(self, X, H=None):
        preact_in = X @ self.W_in.T
        if H is None:
            H = np.tanh(preact_in)
        else:
            H = np.tanh(preact_in + H @ self.W.T)
        out = H @ self.W_out.T
        return out, H

    def teacher_forcing(self, X: np.array):
        hidden = []
        for x in X:
            hidden.append([])
            H = None
            for x_i in x:
                _, H = self.predict_one(x_i, H)
                hidden[-1].append(H)
        return np.array(hidden)

    def predict(self, X: np.array, times: int):
        predictions = []
        out, H = None, None
        for i in range(times):
            out, H = self.predict_one(X if out is None else out, H)
            predictions.append(out)
        return np.array(predictions)
