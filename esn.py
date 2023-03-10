import numpy as np


class ESN:

    def __init__(self, n_inputs: int, n_reservoir: int = 500, seed: int = 0):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_inputs
        self.random_state_ = np.random.RandomState(seed)

        # init weights
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * radius
        self.W_out = self.random_state_.rand(self.n_outputs, self.n_reservoir) * 2 - 1

    def train(self, X: np.array, Y: np.array, _lambda: float = 0.0001):
        _, H = self.teacher_forcing(X)
        H = H.reshape(-1, H.shape[-1])
        Y = Y.reshape(H.shape[0], -1)
        pseudo_inverse = np.linalg.pinv(H.T @ H + _lambda * np.eye(H.T.shape[0]))
        self.W_out = Y.T @ H @ pseudo_inverse

    def predict_one(self, X, H_old=None):
        preact_in = X @ self.W_in.T
        if H_old is None:
            H_old = np.tanh(preact_in)
        else:
            H_old = np.tanh(preact_in + H_old @ self.W.T)
        out = H_old @ self.W_out.T
        return out, H_old

    def teacher_forcing(self, X: np.array):
        hidden, predictions = [], []
        for x in X:
            h, p = [], []
            out, H_old = None, None
            for x_i in x:
                out, H_old = self.predict_one(x_i, H_old)
                h.append(H_old)
                p.append(out)
            hidden.append(h)
            predictions.append(p)
        return np.array(predictions), np.array(hidden)

    def predict(self, X: np.array, times: int):
        hidden = []
        predictions = []
        out, H_old = None, None
        for i in range(times):
            out, H_old = self.predict_one(X if out is None else out, H_old)
            hidden.append(H_old)
            predictions.append(out)
        return np.array(predictions), np.array(hidden)
