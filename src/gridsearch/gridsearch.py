import itertools
import json
import pickle
from typing import Callable
import numpy as np
from tqdm import tqdm
from src.loss.mse import mse

from src.model.esn import ESN


def create_and_train_by_config(X_TR: np.array, Y_TR: np.array, config: dict, model_constructor: Callable = ESN) -> tuple[ESN, list]:
    """
    Function able to create an ESN model from a configuration and train it.

    X_TR: Input time series.
    Y_TR: Output values.
    config: Configuration containing hyperparams.
    model_constructor: Function able to construct the model.

    returns:
        tuple[ESN, list]: Trained model and list of hidden states.
    """
    model = model_constructor(
        input_size=X_TR.shape[-1], 
        output_size=Y_TR.shape[-1], 
        **{k: v for k, v in config.items() if k not in ['reg', 'transient']},
    )
    states = model.train(
        X_TR, 
        Y_TR,
        reg=config['reg'],
        transient=config['transient'],
    )
    return model, states


def run_gridsearch(
        configs: dict,
        TR: tuple[np.array, np.array],
        TS: tuple[np.array, np.array],
        vl_perc: float = 0.2,
        attempts_for_config: int = 1,
        save_name: str = None,
        model_constructor: Callable = ESN,
) -> tuple:
    """
    Gridsearch function able to find the best hyperparameters configuration, train the model with the best config and test it.

    train_func: Function able to create a model and train it given a config, a train and validation set and a number of epochs.
    configs: Hyperparameters configurations to investigate to find the best one that minimizes the loss on validation set. In particular this is a dictionary of lists for each hyperparam to investigate that is transformed by this function in a list of dictionaries.
    TR: Training set data (X, Y).
    TS: test set data (X, Y).
    epochs: Number of epochs of training both for model selection and model evaluation.
    vl_perc: Percentage of training set used for validation.
    attempts_for_config: Number of attempts to do for each configuration. The loss that it's minimized is the mean of each loss of each attempt.
    save_name: Name used for saving the file related to mse info (mse of training, validation and test set).
    model_constructor: Function able to construct the model.

    returns: A tuple of 4 variables related to the result of training function during the model evaluation phase (training on entire training set and test on test set).
    """
    configs = [dict(zip(configs.keys(), t)) for t in itertools.product(*configs.values())]
    best_config, best_loss = {}, None
    X_TR, Y_TR = TR
    X_TS, Y_TS = TS
    vl_size = int(vl_perc * len(X_TR))
    X_VL_TR, Y_VL_TR = X_TR[:-vl_size], Y_TR[:-vl_size]
    X_VL_TS, Y_VL_TS = X_TR[-vl_size:], Y_TR[-vl_size:]
    for i, config in enumerate(tqdm(configs)):
        vl_loss = 0
        for j in range(attempts_for_config):
            model, states = create_and_train_by_config(X_VL_TR, Y_VL_TR, config, model_constructor=model_constructor)
            preds = model(X_VL_TS)
            vl_loss += mse(Y_VL_TS, preds)
        vl_loss /= attempts_for_config
        print(f'{i + 1}/{len(configs)} - Tried config {config} with loss {vl_loss}')
        if best_loss is None or vl_loss < best_loss:
            best_config = config
            best_loss = vl_loss
    print(f'Best config: {best_config} with loss {best_loss}')
    print('Retraining...')
    model, states = create_and_train_by_config(X_TR, Y_TR, best_config, model_constructor=model_constructor)
    tr_preds = model(X_TR, states=states)
    ts_preds = model(X_TS)
    tr_loss = mse(Y_TR, tr_preds)
    ts_loss = mse(Y_TS, ts_preds)
    mse_data = {
        'train_mse': tr_loss,
        'validation_mse': best_loss,
        'test_mse': ts_loss
    }
    print(mse_data)
    if save_name is not None:
        with open(f'esn/variables/{save_name}_best_hyperparams.json', 'w') as file:
            json.dump(best_config, file)
        with open(f'esn/variables/{save_name}_mse_data.json', 'w') as file:
            json.dump(mse_data, file)
        with open(f'esn/variables/{save_name}_model.pickle', 'wb') as file:
            pickle.dump(model, file)

    return tr_loss, ts_loss, tr_preds, ts_preds