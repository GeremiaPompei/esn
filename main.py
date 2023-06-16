from src.dataloader.dataloader import cardano, sincode, noisy_sin, mg17
from src.gridsearch.gridsearch import run_gridsearch
from src.model.esn import ESN
from src.model.eusn import EuSN
from src.plots.plotting import general_plot
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)

TR_X, TS_X, TR_Y, TS_Y = sincode()

tr_loss, ts_loss, tr_preds, ts_preds = run_gridsearch(
    configs=dict(
        hidden_size=[50, 100, 200],
        input_scaling=[0.1, 0.5],
        spectral_radius=[0.8, 0.9],
        #leakage_rate=[0.1, 0.5],
        step_size=[0.01, 0.001],
        diffusion_coefficient=[0.01, 0.001],
        sparsity=[0.9, 0.5],
        reg=[1e-5],
        transient=[100, 150],
    ),
    TR=(TR_X, TR_Y),
    TS=(TS_X, TS_Y),
    vl_perc=0.2,
    attempts_for_config=5,
    model_constructor=EuSN,
)

general_plot(TR_Y.reshape(-1), tr_preds, TS_Y.reshape(-1), ts_preds)
