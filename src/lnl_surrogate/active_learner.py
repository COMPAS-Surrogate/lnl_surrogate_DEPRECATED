""" BO with Trieste"""
import matplotlib.pyplot as plt
from typing import List
import trieste
from trieste.observer import Observer
from trieste.objectives import mk_observer
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.acquisition.function import PredictiveVariance, ExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.experimental.plotting import plot_regret, plot_gp_2d
from trieste.space import SearchSpace
from trieste.acquisition import AcquisitionFunction
import tensorflow as tf
import os

from lnl_computer.observation.mock_observation import MockObservation
from lnl_computer.mock_data import MockData, generate_mock_data
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.cosmic_integration.star_formation_paramters import get_star_formation_prior

import numpy as np


def _generate_lnl_observer(mcz_obs: np.ndarray, compas_h5_filename: str, params: List[str]) -> Observer:
    def _f(x):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        lnls = [
            McZGrid.lnl(
                mcz_obs=mcz_obs,
                duration=1,
                compas_h5_path=compas_h5_filename,
                sf_sample={params[i]: _xi[i] for i in range(len(params))},
                n_bootstraps=0,
            )[0] * -1 for _xi in x
        ]
        _t = tf.convert_to_tensor(lnls, dtype=tf.float64)
        return tf.reshape(_t, (-1, 1))

    return mk_observer(_f)


def _get_search_space(params: List[str]) -> SearchSpace:
    prior = get_star_formation_prior()
    param_mins = [prior[p].minimum for p in params]
    param_maxs = [prior[p].maximum for p in params]
    return trieste.space.Box(param_mins, param_maxs)


def _get_gp_model(data, search_space, likelihood_variance=10):
    gpflow_model = build_gpr(data, search_space, likelihood_variance=likelihood_variance)
    model = GaussianProcessRegression(gpflow_model)
    return model


def _get_deepgp_model(data, search_space, likelihood_variance=10):
    gpflow_model = build_gpr(data, search_space, likelihood_variance=likelihood_variance)
    model = GaussianProcessRegression(gpflow_model)
    return model


def get_model(model_type, data, search_space):
    if model_type == "deepgp":
        model = _get_deepgp_model(data, search_space)
    elif model_type == 'gp':
        model = _get_gp_model(data, search_space)
    else:
        raise ValueError("Model not found")
    return model


def train_and_save_lnl_surrogate(
        model_type,
        mcz_obs: np.ndarray,
        compas_h5_filename: str,
        params: List[str],
        acquisition_fns: List[AcquisitionFunction],
        n_init: int = 5,
        n_rounds: int = 5,
        n_pts_per_round: int = 10,
        outdir: str = 'outdir',
        model_plotter=None
):
    search_space = _get_search_space(params)
    observer = _generate_lnl_observer(mcz_obs, compas_h5_filename, params)
    x0 = search_space.sample(n_init)
    init_data = observer(x0)

    # Set up TF logging
    os.makedirs(outdir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(outdir)
    trieste.logging.set_tensorboard_writer(summary_writer)
    # visualise plots/training progress with tensorboard --logdir=outdir

    model = get_model(model_type, init_data, search_space)
    learning_rules = [EfficientGlobalOptimization(aq_fn) for aq_fn in acquisition_fns]
    bo = BayesianOptimizer(observer, search_space)
    data = init_data
    for round_idx in range(n_rounds):
        rule = learning_rules[round_idx % len(learning_rules)]
        result = bo.optimize(n_pts_per_round, data, model, rule, track_state=False)
        data = result.try_get_final_dataset()
        model = result.try_get_final_model()
        if model_plotter:
            model_plotter(model, data, search_space).savefig(f"{outdir}/round_{round_idx}.png")

    # TODO: save final model + dataset
    return result