import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lnl_surrogate.models import load_model

FUNCTIONS = dict(
    quadratic=lambda x: np.sum(x**2, axis=-1),
)
RESOLUTION = 50


def generate_true_data(func: Callable) -> np.ndarray:
    """Generates tabulated data for a 2D function
    RES*RES rows of (x1,x2)->y"""
    lin = np.linspace(-1, 1, RESOLUTION)
    x1, x2 = np.meshgrid(lin, lin)
    xx = np.vstack((x1.flatten(), x2.flatten())).T
    yy = func(xx).reshape(-1, 1)
    return xx, yy


def generate_data(func: Callable, n=50, add_unc=False) -> np.ndarray:
    xx, yy = generate_true_data(func)

    # sample n points from the true function
    idx = np.random.choice(np.arange(len(xx)), n)
    x_obs = xx[idx].T
    y_obs = yy[idx].T

    if add_unc:
        # uncertainty is inversely proportional to function
        dy = 0.01 + 1 / y_obs
        noise = np.random.normal(0, dy)
        y_obs += noise
        return np.array(
            [x_obs[0], x_obs[1], y_obs.ravel(), dy.ravel()]
        ).reshape((4, -1, 1))

    return np.array([x_obs[0], x_obs[1], y_obs.ravel()]).reshape((3, -1, 1))


@pytest.mark.parametrize(
    "model_type, n, func_name, add_unc",
    [
        ("gpflow", 20, "quadratic", False),
        ("gpflow", 20, "quadratic", True),
        ("sklearngp", 10, "quadratic", False),
        ("sklearngp", 10, "quadratic", True),
        ("sklearngp", 20, "quadratic", False),
        ("sklearngp", 20, "quadratic", True),
    ],
)
def test_model_for_2d_data(tmpdir, model_type, n, func_name, add_unc):
    np.random.seed(0)
    flabel = f"{model_type}_{func_name}_unc{add_unc}_n{n}"
    outdir = f"{tmpdir}/out_2d"
    os.makedirs(outdir, exist_ok=True)

    func = FUNCTIONS[func_name]
    data = generate_data(func, n=n, add_unc=add_unc)

    # build and train model
    model_class = load_model(model_type)
    model = model_class()
    kwgs = dict(unc=None, verbose=True, savedir=f"{outdir}/{flabel}")
    if add_unc:
        kwgs["unc"] = data[-1]
    train_in = np.vstack((data[0].flatten(), data[1].flatten())).T
    train_out = data[2]
    model.train(train_in, train_out, **kwgs)

    # make plots
    data_true = generate_true_data(func)
    x = data_true[0]
    y = data_true[1].reshape(-1, RESOLUTION)
    low_y, mean_y, up_y = model(x)
    mean_y = mean_y.reshape(-1, RESOLUTION)
    x = np.unique(x[:, 0])
    x1, x2 = np.meshgrid(x, x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    cbar_lvls = np.linspace(np.min(y), np.max(y), 20)
    ax.contourf(x1, x2, y, levels=cbar_lvls)
    ax.scatter(data[0].flatten(), data[1].flatten(), c="red", s=1)
    ax.set_title("True")
    ax = axes[1]
    ax.contourf(x1, x2, mean_y, levels=cbar_lvls)
    ax.scatter(data[0].flatten(), data[1].flatten(), c="red", s=1)
    ax.set_title("Surrogate")
    fig.suptitle(flabel)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{flabel}.png")
