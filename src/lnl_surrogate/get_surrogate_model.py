import os
import random
import shutil
from glob import glob
from typing import List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from .logger import logger
from .models import ModelType
from .models.base_model import Model


def get_surrogate_model(
    model_class: Union[str, ModelType],
    model_dir: str,
    training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    clean=False,
):
    """
    Get the ML surrogate model
    """
    model_class = ModelType.from_str(model_class)

    if clean and model_class.saved_model_exists(model_dir):
        shutil.rmtree(model_dir)

    if model_class.saved_model_exists(model_dir):
        logger.info(f"Loading model from {model_dir}")
        return model_class.load(model_dir)

    in_data = training_data.inputs
    out_data = training_data.outputs.reshape(-1, 1)
    logger.info(
        f"Training model {model_dir}: IN[{in_data.shape}]--> OUT[{out_data.shape}]"
    )
    model = model_class()
    plt_kwgs = dict(
        labels=training_data.labels,
        truths=training_data.truths,
    )

    metrics = model.train(
        in_data,
        out_data,
        unc=training_data.unc,
        verbose=True,
        savedir=model_dir,
        extra_kwgs=plt_kwgs,
    )
    logger.info(f"Surrogate metrics: {metrics}")

    # check if true lnl inside the range of pred_lnl
    # pred_lnl = np.array(model_class(training_data.true_vals)).ravel()
    # check = (pred_lnl[0] <= tru_lnl) & (tru_lnl <= pred_lnl[2])
    # diff = np.max(
    #     [np.abs(pred_lnl[1] - pred_lnl[0]), np.abs(pred_lnl[1] - pred_lnl[2])]
    # )
    # pred_str = f"{pred_lnl[1]:.2f} +/- {diff:.2f}"
    # check_str = "✔" * 3 if check else "❌" * 3
    # logger.info(
    #     f"{check_str} True LnL: {tru_lnl:.2f}, Surrogate LnL: {pred_str} {check_str}"
    # )
    # logger.success("Trained and saved Model")

    return model
