import numpy as np
import pandas as pd

from .models.base_model import Model
from scipy.optimize import minimize
from .logger import logger

from typing import Callable, List

from .acquisition.ei import expected_improvement


def query_points(
        trained_model: Model,
        training_in,
        priors,
        acquisition_function: Callable = expected_improvement,
        acquisition_args: List = [],
        n_pts_to_query: int = 1,
        minimize_kwargs: dict = {},
) -> np.ndarray:
    if not trained_model.trained:
        raise ValueError("Model is not trained yet")

    logger.info(f"Querying new points using {acquisition_function.__name__} and model {trained_model}")

    # Get the current acquisition function values + the best
    current_acf = acquisition_function(
        training_in,
        trained_model,
        *acquisition_args,
    )

    pts = np.zeros((n_pts_to_query, priors.n_params))
    for i in range(n_pts_to_query):

        res = minimize(
            fun=acquisition_function,
            x0=priors.sample_val(),
            bounds=priors.bounds,
            method="L-BFGS-B",
            args=(trained_model, current_acf,  *acquisition_args),
            **minimize_kwargs,
        )

        pts[i] = res.x

    return pts
