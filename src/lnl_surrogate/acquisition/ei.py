r"""
    Expected Improvement

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Expected Improvement at an input point :math:`x` is defined as:

    .. math::
        EI(x) =
        \begin{cases}
        (\mu(x) - f^+) \Phi(Z) + \sigma(x) \phi(Z) & \text{if } \sigma(x) > 0 \\
        0 & \text{if } \sigma(x) = 0
        \end{cases}

    where
        :math:`\mu(x)` is the predictive mean,
        :math:`\sigma(x)` is the predictive standard deviation,
        :math:`f^+` is the value of the best observed sample.
        :math:`\Phi` is the cumulative distribution function (CDF) of the standard normal distribution.
        :math:`\xi` is a small positive "jitter"/tradeoff term to encourage more exploration,
        :math:`Z` is defined as:

    .. math::

        Z = \frac{\mu(x) - f^+}{\sigma(x)}

    provided :math:`\sigma(x) > 0`.

"""
import numpy as np
from scipy.special import (
    ndtr as norm_cdf,  # Cumulative distribution of the standard normal distribution.
)
from scipy.stats import norm


def EI(mean: float, std: float, max_val: float, tradeoff: float) -> float:
    z = (mean - max_val - tradeoff) / std
    return (z * std) * norm_cdf(z) + std * norm.pdf(z)


def expected_improvement(
    x, model, evaluated_loss=[], greater_is_better=True, n_params=1
):
    """expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = model._model.predict(x_to_predict, return_std=True)

    if len(evaluated_loss) > 0:
        if greater_is_better:
            loss_optimum = np.max(evaluated_loss)
        else:
            loss_optimum = np.min(evaluated_loss)
    else:
        loss_optimum = 0

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide="ignore"):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(
            Z
        ) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement
