r"""
    Probability of Improvement

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Probability of Improvement at an input point :math:`x` is defined as:

    .. math::

        PI(x) = \Phi\left(\frac{\mu(x) - f^+ - \xi}{\sigma(x)}\right)

    where
        :math:`\mu(x)` is the predictive mean,
        :math:`\sigma(x)` is the predictive standard deviation,
        :math:`f^+` is the value of the best observed sample,
        :math:`\xi` is a small positive "jitter"/tradeoff term to encourage more exploration,
        :math:`\Phi` is the cumulative distribution function (CDF) of the standard normal distribution.
    """

from scipy.special import (
    ndtr as norm_cdf,  # Cumulative distribution of the standard normal distribution.
)
from scipy.stats import norm


def PI(mean: float, std: float, max_val: float, tradeoff: float) -> float:
    return norm_cdf((mean - max_val - tradeoff) / std)
