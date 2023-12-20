r"""
    Upper confidence bound

    Given a probabilistic model :math:`m` that models the objective function :math:`f`,
    the Upper Confidence Bound at an input point :math:`x` is defined as:

    .. math::

        UCB(x) = \mu(x) + \beta \sigma(x)

    where
        :math:`\mu(x)` is the predictive mean,
        :math:`\sigma(x)` is the predictive standard deviation,
        :math:`\beta` is the exploration-exploitation trade-off parameter.
"""

def UCB(mean:float, std:float, beta:float)->float:
    return mean + beta*std
