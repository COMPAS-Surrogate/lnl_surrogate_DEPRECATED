from enum import Enum, auto

from .ei import EI
from .pi import PI
from .ucb import UCB


class AcquisitionType(Enum):
    EI = auto()
    PI = auto()
    UCB = auto()

    # get the acquisition function given the type
    def get_acquisition_function(self):
        if self == AcquisitionType.EI:
            return EI
        elif self == AcquisitionType.PI:
            return PI
        elif self == AcquisitionType.UCB:
            return UCB
        else:
            raise ValueError("AcquisitionType not recognized")
