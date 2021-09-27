import numpy as np
from typing import Sequence


class Mean:
    def __init__(self, hyp=None):
        self.hyp = hyp

    def __call__(self, shape, *args):
        return self.impl(shape, *args)

    def impl(self):
        raise NotImplementedError


class ZeroMean(Mean):
    def impl(self, shape, *args):
        return np.zeros(shape)


if __name__ == "__main__":
    # from nanod.predict.gp import means

    mu = ZeroMean()
    x = np.linspace(0, 1)
    print(mu(x))
