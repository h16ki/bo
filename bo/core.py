import numpy as np
from typing import Union
from dataclasses import dataclass


@dataclass
class Scalar:
    var: Union[int, float]


class Distribution:
    def __call__(self, x, *args, **kwargs):
        return self.pdf(x, *args, **kwargs)

    def pdf(self, var, *args, **kwargs):
        raise NotImplementedError

    def sampling(self, N):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        self.args = args
        try:
            mean, cov = args
        except:
            mean = kwargs.pop("mean", None)
            cov = kwargs.pop("cov", None)

        self.mean = mean
        self.cov = cov


class Normal(Distribution):
    def pdf(self, var):
        z = var - self.mean
        if (dim:=np.size(var)) == 1:
            normalize = np.sqrt(2.0 * np.pi * self.cov) ** (-1)
            return normalize * np.exp(-0.5 * z ** 2 / self.cov)

        else:
            normalize = np.linalg.det(2.0 * np.pi * self.cov) ** (-0.5 * dim)
            return normalize * np.exp(-0.5 * z @ np.linalg.solve(self.cov, z))


