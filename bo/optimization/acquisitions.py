import optim
import numpy as np
from scipy.optimize import minimize


class Acquisition:
    """Base class of Acquisition funtion"""
    def __init__(self, gp):
        self.gp = gp

    def __call__(self):
        pass

    def optimize(self, domain, method="BFGS", split=100):
        """
        """
        domain = np.asarray(domain, dtype=np.float64)

    def acquisition(self, x):
        raise NotImplementedError


class UCB(Acquisition):
    """Upper Confidential Bound"""
    def acquisition(self, x, kappa=1.0):
        mean, covariance = self.gp(x)
        std = np.sqrt(covariance.diag)
        return mean + kappa * std


class EI(Acquisition):
    """Expectation Improvement"""
    ...


class PI(Acquisition):
    """Probability Improvement"""
    ...


if __name__ == "__main__":
    from bo.prediction import gaussian_process as gp
    from bo.prediction.gaussian_process import GaussianProcessRegression
    from bo.prediction import kernels, means

    kern = kernels.RBF(gamma=1.0)
    mean = means.ZeroMean()
    train_x = [[1, 1], [2, 2], [3, 3], [4, 4]]
    train_y = [1, 2, 3, 4]

    gp = GaussianProcessRegression(mean, kern, train_x, train_y)
    acquisition = UCB(gp)
    candidate = acquisition.optimize(method="BFGS")

