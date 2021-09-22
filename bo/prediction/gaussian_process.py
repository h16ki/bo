import numpy as np
import scipy as sp


class Regressor:
    """
  Base class to prediction probability.
  """

    def __init__(self, train_inputs, train_outputs):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.__means = None

    def __call__(self, inputs, *args, **kwargs):
        return self.pdf(inputs)

    def pdf(self, inputs, *args, **kwargs):
        raise NotImplementedError


class GaussianProcessRegression(Regressor):
    """Gaussian process regression"""

    def __init__(self, mean, covariance, train_inputs, train_outputs, **kwargs):
        """
    Args:
      means: bo.predict.means,
        Mean function
      covariance: bo.predict.kernels: 
        Covariance function
      train_inputs: numpy.arraylike
        Data set of train inputs
      train_outputs: numpy.arraylike
        Data set of train inputs and outputs
    """
        self.meanfunc = mean
        self.covariance = covariance
        self.train_inputs = np.array(train_inputs, dtype=np.float64)
        self.train_outputs = np.array(train_outputs, dtype=np.float64)

        noise = kwargs.pop("noise", None)
        if noise:
            self.noise = self.get_noise(len(train_inputs), noise)
        else:
            # add a small noise (std=0.1) to avoid a negative eigenvalue
            self.noise = self.get_noise(len(train_inputs))

        self.mean = mean(train_inputs)
        print("debug:", mean)
        self.Kt = lambda inputs: covariance(inputs, train_inputs)
        self.Ktt = covariance(train_inputs, train_inputs)
        print("debug:", self.Ktt)

    def pdf(self, inputs):
        bare_mean = self.meanfunc(inputs)
        bare_covariance = self.covariance(inputs, inputs) + self.get_noise(len(inputs))
        offdiag_covariance = self.Kt(inputs)
        z = self.train_outputs - self.mean
        train_covariance = self.Ktt + self.noise
        # mean = self.meanfunc(inputs) + self.Kt(inputs) @ self.precision @ (self.train_outputs - self.mu)
        conditional_mean = bare_mean + offdiag_covariance @ np.linalg.solve(
            train_covariance, z
        )
        # cov = self.covariance(inputs, inputs) - self.Kt(inputs) @ self.precision @ self.Kt(inputs).T
        # temp = sp.linalg.solve(self.Ktt, self.Kt(inputs).T)
        # cov = self.covariance(inputs, inputs) + self.white_noise(len(inputs)) - self.Kt(inputs) @ self.precision @ self.Kt(inputs).T
        
        conditional_covariance = bare_covariance - offdiag_covariance @ np.linalg.solve(
            train_covariance, offdiag_covariance.T
        )


        # conditional_mean = offdiag_covariance
        return conditional_mean, conditional_covariance

    def get_noise(self, N, cov=0.01):
        return cov * np.eye(N)



if __name__ == "__main__":
    print("\n-----------------------------")
    print("> DEBUG Information")
    print("> Gaussian process regression")
    print("-----------------------------\n")
    import kernels, means


    cov = kernels.RBF(gamma=1.0)
    mean = means.ZeroMean()

    x = test_x = np.pi * np.linspace(-1, 1, 100)
    y = np.sin(x)

    # 1d chech
    print("Gaussian process regression for single-valiable black box function.")
    print("Data space")
    print("----------------------------")

    train_x = [-1, 0, 1]
    train_y = np.sin(train_x)

    print("input:", train_x)
    print("output:", train_y)
    print("----------------------------")

    gpr = GaussianProcessRegression(mean, cov, train_x, train_y)
    print("")
    pmean, pcov = gpr([0.0])
    print("conditional mean:", pmean)
    print("conditional covariance:", pcov)
    # pred_mean, pred_cov = gpr(test_x)
    # print(max(pred_mean - y))
    # print(min(pred_mean - y))

    # test_y = pred.mean
    # one_sigma = pred.variance(scale=1)
    # two_sigma = pred.variance(scale=2)
    # foo = pred.std(scale=1.0)



    # 2d check
    print("Gaussian process regression for single-valiable black box function.")
    train_x = [[-1, -1], [0, 0], [1, 1]]
    train_y = (lambda x: np.sin(x[0]) * np.sin(x[1]))(train_x)

    gpr = GaussianProcessRegression(mean, cov, train_x, train_y)
    pmean, pcov = gpr([[0.0, 0.0], [1, 1]])
    print("conditional mean:", pmean.shape)
    print("conditional covariance:", pcov.shape)

    foo = [0, 0]
    bar = kernels.RBF(1.0)(train_x, train_x)
    print(bar)
