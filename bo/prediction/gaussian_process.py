from typing import DefaultDict
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

    def loglikelihood(self):
        print(f"{self.Ktt}")
        pass


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
        self.mean = mean
        self.covariance = covariance
        self.train_inputs = np.array(train_inputs, dtype=np.float64)
        self.train_outputs = np.array(train_outputs, dtype=np.float64)

        noise = kwargs.pop("noise", None)
        if noise:
            self.noise = self.get_noise(len(train_inputs), noise)
        else:
            # add a small noise (std=0.1) to avoid a negative eigenvalue
            self.noise = self.get_noise(len(train_inputs))

        output_shape = np.shape(train_outputs)
        self.mu = mean(shape=output_shape)
        # print("debug:", self.mu)
        self.Kt = lambda inputs: covariance(inputs, train_inputs)
        self.Ktt = covariance(train_inputs, train_inputs)
        # print("debug:", self.Ktt)

    def pdf(self, inputs):
        output_shape = len(inputs)
        prior_mean = self.mean(shape=output_shape)
        prior_covariance = self.covariance(inputs, inputs) + self.get_noise(len(inputs))

        offdiag_covariance = self.Kt(inputs)
        z = self.train_outputs - self.mu
        train_covariance = self.Ktt + self.noise
        conditional_mean = prior_mean + offdiag_covariance @ np.linalg.solve(
            train_covariance, z
        )
        # cov = self.covariance(inputs, inputs) - self.Kt(inputs) @ self.precision @ self.Kt(inputs).T
        # temp = sp.linalg.solve(self.Ktt, self.Kt(inputs).T)
        # cov = self.covariance(inputs, inputs) + self.white_noise(len(inputs)) - self.Kt(inputs) @ self.precision @ self.Kt(inputs).T
        
        conditional_covariance = prior_covariance - offdiag_covariance @ np.linalg.solve(
            train_covariance, offdiag_covariance.T
        )


        # conditional_mean = offdiag_covariance
        return conditional_mean, conditional_covariance

    def get_noise(self, N, cov=0.0):
        return cov * np.eye(N)

    def __enter__(self):
        print(f"Automatic Relevance Determination.")
        print(f"Initial parameter guess: {self.covariance.args}")
        # dLdp = self.loglikelihood().param_grad()
        x = np.asarray(self.train_inputs)
        
        step = 1_000
        tol = 1e-6
        p = self.covariance.args
        dp = 1
        r = 0.1
        for _ in range(step):
            dKdp = - np.outer(x, x) @ self.Ktt
            dLdp = -np.trace(np.linalg.solve(self.Ktt, dKdp)) + \
                (np.linalg.solve(self.Ktt, self.train_outputs).T @ dKdp @ np.linalg.solve(self.Ktt, self.train_outputs))

            if dp < tol:
                break
            else:
                dp = p
                p = p - r * dLdp    # GD method
                dp = abs(dp - p)

            self.covariance.args = p

        print(f"{self.covariance.args}")




    def __exit__(self, exc_type, exc_value, traceback):
        # print(f'exit: {exc_type}, {exc_value}, {traceback}')
        return self



if __name__ == "__main__":
    print("\n-----------------------------")
    print("> DEBUG Information")
    print("> Gaussian process regression")
    print("-----------------------------\n")
    import kernels, means



    x = test_x = np.pi * np.linspace(-1, 1, 100)
    y = np.sin(x)

    # 1d chech
    print("Gaussian process regression for single-valiable black box function.")
    print("Data space")
    print("----------------------------")

    # train_x = [-1, 0, 1]
    np.random.seed(1)
    train_x = np.random.uniform(-3, 3, 5)
    train_y = np.sin(train_x)
    output_shape = np.shape(train_y)

    cov = kernels.RBF(args=1.0)
    mean = means.ZeroMean()

    print("input:", train_x)
    print("output:", train_y)
    print("----------------------------")

    with GaussianProcessRegression(mean, cov, train_x, train_y) as gp:
        print("")
        pmean, pcov = gp([0.0])
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
    print("\n2d inputs case")
    train_x = [[-1, -1], [0, 0], [1, 1]]
    train_y = list(map(lambda x: np.sin(x[0]) * np.sin(x[1]), train_x))
    output_shape = np.shape(train_y)

    cov = kernels.RBF(gamma=1.0)
    mean = means.ZeroMean()

    # prediction at a point (0.0, 0.0)
    gp = GaussianProcessRegression(mean, cov, train_x, train_y)
    points = ((0.0, 0.0),)
    pmean, pcov = gp(points)
    print("conditional mean:", pmean)
    print("conditional covariance:", pcov)

