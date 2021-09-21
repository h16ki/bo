import numpy as np
import scipy as sp

class PredictionBase:
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



class GPRegression(PredictionBase):
  """
  Gaussian process regression

  Args:
    means: nanod.predict.gp.means,
      Mean function
    covariance: nanod.predict.gp.kernels: 
      Covariance function
    train: nanod.train 
      Data set of train inputs and outputs
  """

  def __init__(self, mean, covariance, train_inputs, train_outputs, **kwargs):
    """
    Args:
      means: nanod.predict.gp.means,
        Mean function
      covariance: nanod.predict.gp.kernels: 
        Covariance function
      train: nanod.train 
        Data set of train inputs and outputs
    """
    self.mean = mean
    self.covariance = covariance
    self.train_inputs = train_inputs
    self.train_outputs = train_outputs

    self.noise = kwargs.pop("noise", None)
    if self.noise:
      self.noiseCov = self.white_noise(len(train_inputs), self.noise)
    else:
      self.noiseCov = self.white_noise(len(train_inputs))

    self.mu = mean(train_inputs)
    self.Kt = lambda inputs: covariance(inputs, train_inputs)
    self.Ktt = covariance(train_inputs, train_inputs)
    # self.precision = sp.linalg.inv(self.Ktt + self.noiseCov)
    self.precision = inverse(self.Ktt + self.noiseCov)

  def pdf(self, inputs, *args, **kwargs):
    mean = self.mean(inputs) + self.Kt(inputs) @ self.precision @ (self.train_outputs - self.mu)
    # cov = self.covariance(inputs, inputs) - self.Kt(inputs) @ self.precision @ self.Kt(inputs).T
    # temp = sp.linalg.solve(self.Ktt, self.Kt(inputs).T)
    cov = self.covariance(inputs, inputs) + self.white_noise(len(inputs)) - self.Kt(inputs) @ self.precision @ self.Kt(inputs).T
    return mean, cov

  def white_noise(self, N, cov=0.01):
    return cov * np.eye(N)



def inverse(matrix):
  row, column = np.shape(matrix)
  print(row, column)
  uni = np.eye(N=row, M=column)
  L = np.linalg.cholesky(matrix)
  t = np.linalg.solve(L, uni)
  matinv = np.linalg.solve(L.T, t)
  return matinv

if __name__ == "__main__":
  print("--- DEBUG ---")
  # from nanod.regression import predictions
  import kernels, means
  x = test_x = np.pi * np.linspace(-1, 1, 100)
  y = np.sin(x)

  train_x = x[::20]
  train_y = np.sin(train_x)

  gamma = 1
  cov = kernels.RBF(gamma)
  mean = means.ZeroMean()

  gpr = GPRegression(mean, cov, train_x, train_y)
  pred_mean, pred_cov = gpr(test_x)
  print(max(pred_mean - y))
  print(min(pred_mean - y))

  # test_y = pred.mean
  # one_sigma = pred.variance(scale=1)
  # two_sigma = pred.variance(scale=2)
  # foo = pred.std(scale=1.0)



