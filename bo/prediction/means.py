import numpy as np

class MeanBase:
  def __init__(self, hyp=None):
    self.hyp = hyp

  def __call__(self, x):
    return self.function(x)

  def function(self, x):
    raise NotImplementedError


class ZeroMean(MeanBase):
  def __init__(self):
    super(ZeroMean, self).__init__()

  def function(self, x):
    n = len(x)
    return np.zeros(n)


if __name__ == "__main__":
  # from nanod.predict.gp import means

  mu = ZeroMean()
  x = np.linspace(0, 1)
  print(mu(x))