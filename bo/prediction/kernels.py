import numpy as np
from numpy import linalg as LA
from scipy.special import gamma, kv
# from typing import ClassVar, Sequence
from numbers import Number



class Kernel:
  def __init__(self, args=None):
    self.args = args

  def __call__(self, x, y):
    x, y = list(map(self.toNpArray, [x, y]))

    row, column = len(x), len(y)
    K = np.empty([row, column])
    for i in range(row):
        for j in range(column):
            K[i][j] = self.kernel(x[i], y[j])

    return K

  def toNpArray(self, value):
    if (s:= np.shape(value)) == ():
        return np.array([value])
    else:
        return np.array(value).reshape(s)

  def kernel(self, x, y, *args, **kwargs):
    return NotImplementedError

  def __add__(self, other):
    return lambda x, y: self(x, y) + other(x, y)

  def __radd__(self, other):
    return lambda x, y: self(x, y) + other(x, y)


class Scale(Kernel):
  def __init__(self, scale: Number, k: Kernel):
    super(Scale, self).__init__(scale)
    self.rescaled_kernel = k

  def kernel(self, x, y, scale):
    return scale * self.rescaled_kernel(x, y)

  def grad(self, x, y):
    return self.rescaled_kernel(x, y)




class RBF(Kernel):
  """RBF kernel."""
  def kernel(self, x, y):
    z = np.asarray(x) - np.asarray(y)
    return np.exp(-self.args * np.dot(z, z))


class Matern(Kernel):
  def __init__(self, nu, hyp):
      super(Matern, self).__init__(hyp)
      self.nu = nu

      if nu == 1/2:
        self.__matern = self.matern1
      elif nu == 3/2:
        self.__matern = self.matern3
      elif nu == 5/2:
        self.__matern = self.matern5
      elif nu == np.inf:
        self.__matern = self.materninf
      else:
        self.__matern = self.matern

  
  def kernel(self, x, y, theta):
    z = np.asarray(x) - np.asarray(y)
    r = LA.norm(z)
    return self.__matern(r, theta)

  def matern(self, r, theta):
    z = np.sqrt(2 * self.nu) * r / theta
    coeff = 2 ** (1-self.nu) / gamma(self.nu)
    return coeff * z**self.nu * kv(self.nu, z)

  def matern1(self, r, theta):
    return np.exp(-r / theta)

  def matern3(self, r, theta):
    return (1 + np.sqrt(3) / theta * r) * np.exp(-np.sqrt(3) / theta * r)

  def matern5(self, r, theta):
    return (1 + np.sqrt(5) / theta * r + 5 / 3 / theta**2 * r**2) * np.exp(-np.sqrt(5) / theta * r)

  def materninf(self, r, theta):
    return np.exp(-0.5 / theta**2 * r**2)

class RationalQuad(Kernel):
  ...

class DotProduct(Kernel):
  def kernel(self, x, y, *args, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.dot(x, y)

class ARD(Kernel):
  def kernel(self, x, y):
    rbf = RBF(1.0)
    



if __name__ == "__main__":
  a = [[1.0, 2.0], [2.1, 4.2], [2.5, 3.0]]
  b = [[4.12, 3.1], [32, 3.3], [-4, 10]]
  dp = DotProduct()
  print(dp(a, b))

  rbf = RBF(1.0)
  print(rbf(a, b))