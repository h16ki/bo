import numpy as np
from typing import Sequence
np.seterr(divide="raise")


class BFGS:
    def __init__(self, xi=0.3, tau=0.9, rec=False):
        self.xi = xi        # Armijo condition
        self.tau = tau      # step-wise
        self.path = None
        self.status = False # is convergent

    def solve(self, fun, *, x0, jac, tol=1e-9, iter_max=1000):
        if isinstance(x0, Sequence):
            dim = len(x0)
            x0 = np.array(x0, dtype=np.float64)
        else:
            dim = 1

        path = [x0]
        I = np.eye(dim)
        hessi = np.eye(dim)     # hess ** (-1)

        k = 0
        # sTy = 1
        for _ in range(iter_max):
            grad = jac(x0)

            # if np.linalg.norm(grad) < tol or np.linalg.norm(s) < tol:
            if np.linalg.norm(grad) < tol:
                self.status = "OK"
                break
            elif k == iter_max - 1:
                break
            else:
                alpha = 1.0
                k += 1

            d = -np.dot(hessi, grad)
            while fun(x0 + alpha * d) > (fun(x0) + self.xi * alpha * np.dot(grad, d)):
                alpha = self.tau * alpha

            s = alpha * d
            y = jac(x0 + s) - jac(x0)
            x0 = x0 + s

            # BFGS: update hessian inverse
            sTy = np.dot(s, y)
            syT = np.outer(s, y)
            ssT = np.outer(s, s)
            try:
                hessi = (I - syT / sTy) @ hessi @ (I - syT / sTy).T + ssT / sTy
            except:
                print(r"The iteration stop.")
                print(f"The diff of jacobian is: {y}")
                print(f"The diff of iteration: {s}")
                break
            
            path.append(x0)

        path = np.asarray(path, dtype=np.float64)
        self.path = path

        return x0