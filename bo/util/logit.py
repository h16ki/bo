import numpy as np

def logit(x):
    """
    Logit function
    Map values from (0, 1) to (-oo, +oo)

    Args
    ----
    x: A real number, (0, 1)

    Returns
    -------
    A real number, (-oo, +oo)
    """
    return np.log(x) - np.log(1 - x)

def ilogit(x):
    """
    Inverse of logit function
    a.k.a logistic function, sigmoid function

    Args
    ----
    x: A real number, (-oo, +oo)

    Returns
    -------
    A real number, (0, 1)
    """
    return 1 / ( 1 + np.exp(-x) )