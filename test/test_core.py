import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from bo.core import Normal
import numpy as np


class TestNormal(unittest.TestCase):
    def test_normal(self):
        x = 0.0
        mean = 0.0
        cov = 1.0
        normal = Normal(mean, cov)
        xbar = normal(x)

        expected = (np.sqrt(2.0 * np.pi * cov)) ** (-1)
        actual = xbar
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
