import os, sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bo.prediction import kernels

foo = kernels.RBF()
if __name__ == '__main__':
    unittest.main()
