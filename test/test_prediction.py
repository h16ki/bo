import os
import sys
import traceback
import unittest

# 以下1lineを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bo.prediction import *


if __name__ == '__main__':
    unittest.main()
