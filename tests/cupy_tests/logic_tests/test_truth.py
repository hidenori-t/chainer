import unittest

from cupy import testing


@testing.gpu
class TestTruth(unittest.TestCase):

    _multiprocess_can_split_ = True
