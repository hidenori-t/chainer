import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestSum(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        self.gy = numpy.array(2, dtype=numpy.float32)

    def check_forward(self, x_data, axis=None):
        x = chainer.Variable(x_data)
        y = functions.sum(x, axis=axis)
        self.assertEqual(y.data.dtype, numpy.float32)
        y_expect = self.x.sum(axis=axis)
        gradient_check.assert_allclose(y_expect, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

        for i in range(self.x.ndim):
            self.check_forward(self.x, axis=i)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

        for i in range(self.x.ndim):
            self.check_forward(cuda.to_gpu(self.x), axis=i)

    def check_backward(self, x_data, y_grad, axis=None):
        x = chainer.Variable(x_data)
        y = functions.sum(x, axis=axis)

        y.grad = y_grad
        y.backward()

        gx_expect = numpy.full_like(self.x, self.gy)
        gradient_check.assert_allclose(gx_expect, x.grad)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(self.x, gy, axis=i)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

        for i in range(self.x.ndim):
            gy = numpy.ones_like(self.x.sum(axis=i)) * self.gy
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(gy), axis=i)


testing.run_module(__name__, __file__)
