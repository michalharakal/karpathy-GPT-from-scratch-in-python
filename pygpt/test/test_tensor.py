import unittest

from pygpt.tensor.tensor import Tensor2D


class MyTestCase(unittest.TestCase):
    def test_dot_product(self):
        tensor_a = Tensor2D([[1, 2, 3], [4, 5, 6]])
        tensor_b = Tensor2D([[7, 8], [9, 10], [11, 12]])
        result = tensor_a.dot_product(tensor_b)
        assert result.data == [[58, 64], [139, 154]]
