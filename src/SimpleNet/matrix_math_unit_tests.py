import unittest
from matrix_math import matrix, map, transpose, from_array, subtract, multiply
import numpy as np


class TestMatrixMath(unittest.TestCase):
    def setUp(self):
        # Create sample matrices for testing
        self.mat_a = matrix(2, 2)
        self.mat_a.values = [[1, 2], [3, 4]]

        self.mat_b = matrix(2, 2)
        self.mat_b.values = [[5, 6], [7, 8]]

    def test_multiply_scalar(self):
        # Test scalar multiplication
        self.mat_a.multiply(2)
        self.assertEqual(self.mat_a.values, [[2, 4], [6, 8]])

    def test_element_wise_multiply(self):
        # Create two matrices for testing
        mat_a = matrix(2, 2)
        mat_a.values = [[1, 2], [3, 4]]

        mat_b = matrix(2, 2)
        mat_b.values = [[5, 6], [7, 8]]

        # Perform element-wise multiplication
        result = element_wise_multiply(mat_a, mat_b)

        # Define the expected result
        expected_values = np.array([[5, 12], [21, 32]])

        # Assert that the result matches the expected values
        self.assertTrue(np.array_equal(result.values, expected_values))

    def test_multiply_matrix(self):
        # Test matrix multiplication
        result = multiply(self.mat_a, self.mat_b)
        expected_values = np.array([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result.values, expected_values))

    def test_transpose(self):
        # Test matrix transpose
        result = transpose(self.mat_a)
        expected_values = np.array([[1, 3], [2, 4]])
        self.assertTrue(np.array_equal(result.values, expected_values))

    def test_from_array(self):
        # Test conversion from array to matrix
        arr = [1, 2, 3, 4]
        result = from_array(arr)
        expected_values = np.array([[1], [2], [3], [4]])
        self.assertTrue(np.array_equal(result.values, expected_values))

    def test_subtract(self):
        # Test matrix subtraction
        result = subtract(self.mat_a, self.mat_b)
        expected_values = np.array([[-4, -4], [-4, -4]])
        self.assertTrue(np.array_equal(result.values, expected_values))

    def test_map(self):
        # Test element-wise mapping
        result = map(self.mat_a, lambda x: x * 2)
        expected_values = np.array([[2, 4], [6, 8]])
        self.assertTrue(np.array_equal(result.values, expected_values))


if __name__ == "__main__":
    unittest.main()
