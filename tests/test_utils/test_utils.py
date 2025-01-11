import unittest
import numpy as np
from framework.utils import Function, quadratic_penalty

class TestUtils(unittest.TestCase):
    def test_function_evaluation(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds and constraints
        dimension = 2
        x_lower = [-5, -5]
        x_upper = [5, 5]
        constraints = [lambda x: x[0] + x[1] - 1]  # x[0] + x[1] <= 1

        # Initialize the Function class
        func = Function(quadratic, dimension, x_lower, x_upper, constraints)

        # Test evaluation
        x = np.array([1, 2])
        objective_value, constraint_violations = func.evaluate(x)
        self.assertAlmostEqual(objective_value, 5)
        self.assertAlmostEqual(constraint_violations[0], 2)

        # Test feasibility
        self.assertFalse(func.is_feasible(x))
        self.assertTrue(func.is_feasible(np.array([0, 0])))

    def test_quadratic_penalty(self):
        constraint_values = [1, -2, 3]
        penalty = quadratic_penalty(constraint_values)
        self.assertAlmostEqual(penalty, 10)

if __name__ == "__main__":
    unittest.main()