import unittest
import numpy as np
from framework.algorithms.real_ga import RGA1Adaptive, RGA4, RGA4AdaptiveV2
from framework.utils import Function

class TestRGA(unittest.TestCase):
    def test_rga_1_adaptive(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test RGA1Adaptive
        rga1 = RGA1Adaptive(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = rga1.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_rga_4(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test RGA4
        rga4 = RGA4(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = rga4.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_rga_4_adaptive_v2(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test RGA4AdaptiveV2
        rga4_adaptive = RGA4AdaptiveV2(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = rga4_adaptive.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

if __name__ == "__main__":
    unittest.main()