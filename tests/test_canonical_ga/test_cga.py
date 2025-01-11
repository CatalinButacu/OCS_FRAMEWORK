import unittest
import numpy as np
from framework.algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
from framework.utils import Function

class TestCGA(unittest.TestCase):
    def test_cga(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test CGA
        cga = CGA(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = cga.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_cga_adaptive_v2(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test CGAAdaptiveV2
        cga_adaptive = CGAAdaptiveV2(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = cga_adaptive.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_cga_greedy(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test CGAGreedy
        cga_greedy = CGAGreedy(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = cga_greedy.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

if __name__ == "__main__":
    unittest.main()