import unittest
import numpy as np
from framework.algorithms.random_search import RandomSearch, PopulationV1Adaptive, PopulationV2, PopulationV3SelfAdaptive
from framework.utils import Function

class TestRandomSearch(unittest.TestCase):
    def test_random_search(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test RandomSearch
        rs = RandomSearch(func, bounds, max_iter=1000)
        best_solution, best_fitness = rs.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_population_v1_adaptive(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test PopulationV1Adaptive
        pv1 = PopulationV1Adaptive(func, bounds, population_size=10, max_iter=1000)
        best_solution, best_fitness = pv1.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_population_v2(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test PopulationV2
        pv2 = PopulationV2(func, bounds, population_size=10, max_iter=1000)
        best_solution, best_fitness = pv2.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_population_v3_self_adaptive(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test PopulationV3SelfAdaptive
        pv3 = PopulationV3SelfAdaptive(func, bounds, population_size=10, max_iter=1000)
        best_solution, best_fitness = pv3.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

if __name__ == "__main__":
    unittest.main()