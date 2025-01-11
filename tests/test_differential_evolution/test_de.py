import unittest
import numpy as np
from framework.algorithms.differential_evolution import DERand2Bin, DEBest1Exp, DEBest2Exp
from framework.utils import Function

class TestDE(unittest.TestCase):
    def test_de_rand_2_bin(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test DERand2Bin
        de_rand = DERand2Bin(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = de_rand.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_de_best_1_exp(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test DEBest1Exp
        de_best1 = DEBest1Exp(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = de_best1.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

    def test_de_best_2_exp(self):
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Test DEBest2Exp
        de_best2 = DEBest2Exp(func, bounds, population_size=50, max_nfe=1000)
        best_solution, best_fitness = de_best2.optimize()
        self.assertAlmostEqual(best_fitness, 0, delta=0.1)
        self.assertTrue(all(np.abs(best_solution) < 0.1))

if __name__ == "__main__":
    unittest.main()