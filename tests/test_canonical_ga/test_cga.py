import unittest
import numpy as np
from framework.algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
from framework.utils import Function

class TestCGA(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        # Define a simple quadratic function
        def quadratic(x):
            return np.sum(x**2)

        # Define bounds
        self.bounds = [(-5, 5), (-5, 5)]

        # Initialize the Function class
        self.func = Function(quadratic, dimension=2, x_lower=[-5, -5], x_upper=[5, 5])

        # Define algorithm parameters
        self.population_size = 100
        self.max_nfe = 5000
        self.pc = 0.9
        self.pm = 0.05
        self.pc_initial = 0.9
        self.pm_initial = 0.05

    def test_cga(self):
        """
        Test the Canonical Genetic Algorithm (CGA).
        """
        # Initialize CGA
        cga = CGA(
            self.func,
            bounds=self.bounds,
            population_size=self.population_size,
            pc=self.pc,
            pm=self.pm,
            max_nfe=self.max_nfe,
        )

        # Run optimization
        best_solution, best_fitness = cga.optimize()

        # Verify results
        self.assertAlmostEqual(best_fitness, 0, delta=0.5)
        self.assertTrue(all(np.abs(best_solution) < 0.5))

    def test_cga_adaptive_v2(self):
        """
        Test the Adaptive Canonical Genetic Algorithm (CGAAdaptiveV2).
        """
        # Initialize CGAAdaptiveV2
        cga_adaptive = CGAAdaptiveV2(
            self.func,
            bounds=self.bounds,
            population_size=self.population_size,
            pc_initial=self.pc_initial,
            pm_initial=self.pm_initial,
            max_nfe=self.max_nfe,
        )

        # Run optimization
        best_solution, best_fitness = cga_adaptive.optimize()

        # Verify results
        self.assertAlmostEqual(best_fitness, 0, delta=0.5)
        self.assertTrue(all(np.abs(best_solution) < 0.5))

    def test_cga_greedy(self):
        """
        Test the Greedy Canonical Genetic Algorithm (CGAGreedy).
        """
        # Initialize CGAGreedy
        cga_greedy = CGAGreedy(
            self.func,
            bounds=self.bounds,
            population_size=self.population_size,
            pc=self.pc,
            pm=self.pm,
            max_nfe=self.max_nfe,
        )

        # Run optimization
        best_solution, best_fitness = cga_greedy.optimize()

        # Verify results
        self.assertAlmostEqual(best_fitness, 0, delta=0.5)
        self.assertTrue(all(np.abs(best_solution) < 0.5))

if __name__ == "__main__":
    unittest.main()