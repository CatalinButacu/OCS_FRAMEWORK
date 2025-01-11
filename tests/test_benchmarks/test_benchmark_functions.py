import unittest
import numpy as np
from framework.benchmarks import BenchmarkFunctions

class TestBenchmarkFunctions(unittest.TestCase):
    def test_sphere(self):
        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.sphere(x), 0)

        x = np.array([1, 2, 3])
        self.assertAlmostEqual(BenchmarkFunctions.sphere(x), 14)

    def test_rastrigin(self):
        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.rastrigin(x), 0)

        x = np.array([1, 2, 3])
        self.assertAlmostEqual(BenchmarkFunctions.rastrigin(x), 14 + 20 * 3)

    def test_rosenbrock(self):
        x = np.array([1, 1, 1])
        self.assertAlmostEqual(BenchmarkFunctions.rosenbrock(x), 0)

        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.rosenbrock(x), 2)

    def test_ackley(self):
        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.ackley(x), 0)

        x = np.array([1, 2, 3])
        self.assertAlmostEqual(BenchmarkFunctions.ackley(x), 20 + np.exp(1) - 20 * np.exp(-0.2 * np.sqrt(14 / 3)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / 3))

    def test_griewank(self):
        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.griewank(x), 0)

        x = np.array([1, 2, 3])
        self.assertAlmostEqual(BenchmarkFunctions.griewank(x), 1 + (1 + 4 + 9) / 4000 - np.cos(1) * np.cos(2 / np.sqrt(2)) * np.cos(3 / np.sqrt(3)))

    def test_schwefel(self):
        x = np.array([420.9687, 420.9687, 420.9687])
        self.assertAlmostEqual(BenchmarkFunctions.schwefel(x), 0, delta=1e-4)

    def test_zakharov(self):
        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.zakharov(x), 0)

        x = np.array([1, 2, 3])
        self.assertAlmostEqual(BenchmarkFunctions.zakharov(x), 14 + (0.5 * 1 + 1 * 2 + 1.5 * 3)**2 + (0.5 * 1 + 1 * 2 + 1.5 * 3)**4)

    def test_michalewicz(self):
        x = np.array([0, 0, 0])
        self.assertAlmostEqual(BenchmarkFunctions.michalewicz(x), 0)

    def test_styblinski_tang(self):
        x = np.array([-2.903534, -2.903534, -2.903534])
        self.assertAlmostEqual(BenchmarkFunctions.styblinski_tang(x), -39.16617 * 3, delta=1e-4)

    def test_dixon_price(self):
        x = np.array([1, 1 / np.sqrt(2), 1 / np.sqrt(4)])
        self.assertAlmostEqual(BenchmarkFunctions.dixon_price(x), 0)

if __name__ == "__main__":
    unittest.main()