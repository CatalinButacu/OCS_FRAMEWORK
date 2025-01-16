import numpy as np

class BenchmarkFunctions:
    @staticmethod
    def _handle_nan(x):
        """
        Replace NaN values in the input array with 0.
        """
        x = np.array(x)  # Ensure x is a numpy array
        x[np.isnan(x)] = 0  # Replace NaN values with 0
        return x

    @staticmethod
    def sphere(x):
        """
        Sphere function.
        Global minimum: f(0, 0, ..., 0) = 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        return np.sum(x * x)

    @staticmethod
    def rastrigin(x):
        """
        Rastrigin function.
        Global minimum: f(0, 0, ..., 0) = 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def rosenbrock(x):
        """
        Rosenbrock function.
        Global minimum: f(1, 1, ..., 1) = 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    @staticmethod
    def ackley(x):
        """
        Ackley function.
        Global minimum: f(0, 0, ..., 0) = 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / len(x)))
        sum_cos_term = -np.exp(np.sum(np.cos(c * x)) / len(x))
        return sum_sq_term + sum_cos_term + a + np.exp(1)

    @staticmethod
    def griewank(x):
        """
        Griewank function.
        Global minimum: f(0, 0, ..., 0) = 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - prod_cos

    @staticmethod
    def schwefel(x):
        """
        Schwefel function.
        Global minimum: f(420.9687, 420.9687, ..., 420.9687) ≈ 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    @staticmethod
    def zakharov(x):
        """
        Zakharov function.
        Global minimum: f(0, 0, ..., 0) = 0.
        """
        x = BenchmarkFunctions._handle_nan(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum1 + sum2**2 + sum2**4

    @staticmethod
    def michalewicz(x, m=10):
        """
        Michalewicz function.
        Global minimum: Depends on dimension and parameter m.
        """
        x = BenchmarkFunctions._handle_nan(x)
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin((i * x**2) / np.pi)**(2 * m))

    @staticmethod
    def styblinski_tang(x):
        """
        Styblinski-Tang function.
        Global minimum: f(-2.903534, -2.903534, ..., -2.903534) ≈ -39.16617 * n.
        """
        x = BenchmarkFunctions._handle_nan(x)
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

    @staticmethod
    def dixon_price(x):
        """
        Dixon-Price function.
        Global minimum: f(x) = 0 at x_i = 2^(-(2^i - 2)/(2^i)) for i = 1, 2, ..., n.
        """
        x = BenchmarkFunctions._handle_nan(x)
        i = np.arange(2, len(x) + 1)
        return (x[0] - 1)**2 + np.sum(i * (2 * x[1:]**2 - x[:-1])**2)