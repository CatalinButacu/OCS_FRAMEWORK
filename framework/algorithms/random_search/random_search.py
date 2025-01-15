import numpy as np
from framework.utils import Function

class RandomSearch:
    def __init__(self, func:Function, bounds, max_iter=1000):
        """
        Initialize the Random Search algorithm.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param max_iter: Maximum number of iterations.
        """
        self.func = func
        self.bounds = bounds
        self.max_iter = max_iter
        self.best_fitness_history = []

    def optimize(self):
        """
        Run the Random Search algorithm.

        :return: The best solution found and its fitness value.
        """
        best_solution = None
        best_fitness = float('inf')
        self.best_fitness_history.append(best_fitness)

        for _ in range(self.max_iter):
            # Generate a random solution within bounds
            solution = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            fitness = self.func(solution)

            # Update the best solution
            if fitness < best_fitness:
                best_solution = solution
                best_fitness = fitness
            
            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

        return best_solution, best_fitness