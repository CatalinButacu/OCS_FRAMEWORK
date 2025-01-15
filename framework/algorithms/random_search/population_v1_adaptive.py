import numpy as np
from framework.utils import Function

class PopulationV1Adaptive:
    def __init__(self, func:Function, bounds, population_size=10, max_iter=1000, alpha_initial=1.0, alpha_decay=0.99):
        """
        Initialize the PopulationV1Adaptive algorithm.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of agents in the population.
        :param max_iter: Maximum number of iterations.
        :param alpha_initial: Initial step size.
        :param alpha_decay: Decay rate for the step size.
        """

        self.func = func
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha_initial = alpha_initial
        self.alpha_decay = alpha_decay
        self.best_fitness_history = []

    def initialize_population(self):
        """
        Initialize the population with random individuals.

        :return: A numpy array of shape (population_size, dimension).
        """
        population = []
        for _ in range(self.population_size):
            individual = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            population.append(individual)
        return np.array(population)
    
    def optimize(self):
        """
        Run the PopulationV1Adaptive algorithm.

        :return: The best solution found and its fitness value.
        """        
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        self.best_fitness_history.append(best_fitness)

        for iteration in range(self.max_iter):
            # Update step size (alpha)
            alpha = self.alpha_initial * (self.alpha_decay ** iteration)

            # Generate new solutions
            new_population = []
            for agent in population:
                # Generate a random direction
                direction = np.random.uniform(-1, 1, size=len(self.bounds))
                direction /= np.linalg.norm(direction)  # Normalize to unit vector

                # Generate a new solution
                new_agent = agent + alpha * direction
                new_agent = np.clip(new_agent, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
                new_population.append(new_agent)

            # Evaluate new population
            new_population = np.array(new_population)
            fitness_values = np.array([self.func(agent) for agent in new_population])

            # Update best solution
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < best_fitness:
                best_solution = new_population[min_fitness_idx]
                best_fitness = fitness_values[min_fitness_idx]

            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

            # Replace population
            population = new_population

        return best_solution, best_fitness
    