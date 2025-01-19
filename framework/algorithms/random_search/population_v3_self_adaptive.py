import numpy as np
from framework.utils import Function

class PopulationV3SelfAdaptive:
    def __init__(self, func: Function, bounds, population_size=10, max_iter=1000, alpha_initial=1.0, alpha_change_rate=0.005):
        """
        Initialize the PopulationV3SelfAdaptive algorithm.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of agents in the population.
        :param max_iter: Maximum number of iterations.
        :param alpha_initial: Initial step size.
        :param alpha_change_rate: Rate at which alpha changes based on fitness.
        """
        self.func = func
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha_initial = alpha_initial
        self.alpha_change_rate = alpha_change_rate
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
        Run the PopulationV3SelfAdaptive algorithm.

        :return: The best solution found and its fitness value.
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate the initial population
        fitness_values = np.array([self.func(agent) for agent in population])
        
        # Initialize best_solution and best_fitness
        min_fitness_idx = np.argmin(fitness_values)
        best_solution = population[min_fitness_idx]
        best_fitness = fitness_values[min_fitness_idx]
        
        # Track the best fitness at each iteration
        self.best_fitness_history.append(best_fitness)
        
        # Initialize alpha
        alpha = self.alpha_initial

        for _ in range(self.max_iter):
            new_population = []
            for agent in population:
                # Generate a random direction
                direction = np.random.uniform(-1, 1, size=len(self.bounds))
                direction /= np.linalg.norm(direction)  # Normalize to unit vector

                # Generate a new agent
                new_agent = agent + alpha * direction
                new_agent = np.clip(new_agent, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
                new_population.append(new_agent)

            # Evaluate new population
            new_population = np.array(new_population)
            fitness_values = np.array([self.func(agent) for agent in new_population])

            # Compute fitness improvement ratio
            current_best_fitness = np.min(fitness_values)
            improvement_ratio = (best_fitness - current_best_fitness) / (best_fitness + 1e-10)

            # Update alpha based on improvement ratio
            if improvement_ratio > 0:
                # If there is improvement, increase alpha slightly
                alpha *= (1 + self.alpha_change_rate)
            else:
                # If no improvement, decrease alpha more aggressively
                alpha *= (1 - self.alpha_change_rate)

            # Ensure alpha stays within reasonable bounds
            alpha = np.clip(alpha, 1e-5, 1.0)

            # Select the best agents
            best_indices = np.argsort(fitness_values)[:self.population_size]
            population = new_population[best_indices]

            # Update best solution
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < best_fitness:
                best_solution = new_population[min_fitness_idx]
                best_fitness = fitness_values[min_fitness_idx]

            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

        return best_solution, best_fitness