import numpy as np
from .cga import CGA
from framework.utils import Function

class CGAGreedy(CGA):
    def __init__(self, func:Function, bounds, population_size=50, pc=0.8, pm=0.1, max_nfe=1000):
        """
        Initialize the CGAGreedy algorithm.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of individuals in the population.
        :param pc: Crossover probability.
        :param pm: Mutation probability.
        :param max_nfe: Maximum number of function evaluations (NFE).
        """
        super().__init__(func, bounds, population_size, pc, pm, max_nfe)

    def optimize(self):
        """
        Run the CGAGreedy algorithm.

        :return: The best solution found and its fitness value.
        """
        # Initialize population
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)

        best_solution = population[np.argmin(fitness_values)]
        best_fitness = np.min(fitness_values)

        while self.nfe < self.max_nfe:
            # Selection
            parents = self.selection(population, fitness_values)

            # Crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])

            # Evaluate new population
            new_population = np.array(new_population)
            new_fitness_values = self.evaluate_population(new_population)

            # Replace the worst child with the best parent
            worst_child_idx = np.argmax(new_fitness_values)
            best_parent_idx = np.argmin(fitness_values)
            new_population[worst_child_idx] = population[best_parent_idx]
            new_fitness_values[worst_child_idx] = fitness_values[best_parent_idx]

            # Update best solution
            min_fitness_idx = np.argmin(new_fitness_values)
            if new_fitness_values[min_fitness_idx] < best_fitness:
                best_solution = new_population[min_fitness_idx]
                best_fitness = new_fitness_values[min_fitness_idx]

            # Replace population
            population = new_population
            fitness_values = new_fitness_values

        return best_solution, best_fitness