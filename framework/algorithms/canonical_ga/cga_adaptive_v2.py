import numpy as np
from .cga import CGA
from framework.utils import Function
from framework.utils.binary_manipulation import binarysolution2float

class CGAAdaptiveV2(CGA):
    def __init__(self, func: Function, bounds, population_size=50, pc_initial=0.8, pm_initial=0.1, max_nfe=1000, bits_per_gene=64):
        """
        Initialize the CGAAdaptiveV2 algorithm with binary representation.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of individuals in the population.
        :param pc_initial: Initial crossover probability.
        :param pm_initial: Initial mutation probability.
        :param max_nfe: Maximum number of function evaluations (NFE).
        :param bits_per_gene: Number of bits used to represent each decision variable.
        """
        super().__init__(func, bounds, population_size, pc_initial, pm_initial, max_nfe, bits_per_gene)
        self.pc_initial = pc_initial
        self.pm_initial = pm_initial

    def adapt_parameters(self, fitness_values):
        """
        Adapt pc and pm based on fitness values.

        :param fitness_values: A numpy array of fitness values.
        """
        avg_fitness = np.mean(fitness_values)
        best_fitness = np.min(fitness_values)

        # Adapt pc and pm
        self.pc = self.pc_initial * (1 - (avg_fitness - best_fitness) / avg_fitness)
        self.pm = self.pm_initial * (1 + (avg_fitness - best_fitness) / avg_fitness)

    def optimize(self):
        """
        Run the CGAAdaptiveV2 algorithm with binary representation.

        :return: The best solution found and its fitness value.
        """
        # Initialize population
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)

        # Track the best solution
        best_index = np.argmin(fitness_values)
        best_solution = binarysolution2float(population[best_index], self.bounds)  # Pass bounds here
        best_fitness = fitness_values[best_index]
        self.best_fitness_history.append(best_fitness)

        for iter in range(self.max_nfe):
            # Adapt pc and pm
            self.adapt_parameters(fitness_values)

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
            new_fitness_values = self.evaluate_population(new_population)

            # Update best solution
            min_fitness_idx = np.argmin(new_fitness_values)
            if new_fitness_values[min_fitness_idx] < best_fitness:
                best_solution = binarysolution2float(new_population[min_fitness_idx], self.bounds)
                best_fitness = new_fitness_values[min_fitness_idx]

            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

            # Replace population
            population = new_population
            fitness_values = new_fitness_values

        return best_solution, best_fitness