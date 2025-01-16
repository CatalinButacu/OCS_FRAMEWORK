import numpy as np
from framework.utils import Function
from framework.utils.binary_manipulation import (
    solution2binary, binarysolution2float, binary_crossover, binary_mutation
)

class CGA:
    def __init__(self, func: Function, bounds, population_size=50, pc=0.8, pm=0.1, max_nfe=1000, bits_per_gene=64):
        """
        Initialize the Canonical Genetic Algorithm (CGA) with binary representation.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of individuals in the population.
        :param pc: Crossover probability.
        :param pm: Mutation probability.
        :param max_nfe: Maximum number of function evaluations (NFE).
        :param bits_per_gene: Number of bits used to represent each decision variable.
        """
        self.func = func
        self.bounds = bounds
        self.population_size = population_size
        self.pc = pc
        self.pm = pm
        self.max_nfe = max_nfe
        self.bits_per_gene = bits_per_gene
        self.nfe = 0
        self.best_fitness_history = []

    def initialize_population(self):
        """
        Initialize the population with random binary individuals.

        :return: A list of binary strings representing the population.
        """
        population = []
        for _ in range(self.population_size):
            individual = []
            for b in self.bounds:
                # Generate random float within bounds and convert to binary
                random_float = np.random.uniform(b[0], b[1])
                binary_val = solution2binary([random_float], self.bounds)  # Pass bounds here
                individual.append(binary_val)
            population.append(''.join(individual))
        return population

    def evaluate_population(self, population):
        """
        Evaluate the fitness of the population.

        :param population: A list of binary strings representing the population.
        :return: A numpy array of fitness values.
        """
        fitness_values = []
        for individual in population:
            # Convert binary individual to real values
            real_values = binarysolution2float(individual, self.bounds)  # Pass bounds here
            # Evaluate fitness
            fitness = self.func(real_values)
            fitness_values.append(fitness)
        self.nfe += len(population)
        return np.array(fitness_values)

    def selection(self, population, fitness_values):
        """
        Select parents using roulette wheel selection.

        :param population: A list of binary strings representing the population.
        :param fitness_values: A numpy array of fitness values.
        :return: Selected parents (list of binary strings).
        """
        
        # Check for NaN or inf in fitness values
        if np.any(np.isnan(fitness_values)) or np.any(np.isinf(fitness_values)):
            fitness_values = np.where(np.isnan(fitness_values) | np.isinf(fitness_values), 1e10, fitness_values)

        # Shift fitness values to make them non-negative
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1e-10  # Add a small epsilon to avoid zero

        # Normalize fitness values (minimization problem)
        fitness_values = 1 / (1 + fitness_values)
        probabilities = fitness_values / np.sum(fitness_values)

        # Ensure probabilities are valid
        if np.any(probabilities < 0) or np.any(np.isnan(probabilities)):
            raise ValueError("Invalid probabilities in selection. Check fitness values.")

        # Select parents
        parent_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        return [population[i] for i in parent_indices]

    def crossover(self, parent1, parent2):
        """
        Perform binary crossover (single-point by default).

        :param parent1: First parent (binary string).
        :param parent2: Second parent (binary string).
        :return: Two offspring (binary strings).
        """
        return binary_crossover(parent1, parent2, crossover_type="single_point")

    def mutation(self, individual):
        """
        Perform binary mutation (bit-flip).

        :param individual: A binary string representing an individual.
        :return: Mutated binary string.
        """
        return binary_mutation(individual, self.pm)

    def optimize(self):
        """
        Run the Canonical Genetic Algorithm with binary representation.

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
            new_fitness_values = self.evaluate_population(new_population)

            # Update best solution
            min_fitness_idx = np.argmin(new_fitness_values)
            if new_fitness_values[min_fitness_idx] < best_fitness:
                best_solution = binarysolution2float(new_population[min_fitness_idx], self.bounds)  # Pass bounds here
                best_fitness = new_fitness_values[min_fitness_idx]

            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

            # Replace population
            population = new_population
            fitness_values = new_fitness_values

        return best_solution, best_fitness