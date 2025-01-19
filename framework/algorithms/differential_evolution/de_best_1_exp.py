import numpy as np
from framework.utils import Function

class DEBest1Exp:
    def __init__(self, func:Function, bounds, population_size=50, F=0.8, CR=0.9, max_nfe=1000):
        """
        Initialize the DE/best/1/exp algorithm.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of individuals in the population.
        :param F: Mutation scaling factor.
        :param CR: Crossover probability.
        :param max_nfe: Maximum number of function evaluations (NFE).
        """
        self.func = func
        self.bounds = bounds
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.max_nfe = max_nfe
        self.nfe = 0
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

    def evaluate_population(self, population):
        """
        Evaluate the fitness of the population.

        :param population: A numpy array of shape (population_size, dimension).
        :return: A numpy array of fitness values.
        """
        fitness_values = np.array([self.func(individual) for individual in population])
        self.nfe += len(population)
        return fitness_values

    def best_1_mutation(self, population, best_idx):
        """
        Perform best/1 mutation.

        :param population: The current population.
        :param best_idx: Index of the best individual.
        :return: Mutant vector.
        """
        # Select 2 distinct random indices (excluding best_idx)
        candidates = [i for i in range(self.population_size) if i != best_idx]
        r1, r2 = np.random.choice(candidates, size=2, replace=False)

        # Generate mutant vector
        mutant = population[best_idx] + self.F * (population[r1] - population[r2])
        return mutant

    def exponential_crossover(self, target, mutant):
        """
        Perform exponential crossover.

        :param target: Target vector.
        :param mutant: Mutant vector.
        :return: Trial vector.
        """
        trial = np.copy(target)
        n = len(self.bounds)
        start = np.random.randint(n)
        L = 0

        while np.random.rand() < self.CR and L < n:
            trial[(start + L) % n] = mutant[(start + L) % n]
            L += 1

        return trial

    def optimize(self):
        """
        Run the DE/best/1/exp algorithm.

        :return: The best solution found and its fitness value.
        """
        # Initialize population
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        self.best_fitness_history.append(best_fitness)

        for _ in range(self.max_nfe):
            new_population = []
            for i in range(self.population_size):
                # Mutation
                mutant = self.best_1_mutation(population, best_idx)

                # Crossover
                trial = self.exponential_crossover(population[i], mutant)

                # Clip to bounds
                trial = np.clip(trial, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

                # Selection
                trial_fitness = self.func(trial)
                self.nfe += 1

                if trial_fitness < fitness_values[i]:
                    new_population.append(trial)
                    fitness_values[i] = trial_fitness
                else:
                    new_population.append(population[i])

                # Update best solution
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
                    best_idx = i
                    
            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

            # Replace population
            population = np.array(new_population)

        return best_solution, best_fitness