import numpy as np
from framework.utils import Function

class DERand2Bin:
    def __init__(self, func:Function, bounds, population_size=50, F=0.8, CR=0.9, max_nfe=1000):
        """
        Initialize the DE/rand/2/bin algorithm.

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

    def rand_2_mutation(self, population, idx):
        """
        Perform rand/2 mutation.

        :param population: The current population.
        :param idx: Index of the target vector.
        :return: Mutant vector.
        """
        # Select 5 distinct random indices (excluding idx)
        candidates = [i for i in range(self.population_size) if i != idx]
        r1, r2, r3, r4, r5 = np.random.choice(candidates, size=5, replace=False)

        # Generate mutant vector
        mutant = population[r1] + self.F * (population[r2] - population[r3]) + self.F * (population[r4] - population[r5])
        return mutant

    def binomial_crossover(self, target, mutant):
        """
        Perform binomial crossover.

        :param target: Target vector.
        :param mutant: Mutant vector.
        :return: Trial vector.
        """
        trial = np.copy(target)
        for i in range(len(self.bounds)):
            if np.random.rand() < self.CR or i == np.random.randint(len(self.bounds)):
                trial[i] = mutant[i]
        return trial

    def optimize(self):
        """
        Run the DE/rand/2/bin algorithm.

        :return: The best solution found and its fitness value.
        """
        # Initialize population
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)

        best_solution = population[np.argmin(fitness_values)]
        best_fitness = np.min(fitness_values)
        self.best_fitness_history.append(best_fitness)

        while self.nfe < self.max_nfe:
            new_population = []
            for i in range(self.population_size):
                # Mutation
                mutant = self.rand_2_mutation(population, i)

                # Crossover
                trial = self.binomial_crossover(population[i], mutant)

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
                    
            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

            # Replace population
            population = np.array(new_population)

        return best_solution, best_fitness