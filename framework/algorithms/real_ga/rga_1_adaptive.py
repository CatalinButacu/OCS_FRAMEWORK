import numpy as np
from framework.utils import Function

class RGA1Adaptive:
    def __init__(self, func:Function, bounds, population_size=50, pc_initial=0.8, pm_initial=0.1, max_nfe=1000):
        """
        Initialize the RGA1Adaptive algorithm.

        :param func: The objective function to minimize (an instance of `Function`).
        :param bounds: A list of tuples [(min, max)] for each dimension.
        :param population_size: The number of individuals in the population.
        :param pc_initial: Initial crossover probability.
        :param pm_initial: Initial mutation probability.
        :param max_nfe: Maximum number of function evaluations (NFE).
        """
        self.func = func
        self.bounds = bounds
        self.population_size = population_size
        self.pc_initial = pc_initial
        self.pm_initial = pm_initial
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

    def selection(self, population, fitness_values):
        """
        Select parents using roulette wheel selection.

        :param population: A numpy array of shape (population_size, dimension).
        :param fitness_values: A numpy array of fitness values.
        :return: Selected parents.
        """
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
        return population[parent_indices]

    def linear_crossover(self, parent1, parent2):
        """
        Perform linear crossover.

        :param parent1: First parent.
        :param parent2: Second parent.
        :return: Two offspring.
        """
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def non_uniform_mutation(self, individual, iteration, max_iter, b=5):
        """
        Perform non-uniform mutation.

        :param individual: An individual in the population.
        :param iteration: Current iteration number.
        :param max_iter: Maximum number of iterations.
        :param b: Parameter controlling the mutation strength.
        :return: Mutated individual.
        """
        for i in range(len(self.bounds)):
            if np.random.rand() < self.pm:
                delta = (1 - (iteration / max_iter)) ** b
                individual[i] += delta * np.random.uniform(-1, 1)
                individual[i] = np.clip(individual[i], self.bounds[i][0], self.bounds[i][1])
        return individual

    def optimize(self):
        """
        Run the RGA1Adaptive algorithm.

        :return: The best solution found and its fitness value.
        """
        # Initialize population
        population = self.initialize_population()
        fitness_values = self.evaluate_population(population)

        best_solution = population[np.argmin(fitness_values)]
        best_fitness = np.min(fitness_values)
        self.best_fitness_history.append(best_fitness)

        iteration = 0
        max_iter = self.max_nfe // self.population_size

        for _ in range(self.max_nfe):
            # Adapt pc and pm based on iteration number
            self.pc = self.pc_initial * (1 - iteration / max_iter)
            self.pm = self.pm_initial * (iteration / max_iter)

            # Selection
            parents = self.selection(population, fitness_values)

            # Crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.linear_crossover(parent1, parent2)
                child1 = self.non_uniform_mutation(child1, iteration, max_iter)
                child2 = self.non_uniform_mutation(child2, iteration, max_iter)
                new_population.extend([child1, child2])

            # Evaluate new population
            new_population = np.array(new_population)
            new_fitness_values = self.evaluate_population(new_population)

            # Update best solution
            min_fitness_idx = np.argmin(new_fitness_values)
            if new_fitness_values[min_fitness_idx] < best_fitness:
                best_solution = new_population[min_fitness_idx]
                best_fitness = new_fitness_values[min_fitness_idx]

            # Track the best fitness at each iteration
            self.best_fitness_history.append(best_fitness)

            # Replace population
            population = new_population
            fitness_values = new_fitness_values
            iteration += 1

        return best_solution, best_fitness