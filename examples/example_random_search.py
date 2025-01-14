import numpy as np
from framework.algorithms.random_search import RandomSearch, PopulationV1Adaptive, PopulationV2, PopulationV3SelfAdaptive
from framework.benchmarks import BenchmarkFunctions

# Define the benchmark function
func = BenchmarkFunctions.sphere
bounds = [(-5, 5), (-5, 5)]  # 2D problem
dimension = len(bounds)

# Initialize the Function class
from framework.utils import Function
function = Function(func, dimension, x_lower=[-5, -5], x_upper=[5, 5])

# Test RandomSearch
print("Running RandomSearch...")
random_search = RandomSearch(function, bounds, max_iter=1000)
best_solution, best_fitness = random_search.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test PopulationV1Adaptive - Radu
print("Running PopulationV1Adaptive...")
population_v1 = PopulationV1Adaptive(function, bounds, population_size=10, max_iter=1000)
best_solution, best_fitness = population_v1.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test PopulationV2 - Roxana
print("Running PopulationV2...")
population_v2 = PopulationV2(function, bounds, population_size=10, max_iter=1000)
best_solution, best_fitness = population_v2.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test PopulationV3SelfAdaptive - Catalin
print("Running PopulationV3SelfAdaptive...")
population_v3 = PopulationV3SelfAdaptive(function, bounds, population_size=10, max_iter=1000)
best_solution, best_fitness = population_v3.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")