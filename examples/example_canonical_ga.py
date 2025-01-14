import numpy as np
from framework.algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
from framework.benchmarks import BenchmarkFunctions

# Define the benchmark function
func = BenchmarkFunctions.rastrigin
bounds = [(-5, 5), (-5, 5)]  # 2D problem
dimension = len(bounds)

# Initialize the Function class
from framework.utils import Function
function = Function(func, dimension, x_lower=[-5, -5], x_upper=[5, 5])

# Test CGA - Roxana
print("Running CGA...")
cga = CGA(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = cga.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test CGAGreedy - Catalin
print("Running CGAGreedy...")
cga_greedy = CGAGreedy(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = cga_greedy.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test CGAAdaptiveV2 - Radu
print("Running CGAAdaptiveV2...")
cga_adaptive = CGAAdaptiveV2(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = cga_adaptive.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

