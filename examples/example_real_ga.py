import numpy as np
from framework.algorithms.real_ga import RGA1Adaptive, RGA4, RGA4AdaptiveV2
from framework.benchmarks import BenchmarkFunctions

# Define the benchmark function
func = BenchmarkFunctions.rosenbrock
bounds = [(-5, 5), (-5, 5)]  # 2D problem
dimension = len(bounds)

# Initialize the Function class
from framework.utils import Function
function = Function(func, dimension, x_lower=[-5, -5], x_upper=[5, 5])

# Test RGA4 - Roxana
print("Running RGA4...") 
rga4 = RGA4(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = rga4.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test RGA1Adaptive - Catalin
print("Running RGA1Adaptive...")
rga1 = RGA1Adaptive(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = rga1.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test RGA4AdaptiveV2 - Radu
print("Running RGA4AdaptiveV2...")
rga4_adaptive = RGA4AdaptiveV2(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = rga4_adaptive.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")