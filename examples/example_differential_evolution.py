import numpy as np
from framework.algorithms.differential_evolution import DERand2Bin, DEBest1Exp, DEBest2Exp
from framework.benchmarks import BenchmarkFunctions

# Define the benchmark function
func = BenchmarkFunctions.ackley
bounds = [(-5, 5), (-5, 5)]  # 2D problem
dimension = len(bounds)

# Initialize the Function class
from framework.utils import Function
function = Function(func, dimension, x_lower=[-5, -5], x_upper=[5, 5])

# Test DERand2Bin
#print("Running DERand2Bin...")
#de_rand = DERand2Bin(function, bounds, population_size=50, max_nfe=1000)
#best_solution, best_fitness = de_rand.optimize()
#print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test DEBest1Exp
print("Running DEBest1Exp...")
de_best1 = DEBest1Exp(function, bounds, population_size=50, max_nfe=1000)
best_solution, best_fitness = de_best1.optimize()
print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")

# Test DEBest2Exp
#print("Running DEBest2Exp...")
#de_best2 = DEBest2Exp(function, bounds, population_size=50, max_nfe=1000)
#best_solution, best_fitness = de_best2.optimize()
#print(f"Best solution: {best_solution}, Best fitness: {best_fitness}")