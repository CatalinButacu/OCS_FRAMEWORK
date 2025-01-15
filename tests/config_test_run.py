from framework.benchmarks import BenchmarkFunctions
from framework.algorithms.random_search import RandomSearch
from framework.algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
from framework.algorithms.real_ga import RGA1Adaptive, RGA4, RGA4AdaptiveV2
from framework.algorithms.differential_evolution import DERand2Bin, DEBest1Exp, DEBest2Exp

# Define benchmark functions (5 test problems as per Lab 1)
benchmark_functions = {
    "Sphere": BenchmarkFunctions.sphere,
    "Rastrigin": BenchmarkFunctions.rastrigin,
    "Rosenbrock": BenchmarkFunctions.rosenbrock,
    "Ackley": BenchmarkFunctions.ackley,
    "Griewank": BenchmarkFunctions.griewank,
}

# Define dimensions
dimensions = [5, 10]  # Test for 5D and 10D problems

# Define stopping criteria
max_nfe = 1000  # Maximum number of function evaluations
runs_per_problem = 10  # Number of independent runs per problem

# Define bounds for the search space
bounds = [(-20, 20)]  # Bounds for each dimension (same for all dimensions)

# Define algorithms to test (only those assigned in Labs 2-5)
algorithms = {
    "CGA": CGA,
    "CGAAdaptiveV2": CGAAdaptiveV2,
    "CGAGreedy": CGAGreedy,
    "RGA1Adaptive": RGA1Adaptive,
    "RGA4": RGA4,
    "RGA4AdaptiveV2": RGA4AdaptiveV2,
    "DERand2Bin": DERand2Bin,
    "DEBest1Exp": DEBest1Exp,
    "DEBest2Exp": DEBest2Exp,
}

# Define algorithm parameters (tuned as per Labs 2-5)
algorithm_parameters = {
    "CGA": {"population_size": 50, "pc": 0.8, "pm": 0.1, "max_nfe": max_nfe},
    "CGAAdaptiveV2": {"population_size": 50, "pc_initial": 0.8, "pm_initial": 0.1, "max_nfe": max_nfe},
    "CGAGreedy": {"population_size": 50, "pc": 0.8, "pm": 0.1, "max_nfe": max_nfe},
    "RGA1Adaptive": {"population_size": 50, "pc_initial": 0.8, "pm_initial": 0.1, "max_nfe": max_nfe},
    "RGA4": {"population_size": 50, "pc": 0.8, "pm": 0.1, "max_nfe": max_nfe},
    "RGA4AdaptiveV2": {"population_size": 50, "pc_initial": 0.8, "pm_initial": 0.1, "max_nfe": max_nfe},
    "DERand2Bin": {"population_size": 50, "F": 0.8, "CR": 0.9, "max_nfe": max_nfe},
    "DEBest1Exp": {"population_size": 50, "F": 0.8, "CR": 0.9, "max_nfe": max_nfe},
    "DEBest2Exp": {"population_size": 50, "F": 0.8, "CR": 0.9, "max_nfe": max_nfe},
}