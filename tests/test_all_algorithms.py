import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from framework.benchmarks import BenchmarkFunctions
from framework.utils import Function
from framework.algorithms.random_search import RandomSearch, PopulationV1Adaptive, PopulationV2, PopulationV3SelfAdaptive
from framework.algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
from framework.algorithms.real_ga import RGA1Adaptive, RGA4, RGA4AdaptiveV2
from framework.algorithms.differential_evolution import DERand2Bin, DEBest1Exp, DEBest2Exp

# --- Step 1: Problem Selection ---
# Define benchmark functions
benchmark_functions = {
    "Sphere": BenchmarkFunctions.sphere,
    "Rastrigin": BenchmarkFunctions.rastrigin,
    "Rosenbrock": BenchmarkFunctions.rosenbrock,
    "Ackley": BenchmarkFunctions.ackley,
    "Griewank": BenchmarkFunctions.griewank,
}

# Define dimensions and stopping criteria
dimensions = [5, 10]  # Test for 5D and 10D problems
max_nfe = 1000  # Maximum number of function evaluations
runs_per_problem = 10  # Number of independent runs per problem

# Define bounds for the search space
bounds = [(-5.12, 5.12)]  # Bounds for each dimension (same for all dimensions)

# --- Step 2: Algorithm Implementation ---
# Define algorithms to test
algorithms = {
    "RandomSearch": RandomSearch,
    "PopulationV1Adaptive": PopulationV1Adaptive,
    "PopulationV2": PopulationV2,
    "PopulationV3SelfAdaptive": PopulationV3SelfAdaptive,
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
    "RandomSearch": {"max_iter": max_nfe},
    "PopulationV1Adaptive": {"population_size": 50, "max_iter": max_nfe, "alpha_initial": 1.0, "alpha_decay": 0.99},
    "PopulationV2": {"population_size": 50, "max_iter": max_nfe, "alpha": 0.5},
    "PopulationV3SelfAdaptive": {"population_size": 50, "max_iter": max_nfe, "alpha_initial": 1.0, "alpha_change_rate": 0.01},
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

# --- Step 3: Run Experiments ---
results = []
convergence_data = {}  # To store convergence data for plotting

for func_name, func in benchmark_functions.items():
    for dim in dimensions:
        # Define bounds for the current dimension
        current_bounds = bounds * dim  # Repeat bounds for each dimension
        for algo_name, algo_class in algorithms.items():
            convergence_data[(func_name, algo_name, dim)] = []  # Initialize convergence data
            for run in range(runs_per_problem):
                print(f"Running {algo_name} on {func_name} (dim={dim}, run={run + 1}/{runs_per_problem})")
                
                # Initialize the function and algorithm
                function = Function(func, dimension=dim, x_lower=[-5.12] * dim, x_upper=[5.12] * dim)
                algorithm = algo_class(function, bounds=current_bounds, **algorithm_parameters[algo_name])
                
                # Run the algorithm and track convergence
                best_solution, best_fitness = algorithm.optimize()
                if hasattr(algorithm, "best_fitness_history"):
                    convergence_data[(func_name, algo_name, dim)].append(algorithm.best_fitness_history)
                
                # Store results
                results.append({
                    "Function": func_name,
                    "Dimension": dim,
                    "Algorithm": algo_name,
                    "Run": run + 1,
                    "Best Fitness": best_fitness,
                })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv("experiment_results.csv", index=False)

# --- Step 4: Statistical Analysis ---
# Compute statistics (min, max, mean, std) for each algorithm on each problem
summary = results_df.groupby(["Function", "Dimension", "Algorithm"]).agg(
    Min_Fitness=("Best Fitness", "min"),
    Max_Fitness=("Best Fitness", "max"),
    Mean_Fitness=("Best Fitness", "mean"),
    Std_Fitness=("Best Fitness", "std"),
).reset_index()

# Save summary to a CSV file
summary.to_csv("experiment_summary.csv", index=False)

# --- Step 5: Convergence Plots ---
def plot_convergence(function_name, algorithm_name, dimension):
    key = (function_name, algorithm_name, dimension)
    if key not in convergence_data or not convergence_data[key]:
        print(f"No convergence data for {function_name}, {algorithm_name}, D={dimension}")
        return
    
    plt.figure(figsize=(10, 6))
    for run, fitness_history in enumerate(convergence_data[key]):
        plt.plot(fitness_history, label=f"Run {run + 1}")
    
    plt.title(f"Convergence Plot: {function_name} ({algorithm_name}, Dimension {dimension})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    plt.savefig(f"convergence_{function_name}_{algorithm_name}_D{dimension}.png")
    plt.show()

# Generate convergence plots for all combinations
for func_name in benchmark_functions:
    for algo_name in algorithms:
        for dim in dimensions:
            plot_convergence(func_name, algo_name, dim)

# --- Step 6: Rank Computation and Plot ---
def compute_and_plot_ranks(data, measure):
    ranks = data.pivot(index="Algorithm", columns="Function", values=measure).rank(axis=0, ascending=True)
    avg_ranks = ranks.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    avg_ranks.sort_values().plot(kind="bar")
    plt.title(f"Average Ranks based on {measure}")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Rank")
    plt.tight_layout()
    plt.savefig(f"rank_plot_{measure}.png")
    plt.show()
    return avg_ranks

# Compute and plot ranks based on mean fitness
compute_and_plot_ranks(summary, "Mean_Fitness")

# --- Step 7: Results Interpretation ---
# Analyze the results and draw conclusions
print("Summary of Results:")
print(summary)

# Determine the best algorithm for each function and dimension
best_algorithms = summary.loc[summary.groupby(["Function", "Dimension"])["Mean_Fitness"].idxmin()]
print("\nBest Algorithms for Each Function and Dimension:")
print(best_algorithms[["Function", "Dimension", "Algorithm", "Mean_Fitness"]])