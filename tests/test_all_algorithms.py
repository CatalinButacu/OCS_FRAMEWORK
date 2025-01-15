import os
import time
import pandas as pd
from framework.utils import Function
from config_test_run import (
    benchmark_functions,
    dimensions,
    runs_per_problem,
    bounds,
    algorithms,
    algorithm_parameters,
)

# Create directories for saving results
os.makedirs("results/csv/convergence", exist_ok=True)
os.makedirs("results/csv/fitness_table", exist_ok=True)
os.makedirs("results/images", exist_ok=True)

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
                function = Function(func, dimension=dim, x_lower=[-20] * dim, x_upper=[20] * dim)
                algorithm = algo_class(function, bounds=current_bounds, **algorithm_parameters[algo_name])
                
                # Run the algorithm and track convergence
                start_time = time.time()
                best_solution, best_fitness = algorithm.optimize()
                execution_time = time.time() - start_time
                
                # Store results
                results.append({
                    "Function": func_name,
                    "Dimension": dim,
                    "Algorithm": algo_name,
                    "Run": run + 1,
                    "Best Fitness": best_fitness,
                    "Execution Time": execution_time,
                })

                # Save convergence data for this run
                convergence_data[(func_name, algo_name, dim)].append(algorithm.best_fitness_history)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV files
results_df.to_csv("results/csv/fitness_table/experiment_results.csv", index=False)

# Save convergence data to CSV files
for key, histories in convergence_data.items():
    func_name, algo_name, dim = key
    convergence_df = pd.DataFrame(histories).T  # Transpose to have iterations as rows
    convergence_df.to_csv(f"results/csv/convergence/{func_name}_{algo_name}_D{dim}.csv", index=False)

# --- Step 4: Statistical Analysis ---
# Compute statistics (min, max, mean, std) for each algorithm on each problem
summary = results_df.groupby(["Function", "Dimension", "Algorithm"]).agg(
    Min_Fitness=("Best Fitness", "min"),
    Max_Fitness=("Best Fitness", "max"),
    Mean_Fitness=("Best Fitness", "mean"),
    Std_Fitness=("Best Fitness", "std"),
    Mean_Execution_Time=("Execution Time", "mean"),
).reset_index()

# Save summary to a CSV file
summary.to_csv("results/csv/fitness_table/experiment_summary.csv", index=False)