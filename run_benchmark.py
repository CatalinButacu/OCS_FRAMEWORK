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
os.makedirs("results/csv/fitness_table", exist_ok=True)
os.makedirs("results/csv/convergence", exist_ok=True)

# --- Step 3: Run Experiments ---
results = []

for func_name, func in benchmark_functions.items():
    for dim in dimensions:
        # Define bounds for the current dimension
        current_bounds = bounds * dim  # Repeat bounds for each dimension
        for algo_name, algo_class in algorithms.items():
            convergence_data = []  # Store convergence data for this algorithm
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
                if hasattr(algorithm, 'best_fitness_history'):
                    # Ensure best_fitness_history only contains the best fitness value per iteration
                    convergence_data.append(algorithm.best_fitness_history)
                else:
                    # If the algorithm doesn't have best_fitness_history, create it manually
                    print(f"[WARNING] {algo_name} doesnt have best_fitness_history...")
                    convergence_data.append([best_fitness])  # Only save the final best fitness

            # Save convergence data to a CSV file
            convergence_df = pd.DataFrame(convergence_data).T  # Transpose to have iterations as rows
            convergence_file = f"results/csv/convergence/{func_name}_{algo_name}_D{dim}.csv"
            convergence_df.to_csv(convergence_file, index=False)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV files
results_df.to_csv("results/csv/fitness_table/experiment_results.csv", index=False)

# --- Step 4: Statistical Analysis ---
# Compute statistics (min, max, mean, std) for each algorithm on each problem
summary = results_df.groupby(["Function", "Dimension", "Algorithm"]).agg(
    Best=("Best Fitness", "min"),  # Renamed to "Best"
    Average=("Best Fitness", "mean"),  # Renamed to "Average"
    Worst=("Best Fitness", "max"),  # Renamed to "Worst"
    Std=("Best Fitness", "std"),  # Renamed to "Std"
    Mean_Execution_Time=("Execution Time", "mean"),
).reset_index()

# Reorder columns to match the table
summary = summary[["Function", "Dimension", "Algorithm", "Best", "Average", "Worst", "Std", "Mean_Execution_Time"]]

# Save summary to a CSV file
summary.to_csv("results/csv/fitness_table/experiment_summary.csv", index=False)

print("Experiment summary CSV file updated and saved successfully!")