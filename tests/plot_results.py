import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from config_test_run import benchmark_functions, dimensions, algorithms

# Set the backend to 'Agg' to avoid TclError
matplotlib.use('Agg')

# Create the results/images directory if it doesn't exist
os.makedirs("results/images", exist_ok=True)

# --- Step 1: Load Results from CSV Files ---
# Load the raw results
results_df = pd.read_csv("results/csv/fitness_table/experiment_results.csv")

# Load the summary statistics
summary_df = pd.read_csv("results/csv/fitness_table/experiment_summary.csv")

# Display the raw results
print("Raw Results:")
print(results_df.head())

# Display the summary statistics
print("\nSummary Statistics:")
print(summary_df.head())

# --- Step 2: Plot Convergence Curves ---
def plot_convergence(func_name, algo_name, dimension):
    # Load convergence data
    convergence_file = f"results/csv/convergence/{func_name}_{algo_name}_D{dimension}.csv"
    if not os.path.exists(convergence_file):
        print(f"No convergence data for {func_name}, {algo_name}, D={dimension}")
        return
    
    convergence_df = pd.read_csv(convergence_file)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    for run in range(convergence_df.shape[1]):
        plt.plot(convergence_df.iloc[:, run], label=f"Run {run + 1}")
    
    plt.title(f"Convergence Plot: {func_name} ({algo_name}, Dimension {dimension})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    
    # Save the plot to the results/images folder
    plot_filename = f"results/images/convergence_{func_name}_{algo_name}_D{dimension}.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Saved convergence plot: {plot_filename}")

# Generate convergence plots for all combinations
for func_name in benchmark_functions:
    for algo_name in algorithms:
        for dim in dimensions:
            plot_convergence(func_name, algo_name, dim)

# --- Step 3: Rank Computation and Plot ---
def compute_and_plot_ranks(data, measure):
    # Ensure no duplicate entries in the data
    data = data.drop_duplicates(subset=["Function", "Dimension", "Algorithm"])
    
    # Pivot the data to compute ranks
    pivot_data = data.pivot(index="Algorithm", columns=["Function", "Dimension"], values=measure)
    
    # Compute ranks (lower is better)
    ranks = pivot_data.rank(axis=0, ascending=True)
    
    # Compute average ranks across all functions and dimensions
    avg_ranks = ranks.mean(axis=1)
    
    # Plot the average ranks
    plt.figure(figsize=(10, 6))
    avg_ranks.sort_values().plot(kind="bar")
    plt.title(f"Average Ranks based on {measure}")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Rank")
    plt.tight_layout()
    
    # Save the rank plot to the results/images folder
    rank_plot_filename = f"results/images/rank_plot_{measure}.png"
    plt.savefig(rank_plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Saved rank plot: {rank_plot_filename}")
    
    return avg_ranks

# Compute and plot ranks based on mean fitness
compute_and_plot_ranks(summary_df, "Mean_Fitness")

# --- Step 4: Display Best Algorithms ---
# Determine the best algorithm for each function and dimension
best_algorithms = summary_df.loc[summary_df.groupby(["Function", "Dimension"])["Mean_Fitness"].idxmin()]
print("\nBest Algorithms for Each Function and Dimension:")
print(best_algorithms[["Function", "Dimension", "Algorithm", "Mean_Fitness"]])

# --- Step 5: Plot Best Performance Evolution for Each Formula ---
def plot_best_performance_evolution_for_function(func_name):
    """
    Plot the evolution of the best fitness over time for each algorithm for a specific function.
    """
    plt.figure(figsize=(12, 8))
    
    for algo_name in algorithms:
        # Load convergence data for the specified function and algorithm
        convergence_file = f"results/csv/convergence/{func_name}_{algo_name}_D{dimensions[0]}.csv"
        if not os.path.exists(convergence_file):
            print(f"No convergence data for {func_name}, {algo_name}, D={dimensions[0]}")
            continue
        
        convergence_df = pd.read_csv(convergence_file)
        
        # Find the best run (the one with the minimum final fitness value)
        best_run = convergence_df.iloc[-1].idxmin()
        best_fitness_history = convergence_df[best_run]
        
        # Plot the best fitness history
        plt.plot(best_fitness_history, label=f"{algo_name}")
    
    plt.title(f"Best Fitness Evolution for {func_name} (Dimension {dimensions[0]})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()
    
    # Save the plot to the results/images folder
    plot_filename = f"results/images/best_evolution_{func_name}.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Saved best evolution plot: {plot_filename}")

# Plot best performance evolution for each benchmark function
for func_name in benchmark_functions:
    plot_best_performance_evolution_for_function(func_name)