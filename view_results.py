import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from config_test_run import benchmark_functions, dimensions, algorithms, algorithm_line_styles

# Set the backend to 'Agg' to avoid Tcl/Tk issues
matplotlib.use('Agg')

# Create the results/images directory if it doesn't exist
os.makedirs("results/images", exist_ok=True)

# --- Step 1: Load Results from CSV Files ---
# Load the raw results
results_df = pd.read_csv("results/csv/fitness_table/experiment_results.csv")

# Load the summary statistics
summary_df = pd.read_csv("results/csv/fitness_table/experiment_summary.csv")

# --- Step 2: Plot Convergence Curves ---
def plot_convergence(func_name, algo_name, dimension):
    """
    Plot convergence for all 10 runs of an algorithm on a specific function and dimension.
    Save the plot in a structured folder.
    """
    # Load convergence data
    convergence_file = f"results/csv/convergence/{func_name}_{algo_name}_D{dimension}.csv"
    if not os.path.exists(convergence_file):
        print(f"No convergence data for {func_name}, {algo_name}, D={dimension}")
        return

    convergence_df = pd.read_csv(convergence_file)

    # Plot convergence for all 10 runs
    plt.figure(figsize=(10, 6))
    for run in range(convergence_df.shape[1]):
        plt.plot(convergence_df.iloc[:, run], label=f"Run {run + 1}")
    plt.title(f"Convergence Plot: {func_name} ({algo_name}, Dimension {dimension})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid()

    # Save the plot in a structured folder
    plot_folder = f"results/images/convergence/{func_name}"
    os.makedirs(plot_folder, exist_ok=True)
    plot_filename = f"{plot_folder}/{algo_name}_D{dimension}.png"
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
    """
    Compute and plot ranks based on the specified measure.
    Algorithms are displayed in the order they appear in the `algorithms` vector.
    RandomSearch is skipped entirely.
    Save the rank plot in a structured folder.
    """
    # Ensure no duplicate entries in the data
    data = data.drop_duplicates(subset=["Function", "Dimension", "Algorithm"])

    # Filter out RandomSearch from the data
    data = data[data["Algorithm"] != "RandomSearch"]

    # Pivot the data to compute ranks
    pivot_data = data.pivot(index="Algorithm", columns=["Function", "Dimension"], values=measure)

    # Compute ranks (lower is better)
    ranks = pivot_data.rank(axis=0, ascending=True)

    # Compute average ranks across all functions and dimensions
    avg_ranks = ranks.mean(axis=1)

    # Sort the average ranks according to the order of algorithms in the `algorithms` vector
    algorithm_order = [algo for algo in algorithms.keys() if algo != "RandomSearch"]
    avg_ranks = avg_ranks.reindex(algorithm_order)

    # Plot the average ranks
    plt.figure(figsize=(10, 6))
    avg_ranks.plot(kind="bar")
    plt.title(f"Average Ranks based on {measure}")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Rank")
    plt.grid(True)
    plt.tight_layout()

    # Save the rank plot in a structured folder
    plot_folder = "results/images/ranks"
    os.makedirs(plot_folder, exist_ok=True)
    rank_plot_filename = f"{plot_folder}/rank_plot_{measure}.png"
    plt.savefig(rank_plot_filename)
    plt.close()  # Close the plot to free up memory
    print(f"Saved rank plot: {rank_plot_filename}")

    return avg_ranks

# Compute and plot ranks based on mean fitness, min fitness, and std fitness
compute_and_plot_ranks(summary_df, "Average")  # Changed from "Mean_Fitness" to "Average"
compute_and_plot_ranks(summary_df, "Best")     # Changed from "Min_Fitness" to "Best"
compute_and_plot_ranks(summary_df, "Std")      # Changed from "Std_Fitness" to "Std"

# --- Step 4: Display Best Algorithms ---
# Determine the best algorithm for each function and dimension
best_algorithms = summary_df.loc[summary_df.groupby(["Function", "Dimension"])["Average"].idxmin()]
print("\nBest Algorithms for Each Function and Dimension:")
print(best_algorithms[["Function", "Dimension", "Algorithm", "Average"]])

# --- Step 5: Plot Best Performance Evolution for Each Formula ---
def plot_best_performance_evolution_for_function(func_name):
    """
    Plot the evolution of the best fitness over time for each algorithm for a specific function.
    Only the first 100 iterations are plotted for each dimension.
    Save the plot in a structured folder.
    """
    for dimIdx, dim in enumerate(dimensions):  # Iterate over all dimensions
        plt.figure(figsize=(12, 8))  # Create a new figure for each dimension

        for algo_name in algorithms:
            if algo_name == "RandomSearch":
                continue  # Skip RandomSearch

            # Load convergence data for the specified function and algorithm
            convergence_file = f"results/csv/convergence/{func_name}_{algo_name}_D{dim}.csv"
            if not os.path.exists(convergence_file):
                print(f"No convergence data for {func_name}, {algo_name}, D={dim}")
                continue

            convergence_df = pd.read_csv(convergence_file)

            # Find the best run (the one with the minimum final fitness value)
            best_run = convergence_df.iloc[-1].idxmin()
            best_fitness_history = convergence_df[best_run]

            # Slice the best fitness history to only include the first 100 iterations
            best_fitness_history = best_fitness_history[:100]

            # Plot the best fitness history for the first 100 iterations with the corresponding line style
            plt.plot(best_fitness_history, label=f"{algo_name}", linestyle=algorithm_line_styles.get(algo_name, "-"))

        # Add title, labels, and legend for the plot
        plt.title(f"Best Fitness Evolution for {func_name} (Dimension {dim}, First 100 Iterations)")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.grid()

        # Save the plot in a structured folder
        plot_folder = f"results/images/best_evolution/{func_name}"
        os.makedirs(plot_folder, exist_ok=True)
        plot_filename = f"{plot_folder}/best_evolution_D{dim}.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free up memory
        print(f"Saved best evolution plot: {plot_filename}")

# --- Step 6: Plot Average Performance Evolution for Each Formula ---
def plot_average_performance_evolution_for_function(func_name):
    """
    Plot the evolution of the average best fitness over time for each algorithm for a specific function.
    Only the first 100 iterations are plotted for each dimension.
    Save the plot in a structured folder.
    """
    for dimIdx, dim in enumerate(dimensions):  # Iterate over all dimensions
        plt.figure(figsize=(12, 8))  # Create a new figure for each dimension

        for algo_name in algorithms:
            if algo_name == "RandomSearch":
                continue  # Skip RandomSearch

            # Load convergence data for the specified function and algorithm
            convergence_file = f"results/csv/convergence/{func_name}_{algo_name}_D{dim}.csv"
            if not os.path.exists(convergence_file):
                print(f"No convergence data for {func_name}, {algo_name}, D={dim}")
                continue

            convergence_df = pd.read_csv(convergence_file)

            # Compute the average best fitness across all runs
            avg_fitness_history = convergence_df.mean(axis=1)

            # Slice the average fitness history to only include the first 100 iterations
            avg_fitness_history = avg_fitness_history[:100]

            # Plot the average fitness history for the first 100 iterations with the corresponding line style
            plt.plot(avg_fitness_history, label=f"{algo_name}", linestyle=algorithm_line_styles.get(algo_name, "-"))

        # Add title, labels, and legend for the plot
        plt.title(f"Average Fitness Evolution for {func_name} (Dimension {dim}, First 100 Iterations)")
        plt.xlabel("Iteration")
        plt.ylabel("Average Best Fitness")
        plt.legend()
        plt.grid()

        # Save the plot in a structured folder
        plot_folder = f"results/images/average_evolution/{func_name}"
        os.makedirs(plot_folder, exist_ok=True)
        plot_filename = f"{plot_folder}/average_evolution_D{dim}.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free up memory
        print(f"Saved average evolution plot: {plot_filename}")

# Plot best performance evolution for each benchmark function (first 100 iterations)
for func_name in benchmark_functions:
    plot_best_performance_evolution_for_function(func_name)

# Plot average performance evolution for each benchmark function (first 100 iterations)
for func_name in benchmark_functions:
    plot_average_performance_evolution_for_function(func_name)