# Utils

This module provides utility functions and classes for optimization, including:

## Function Class
- **Function**: A class to handle objective functions, constraints, and evaluations.
  - Methods:
    - `evaluate(x)`: Evaluates the objective function at a given point.
    - `is_feasible(x)`: Checks if a point satisfies all constraints.

## Data Handling
- **save_results(results, filename, format="json")**: Saves optimization results to a file.
- **load_results(filename, format="json")**: Loads optimization results from a file.
- **array_to_dict(array, keys)**: Converts a numpy array to a dictionary with specified keys.

## Constraints
- **quadratic_penalty(constraint_values)**: Computes the quadratic penalty for constraint violations.
- **linear_penalty(constraint_values)**: Computes the linear penalty for constraint violations.
- **logarithmic_barrier(constraint_values)**: Computes the logarithmic barrier penalty for constraint violations.