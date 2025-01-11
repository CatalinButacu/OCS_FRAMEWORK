import numpy as np

def quadratic_penalty(constraint_values):
    """
    Compute the quadratic penalty for constraint violations.

    :param constraint_values: A list of constraint values (positive values indicate violations).
    :return: The penalty value.
    """
    return np.sum(np.maximum(0, constraint_values) ** 2)

def linear_penalty(constraint_values):
    """
    Compute the linear penalty for constraint violations.

    :param constraint_values: A list of constraint values (positive values indicate violations).
    :return: The penalty value.
    """
    return np.sum(np.maximum(0, constraint_values))

def logarithmic_barrier(constraint_values):
    """
    Compute the logarithmic barrier penalty for constraint violations.

    :param constraint_values: A list of constraint values (positive values indicate violations).
    :return: The penalty value.
    """
    return -np.sum(np.log(-constraint_values[constraint_values < 0]))