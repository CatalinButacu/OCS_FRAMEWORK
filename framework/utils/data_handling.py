import numpy as np
import json
import pickle

def save_results(results, filename, format="json"):
    """
    Save optimization results to a file.

    :param results: A dictionary containing the results.
    :param filename: The name of the file to save the results.
    :param format: The file format ("json" or "pkl").
    """
    if format == "json":
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
    elif format == "pkl":
        with open(filename, "wb") as f:
            pickle.dump(results, f)
    else:
        raise ValueError("Unsupported file format. Use 'json' or 'pkl'.")

def load_results(filename, format="json"):
    """
    Load optimization results from a file.

    :param filename: The name of the file to load the results from.
    :param format: The file format ("json" or "pkl").
    :return: The loaded results.
    """
    if format == "json":
        with open(filename, "r") as f:
            return json.load(f)
    elif format == "pkl":
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use 'json' or 'pkl'.")

def array_to_dict(array, keys):
    """
    Convert a numpy array to a dictionary with specified keys.

    :param array: A numpy array.
    :param keys: A list of keys corresponding to the array elements.
    :return: A dictionary with keys and array values.
    """
    if len(array) != len(keys):
        raise ValueError("Length of array and keys must match.")
    return {key: value for key, value in zip(keys, array)}