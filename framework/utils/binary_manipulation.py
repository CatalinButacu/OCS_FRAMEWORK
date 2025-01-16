import struct
import numpy as np

def bin2float(binary_list):
    """
    Convert a binary string (list of bits) to a float (64-bit representation).
    """
    rez = ''.join(binary_list)
    h = int(rez, 2).to_bytes(8, byteorder="big")
    return struct.unpack('>d', h)[0]

def float2bin(float_val):
    """
    Convert a float to a binary string (64-bit representation).
    """
    [d] = struct.unpack(">Q", struct.pack(">d", float_val))
    return list(f'{d:064b}')

def solution2binary(solution_x, bounds):
    """
    Convert a list of float values to a binary string.

    :param solution_x: List of float values.
    :param bounds: List of tuples [(min, max)] for each dimension.
    :return: Binary string.
    """
    binary_sol = []
    for val, (lower, upper) in zip(solution_x, bounds):
        # Normalize the value to [0, 1] range
        normalized_val = (val - lower) / (upper - lower)
        # Convert to binary representation
        binary_val = float2bin(normalized_val)
        binary_sol += binary_val
    return ''.join(binary_sol)

def binarysolution2float(binary_sol, bounds):
    """
    Convert a binary string to a list of float values.

    :param binary_sol: Binary string.
    :param bounds: List of tuples [(min, max)] for each dimension.
    :return: List of float values.
    """
    n = 64  # Number of bits per float value
    solution_x = []
    bitss = [binary_sol[i:i+n] for i in range(0, len(binary_sol), n)]
    for bits, (lower, upper) in zip(bitss, bounds):
        # Convert binary to normalized float in [0, 1]
        normalized_val = bin2float(bits)
        # Map normalized value to the original range [lower, upper]
        real_value = lower + normalized_val * (upper - lower)
        # Clip the value to bounds
        real_value = np.clip(real_value, lower, upper)
        solution_x.append(real_value)
    return solution_x

def binary_crossover(parent1, parent2, crossover_type="single_point"):
    """
    Perform crossover on two binary strings.
    :param parent1: First parent (binary string).
    :param parent2: Second parent (binary string).
    :param crossover_type: Type of crossover ("single_point", "two_point", or "uniform").
    :return: Two offspring (binary strings).
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parent binary strings must have the same length.")

    if crossover_type == "single_point":
        point = np.random.randint(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    elif crossover_type == "two_point":
        points = sorted(np.random.choice(range(1, len(parent1)), size=2, replace=False))
        child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
        child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    elif crossover_type == "uniform":
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        child1 = ''.join(child1)
        child2 = ''.join(child2)
    else:
        raise ValueError("Invalid crossover type. Use 'single_point', 'two_point', or 'uniform'.")
    return child1, child2

def binary_mutation(individual, mutation_rate):
    """
    Perform bit-flip mutation on a binary string.
    :param individual: Binary string.
    :param mutation_rate: Probability of flipping each bit.
    :return: Mutated binary string.
    """
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = '1' if mutated_individual[i] == '0' else '0'
    return ''.join(mutated_individual)

def int2binary(integer, num_bits):
    """
    Convert an integer to a binary string of fixed length.
    :param integer: Integer value.
    :param num_bits: Number of bits in the binary string.
    :return: Binary string.
    """
    return format(integer, f'0{num_bits}b')

def binary2int(binary_string):
    """
    Convert a binary string to an integer.
    :param binary_string: Binary string.
    :return: Integer value.
    """
    return int(binary_string, 2)