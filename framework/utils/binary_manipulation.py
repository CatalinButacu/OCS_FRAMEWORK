import numpy as np

def float_to_binary(value, bits_per_gene=64):
    """
    Convert a float value to a binary string with a fixed number of bits.
    :param value: Float value in the range [0, 1].
    :param bits_per_gene: Number of bits to represent the value.
    :return: Binary string.
    """
    # Scale the value to the range [0, 2^bits_per_gene - 1]
    scaled_value = int(value * (2**bits_per_gene - 1))
    # Convert to binary string
    binary_string = format(scaled_value, f'0{bits_per_gene}b')
    return binary_string

def binary_to_float(binary_string, bits_per_gene=64):
    """
    Convert a binary string to a float value.
    :param binary_string: Binary string.
    :param bits_per_gene: Number of bits used to represent the value.
    :return: Float value in the range [0, 1].
    """
    # Convert binary string to integer
    scaled_value = int(binary_string, 2)
    # Scale back to the range [0, 1]
    value = scaled_value / (2**bits_per_gene - 1)
    return value

def solution2binary(solution_x, bounds, bits_per_gene=64):
    """
    Convert a list of float values to a binary string.
    :param solution_x: List of float values.
    :param bounds: List of tuples [(min, max)] for each dimension.
    :param bits_per_gene: Number of bits per gene.
    :return: Binary string.
    """
    binary_sol = []
    for val, (lower, upper) in zip(solution_x, bounds):
        # Normalize the value to [0, 1] range
        normalized_val = (val - lower) / (upper - lower)
        # Convert to binary representation
        binary_val = float_to_binary(normalized_val, bits_per_gene)
        binary_sol.append(binary_val)
    return ''.join(binary_sol)

def binarysolution2float(binary_sol, bounds, bits_per_gene=64):
    """
    Convert a binary string to a list of float values.
    :param binary_sol: Binary string.
    :param bounds: List of tuples [(min, max)] for each dimension.
    :param bits_per_gene: Number of bits per gene.
    :return: List of float values.
    """
    solution_x = []
    bitss = [binary_sol[i:i+bits_per_gene] for i in range(0, len(binary_sol), bits_per_gene)]
    for bits, (lower, upper) in zip(bitss, bounds):
        # Convert binary to normalized float in [0, 1]
        normalized_val = binary_to_float(bits, bits_per_gene)
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