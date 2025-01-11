import numpy as np

class Function:
    def __init__(self, function, dimension, x_lower=None, x_upper=None, constraints=None):
        """
        Initialize the Function class.

        :param function: The objective function to minimize.
        :param dimension: The number of decision variables.
        :param x_lower: Lower bounds for decision variables (list or numpy array).
        :param x_upper: Upper bounds for decision variables (list or numpy array).
        :param constraints: A list of constraint functions (optional).
        """
        self.function = function
        self.dimension = dimension
        self.x_lower = np.array(x_lower) if x_lower is not None else np.array([-np.inf] * dimension)
        self.x_upper = np.array(x_upper) if x_upper is not None else np.array([np.inf] * dimension)
        self.constraints = constraints if constraints is not None else []

        # Validate bounds
        if len(self.x_lower) != dimension or len(self.x_upper) != dimension:
            raise ValueError("Length of bounds must match the dimension of the problem.")

    def evaluate(self, x):
        """
        Evaluate the objective function at a given point.

        :param x: A numpy array representing the decision variables.
        :return: The objective function value.
        """
        x = np.array(x)
        if x.shape != (self.dimension,):
            raise ValueError(f"Input x must have shape ({self.dimension},).")

        # Clip x to bounds
        x = np.clip(x, self.x_lower, self.x_upper)

        # Evaluate the objective function
        objective_value = self.function(x)

        # Evaluate constraints (if any)
        constraint_violations = [constraint(x) for constraint in self.constraints]

        return objective_value, constraint_violations

    def is_feasible(self, x):
        """
        Check if a given point is feasible (satisfies all constraints).

        :param x: A numpy array representing the decision variables.
        :return: True if feasible, False otherwise.
        """
        _, constraint_violations = self.evaluate(x)
        return all(violation <= 0 for violation in constraint_violations)

    def __call__(self, x):
        """
        Allow the Function object to be called like a function.

        :param x: A numpy array representing the decision variables.
        :return: The objective function value.
        """
        return self.evaluate(x)[0]