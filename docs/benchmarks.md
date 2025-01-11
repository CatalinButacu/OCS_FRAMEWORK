# Benchmarks

This module provides a set of standard benchmark functions for testing optimization algorithms. Each function has a known global minimum, making it easy to evaluate algorithm performance.

## Available Benchmark Functions

1. **Sphere Function**
   - Formula: \( f(x) = \sum_{i=1}^n x_i^2 \)
   - Global Minimum: \( f(0, 0, ..., 0) = 0 \)

2. **Rastrigin Function**
   - Formula: \( f(x) = 10n + \sum_{i=1}^n (x_i^2 - 10 \cos(2\pi x_i)) \)
   - Global Minimum: \( f(0, 0, ..., 0) = 0 \)

3. **Rosenbrock Function**
   - Formula: \( f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2] \)
   - Global Minimum: \( f(1, 1, ..., 1) = 0 \)

4. **Ackley Function**
   - Formula: \( f(x) = -20 \exp(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}) - \exp(\frac{1}{n} \sum_{i=1}^n \cos(2\pi x_i)) + 20 + e \)
   - Global Minimum: \( f(0, 0, ..., 0) = 0 \)

5. **Griewank Function**
   - Formula: \( f(x) = 1 + \frac{1}{4000} \sum_{i=1}^n x_i^2 - \prod_{i=1}^n \cos(\frac{x_i}{\sqrt{i}}) \)
   - Global Minimum: \( f(0, 0, ..., 0) = 0 \)

6. **Schwefel Function**
   - Formula: \( f(x) = 418.9829n - \sum_{i=1}^n x_i \sin(\sqrt{|x_i|}) \)
   - Global Minimum: \( f(420.9687, 420.9687, ..., 420.9687) \approx 0 \)

7. **Zakharov Function**
   - Formula: \( f(x) = \sum_{i=1}^n x_i^2 + (\sum_{i=1}^n 0.5ix_i)^2 + (\sum_{i=1}^n 0.5ix_i)^4 \)
   - Global Minimum: \( f(0, 0, ..., 0) = 0 \)

8. **Michalewicz Function**
   - Formula: \( f(x) = -\sum_{i=1}^n \sin(x_i) \sin^{20}(\frac{ix_i^2}{\pi}) \)
   - Global Minimum: Depends on dimension and parameter \( m \).

9. **Styblinski-Tang Function**
   - Formula: \( f(x) = \frac{1}{2} \sum_{i=1}^n (x_i^4 - 16x_i^2 + 5x_i) \)
   - Global Minimum: \( f(-2.903534, -2.903534, ..., -2.903534) \approx -39.16617n \)

10. **Dixon-Price Function**
    - Formula: \( f(x) = (x_1 - 1)^2 + \sum_{i=2}^n i(2x_i^2 - x_{i-1})^2 \)
    - Global Minimum: \( f(x) = 0 \) at \( x_i = 2^{-\frac{2^i - 2}{2^i}} \) for \( i = 1, 2, ..., n \).