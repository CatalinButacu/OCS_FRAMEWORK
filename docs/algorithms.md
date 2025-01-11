# Algorithms

This module contains implementations of various optimization algorithms, including:

## Random Search
- **RandomSearch**: Basic random search algorithm.
- **PopulationV1Adaptive**: Adaptive step size based on iteration number.
- **PopulationV2**: Multiple new agents per current agent, replace only the best one.
- **PopulationV3SelfAdaptive**: Adaptive step size based on fitness.

## Canonical Genetic Algorithm (CGA)
- **CGA**: Basic canonical genetic algorithm.
- **CGAAdaptiveV2**: Adaptive crossover and mutation probabilities based on fitness.
- **CGAGreedy**: Replaces the worst child with the best parent.

## Real-Coded Genetic Algorithm (RGA)
- **RGA1Adaptive**: Adaptive `pc` and `pm` based on iteration number, Linear Crossover, Non-Uniform Mutation.
- **RGA4**: Fixed `pc` and `pm`, BLX-0.1 Crossover, Gaussian Mutation.
- **RGA4AdaptiveV2**: Adaptive `pc` and `pm` based on fitness, BLX-0.1 Crossover, Gaussian Mutation.

## Differential Evolution (DE)
- **DERand2Bin**: rand/2 mutation, binomial crossover.
- **DEBest1Exp**: best/1 mutation, exponential crossover.
- **DEBest2Exp**: best/2 mutation, exponential crossover.
