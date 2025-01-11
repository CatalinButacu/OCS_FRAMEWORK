# framework/__init__.py

# Expose the main classes and functions from each module
from .utils import Function
from .benchmarks import BenchmarkFunctions
from .algorithms.random_search import RandomSearch, PopulationV1Adaptive, PopulationV2, PopulationV3SelfAdaptive
from .algorithms.canonical_ga import CGA, CGAAdaptiveV2, CGAGreedy
from .algorithms.real_ga import RGA1Adaptive, RGA4, RGA4AdaptiveV2
from .algorithms.differential_evolution import DERand2Bin, DEBest1Exp, DEBest2Exp

# Define what gets imported when using `from framework import *`
__all__ = [
    # Utils
    "Function",

    # Benchmarks
    "BenchmarkFunctions",

    # Random Search Algorithms
    "RandomSearch",
    "PopulationV1Adaptive",
    "PopulationV2",
    "PopulationV3SelfAdaptive",

    # Canonical GA Algorithms
    "CGA",
    "CGAAdaptiveV2",
    "CGAGreedy",

    # Real-Coded GA Algorithms
    "RGA1Adaptive",
    "RGA4",
    "RGA4AdaptiveV2",

    # Differential Evolution Algorithms
    "DERand2Bin",
    "DEBest1Exp",
    "DEBest2Exp",
]