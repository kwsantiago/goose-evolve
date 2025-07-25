"""
Evolution module for Goose Evolve.
Provides genetic algorithm-based evolution capabilities.
"""

from .engine import EvolutionConfig, EvolutionEngine, EvolutionResult
from .fitness import (
    CompositeFitnessEvaluator,
    FitnessConfig,
    FitnessEvaluator,
    FitnessFunction,
)
from .variants import VariantGenerator

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "VariantGenerator",
    "FitnessEvaluator",
    "FitnessConfig",
    "FitnessFunction",
    "CompositeFitnessEvaluator",
]
