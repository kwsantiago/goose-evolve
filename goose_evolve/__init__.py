"""
Goose Evolve - Self-improvement MCP extension for Goose agents.
"""

from evolution import (
    EvolutionConfig,
    EvolutionEngine,
    EvolutionResult,
    FitnessConfig,
    FitnessEvaluator,
    FitnessFunction,
    VariantGenerator,
)
from monitoring import MetricsCollector

__version__ = "0.1.0"

__all__ = [
    "EvolutionEngine",
    "EvolutionConfig",
    "EvolutionResult",
    "VariantGenerator",
    "FitnessEvaluator",
    "FitnessConfig",
    "FitnessFunction",
    "MetricsCollector",
]
