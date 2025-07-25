"""
Fitness Evaluator implementation for Goose Evolve.
Provides pluggable fitness evaluation with weighted sum scoring.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from evolution.interfaces import FitnessEvaluator as FitnessEvaluatorBase
from evolution.interfaces import TestResults, Variant

logger = logging.getLogger(__name__)


@dataclass
class FitnessFunction:
    """Configuration for a fitness function"""

    name: str
    function: Callable[[Variant, TestResults], float]
    weight: float
    normalize: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation"""

    functions: Dict[str, FitnessFunction] = field(default_factory=dict)
    default_weights: Dict[str, float] = field(
        default_factory=lambda: {"speed": 0.4, "accuracy": 0.4, "efficiency": 0.2}
    )


class FitnessEvaluator(FitnessEvaluatorBase):
    """
    Simple fitness evaluation with weighted sum scoring.
    Pluggable fitness functions without complex multi-objective optimization.
    """

    def __init__(self, config: Optional[FitnessConfig] = None):
        self.config = config or FitnessConfig()
        self.functions: Dict[str, FitnessFunction] = {}
        self.weights: Dict[str, float] = self.config.default_weights.copy()
        self._normalization_cache: Dict[str, Dict[str, float]] = {}

        # Register default fitness functions
        self._register_default_functions()

    def _register_default_functions(self):
        """Register the 3 default fitness functions"""
        self.register_fitness_function(
            "speed", self._speed_fitness, self.weights.get("speed", 0.4)
        )
        self.register_fitness_function(
            "accuracy", self._accuracy_fitness, self.weights.get("accuracy", 0.4)
        )
        self.register_fitness_function(
            "efficiency", self._token_fitness, self.weights.get("efficiency", 0.2)
        )

    async def evaluate(self, variant: Variant, results: TestResults) -> float:
        """
        Evaluate fitness using weighted sum of registered functions.
        Returns normalized score between 0 and 1.
        """
        if not self.functions:
            logger.warning("No fitness functions registered, returning 0.0")
            return 0.0

        total_score = 0.0
        total_weight = 0.0
        function_scores = {}

        try:
            # Calculate individual fitness scores
            for name, fitness_func in self.functions.items():
                try:
                    raw_score = fitness_func.function(variant, results)

                    # Normalize if enabled
                    if fitness_func.normalize:
                        normalized_score = self._normalize_score(
                            name, raw_score, fitness_func
                        )
                    else:
                        normalized_score = raw_score

                    function_scores[name] = normalized_score
                    weighted_score = normalized_score * fitness_func.weight
                    total_score += weighted_score
                    total_weight += fitness_func.weight

                    logger.debug(
                        f"Function {name}: raw={raw_score:.3f}, "
                        f"normalized={normalized_score:.3f}, "
                        f"weighted={weighted_score:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Error evaluating fitness function '{name}': {e}")
                    # Continue with other functions
                    continue

            # Calculate final weighted average
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                logger.warning("No valid fitness functions, returning 0.0")
                final_score = 0.0

            # Clamp to [0, 1] range
            final_score = max(0.0, min(1.0, final_score))

            logger.info(
                f"Variant {variant.id}: fitness={final_score:.3f} "
                f"(functions: {function_scores})"
            )

            return final_score

        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            return 0.0

    def register_fitness_function(
        self,
        name: str,
        fn: Callable[[Variant, TestResults], float],
        weight: float = 1.0,
    ):
        """
        Register a custom fitness function.

        Args:
            name: Unique name for the function
            fn: Function that takes (Variant, TestResults) and returns float
            weight: Weight for weighted sum (default: 1.0)
        """
        if not callable(fn):
            raise ValueError(f"Fitness function '{name}' must be callable")

        if weight < 0:
            raise ValueError(f"Weight for '{name}' must be non-negative")

        fitness_func = FitnessFunction(
            name=name, function=fn, weight=weight, normalize=True
        )

        self.functions[name] = fitness_func
        self.weights[name] = weight

        # Clear normalization cache for this function
        if name in self._normalization_cache:
            del self._normalization_cache[name]

        logger.info(f"Registered fitness function '{name}' with weight {weight}")

    def update_weights(self, weights: Dict[str, float]):
        """Update weights for existing fitness functions"""
        for name, weight in weights.items():
            if name in self.functions:
                self.functions[name].weight = weight
                self.weights[name] = weight
                logger.info(f"Updated weight for '{name}' to {weight}")
            else:
                logger.warning(f"Cannot update weight for unknown function '{name}'")

    def get_registered_functions(self) -> List[str]:
        """Get list of registered fitness function names"""
        return list(self.functions.keys())

    def _normalize_score(
        self, function_name: str, raw_score: float, fitness_func: FitnessFunction
    ) -> float:
        """
        Normalize score to [0, 1] range using adaptive normalization.
        Uses exponential moving average to track min/max values.
        """
        # Use explicit bounds if provided, skip adaptive normalization
        if fitness_func.min_value is not None and fitness_func.max_value is not None:
            min_val = fitness_func.min_value
            max_val = fitness_func.max_value

            if max_val <= min_val:
                return 0.5  # Return middle value if no range

            normalized = (raw_score - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))

        # Initialize cache for adaptive normalization
        if function_name not in self._normalization_cache:
            self._normalization_cache[function_name] = {
                "min": raw_score,
                "max": raw_score,
                "count": 1,
            }
            # For first score, try to make reasonable assumptions based on score value
            if raw_score <= 0.1:
                return 0.0  # Very low scores
            elif raw_score >= 0.9:
                return 1.0  # Very high scores
            else:
                return 0.5  # Middle value for first score

        cache = self._normalization_cache[function_name]

        # Update min/max with simple tracking (no exponential average for clearer behavior)
        cache["min"] = min(cache["min"], raw_score)
        cache["max"] = max(cache["max"], raw_score)
        cache["count"] += 1

        min_val = cache["min"]
        max_val = cache["max"]

        # Normalize to [0, 1]
        if max_val <= min_val:
            return 0.5  # Return middle value if no range

        normalized = (raw_score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    # Default fitness functions

    def _speed_fitness(self, variant: Variant, results: TestResults) -> float:
        """
        Speed fitness: Lower response time is better.
        Returns inverted and scaled response time.
        """
        if results.avg_response_time <= 0:
            return 1.0  # Maximum score for instant response

        # Use reciprocal with scaling
        # Assumes reasonable response times are 0.1s to 10s
        target_time = 1.0  # 1 second target
        return target_time / (target_time + results.avg_response_time)

    def _accuracy_fitness(self, variant: Variant, results: TestResults) -> float:
        """
        Accuracy fitness: Higher success rate is better.
        Penalizes errors heavily.
        """
        base_score = results.success_rate

        # Penalize errors (error_count relative to total attempts)
        # Assume success_rate is based on some number of attempts
        if results.error_count > 0:
            error_penalty = min(0.5, results.error_count * 0.1)
            base_score = max(0.0, base_score - error_penalty)

        return base_score

    def _token_fitness(self, variant: Variant, results: TestResults) -> float:
        """
        Token efficiency fitness: Lower token usage is better.
        Balances performance and resource usage.
        """
        cpu_usage = results.resource_usage.get("cpu_percent", 50.0)
        memory_usage = results.resource_usage.get("memory_mb", 100.0)

        # Normalize typical usage ranges
        # CPU: 0-100%, target ~25%
        # Memory: 0-500MB, target ~100MB
        cpu_efficiency = max(0.0, 1.0 - (cpu_usage / 100.0))
        memory_efficiency = max(0.0, 1.0 - min(1.0, memory_usage / 500.0))

        # Weighted average (CPU more important for efficiency)
        return 0.7 * cpu_efficiency + 0.3 * memory_efficiency


class CompositeFitnessEvaluator(FitnessEvaluator):
    """
    Extended fitness evaluator with advanced composition strategies.
    Future extension point for multi-objective optimization.
    """

    def __init__(self, config: Optional[FitnessConfig] = None):
        super().__init__(config)
        self.pareto_enabled = False  # Future: enable Pareto optimization

    # Future: implement Pareto frontier methods, NSGA-II, etc.
