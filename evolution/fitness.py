"""
Fitness Evaluator implementation for Goose Evolve.
Provides pluggable fitness evaluation with weighted sum scoring.
"""

import asyncio
import logging
import math
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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
class FitnessRecord:
    """Historical fitness record for tracking over time"""

    variant_id: str
    fitness_score: float
    individual_scores: Dict[str, float]
    timestamp: datetime
    generation: int


@dataclass
class FitnessExplanation:
    """Detailed breakdown of fitness score calculation"""

    overall_score: float
    individual_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    weights: Dict[str, float]
    explanations: Dict[str, str]
    baseline_comparison: Optional[Dict[str, float]] = None


@dataclass
class ConfidenceInterval:
    """Statistical confidence interval for fitness scores"""

    lower_bound: float
    upper_bound: float
    confidence_level: float
    sample_size: int
    standard_error: float


@dataclass
class NormalizationStrategy:
    """Custom normalization strategy"""

    name: str
    normalize_fn: Callable[[float, Dict[str, Any]], float]
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation"""

    functions: Dict[str, FitnessFunction] = field(default_factory=dict)
    default_weights: Dict[str, float] = field(
        default_factory=lambda: {"speed": 0.4, "accuracy": 0.4, "efficiency": 0.2}
    )
    enable_pareto: bool = False
    cache_enabled: bool = True
    history_enabled: bool = True
    confidence_level: float = 0.95


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
        
        # New features
        self._fitness_history: Dict[str, List[FitnessRecord]] = defaultdict(list)
        self._fitness_cache: Dict[str, Tuple[float, FitnessExplanation]] = {}
        self._baseline_results: Optional[TestResults] = None
        self._normalization_strategies: Dict[str, NormalizationStrategy] = {}
        self._dynamic_weights: Dict[str, Callable[[Dict[str, Any]], float]] = {}
        
        # Register default normalization strategies
        self._register_default_normalization_strategies()
        
        # Register default fitness functions
        self._register_default_functions()
    
    def _register_default_normalization_strategies(self):
        """Register default normalization strategies"""
        # Min-max normalization (current default)
        self.register_normalization_strategy(
            "minmax", 
            lambda score, params: max(0.0, min(1.0, 
                (score - params.get('min', 0)) / max(params.get('max', 1) - params.get('min', 0), 1e-6))),
            "Min-max normalization to [0,1] range"
        )
        
        # Z-score normalization
        self.register_normalization_strategy(
            "zscore",
            lambda score, params: max(0.0, min(1.0, 
                0.5 + (score - params.get('mean', 0)) / (2 * max(params.get('std', 1), 1e-6)))),
            "Z-score normalization with sigmoid mapping"
        )
        
        # Logarithmic normalization for highly skewed data
        self.register_normalization_strategy(
            "log",
            lambda score, params: max(0.0, min(1.0,
                math.log(max(score, 1e-6)) / max(math.log(params.get('max', math.e)), 1))),
            "Logarithmic normalization for skewed distributions"
        )

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
        Supports caching, history tracking, and dynamic weights.
        """
        if not self.functions:
            logger.warning("No fitness functions registered, returning 0.0")
            return 0.0
        
        # Check cache first
        cache_key = self.get_cache_key(variant, results)
        if self.config.cache_enabled and cache_key in self._fitness_cache:
            cached_score, _ = self._fitness_cache[cache_key]
            logger.debug(f"Using cached fitness score for variant {variant.id}: {cached_score:.3f}")
            return cached_score
        
        # Apply dynamic weights if configured
        context = {
            "generation": variant.generation,
            "success_rate": results.success_rate,
            "response_time": results.avg_response_time,
            "error_count": results.error_count
        }
        self._apply_dynamic_weights(context)

        total_score = 0.0
        total_weight = 0.0
        function_scores = {}
        raw_scores = {}

        try:
            # Calculate individual fitness scores
            for name, fitness_func in self.functions.items():
                try:
                    raw_score = fitness_func.function(variant, results)
                    raw_scores[name] = raw_score

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
                        f"weighted={weighted_score:.3f} (weight={fitness_func.weight:.3f})"
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
            
            # Record in history
            self._record_fitness_history(variant, final_score, raw_scores)
            
            # Cache the result with explanation
            if self.config.cache_enabled:
                explanation = self.explain_fitness(variant, results)
                self._fitness_cache[cache_key] = (final_score, explanation)

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
    
    def get_cache_key(self, variant: Variant, results: TestResults) -> str:
        """Generate cache key for fitness evaluation"""
        # Create a hash of the key components
        key_components = [
            str(variant.id),
            variant.prompt,
            str(results.success_rate),
            str(results.avg_response_time),
            str(results.error_count),
            str(sorted(results.resource_usage.items()))
        ]
        return "|".join(key_components)

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
    
    # Multi-objective optimization and analytics methods
    
    def calculate_pareto_frontier(self, population: List[Variant]) -> List[Variant]:
        """Find non-dominated solutions using Pareto optimality"""
        if not population:
            return []
        
        # For multi-objective optimization, we need individual objective scores
        # This requires evaluating each variant first
        logger.info(f"Calculating Pareto frontier for {len(population)} variants")
        
        pareto_front = []
        
        for i, variant1 in enumerate(population):
            is_dominated = False
            
            for j, variant2 in enumerate(population):
                if i == j:
                    continue
                
                # Check if variant1 is dominated by variant2
                if self._dominates(variant2, variant1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(variant1)
        
        logger.info(f"Pareto frontier contains {len(pareto_front)} non-dominated solutions")
        return pareto_front
    
    def _dominates(self, variant1: Variant, variant2: Variant) -> bool:
        """Check if variant1 dominates variant2 in multi-objective space"""
        # For this to work properly, we need the individual objective scores
        # This is a simplified version that uses fitness_score as primary objective
        # In a full implementation, you'd compare all objectives
        
        if variant1.fitness_score is None or variant2.fitness_score is None:
            return False
        
        # Simple dominance: better in at least one objective, not worse in any
        return variant1.fitness_score >= variant2.fitness_score
    
    def explain_fitness(self, variant: Variant, results: TestResults) -> FitnessExplanation:
        """Provide detailed breakdown of fitness score calculation"""
        individual_scores = {}
        normalized_scores = {}
        explanations = {}
        current_weights = {}
        
        # Calculate individual scores for each function
        for name, fitness_func in self.functions.items():
            try:
                raw_score = fitness_func.function(variant, results)
                individual_scores[name] = raw_score
                
                # Get normalized score
                if fitness_func.normalize:
                    normalized_score = self._normalize_score(name, raw_score, fitness_func)
                else:
                    normalized_score = raw_score
                
                normalized_scores[name] = normalized_score
                current_weights[name] = fitness_func.weight
                
                # Generate explanation
                explanations[name] = self._generate_score_explanation(name, raw_score, normalized_score, fitness_func)
                
            except Exception as e:
                logger.error(f"Error explaining fitness function '{name}': {e}")
                individual_scores[name] = 0.0
                normalized_scores[name] = 0.0
                current_weights[name] = fitness_func.weight
                explanations[name] = f"Error: {str(e)}"
        
        # Calculate overall score
        total_score = sum(normalized_scores[name] * current_weights[name] for name in normalized_scores)
        total_weight = sum(current_weights.values())
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Add baseline comparison if available
        baseline_comparison = None
        if self._baseline_results:
            baseline_comparison = self._compare_to_baseline(results)
        
        return FitnessExplanation(
            overall_score=overall_score,
            individual_scores=individual_scores,
            normalized_scores=normalized_scores,
            weights=current_weights,
            explanations=explanations,
            baseline_comparison=baseline_comparison
        )
    
    def _generate_score_explanation(self, name: str, raw_score: float, normalized_score: float, fitness_func: FitnessFunction) -> str:
        """Generate human-readable explanation for a fitness score"""
        if name == "speed":
            return f"Response time: {1.0/max(raw_score, 1e-6):.2f}s → Speed score: {normalized_score:.3f}"
        elif name == "accuracy":
            return f"Success rate: {raw_score*100:.1f}% → Accuracy score: {normalized_score:.3f}"
        elif name == "efficiency":
            return f"Resource efficiency: {raw_score*100:.1f}% → Efficiency score: {normalized_score:.3f}"
        else:
            return f"Raw: {raw_score:.3f} → Normalized: {normalized_score:.3f}"
    
    def set_baseline(self, baseline: TestResults) -> None:
        """Set reference baseline for comparison"""
        self._baseline_results = baseline
        logger.info(f"Baseline set: success_rate={baseline.success_rate:.3f}, response_time={baseline.avg_response_time:.3f}s")
    
    def _compare_to_baseline(self, results: TestResults) -> Dict[str, float]:
        """Compare current results to baseline"""
        if not self._baseline_results:
            return {}
        
        baseline = self._baseline_results
        comparison = {}
        
        # Speed comparison (lower is better)
        if baseline.avg_response_time > 0:
            speed_improvement = (baseline.avg_response_time - results.avg_response_time) / baseline.avg_response_time
            comparison["speed_improvement"] = speed_improvement
        
        # Accuracy comparison (higher is better)
        accuracy_improvement = results.success_rate - baseline.success_rate
        comparison["accuracy_improvement"] = accuracy_improvement
        
        # Error comparison (lower is better)
        error_improvement = baseline.error_count - results.error_count
        comparison["error_improvement"] = error_improvement
        
        return comparison
    
    def get_fitness_history(self, variant_id: str) -> List[FitnessRecord]:
        """Get historical fitness tracking for a variant"""
        return self._fitness_history.get(variant_id, [])
    
    def _record_fitness_history(self, variant: Variant, fitness_score: float, individual_scores: Dict[str, float]):
        """Record fitness score in history"""
        if not self.config.history_enabled:
            return
        
        record = FitnessRecord(
            variant_id=str(variant.id),
            fitness_score=fitness_score,
            individual_scores=individual_scores.copy(),
            timestamp=datetime.now(),
            generation=variant.generation
        )
        
        self._fitness_history[str(variant.id)].append(record)
        
        # Limit history size to prevent memory bloat
        if len(self._fitness_history[str(variant.id)]) > 100:
            self._fitness_history[str(variant.id)] = self._fitness_history[str(variant.id)][-50:]
    
    async def evaluate_parallel(self, variants: List[Variant], results_list: List[TestResults]) -> List[float]:
        """Evaluate fitness for multiple variants in parallel"""
        if len(variants) != len(results_list):
            raise ValueError("Number of variants must match number of test results")
        
        logger.info(f"Starting parallel evaluation of {len(variants)} variants")
        
        # Create tasks for parallel evaluation
        tasks = []
        for variant, results in zip(variants, results_list):
            task = asyncio.create_task(self.evaluate(variant, results))
            tasks.append(task)
        
        # Wait for all evaluations to complete
        fitness_scores = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_scores = []
        for i, score in enumerate(fitness_scores):
            if isinstance(score, Exception):
                logger.error(f"Error evaluating variant {variants[i].id}: {score}")
                final_scores.append(0.0)
            else:
                final_scores.append(score)
        
        logger.info(f"Completed parallel evaluation: avg_score={statistics.mean(final_scores):.3f}")
        return final_scores
    
    def create_composite_function(self, name: str, logic: str, component_functions: List[str]) -> FitnessFunction:
        """Create composite fitness function with logic operators"""
        if not component_functions:
            raise ValueError("Composite function requires at least one component function")
        
        # Validate component functions exist
        for func_name in component_functions:
            if func_name not in self.functions:
                raise ValueError(f"Component function '{func_name}' not found")
        
        def composite_fn(variant: Variant, results: TestResults) -> float:
            # Evaluate component functions
            component_scores = {}
            for func_name in component_functions:
                try:
                    score = self.functions[func_name].function(variant, results)
                    component_scores[func_name] = score
                except Exception as e:
                    logger.warning(f"Error in component function '{func_name}': {e}")
                    component_scores[func_name] = 0.0
            
            # Apply logic operators
            try:
                return self._apply_composite_logic(logic, component_scores)
            except Exception as e:
                logger.error(f"Error applying composite logic '{logic}': {e}")
                return 0.0
        
        composite_function = FitnessFunction(
            name=name,
            function=composite_fn,
            weight=1.0,
            normalize=True
        )
        
        logger.info(f"Created composite function '{name}' with logic: {logic}")
        return composite_function
    
    def _apply_composite_logic(self, logic: str, scores: Dict[str, float]) -> float:
        """Apply logic operators to component scores"""
        # Simple logic operators implementation
        if logic == "AND":
            return min(scores.values()) if scores else 0.0
        elif logic == "OR":
            return max(scores.values()) if scores else 0.0
        elif logic == "AVERAGE":
            return statistics.mean(scores.values()) if scores else 0.0
        elif logic == "WEIGHTED_SUM":
            # Use existing weights
            total = sum(scores[name] * self.functions[name].weight for name in scores if name in self.functions)
            total_weight = sum(self.functions[name].weight for name in scores if name in self.functions)
            return total / total_weight if total_weight > 0 else 0.0
        elif "*" in logic or "+" in logic or "-" in logic:
            # Simple arithmetic expressions
            return self._evaluate_arithmetic_expression(logic, scores)
        else:
            raise ValueError(f"Unsupported composite logic: {logic}")
    
    def _evaluate_arithmetic_expression(self, expression: str, scores: Dict[str, float]) -> float:
        """Safely evaluate arithmetic expressions with component scores"""
        # This is a simplified implementation - in production you'd want a proper expression parser
        # Replace function names with their scores
        safe_expr = expression
        for name, score in scores.items():
            safe_expr = safe_expr.replace(name, str(score))
        
        # Only allow safe operations
        allowed_chars = set('0123456789.+-*/ ()')
        if not all(c in allowed_chars for c in safe_expr):
            raise ValueError(f"Unsafe expression: {expression}")
        
        try:
            result = eval(safe_expr)
            return max(0.0, min(10.0, float(result)))  # Clamp to reasonable range
        except Exception as e:
            raise ValueError(f"Invalid arithmetic expression '{expression}': {e}")
    
    def set_dynamic_weights(self, weight_functions: Dict[str, Callable[[Dict[str, Any]], float]]):
        """Set functions to dynamically adjust weights based on context"""
        self._dynamic_weights = weight_functions.copy()
        logger.info(f"Set dynamic weight functions for: {list(weight_functions.keys())}")
    
    def _apply_dynamic_weights(self, context: Dict[str, Any]):
        """Apply dynamic weight adjustments based on context"""
        for func_name, weight_fn in self._dynamic_weights.items():
            if func_name in self.functions:
                try:
                    new_weight = weight_fn(context)
                    self.functions[func_name].weight = max(0.0, new_weight)
                    self.weights[func_name] = self.functions[func_name].weight
                except Exception as e:
                    logger.warning(f"Error applying dynamic weight for '{func_name}': {e}")
    
    def register_normalization_strategy(self, name: str, normalize_fn: Callable[[float, Dict[str, Any]], float], description: str = ""):
        """Register a custom normalization strategy"""
        strategy = NormalizationStrategy(
            name=name,
            normalize_fn=normalize_fn,
            description=description
        )
        self._normalization_strategies[name] = strategy
        logger.info(f"Registered normalization strategy '{name}': {description}")
    
    def calculate_confidence_interval(self, variant_id: str, confidence_level: float = None) -> Optional[ConfidenceInterval]:
        """Calculate confidence interval for fitness scores"""
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        history = self.get_fitness_history(variant_id)
        if len(history) < 2:
            return None
        
        scores = [record.fitness_score for record in history]
        n = len(scores)
        mean_score = statistics.mean(scores)
        
        if n < 2:
            return None
        
        std_dev = statistics.stdev(scores)
        standard_error = std_dev / math.sqrt(n)
        
        # Use t-distribution for small samples
        if n < 30:
            # Simplified t-value approximation (use 2.0 for ~95% confidence)
            t_value = 2.0 if confidence_level >= 0.95 else 1.96
        else:
            # Normal distribution z-value
            t_value = 1.96 if confidence_level >= 0.95 else 1.645
        
        margin_of_error = t_value * standard_error
        
        return ConfidenceInterval(
            lower_bound=max(0.0, mean_score - margin_of_error),
            upper_bound=min(1.0, mean_score + margin_of_error),
            confidence_level=confidence_level,
            sample_size=n,
            standard_error=standard_error
        )
    
    def clear_cache(self):
        """Clear fitness evaluation cache"""
        self._fitness_cache.clear()
        logger.info("Fitness evaluation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._fitness_cache),
            "history_variants": len(self._fitness_history),
            "total_records": sum(len(records) for records in self._fitness_history.values())
        }


class CompositeFitnessEvaluator(FitnessEvaluator):
    """
    Composite fitness evaluator with multi-objective optimization capabilities.
    Provides extensible composition strategies for complex fitness evaluation.
    """

    def __init__(self, config: Optional[FitnessConfig] = None):
        super().__init__(config)
        self.pareto_enabled = False  # Future: enable Pareto optimization

    # Future: implement Pareto frontier methods, NSGA-II, etc.
