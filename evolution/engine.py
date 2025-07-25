"""
Evolution Engine implementation for Goose Evolve.
Central orchestrator for the genetic algorithm evolution cycle.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from evolution.interfaces import (
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
)
from evolution.interfaces import EvolutionEngine as EvolutionEngineBase
from evolution.interfaces import (
    EvolutionTriggerEvent,
    SelectionMethod,
    TestResults,
    ValidationResult,
    Variant,
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Result of an evolution cycle"""

    cycle_id: UUID
    trigger: EvolutionTriggerEvent
    generation: int
    population_size: int
    winners: List[Variant]
    avg_fitness: float
    best_fitness: float
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionConfig:
    """Configuration for evolution engine"""

    population_size: int = DEFAULT_POPULATION_SIZE
    tournament_size: int = 3
    selection_pressure: float = 0.8
    elite_count: int = 5
    mutation_rate: float = DEFAULT_MUTATION_RATE
    max_generations: int = 10
    fitness_threshold: float = 0.95
    parallel_evaluations: int = 10


class EvolutionEngine(EvolutionEngineBase):
    """
    Central evolution orchestrator implementing genetic algorithm.
    Manages population evolution cycles with tournament selection.
    """

    def __init__(
        self,
        variant_generator,
        safety_validator,
        sandbox_manager,
        fitness_evaluator,
        config: Optional[EvolutionConfig] = None,
    ):
        self.variant_generator = variant_generator
        self.safety_validator = safety_validator
        self.sandbox_manager = sandbox_manager
        self.fitness_evaluator = fitness_evaluator
        self.config = config or EvolutionConfig()

        self.current_generation = 0
        self.event_listeners: List[Any] = []

    async def start_evolution_cycle(
        self, trigger: EvolutionTriggerEvent
    ) -> Dict[str, Any]:
        """
        Start a complete evolution cycle.

        Steps:
        1. Generate population
        2. Validate variants
        3. Test in sandbox
        4. Evaluate fitness
        5. Select survivors
        6. Return winners
        """
        start_time = datetime.now()
        cycle_id = uuid4()

        logger.info(
            f"Starting evolution cycle {cycle_id} triggered by {trigger.trigger_type}"
        )
        self._emit_event(
            "evolution_started", {"cycle_id": cycle_id, "trigger": trigger}
        )

        try:
            # Generate initial population
            base_agent = await self._get_base_agent()
            population = await self._generate_population(base_agent)

            # Main evolution loop
            best_variants = []
            avg_fitness = 0.0
            best_fitness = 0.0
            for generation in range(self.config.max_generations):
                self.current_generation = generation
                logger.info(f"Generation {generation}/{self.config.max_generations}")

                # Validate all variants
                valid_population = await self._validate_population(population)

                if not valid_population:
                    logger.warning("No valid variants in population")
                    break

                # Test variants in sandbox
                test_results = await self._test_population(valid_population)

                # Evaluate fitness
                evaluated_population = await self.evaluate_population(
                    valid_population, test_results
                )

                # Check for fitness threshold
                best_fitness = max(score for _, score in evaluated_population)
                avg_fitness = sum(score for _, score in evaluated_population) / len(
                    evaluated_population
                )

                logger.info(
                    f"Generation {generation} - Best: {best_fitness:.3f}, Avg: {avg_fitness:.3f}"
                )

                # Select top performers
                best_variants = [
                    v
                    for v, s in evaluated_population
                    if s >= self.config.fitness_threshold
                ]

                if best_variants or best_fitness >= self.config.fitness_threshold:
                    logger.info("Fitness threshold reached, ending evolution")
                    break

                # Select survivors for next generation
                survivors = self.select_survivors(evaluated_population)

                # Generate next generation
                population = await self._generate_next_generation(survivors)

            # Calculate final metrics
            duration = (datetime.now() - start_time).total_seconds()

            result = {
                "cycle_id": str(cycle_id),
                "trigger": trigger,
                "generation": self.current_generation,
                "population_size": len(population),
                "winners": best_variants[: self.config.elite_count],
                "avg_fitness": avg_fitness,
                "best_fitness": best_fitness,
                "duration_seconds": duration,
            }

            self._emit_event(
                "evolution_completed", {"cycle_id": cycle_id, "result": result}
            )

            return result

        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            self._emit_event(
                "evolution_failed", {"cycle_id": cycle_id, "error": str(e)}
            )
            raise

    async def evaluate_population(
        self,
        population: List[Variant],
        test_results: Optional[Dict[UUID, TestResults]] = None,
    ) -> List[Tuple[Variant, float]]:
        """
        Evaluate fitness of population in parallel.
        Returns list of (variant, fitness_score) tuples.
        """
        semaphore = asyncio.Semaphore(self.config.parallel_evaluations)

        async def evaluate_variant(variant: Variant) -> Tuple[Variant, float]:
            async with semaphore:
                if test_results is None or variant.id not in test_results:
                    logger.warning(f"No test results for variant {variant.id}")
                    return (variant, 0.0)

                results = test_results[variant.id]
                fitness = await self.fitness_evaluator.evaluate(variant, results)
                variant.fitness_score = fitness
                return (variant, fitness)

        # Evaluate all variants in parallel
        tasks = [evaluate_variant(v) for v in population]
        evaluated = await asyncio.gather(*tasks)

        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x[1], reverse=True)

        return evaluated

    def select_survivors(self, evaluated: List[Tuple[Variant, float]]) -> List[Variant]:
        """
        Select survivors using tournament selection.
        Maintains diversity while favoring high fitness.
        """
        survivors = []
        population = [v for v, _ in evaluated]
        fitness_map = {v.id: score for v, score in evaluated}

        # Always keep elite individuals
        elite_count = min(self.config.elite_count, len(evaluated))
        survivors.extend([v for v, _ in evaluated[:elite_count]])

        # Tournament selection for remaining slots
        remaining_slots = (
            int(self.config.population_size * self.config.selection_pressure)
            - elite_count
        )

        for _ in range(max(0, remaining_slots)):
            # Select tournament participants
            tournament_size = min(self.config.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)

            # Winner is highest fitness
            winner = max(tournament, key=lambda v: fitness_map.get(v.id, 0))
            survivors.append(winner)

        # Remove duplicates while preserving order
        seen = set()
        unique_survivors = []
        for v in survivors:
            if v.id not in seen:
                seen.add(v.id)
                unique_survivors.append(v)

        logger.info(
            f"Selected {len(unique_survivors)} survivors from {len(evaluated)} variants"
        )

        return unique_survivors

    async def _get_base_agent(self) -> Dict[str, Any]:
        """Get current agent configuration as base for evolution"""
        # Note: Agent retrieval from MCP to be implemented in integration phase
        return {
            "prompt": "You are a helpful assistant.",
            "configuration": {"temperature": 0.7},
        }

    async def _generate_population(self, base_agent: Dict[str, Any]) -> List[Variant]:
        """Generate initial population from base agent"""
        population = self.variant_generator.generate_population(
            base_agent["prompt"], self.config.population_size
        )

        logger.info(f"Generated initial population of {len(population)} variants")
        return population

    async def _validate_population(self, population: List[Variant]) -> List[Variant]:
        """Validate all variants, returning only valid ones"""
        valid_variants = []

        for variant in population:
            result = await self.safety_validator.validate(variant)
            if result.is_valid:
                valid_variants.append(variant)
            else:
                logger.warning(
                    f"Variant {variant.id} failed validation: {result.errors}"
                )

        logger.info(f"Validated {len(valid_variants)}/{len(population)} variants")
        return valid_variants

    async def _test_population(
        self, population: List[Variant]
    ) -> Dict[UUID, TestResults]:
        """Test all variants in sandbox, returning results map"""
        semaphore = asyncio.Semaphore(self.config.parallel_evaluations)
        results = {}

        async def test_variant(variant: Variant) -> Tuple[UUID, TestResults]:
            async with semaphore:
                try:
                    test_result = await self.sandbox_manager.test_variant(variant)
                    return (variant.id, test_result)
                except Exception as e:
                    logger.error(f"Failed to test variant {variant.id}: {e}")
                    # Return failed test result
                    return (
                        variant.id,
                        TestResults(
                            variant_id=variant.id,
                            success_rate=0.0,
                            avg_response_time=float("inf"),
                            error_count=1,
                            resource_usage={},
                        ),
                    )

        # Test all variants in parallel
        tasks = [test_variant(v) for v in population]
        test_results = await asyncio.gather(*tasks)

        for variant_id, result in test_results:
            results[variant_id] = result

        logger.info(f"Tested {len(results)} variants in sandbox")
        return results

    async def _generate_next_generation(
        self, survivors: List[Variant]
    ) -> List[Variant]:
        """Generate next generation from survivors"""
        next_generation = []

        # Keep elite individuals
        elite_count = min(self.config.elite_count, len(survivors))
        next_generation.extend(survivors[:elite_count])

        # Fill remaining population with offspring
        while len(next_generation) < self.config.population_size:
            # Select parents
            if len(survivors) >= 2:
                parents = random.sample(survivors, 2)
                offspring = self.variant_generator.crossover(parents[0], parents[1])
                next_generation.extend(offspring)
            else:
                # If not enough survivors, mutate existing ones
                parent = random.choice(survivors)
                mutated = self.variant_generator.mutate_variant(parent)
                next_generation.append(mutated)

        # Trim to exact population size
        next_generation = next_generation[: self.config.population_size]

        # Update generation numbers
        for variant in next_generation:
            if variant.generation == self.current_generation:
                variant.generation = self.current_generation + 1

        logger.info(f"Generated {len(next_generation)} variants for next generation")
        return next_generation

    def add_event_listener(self, event_type: str, callback: Callable):
        """Add event listener for evolution events"""
        self.event_listeners.append((event_type, callback))

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all registered listeners"""
        for listener_type, callback in self.event_listeners:
            if listener_type == event_type:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event listener error: {e}")
