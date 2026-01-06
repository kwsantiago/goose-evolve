"""
Evolution Engine implementation for evolve-mcp.
Central orchestrator for the genetic algorithm evolution cycle.
"""

import asyncio
import gc
import json
import logging
import math
import pickle
import random
import statistics
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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


class EvolutionStrategy(ABC):
    """Abstract base class for evolution strategies"""

    @abstractmethod
    async def evolve_generation(
        self,
        population: List[Variant],
        fitness_scores: List[float],
        config: "EvolutionConfig",
    ) -> List[Variant]:
        """Evolve a single generation using this strategy"""
        pass

    @abstractmethod
    def select_survivors(
        self, evaluated: List[Tuple[Variant, float]], config: "EvolutionConfig"
    ) -> List[Variant]:
        """Select survivors for the next generation"""
        pass

    @abstractmethod
    def should_terminate(
        self,
        generation: int,
        fitness_history: List[List[float]],
        config: "EvolutionConfig",
    ) -> bool:
        """Determine if evolution should terminate"""
        pass


class GeneticAlgorithmStrategy(EvolutionStrategy):
    """Traditional genetic algorithm strategy"""

    def __init__(self, variant_generator):
        self.variant_generator = variant_generator

    async def evolve_generation(
        self,
        population: List[Variant],
        fitness_scores: List[float],
        config: "EvolutionConfig",
    ) -> List[Variant]:
        """Evolve using GA operations"""
        # Create evaluated population
        evaluated = list(zip(population, fitness_scores))
        evaluated.sort(key=lambda x: x[1], reverse=True)

        # Select survivors
        survivors = self.select_survivors(evaluated, config)

        # Generate next generation
        next_generation = []

        # Keep elite individuals
        elite_count = min(config.elite_count, len(survivors))
        next_generation.extend(survivors[:elite_count])

        # Fill remaining population with offspring
        while len(next_generation) < config.population_size:
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
        return next_generation[: config.population_size]

    def select_survivors(
        self, evaluated: List[Tuple[Variant, float]], config: "EvolutionConfig"
    ) -> List[Variant]:
        """Tournament selection for GA"""
        survivors = []
        population = [v for v, _ in evaluated]
        fitness_map = {v.id: score for v, score in evaluated}

        # Always keep elite individuals
        elite_count = min(config.elite_count, len(evaluated))
        survivors.extend([v for v, _ in evaluated[:elite_count]])

        # Tournament selection for remaining slots
        remaining_slots = (
            int(config.population_size * config.selection_pressure) - elite_count
        )

        for _ in range(max(0, remaining_slots)):
            tournament_size = min(config.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda v: fitness_map.get(v.id, 0))
            survivors.append(winner)

        # Remove duplicates while preserving order
        seen = set()
        unique_survivors = []
        for v in survivors:
            if v.id not in seen:
                seen.add(v.id)
                unique_survivors.append(v)

        return unique_survivors

    def should_terminate(
        self,
        generation: int,
        fitness_history: List[List[float]],
        config: "EvolutionConfig",
    ) -> bool:
        """GA termination criteria"""
        # Check generation limit
        if generation >= config.max_generations:
            return True

        # Check fitness threshold
        if fitness_history and fitness_history[-1]:
            best_fitness = max(fitness_history[-1])
            if best_fitness >= config.fitness_threshold:
                return True

        return False


class EvolutionStrategyAlgorithm(EvolutionStrategy):
    """Evolution Strategy (ES) algorithm - future implementation"""

    def __init__(self, variant_generator):
        self.variant_generator = variant_generator

    async def evolve_generation(
        self,
        population: List[Variant],
        fitness_scores: List[float],
        config: "EvolutionConfig",
    ) -> List[Variant]:
        """ES evolution - placeholder for future implementation"""
        raise NotImplementedError("Evolution Strategy algorithm not yet implemented")

    def select_survivors(
        self, evaluated: List[Tuple[Variant, float]], config: "EvolutionConfig"
    ) -> List[Variant]:
        """ES selection - placeholder"""
        raise NotImplementedError("Evolution Strategy selection not yet implemented")

    def should_terminate(
        self,
        generation: int,
        fitness_history: List[List[float]],
        config: "EvolutionConfig",
    ) -> bool:
        """ES termination - placeholder"""
        raise NotImplementedError("Evolution Strategy termination not yet implemented")


@dataclass
class EventBusConfig:
    """Configuration for the event bus"""

    max_queue_size: int = 1000
    max_concurrent_handlers: int = 10
    handler_timeout: float = 30.0
    backpressure_strategy: str = "drop_oldest"  # "drop_oldest", "drop_newest", "block"


class AsyncEventBus:
    """Async event bus with backpressure handling"""

    def __init__(self, config: Optional[EventBusConfig] = None):
        self.config = config or EventBusConfig()
        self.handlers: Dict[str, List[Callable]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_handlers)
        self.running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = {
            "events_processed": 0,
            "events_dropped": 0,
            "handler_errors": 0,
            "queue_overflows": 0,
        }

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type"""
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(handler)
                if not self.handlers[event_type]:
                    del self.handlers[event_type]
            except ValueError:
                pass

    async def emit(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Emit an event with backpressure handling"""
        event = {"type": event_type, "data": data, "timestamp": datetime.now()}

        try:
            self.event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._stats["queue_overflows"] += 1

            if self.config.backpressure_strategy == "drop_oldest":
                try:
                    self.event_queue.get_nowait()  # Remove oldest event
                    self.event_queue.put_nowait(event)
                    self._stats["events_dropped"] += 1
                    return True
                except asyncio.QueueEmpty:
                    pass
            elif self.config.backpressure_strategy == "drop_newest":
                self._stats["events_dropped"] += 1
                return False
            elif self.config.backpressure_strategy == "block":
                await self.event_queue.put(event)
                return True

            return False

    async def start(self) -> None:
        """Start the event processor"""
        if not self.running:
            self.running = True
            self._processor_task = asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the event processor"""
        self.running = False
        if self._processor_task:
            await self._processor_task

    async def _process_events(self) -> None:
        """Process events from the queue"""
        while self.running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._handle_event(event)
                self.event_queue.task_done()
                self._stats["events_processed"] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle a single event with concurrency control"""
        event_type = event["type"]
        if event_type not in self.handlers:
            return

        async def handle_with_semaphore(handler):
            async with self.semaphore:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await asyncio.wait_for(
                            handler(event["data"]), timeout=self.config.handler_timeout
                        )
                    else:
                        handler(event["data"])
                except Exception as e:
                    self._stats["handler_errors"] += 1
                    logger.error(f"Event handler error for {event_type}: {e}")

        # Execute all handlers concurrently
        tasks = [
            handle_with_semaphore(handler) for handler in self.handlers[event_type]
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self._stats,
            "queue_size": self.event_queue.qsize(),
            "active_handlers": sum(
                len(handlers) for handlers in self.handlers.values()
            ),
            "subscribed_event_types": list(self.handlers.keys()),
        }


class EvolutionError(Exception):
    """Base exception for evolution operations"""

    pass


class ValidationError(EvolutionError):
    """Raised when variant validation fails"""

    pass


class SandboxError(EvolutionError):
    """Raised when sandbox testing fails"""

    pass


class FitnessEvaluationError(EvolutionError):
    """Raised when fitness evaluation fails"""

    pass


class ResourceExhaustionError(EvolutionError):
    """Raised when system resources are exhausted"""

    pass


class ConvergenceError(EvolutionError):
    """Raised when population converges without meeting criteria"""

    pass


@dataclass
class DiversityMetrics:
    """Population diversity measurements"""

    genetic_diversity: float  # 0-1, higher is more diverse
    phenotypic_diversity: float  # 0-1, based on fitness distribution
    unique_variants: int
    duplicate_ratio: float
    entropy: float


@dataclass
class EvolutionMetrics:
    """Performance metrics for evolution cycles"""

    cycle_id: UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    generation_times: List[float] = field(default_factory=list)
    validation_times: List[float] = field(default_factory=list)
    testing_times: List[float] = field(default_factory=list)
    evaluation_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    diversity_history: List[DiversityMetrics] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class CheckpointData:
    """Evolution state for persistence"""

    cycle_id: UUID
    trigger: Any  # EvolutionTriggerEvent
    current_generation: int
    population: List[Any]  # List[Variant]
    config: Any  # EvolutionConfig
    metrics: Any  # EvolutionMetrics
    fitness_history: List[List[float]]
    timestamp: datetime


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

    # New configuration options
    min_diversity_threshold: float = 0.1
    convergence_window: int = 3
    convergence_threshold: float = 0.01
    max_memory_mb: float = 1000.0
    checkpoint_interval: int = 5
    enable_auto_checkpoints: bool = True
    error_retry_attempts: int = 3
    error_retry_delay: float = 1.0


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
        config: Optional["EvolutionConfig"] = None,
        strategy: Optional["EvolutionStrategy"] = None,
    ):
        self.variant_generator = variant_generator
        self.safety_validator = safety_validator
        self.sandbox_manager = sandbox_manager
        self.fitness_evaluator = fitness_evaluator
        self.config = config or EvolutionConfig()
        self.strategy = strategy or GeneticAlgorithmStrategy(variant_generator)

        self.current_generation = 0
        self.event_listeners: List[Any] = []
        self.event_bus = AsyncEventBus()
        self._event_bus_started = False
        self.current_cycle_id: Optional[UUID] = None
        self.current_population: List[Variant] = []
        self.metrics: Optional[EvolutionMetrics] = None
        self.fitness_history: List[List[float]] = []
        self.is_cancelled: bool = False
        self._cleanup_tasks: Set[asyncio.Task] = set()

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

            # Initialize metrics tracking
            self.current_cycle_id = cycle_id
            self.metrics = EvolutionMetrics(cycle_id=cycle_id, start_time=start_time)
            self.current_population = population

            # Start event bus if not already started
            if not self._event_bus_started:
                await self.event_bus.start()
                self._event_bus_started = True

            # Main evolution loop
            best_variants = []
            avg_fitness = 0.0
            best_fitness = 0.0
            for generation in range(self.config.max_generations):
                if self.is_cancelled:
                    logger.info("Evolution cancelled by user")
                    break

                generation_start = time.time()
                self.current_generation = generation
                logger.info(f"Generation {generation}/{self.config.max_generations}")

                # Validate all variants
                validation_start = time.time()
                valid_population = await self._validate_population(population)
                self.metrics.validation_times.append(time.time() - validation_start)

                if not valid_population:
                    logger.warning("No valid variants in population")
                    break

                # Test variants in sandbox
                testing_start = time.time()
                test_results = await self._test_population(valid_population)
                self.metrics.testing_times.append(time.time() - testing_start)

                # Evaluate fitness
                evaluation_start = time.time()
                evaluated_population = await self.evaluate_population(
                    valid_population, test_results
                )
                self.metrics.evaluation_times.append(time.time() - evaluation_start)

                # Check for fitness threshold
                best_fitness = max(score for _, score in evaluated_population)
                avg_fitness = sum(score for _, score in evaluated_population) / len(
                    evaluated_population
                )

                logger.info(
                    f"Generation {generation} - Best: {best_fitness:.3f}, Avg: {avg_fitness:.3f}"
                )

                # Record fitness history
                generation_fitness = [score for _, score in evaluated_population]
                self.fitness_history.append(generation_fitness)

                # Update metrics
                generation_time = time.time() - generation_start
                self.metrics.generation_times.append(generation_time)
                diversity = self.get_diversity_metrics()
                self.metrics.diversity_history.append(diversity)
                self.metrics.convergence_history.append(best_fitness)

                # Resource cleanup
                await self._cleanup_resources()

                # Auto-checkpoint if enabled
                if (
                    self.config.enable_auto_checkpoints
                    and generation % self.config.checkpoint_interval == 0
                ):
                    checkpoint_path = f"auto_checkpoint_{cycle_id}_{generation}.pkl"
                    try:
                        await self.save_checkpoint(checkpoint_path)
                    except Exception as e:
                        logger.warning(f"Auto-checkpoint failed: {e}")

                # Check convergence
                if self._check_convergence():
                    logger.info("Population converged, ending evolution")
                    break

                # Select top performers
                best_variants = [
                    v
                    for v, s in evaluated_population
                    if s >= self.config.fitness_threshold
                ]

                # Check strategy-specific termination criteria
                if self.strategy.should_terminate(
                    generation, self.fitness_history, self.config
                ):
                    logger.info("Strategy termination criteria met, ending evolution")
                    break

                if best_variants or best_fitness >= self.config.fitness_threshold:
                    logger.info("Fitness threshold reached, ending evolution")
                    break

                # Use strategy to generate next generation
                generation_fitness = [score for _, score in evaluated_population]
                population = await self.strategy.evolve_generation(
                    [v for v, _ in evaluated_population],
                    generation_fitness,
                    self.config,
                )
                self.current_population = population

            # Finalize metrics
            if self.metrics:
                self.metrics.end_time = datetime.now()

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
            error_type = type(e).__name__
            logger.error(f"Evolution cycle failed ({error_type}): {e}")

            # Attempt recovery based on error type
            try:
                recovery_result = await self._attempt_error_recovery(e, cycle_id)
                if recovery_result:
                    logger.info("Successfully recovered from error")
                    return recovery_result
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")

            self._emit_event(
                "evolution_failed",
                {
                    "cycle_id": cycle_id,
                    "error": str(e),
                    "error_type": error_type,
                    "recovery_attempted": True,
                },
            )
            raise EvolutionError(f"Evolution cycle failed: {e}") from e

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
        population: List[Variant] = self.variant_generator.generate_population(
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
        self.event_bus.subscribe(event_type, callback)

    def remove_event_listener(self, event_type: str, callback: Callable):
        """Remove event listener"""
        try:
            self.event_listeners.remove((event_type, callback))
            self.event_bus.unsubscribe(event_type, callback)
        except ValueError:
            pass

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to all registered listeners with backpressure handling"""
        # Legacy synchronous event handling for backward compatibility
        for listener_type, callback in self.event_listeners:
            if listener_type == event_type:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Event listener error: {e}")

        # New async event bus with backpressure
        asyncio.create_task(self._emit_async_event(event_type, data))

    async def _emit_async_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event via async event bus"""
        # Ensure event bus is started
        if not self._event_bus_started:
            await self.event_bus.start()
            self._event_bus_started = True

        success = await self.event_bus.emit(event_type, data)
        if not success:
            logger.warning(f"Failed to emit event {event_type} due to backpressure")

    async def save_checkpoint(self, path: str) -> None:
        """Persist current evolution state"""
        if not self.current_cycle_id or not self.current_population:
            raise ValueError("No active evolution cycle to checkpoint")

        checkpoint_data = CheckpointData(
            cycle_id=self.current_cycle_id,
            trigger=None,  # Will be set from actual trigger
            current_generation=self.current_generation,
            population=self.current_population,
            config=self.config,
            metrics=self.metrics,
            fitness_history=self.fitness_history,
            timestamp=datetime.now(),
        )

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            raise EvolutionError(f"Failed to save checkpoint: {e}") from e

    async def resume_from_checkpoint(self, path: str) -> None:
        """Resume interrupted evolution"""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data: CheckpointData = pickle.load(f)

            self.current_cycle_id = checkpoint_data.cycle_id
            self.current_generation = checkpoint_data.current_generation
            self.current_population = checkpoint_data.population
            self.config = checkpoint_data.config
            self.metrics = checkpoint_data.metrics
            self.fitness_history = checkpoint_data.fitness_history

            logger.info(
                f"Resumed from checkpoint at generation {self.current_generation}"
            )
        except Exception as e:
            raise EvolutionError(f"Failed to resume from checkpoint: {e}") from e

    def get_diversity_metrics(self) -> DiversityMetrics:
        """Calculate population diversity"""
        if not self.current_population:
            return DiversityMetrics(
                genetic_diversity=0.0,
                phenotypic_diversity=0.0,
                unique_variants=0,
                duplicate_ratio=1.0,
                entropy=0.0,
            )

        # Calculate genetic diversity (based on prompt similarity)
        unique_prompts = set(v.prompt for v in self.current_population)
        genetic_diversity = len(unique_prompts) / len(self.current_population)

        # Calculate phenotypic diversity (based on fitness scores)
        fitness_scores = [
            v.fitness_score
            for v in self.current_population
            if v.fitness_score is not None
        ]
        if fitness_scores:
            phenotypic_diversity = (
                statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0
            )
        else:
            phenotypic_diversity = 0.0

        # Calculate entropy
        if unique_prompts:
            prompt_counts: Dict[str, int] = {}
            for variant in self.current_population:
                prompt_counts[variant.prompt] = prompt_counts.get(variant.prompt, 0) + 1

            total = len(self.current_population)
            entropy = -sum(
                (count / total) * math.log2(count / total)
                for count in prompt_counts.values()
                if count > 0
            )
        else:
            entropy = 0.0

        return DiversityMetrics(
            genetic_diversity=genetic_diversity,
            phenotypic_diversity=min(phenotypic_diversity, 1.0),
            unique_variants=len(unique_prompts),
            duplicate_ratio=1.0 - genetic_diversity,
            entropy=entropy,
        )

    async def cancel_evolution(self) -> None:
        """Gracefully stop evolution"""
        self.is_cancelled = True
        logger.info("Evolution cancellation requested")

        # Cancel all running tasks
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()

        # Clean up resources
        await self._cleanup_resources()

        # Stop event bus
        await self.event_bus.stop()

        # Save emergency checkpoint if possible
        if self.current_cycle_id and self.config.enable_auto_checkpoints:
            try:
                checkpoint_path = f"emergency_checkpoint_{self.current_cycle_id}.pkl"
                await self.save_checkpoint(checkpoint_path)
                logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

    def validate_config(self, config: "EvolutionConfig") -> ValidationResult:
        """Validate configuration parameters"""
        errors = []

        if config.population_size <= 0:
            errors.append("Population size must be positive")
        if config.tournament_size <= 0:
            errors.append("Tournament size must be positive")
        if config.tournament_size > config.population_size:
            errors.append("Tournament size cannot exceed population size")
        if not 0 <= config.selection_pressure <= 1:
            errors.append("Selection pressure must be between 0 and 1")
        if config.elite_count < 0:
            errors.append("Elite count cannot be negative")
        if config.elite_count >= config.population_size:
            errors.append("Elite count must be less than population size")
        if not 0 <= config.mutation_rate <= 1:
            errors.append("Mutation rate must be between 0 and 1")
        if config.max_generations <= 0:
            errors.append("Max generations must be positive")
        if not 0 <= config.fitness_threshold <= 1:
            errors.append("Fitness threshold must be between 0 and 1")
        if config.parallel_evaluations <= 0:
            errors.append("Parallel evaluations must be positive")
        if not 0 <= config.min_diversity_threshold <= 1:
            errors.append("Min diversity threshold must be between 0 and 1")
        if config.convergence_window <= 0:
            errors.append("Convergence window must be positive")
        if config.max_memory_mb <= 0:
            errors.append("Max memory limit must be positive")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def _attempt_error_recovery(
        self, error: Exception, cycle_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover from evolution errors"""
        for attempt in range(self.config.error_retry_attempts):
            logger.info(
                f"Recovery attempt {attempt + 1}/{self.config.error_retry_attempts}"
            )

            try:
                await asyncio.sleep(self.config.error_retry_delay * (attempt + 1))

                if isinstance(error, ValidationError):
                    # Generate new population if validation failed
                    base_agent = await self._get_base_agent()
                    self.current_population = await self._generate_population(
                        base_agent
                    )
                    return None  # Continue with new population

                elif isinstance(error, ResourceExhaustionError):
                    # Clean up resources and reduce population
                    await self._cleanup_resources()
                    self.config.population_size = max(
                        10, self.config.population_size // 2
                    )
                    logger.info(
                        f"Reduced population size to {self.config.population_size}"
                    )
                    return None

                elif isinstance(error, SandboxError):
                    # Retry with reduced parallelism
                    self.config.parallel_evaluations = max(
                        1, self.config.parallel_evaluations // 2
                    )
                    logger.info(
                        f"Reduced parallel evaluations to {self.config.parallel_evaluations}"
                    )
                    return None

            except Exception as recovery_error:
                logger.warning(
                    f"Recovery attempt {attempt + 1} failed: {recovery_error}"
                )
                continue

        return None  # Recovery failed

    async def _cleanup_resources(self) -> None:
        """Clean up memory and resources after each generation"""
        # Clear completed tasks
        self._cleanup_tasks = {t for t in self._cleanup_tasks if not t.done()}

        # Force garbage collection
        gc.collect()

        # Check memory usage
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            logger.warning("psutil not available, skipping memory monitoring")
            memory_mb = 0.0

        if memory_mb > self.config.max_memory_mb:
            logger.warning(
                f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.config.max_memory_mb}MB)"
            )
            raise ResourceExhaustionError(f"Memory limit exceeded: {memory_mb:.1f}MB")

        if self.metrics:
            self.metrics.memory_usage.append(memory_mb)

    def get_event_bus_stats(self) -> Dict[str, Any]:
        """Get event bus performance statistics"""
        return self.event_bus.get_stats()

    def _check_convergence(self) -> bool:
        """Check if population has converged"""
        if len(self.fitness_history) < self.config.convergence_window:
            return False

        recent_generations = self.fitness_history[-self.config.convergence_window :]

        # Check if best fitness has plateaued
        best_fitnesses = [max(generation) for generation in recent_generations]
        fitness_variance = (
            statistics.variance(best_fitnesses)
            if len(best_fitnesses) > 1
            else float("inf")
        )

        # Check diversity
        diversity = self.get_diversity_metrics()

        is_converged = (
            fitness_variance < self.config.convergence_threshold
            and diversity.genetic_diversity < self.config.min_diversity_threshold
        )

        if is_converged:
            logger.info(
                f"Convergence detected: fitness_variance={fitness_variance:.4f}, diversity={diversity.genetic_diversity:.4f}"
            )

        return is_converged

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the current cycle"""
        if not self.metrics:
            return {}

        total_duration = 0.0
        if self.metrics.end_time:
            total_duration = (
                self.metrics.end_time - self.metrics.start_time
            ).total_seconds()

        return {
            "cycle_id": str(self.metrics.cycle_id),
            "total_duration_seconds": total_duration,
            "generations_completed": self.current_generation,
            "avg_generation_time": (
                statistics.mean(self.metrics.generation_times)
                if self.metrics.generation_times
                else 0.0
            ),
            "avg_validation_time": (
                statistics.mean(self.metrics.validation_times)
                if self.metrics.validation_times
                else 0.0
            ),
            "avg_testing_time": (
                statistics.mean(self.metrics.testing_times)
                if self.metrics.testing_times
                else 0.0
            ),
            "avg_evaluation_time": (
                statistics.mean(self.metrics.evaluation_times)
                if self.metrics.evaluation_times
                else 0.0
            ),
            "peak_memory_usage_mb": (
                max(self.metrics.memory_usage) if self.metrics.memory_usage else 0.0
            ),
            "avg_memory_usage_mb": (
                statistics.mean(self.metrics.memory_usage)
                if self.metrics.memory_usage
                else 0.0
            ),
            "final_diversity": (
                self.get_diversity_metrics().__dict__
                if self.current_population
                else None
            ),
            "convergence_achieved": self._check_convergence(),
            "fitness_improvement": self._calculate_fitness_improvement(),
        }

    def _calculate_fitness_improvement(self) -> float:
        """Calculate overall fitness improvement from start to current generation"""
        if len(self.fitness_history) < 2:
            return 0.0

        initial_best = max(self.fitness_history[0]) if self.fitness_history[0] else 0.0
        current_best = (
            max(self.fitness_history[-1]) if self.fitness_history[-1] else 0.0
        )

        return current_best - initial_best
