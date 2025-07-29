"""
Unit tests for EvolutionEngine.
Tests core evolution functionality with mocked components.
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from evolution.engine import EvolutionConfig, EvolutionEngine, EvolutionResult
from evolution.interfaces import (
    EvolutionTriggerEvent,
    TestResults,
    ValidationResult,
    Variant,
)


class MockVariantGenerator:
    """Mock variant generator for testing"""

    def generate_population(self, base_prompt: str, size: int) -> List[Variant]:
        """Generate mock population"""
        return [
            Variant(
                id=uuid4(),
                parent_ids=[],
                generation=0,
                prompt=f"{base_prompt} - variant {i}",
                configuration={"temperature": 0.7 + i * 0.01},
                created_at=datetime.now(),
            )
            for i in range(size)
        ]

    def crossover(self, p1: Variant, p2: Variant) -> Tuple[Variant, Variant]:
        """Mock crossover"""
        child1 = Variant(
            id=uuid4(),
            parent_ids=[p1.id, p2.id],
            generation=max(p1.generation, p2.generation) + 1,
            prompt=f"Crossover of {p1.id} and {p2.id}",
            configuration=p1.configuration,
            created_at=datetime.now(),
        )
        child2 = Variant(
            id=uuid4(),
            parent_ids=[p1.id, p2.id],
            generation=max(p1.generation, p2.generation) + 1,
            prompt=f"Crossover 2 of {p1.id} and {p2.id}",
            configuration=p2.configuration,
            created_at=datetime.now(),
        )
        return (child1, child2)

    def mutate_variant(self, variant: Variant) -> Variant:
        """Mock mutation"""
        return Variant(
            id=uuid4(),
            parent_ids=[variant.id],
            generation=variant.generation + 1,
            prompt=f"{variant.prompt} - mutated",
            configuration=variant.configuration,
            created_at=datetime.now(),
        )


class MockSafetyValidator:
    """Mock safety validator for testing"""

    def __init__(self, valid_ratio: float = 0.9):
        self.valid_ratio = valid_ratio
        self.call_count = 0

    async def validate(self, variant: Variant) -> ValidationResult:
        """Mock validation - some variants fail"""
        self.call_count += 1
        # Use simple modulo to achieve the valid ratio
        random.seed(hash(variant.id))
        is_valid = random.random() < self.valid_ratio
        return ValidationResult(
            is_valid=is_valid, errors=[] if is_valid else ["Mock validation error"]
        )


class MockSandboxManager:
    """Mock sandbox manager for testing"""

    async def test_variant(self, variant: Variant) -> TestResults:
        """Mock testing - returns varied results"""
        # Simulate some randomness in test results
        random.seed(hash(variant.id))

        return TestResults(
            variant_id=variant.id,
            success_rate=random.uniform(0.6, 1.0),
            avg_response_time=random.uniform(0.1, 2.0),
            error_count=random.randint(0, 3),
            resource_usage={
                "memory": random.uniform(100, 500),
                "cpu": random.uniform(10, 90),
            },
        )


class MockFitnessEvaluator:
    """Mock fitness evaluator for testing"""

    async def evaluate(self, variant: Variant, results: TestResults) -> float:
        """Mock fitness evaluation"""
        # Simple fitness based on success rate and response time
        fitness = results.success_rate * 0.7
        fitness += (2.0 - results.avg_response_time) / 2.0 * 0.3
        fitness -= results.error_count * 0.1
        return max(0.0, min(1.0, fitness))


@pytest.fixture
def evolution_config():
    """Test configuration"""
    return EvolutionConfig(
        population_size=10,
        tournament_size=3,
        selection_pressure=0.8,
        elite_count=2,
        mutation_rate=0.1,
        max_generations=3,
        fitness_threshold=0.95,
        parallel_evaluations=5,
    )


@pytest.fixture
def evolution_engine(evolution_config):
    """Create engine with mocked dependencies"""
    return EvolutionEngine(
        variant_generator=MockVariantGenerator(),
        safety_validator=MockSafetyValidator(),
        sandbox_manager=MockSandboxManager(),
        fitness_evaluator=MockFitnessEvaluator(),
        config=evolution_config,
    )


@pytest.fixture
def evolution_trigger():
    """Test trigger event"""
    return EvolutionTriggerEvent(
        trigger_type="threshold",
        metrics_snapshot={"success_rate": 0.75, "avg_response_time": 1.5},
        timestamp=datetime.now(),
    )


@pytest.mark.asyncio
async def test_evolution_cycle_success(evolution_engine, evolution_trigger):
    """Test successful evolution cycle"""
    result = await evolution_engine.start_evolution_cycle(evolution_trigger)

    assert isinstance(result, dict)
    assert result["trigger"] == evolution_trigger
    assert result["population_size"] == 10
    assert result["generation"] <= 3  # Should not exceed max_generations
    assert 0 <= result["avg_fitness"] <= 1.0
    assert 0 <= result["best_fitness"] <= 1.0
    assert result["best_fitness"] >= result["avg_fitness"]
    assert result["duration_seconds"] > 0


@pytest.mark.asyncio
async def test_evaluate_population(evolution_engine):
    """Test population evaluation"""
    # Create test population
    population = [
        Variant(
            id=uuid4(),
            parent_ids=[],
            generation=0,
            prompt=f"Test variant {i}",
            configuration={},
            created_at=datetime.now(),
        )
        for i in range(5)
    ]

    # Create test results
    test_results = {
        v.id: TestResults(
            variant_id=v.id,
            success_rate=0.8 + i * 0.02,
            avg_response_time=1.0 - i * 0.1,
            error_count=0,
            resource_usage={},
        )
        for i, v in enumerate(population)
    }

    evaluated = await evolution_engine.evaluate_population(population, test_results)

    assert len(evaluated) == 5
    assert all(isinstance(item, tuple) for item in evaluated)
    assert all(isinstance(item[0], Variant) for item in evaluated)
    assert all(isinstance(item[1], float) for item in evaluated)
    assert all(0 <= score <= 1.0 for _, score in evaluated)

    # Check sorted by fitness (descending)
    scores = [score for _, score in evaluated]
    assert scores == sorted(scores, reverse=True)


def test_select_survivors(evolution_engine):
    """Test survivor selection"""
    # Create evaluated population
    population = [
        Variant(
            id=uuid4(),
            parent_ids=[],
            generation=0,
            prompt=f"Variant {i}",
            configuration={},
            created_at=datetime.now(),
        )
        for i in range(10)
    ]

    evaluated = [(v, 0.5 + i * 0.05) for i, v in enumerate(population)]
    # Sort by fitness descending (as expected by select_survivors)
    evaluated.sort(key=lambda x: x[1], reverse=True)

    survivors = evolution_engine.select_survivors(evaluated)

    assert len(survivors) > 0
    assert len(survivors) <= 10

    # Check elite individuals are preserved (first 2 after sorting)
    elite_ids = {v.id for v, _ in evaluated[:2]}  # Top 2
    survivor_ids = {v.id for v in survivors}
    assert elite_ids.issubset(survivor_ids)


@pytest.mark.asyncio
async def test_event_emission(evolution_engine, evolution_trigger):
    """Test event emission during evolution"""
    events = []

    def capture_event(data):
        events.append(data)

    evolution_engine.add_event_listener("evolution_started", capture_event)
    evolution_engine.add_event_listener("evolution_completed", capture_event)

    result = await evolution_engine.start_evolution_cycle(evolution_trigger)

    assert len(events) == 2
    assert events[0]["trigger"] == evolution_trigger
    assert events[1]["result"] == result


@pytest.mark.asyncio
async def test_fitness_threshold_early_stop(evolution_engine, evolution_trigger):
    """Test early stopping when fitness threshold is reached"""

    # Mock fitness evaluator to return high scores
    async def high_fitness_evaluate(variant, results):
        return 0.96  # Above threshold

    evolution_engine.fitness_evaluator.evaluate = high_fitness_evaluate

    result = await evolution_engine.start_evolution_cycle(evolution_trigger)

    assert result["generation"] < 3  # Should stop early
    assert result["best_fitness"] >= 0.95
    assert len(result["winners"]) > 0


@pytest.mark.asyncio
async def test_no_valid_variants(evolution_engine, evolution_trigger):
    """Test handling when no variants pass validation"""
    # Make all variants invalid
    evolution_engine.safety_validator = MockSafetyValidator(valid_ratio=0.0)

    result = await evolution_engine.start_evolution_cycle(evolution_trigger)

    assert result["generation"] == 0
    assert len(result["winners"]) == 0


@pytest.mark.asyncio
async def test_parallel_evaluation(evolution_engine):
    """Test parallel evaluation respects semaphore limit"""
    call_times = []

    async def slow_evaluate(variant, results):
        start = datetime.now()
        await asyncio.sleep(0.1)
        call_times.append((start, datetime.now()))
        return 0.5

    evolution_engine.fitness_evaluator.evaluate = slow_evaluate
    evolution_engine.config.parallel_evaluations = 2

    # Create small population
    population = [
        Variant(
            id=uuid4(),
            parent_ids=[],
            generation=0,
            prompt=f"Variant {i}",
            configuration={},
            created_at=datetime.now(),
        )
        for i in range(4)
    ]

    test_results = {v.id: Mock(spec=TestResults) for v in population}

    await evolution_engine.evaluate_population(population, test_results)

    # Check that at most 2 evaluations run concurrently
    concurrent_count = 0
    max_concurrent = 0

    for start1, end1 in call_times:
        concurrent = sum(
            1
            for start2, end2 in call_times
            if start2 <= start1 <= end2 and (start1, end1) != (start2, end2)
        )
        max_concurrent = max(max_concurrent, concurrent + 1)

    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_error_handling(evolution_engine, evolution_trigger):
    """Test error handling during evolution"""

    # Make fitness evaluator raise an error to test error propagation
    async def failing_evaluate(variant, results):
        raise Exception("Evaluation failure")

    evolution_engine.fitness_evaluator.evaluate = failing_evaluate

    with pytest.raises(Exception):
        await evolution_engine.start_evolution_cycle(evolution_trigger)


def test_configuration_defaults():
    """Test configuration default values"""
    config = EvolutionConfig()

    assert config.population_size == 50
    assert config.tournament_size == 3
    assert config.selection_pressure == 0.8
    assert config.elite_count == 5
    assert config.mutation_rate == 0.1
    assert config.max_generations == 10
    assert config.fitness_threshold == 0.95
    assert config.parallel_evaluations == 10


@pytest.mark.asyncio
async def test_config_validation(evolution_engine):
    """Test configuration validation"""
    # Valid config
    valid_config = EvolutionConfig(population_size=10, tournament_size=3)
    result = evolution_engine.validate_config(valid_config)
    assert result.is_valid
    assert len(result.errors) == 0
    
    # Invalid config
    invalid_config = EvolutionConfig(population_size=-1, tournament_size=0)
    result = evolution_engine.validate_config(invalid_config)
    assert not result.is_valid
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_diversity_metrics(evolution_engine):
    """Test population diversity calculation"""
    from uuid import uuid4
    from datetime import datetime
    
    # Create population with some diversity
    population = [
        Variant(uuid4(), [], 0, "prompt 1", {}, datetime.now(), fitness_score=0.8),
        Variant(uuid4(), [], 0, "prompt 2", {}, datetime.now(), fitness_score=0.7),
        Variant(uuid4(), [], 0, "prompt 1", {}, datetime.now(), fitness_score=0.9),  # duplicate
    ]
    evolution_engine.current_population = population
    
    diversity = evolution_engine.get_diversity_metrics()
    assert 0 <= diversity.genetic_diversity <= 1
    assert diversity.unique_variants == 2  # Two unique prompts
    assert diversity.duplicate_ratio > 0


@pytest.mark.asyncio
async def test_checkpoint_functionality(evolution_engine):
    """Test checkpoint save and resume"""
    import tempfile
    import os
    from uuid import uuid4
    from datetime import datetime
    
    # Set up some state
    evolution_engine.current_cycle_id = uuid4()
    evolution_engine.current_generation = 2
    evolution_engine.current_population = [Variant(uuid4(), [], 0, "test", {}, datetime.now())]
    
    with tempfile.NamedTemporaryFile(delete=False) as f:
        checkpoint_path = f.name
    
    try:
        # Save checkpoint
        await evolution_engine.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path)
        
        # Create new engine and resume
        new_engine = EvolutionEngine(
            MockVariantGenerator(),
            MockSafetyValidator(),
            MockSandboxManager(),
            MockFitnessEvaluator()
        )
        
        await new_engine.resume_from_checkpoint(checkpoint_path)
        
        # Verify state was restored
        assert new_engine.current_cycle_id == evolution_engine.current_cycle_id
        assert new_engine.current_generation == evolution_engine.current_generation
        assert len(new_engine.current_population) == len(evolution_engine.current_population)
        
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)
