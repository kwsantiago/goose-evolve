"""Basic integration test to verify components work together."""

import asyncio
import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock

from evolution.engine import EvolutionEngine, EvolutionConfig, GeneticAlgorithmStrategy
from evolution.fitness import FitnessEvaluator, TestResults
from evolution.variants import VariantGenerator, Variant
from evolution.interfaces import EvolutionTriggerEvent, ValidationResult


class TestBasicIntegration:
    """Basic integration tests to verify setup."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test that evolution engine can be initialized."""
        # Create components
        variant_generator = VariantGenerator()
        fitness_evaluator = FitnessEvaluator()
        
        # Mock components
        mock_safety_validator = AsyncMock()
        mock_safety_validator.validate.return_value = ValidationResult(
            is_valid=True, errors=[]
        )
        
        mock_sandbox_manager = AsyncMock()
        
        # Create config
        config = EvolutionConfig(
            population_size=10,
            mutation_rate=0.1,
            max_generations=2
        )
        
        # Initialize engine
        engine = EvolutionEngine(
            variant_generator=variant_generator,
            safety_validator=mock_safety_validator,
            sandbox_manager=mock_sandbox_manager,
            fitness_evaluator=fitness_evaluator,
            config=config,
            strategy=GeneticAlgorithmStrategy(variant_generator)
        )
        
        assert engine is not None
        assert engine.config.population_size == 10
    
    @pytest.mark.asyncio
    async def test_simple_evolution_cycle(self):
        """Test a simple evolution cycle."""
        # Create components
        variant_generator = VariantGenerator()
        fitness_evaluator = FitnessEvaluator()
        
        # Mock components
        mock_safety_validator = AsyncMock()
        mock_safety_validator.validate.return_value = ValidationResult(
            is_valid=True, errors=[]
        )
        
        mock_sandbox_manager = AsyncMock()
        # Return a simple test result
        mock_sandbox_manager.test_variant.return_value = TestResults(
            variant_id=uuid.uuid4(),
            success_rate=0.95,
            avg_response_time=1.0,
            error_count=0,
            resource_usage={"memory_mb": 100}
        )
        
        # Create config
        config = EvolutionConfig(
            population_size=5,
            mutation_rate=0.1,
            max_generations=1  # Just one generation
        )
        
        # Initialize engine
        engine = EvolutionEngine(
            variant_generator=variant_generator,
            safety_validator=mock_safety_validator,
            sandbox_manager=mock_sandbox_manager,
            fitness_evaluator=fitness_evaluator,
            config=config,
            strategy=GeneticAlgorithmStrategy(variant_generator)
        )
        
        # Create trigger
        trigger = EvolutionTriggerEvent(
            trigger_type="manual",
            metrics_snapshot={},
            timestamp=datetime.now()
        )
        
        # Run evolution with timeout
        try:
            result = await asyncio.wait_for(
                engine.start_evolution_cycle(trigger),
                timeout=10.0  # 10 second timeout
            )
            print(f"Evolution result: {result}")
            assert result["status"] in ["completed", "converged"]
        except asyncio.TimeoutError:
            pytest.fail("Evolution cycle timed out after 10 seconds")