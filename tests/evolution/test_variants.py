"""
Unit tests for VariantGenerator class.
Tests prompt mutations, crossover operations, and validation.
"""

import re
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from evolution.interfaces import CrossoverType, PromptMutationType, Variant
from evolution.variants import (
    ComplexityMetrics,
    MutationEvent,
    PromptValidator,
    VariantGenerator,
)


class TestVariantGenerator:
    """Test suite for VariantGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VariantGenerator(mutation_rate=0.2)
        self.base_prompt = (
            "You are a helpful assistant. Please provide accurate information."
        )

    def test_initialization(self):
        """Test VariantGenerator initialization."""
        generator = VariantGenerator(mutation_rate=0.3)
        assert generator.mutation_rate == 0.3
        assert "instruction_prefixes" in generator.mutation_templates
        assert "cot_triggers" in generator.mutation_templates
        assert generator.generation_stats["total_generated"] == 0

    def test_generate_population_basic(self):
        """Test basic population generation."""
        population = self.generator.generate_population(self.base_prompt, 10)

        assert len(population) == 10
        assert all(isinstance(v, Variant) for v in population)
        assert population[0].prompt == self.base_prompt  # First should be base
        assert all(v.generation == 0 for v in population)
        assert self.generator.generation_stats["total_generated"] == 10

    def test_generate_population_mutations(self):
        """Test that generated population contains mutations."""
        population = self.generator.generate_population(self.base_prompt, 5)

        # At least some variants should be different from base
        unique_prompts = set(v.prompt for v in population)
        assert len(unique_prompts) > 1, "Population should contain mutations"

    def test_mutate_prompt_paraphrase(self):
        """Test paraphrase mutation."""
        mutated = self.generator.mutate_prompt(
            self.base_prompt, PromptMutationType.PARAPHRASE
        )

        # Should be different (with high probability)
        assert isinstance(mutated, str)
        assert len(mutated) > 0

    def test_mutate_prompt_instruction_add(self):
        """Test instruction addition mutation."""
        mutated = self.generator.mutate_prompt(
            self.base_prompt, PromptMutationType.INSTRUCTION_ADD
        )

        assert isinstance(mutated, str)
        assert len(mutated) >= len(self.base_prompt)

        # Should contain one of the instruction templates
        instruction_found = any(
            instr in mutated
            for instr in self.generator.mutation_templates["instruction_additions"]
        )
        assert instruction_found, "Should contain instruction addition"

    def test_mutate_prompt_context_expand(self):
        """Test context expansion mutation."""
        mutated = self.generator.mutate_prompt(
            self.base_prompt, PromptMutationType.CONTEXT_EXPAND
        )

        assert isinstance(mutated, str)
        assert len(mutated) > len(self.base_prompt)

        # Should contain context expansion
        expansion_found = any(
            exp in mutated
            for exp in self.generator.mutation_templates["context_expansions"]
        )
        assert expansion_found, "Should contain context expansion"

    def test_mutate_prompt_cot_injection(self):
        """Test chain-of-thought injection mutation."""
        mutated = self.generator.mutate_prompt(
            self.base_prompt, PromptMutationType.COT_INJECTION
        )

        assert isinstance(mutated, str)
        assert len(mutated) > len(self.base_prompt)

        # Should contain CoT trigger
        cot_found = any(
            cot in mutated for cot in self.generator.mutation_templates["cot_triggers"]
        )
        assert cot_found, "Should contain CoT trigger"

    def test_mutate_prompt_tone_shift(self):
        """Test tone shift mutation."""
        mutated = self.generator.mutate_prompt(
            self.base_prompt, PromptMutationType.TONE_SHIFT
        )

        assert isinstance(mutated, str)
        assert len(mutated) > len(self.base_prompt)

        # Should contain tone modifier
        tone_found = False
        for tone_list in self.generator.mutation_templates["tone_modifiers"].values():
            if any(modifier in mutated for modifier in tone_list):
                tone_found = True
                break
        assert tone_found, "Should contain tone modifier"

    def test_mutate_prompt_exception_handling(self):
        """Test mutation exception handling."""
        # Mock a mutation method to raise exception
        with patch.object(
            self.generator, "_paraphrase_mutation", side_effect=Exception("Test error")
        ):
            mutated = self.generator.mutate_prompt(
                self.base_prompt, PromptMutationType.PARAPHRASE
            )
            # Should return original prompt on exception
            assert mutated == self.base_prompt

    def test_crossover_basic(self):
        """Test basic crossover operation."""
        parent1 = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt="First sentence. Second sentence. Third sentence.",
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        parent2 = Variant(
            id=UUID("87654321-4321-8765-4321-876543218765"),
            parent_ids=[],
            generation=0,
            prompt="Alpha sentence. Beta sentence. Gamma sentence.",
            configuration={"temperature": 0.9},
            created_at=datetime.now(),
        )

        offspring1, offspring2 = self.generator.crossover(parent1, parent2)

        # Check offspring properties
        assert isinstance(offspring1, Variant)
        assert isinstance(offspring2, Variant)
        assert offspring1.generation == 1
        assert offspring2.generation == 1
        assert parent1.id in offspring1.parent_ids
        assert parent2.id in offspring1.parent_ids

    def test_crossover_empty_sentences(self):
        """Test crossover with problematic sentence splitting."""
        parent1 = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt="Short",
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        parent2 = Variant(
            id=UUID("87654321-4321-8765-4321-876543218765"),
            parent_ids=[],
            generation=0,
            prompt="",
            configuration={"temperature": 0.9},
            created_at=datetime.now(),
        )

        # Should handle gracefully
        offspring1, offspring2 = self.generator.crossover(parent1, parent2)
        assert isinstance(offspring1, Variant)
        assert isinstance(offspring2, Variant)

    def test_mutate_variant(self):
        """Test variant mutation."""
        original = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt=self.base_prompt,
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        mutated = self.generator.mutate_variant(original)

        assert mutated.id != original.id
        assert original.id in mutated.parent_ids
        assert mutated.generation == 1
        assert self.generator.generation_stats["mutations_applied"] == 1

    def test_validate_variant_valid(self):
        """Test validation of valid variant."""
        variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt=self.base_prompt,
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(variant)
        assert is_valid
        assert len(errors) == 0

    def test_validate_variant_too_short(self):
        """Test validation of too-short prompt."""
        variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt="Hi",  # Too short
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(variant)
        assert not is_valid
        assert any("too short" in error.lower() for error in errors)

    def test_validate_variant_too_long(self):
        """Test validation of too-long prompt."""
        variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt="x" * 6000,  # Too long
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(variant)
        assert not is_valid
        assert any("too long" in error.lower() for error in errors)

    def test_validate_variant_forbidden_patterns(self):
        """Test validation with forbidden patterns."""
        variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt="This contains <script>alert('xss')</script> malicious code.",
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(variant)
        assert not is_valid
        assert any("forbidden pattern" in error.lower() for error in errors)

    def test_validate_configuration_invalid_temperature(self):
        """Test configuration validation with invalid temperature."""
        variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt=self.base_prompt,
            configuration={"temperature": 5.0},  # Invalid
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(variant)
        assert not is_valid
        assert any("temperature" in error.lower() for error in errors)

    def test_validate_configuration_invalid_max_tokens(self):
        """Test configuration validation with invalid max_tokens."""
        variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt=self.base_prompt,
            configuration={"max_tokens": -1},  # Invalid
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(variant)
        assert not is_valid
        assert any("max_tokens" in error.lower() for error in errors)

    def test_filter_valid_variants(self):
        """Test filtering of variant list."""
        valid_variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt=self.base_prompt,
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        invalid_variant = Variant(
            id=UUID("87654321-4321-8765-4321-876543218765"),
            parent_ids=[],
            generation=0,
            prompt="Hi",  # Too short
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        variants = [valid_variant, invalid_variant]
        valid_only = self.generator.filter_valid_variants(variants)

        assert len(valid_only) == 1
        assert valid_only[0].id == valid_variant.id

    def test_mutation_rate_configuration(self):
        """Test mutation rate configuration methods."""
        # Test setting valid rate
        self.generator.set_mutation_rate(0.5)
        assert self.generator.get_mutation_rate() == 0.5

        # Test invalid rates
        with pytest.raises(ValueError):
            self.generator.set_mutation_rate(-0.1)

        with pytest.raises(ValueError):
            self.generator.set_mutation_rate(1.5)

    def test_validation_config_update(self):
        """Test validation configuration updates."""
        new_config = {"min_prompt_length": 20, "max_prompt_length": 1000}

        self.generator.update_validation_config(new_config)
        config = self.generator.get_validation_config()

        assert config["min_prompt_length"] == 20
        assert config["max_prompt_length"] == 1000

        # Test invalid key
        with pytest.raises(ValueError):
            self.generator.update_validation_config({"invalid_key": "value"})

    def test_statistics_tracking(self):
        """Test generation statistics tracking."""
        # Reset stats
        self.generator.reset_stats()
        stats = self.generator.get_stats()
        assert all(count == 0 for count in stats.values())

        # Generate population
        population = self.generator.generate_population(self.base_prompt, 5)
        stats = self.generator.get_stats()
        assert stats["total_generated"] == 5

        # Perform mutation
        self.generator.mutate_variant(population[0])
        stats = self.generator.get_stats()
        assert stats["mutations_applied"] == 1

        # Perform crossover
        self.generator.crossover(population[0], population[1])
        stats = self.generator.get_stats()
        assert stats["crossovers_performed"] == 1

    def test_split_sentences(self):
        """Test sentence splitting utility."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.generator._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_combine_configurations(self):
        """Test configuration combination for crossover."""
        config1 = {"temperature": 0.7, "max_tokens": 100}
        config2 = {"temperature": 0.9, "top_p": 0.8}

        combined = self.generator._combine_configurations(config1, config2)

        # Should have all keys
        assert "temperature" in combined
        assert "max_tokens" in combined
        assert "top_p" in combined

        # Temperature should be averaged
        assert combined["temperature"] == 0.8

    def test_mutate_configuration(self):
        """Test configuration mutation."""
        original_config = {"temperature": 0.7}
        mutated_config = self.generator._mutate_configuration(original_config)

        # Should still have temperature
        assert "temperature" in mutated_config

        # Temperature should be different (with high probability)
        # Note: This test might occasionally fail due to randomness
        assert isinstance(mutated_config["temperature"], (int, float))
        assert 0.0 <= mutated_config["temperature"] <= 2.0

    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty prompt
        population = self.generator.generate_population("", 1)
        assert len(population) == 1

        # Single character prompt
        population = self.generator.generate_population("X", 1)
        assert len(population) == 1

        # Very long prompt
        long_prompt = "This is a test. " * 100
        population = self.generator.generate_population(long_prompt, 1)
        assert len(population) == 1


class TestMutationMethods:
    """Test individual mutation methods in isolation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VariantGenerator()
        self.test_prompt = "You are a helpful assistant."

    def test_paraphrase_mutation_variations(self):
        """Test paraphrase mutation produces variations."""
        results = set()

        # Run multiple times to test randomness
        for _ in range(20):
            mutated = self.generator._paraphrase_mutation(self.test_prompt)
            results.add(mutated)

        # Should produce some variations
        assert len(results) > 1, "Paraphrase should produce variations"

    def test_instruction_add_positions(self):
        """Test instruction addition at different positions."""
        # Test multiple times to hit different positions
        results = []
        for _ in range(10):
            mutated = self.generator._instruction_add_mutation(self.test_prompt)
            results.append(mutated)

        # All should be different from original
        assert all(result != self.test_prompt for result in results)

    def test_context_expand_consistency(self):
        """Test context expansion consistency."""
        mutated = self.generator._context_expand_mutation(self.test_prompt)

        # Should start with context expansion
        assert any(
            mutated.startswith(exp)
            for exp in self.generator.mutation_templates["context_expansions"]
        )

    def test_tone_shift_all_tones(self):
        """Test that all tone types can be applied."""
        tone_types = list(self.generator.mutation_templates["tone_modifiers"].keys())

        for tone in tone_types:
            # Temporarily modify to test specific tone
            original_choice = __import__("random").choice

            def mock_choice(seq):
                if seq == tone_types:
                    return tone
                return original_choice(seq)

            with patch("random.choice", side_effect=mock_choice):
                mutated = self.generator._tone_shift_mutation(self.test_prompt)

                # Should contain modifier from this tone
                tone_found = any(
                    modifier in mutated
                    for modifier in self.generator.mutation_templates["tone_modifiers"][
                        tone
                    ]
                )
                assert tone_found, f"Should contain {tone} tone modifier"


class TestVariantGeneratorAdvanced:
    """Test suite for advanced VariantGenerator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = VariantGenerator(mutation_rate=0.2)
        self.base_prompt = (
            "You are a helpful assistant. Please provide accurate information."
        )
        self.base_variant = Variant(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            parent_ids=[],
            generation=0,
            prompt=self.base_prompt,
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

    def test_deterministic_mode_basic(self):
        """Test basic deterministic mode functionality."""
        # Initially not deterministic
        assert not self.generator.is_deterministic()
        assert self.generator.get_random_seed() is None

        # Set deterministic mode
        self.generator.set_random_seed(42)
        assert self.generator.is_deterministic()
        assert self.generator.get_random_seed() == 42

        # Disable deterministic mode
        self.generator.disable_deterministic_mode()
        assert not self.generator.is_deterministic()
        assert self.generator.get_random_seed() is None

    def test_deterministic_mode_reproducibility(self):
        """Test that deterministic mode produces reproducible results."""
        seed = 123

        # Generate population with seed
        self.generator.set_random_seed(seed)
        population1 = self.generator.generate_population(self.base_prompt, 5)

        # Generate again with same seed
        self.generator.set_random_seed(seed)
        population2 = self.generator.generate_population(self.base_prompt, 5)

        # Should be identical
        assert len(population1) == len(population2)
        for v1, v2 in zip(population1, population2):
            assert v1.prompt == v2.prompt
            assert v1.configuration == v2.configuration

    def test_deterministic_mode_mutations(self):
        """Test deterministic mutations."""
        seed = 456

        # Mutate with seed
        self.generator.set_random_seed(seed)
        mutated1 = self.generator.mutate_variant(self.base_variant)

        # Mutate again with same seed
        self.generator.set_random_seed(seed)
        mutated2 = self.generator.mutate_variant(self.base_variant)

        # Should be identical (except for UUIDs and timestamps)
        assert mutated1.prompt == mutated2.prompt
        assert mutated1.configuration == mutated2.configuration

    def test_deterministic_mode_invalid_seed(self):
        """Test error handling for invalid seeds."""
        with pytest.raises(ValueError):
            self.generator.set_random_seed("not_an_int")

        with pytest.raises(ValueError):
            self.generator.set_random_seed(3.14)

    def test_mutation_history_tracking(self):
        """Test mutation history tracking."""
        # Initially no history
        history = self.generator.get_mutation_history(self.base_variant)
        assert len(history) == 0

        # Perform mutation
        mutated = self.generator.mutate_variant(self.base_variant)

        # Check history was recorded
        history = self.generator.get_mutation_history(mutated)
        assert len(history) == 1
        assert isinstance(history[0], MutationEvent)
        assert history[0].source_variant_id == self.base_variant.id
        assert history[0].target_variant_id == mutated.id
        assert history[0].mutation_type in [mt.value for mt in PromptMutationType]

    def test_mutation_history_crossover(self):
        """Test mutation history tracking for crossover."""
        parent2 = Variant(
            id=UUID("87654321-4321-8765-4321-876543218765"),
            parent_ids=[],
            generation=0,
            prompt="Different prompt for testing.",
            configuration={"temperature": 0.9},
            created_at=datetime.now(),
        )

        offspring1, offspring2 = self.generator.crossover(self.base_variant, parent2)

        # Check crossover history
        history1 = self.generator.get_mutation_history(offspring1)
        history2 = self.generator.get_mutation_history(offspring2)

        assert len(history1) == 1
        assert len(history2) == 1
        assert "crossover" in history1[0].mutation_type
        assert "crossover" in history2[0].mutation_type

    def test_complexity_analysis_basic(self):
        """Test basic prompt complexity analysis."""
        simple_prompt = "Hello world."
        metrics = self.generator.analyze_prompt_complexity(simple_prompt)

        assert isinstance(metrics, ComplexityMetrics)
        assert metrics.sentence_count == 1
        assert metrics.word_count == 2
        assert metrics.avg_sentence_length == 2.0
        assert 0.0 <= metrics.lexical_diversity <= 1.0
        assert 0.0 <= metrics.complexity_score <= 1.0

    def test_complexity_analysis_complex(self):
        """Test complexity analysis with complex prompt."""
        complex_prompt = (
            "Please analyze the given data carefully. "
            "Make sure to consider multiple perspectives and provide detailed explanations. "
            "Think step by step through the problem and use concrete examples where helpful."
        )

        metrics = self.generator.analyze_prompt_complexity(complex_prompt)

        assert metrics.sentence_count == 3
        assert metrics.word_count > 20
        assert metrics.instruction_density > 0.1  # Should have high instruction density
        assert metrics.complexity_score > 0.3  # Should be relatively complex

    def test_complexity_analysis_empty(self):
        """Test complexity analysis with empty/whitespace prompt."""
        empty_metrics = self.generator.analyze_prompt_complexity("")
        whitespace_metrics = self.generator.analyze_prompt_complexity("   \n\t  ")

        for metrics in [empty_metrics, whitespace_metrics]:
            assert metrics.sentence_count == 0
            assert metrics.word_count == 0
            assert metrics.avg_sentence_length == 0.0
            assert metrics.lexical_diversity == 0.0
            assert metrics.complexity_score == 0.0

    def test_ab_pair_generation(self):
        """Test A/B pair generation."""
        variant_a, variant_b = self.generator.generate_ab_pair(self.base_variant)

        # Check basic properties
        assert isinstance(variant_a, Variant)
        assert isinstance(variant_b, Variant)
        assert variant_a.id != variant_b.id
        assert self.base_variant.id in variant_a.parent_ids
        assert self.base_variant.id in variant_b.parent_ids
        assert variant_a.generation == self.base_variant.generation + 1
        assert variant_b.generation == self.base_variant.generation + 1

        # Check that they're different from base and each other
        assert variant_a.prompt != self.base_variant.prompt
        assert variant_b.prompt != self.base_variant.prompt
        assert variant_a.prompt != variant_b.prompt

        # Check history was recorded
        history_a = self.generator.get_mutation_history(variant_a)
        history_b = self.generator.get_mutation_history(variant_b)
        assert len(history_a) == 1
        assert len(history_b) == 1
        assert "ab_test" in history_a[0].mutation_type
        assert "ab_test" in history_b[0].mutation_type

    def test_ab_pair_deterministic(self):
        """Test A/B pair generation in deterministic mode."""
        seed = 789

        # Generate A/B pair with seed
        self.generator.set_random_seed(seed)
        pair1_a, pair1_b = self.generator.generate_ab_pair(self.base_variant)

        # Generate again with same seed
        self.generator.set_random_seed(seed)
        pair2_a, pair2_b = self.generator.generate_ab_pair(self.base_variant)

        # Should be identical
        assert pair1_a.prompt == pair2_a.prompt
        assert pair1_b.prompt == pair2_b.prompt
        assert pair1_a.configuration == pair2_a.configuration
        assert pair1_b.configuration == pair2_b.configuration

    def test_custom_validator_registration(self):
        """Test custom validator registration and usage."""

        class TestValidator(PromptValidator):
            def validate(self, prompt: str) -> tuple[bool, list[str]]:
                if "forbidden" in prompt.lower():
                    return False, ["Contains forbidden word"]
                return True, []

            def get_name(self) -> str:
                return "test_validator"

        validator = TestValidator()

        # Register validator
        self.generator.register_validator(validator)
        assert "test_validator" in self.generator.get_registered_validators()

        # Test validation with custom validator
        valid_variant = Variant(
            id=UUID("11111111-1111-1111-1111-111111111111"),
            parent_ids=[],
            generation=0,
            prompt="This is a valid prompt.",
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        invalid_variant = Variant(
            id=UUID("22222222-2222-2222-2222-222222222222"),
            parent_ids=[],
            generation=0,
            prompt="This contains a forbidden word.",
            configuration={"temperature": 0.7},
            created_at=datetime.now(),
        )

        is_valid, errors = self.generator.validate_variant(valid_variant)
        assert is_valid

        is_invalid, errors = self.generator.validate_variant(invalid_variant)
        assert not is_invalid
        assert any("test_validator" in error for error in errors)

    def test_custom_validator_duplicate_name(self):
        """Test error handling for duplicate validator names."""

        class TestValidator1(PromptValidator):
            def validate(self, prompt: str) -> tuple[bool, list[str]]:
                return True, []

            def get_name(self) -> str:
                return "duplicate_name"

        class TestValidator2(PromptValidator):
            def validate(self, prompt: str) -> tuple[bool, list[str]]:
                return True, []

            def get_name(self) -> str:
                return "duplicate_name"

        self.generator.register_validator(TestValidator1())

        with pytest.raises(ValueError, match="already registered"):
            self.generator.register_validator(TestValidator2())

    def test_custom_validator_unregistration(self):
        """Test custom validator unregistration."""

        class TestValidator(PromptValidator):
            def validate(self, prompt: str) -> tuple[bool, list[str]]:
                return True, []

            def get_name(self) -> str:
                return "removable_validator"

        validator = TestValidator()
        self.generator.register_validator(validator)
        assert "removable_validator" in self.generator.get_registered_validators()

        # Unregister existing validator
        result = self.generator.unregister_validator("removable_validator")
        assert result is True
        assert "removable_validator" not in self.generator.get_registered_validators()

        # Try to unregister non-existent validator
        result = self.generator.unregister_validator("non_existent")
        assert result is False

    def test_custom_validator_exception_handling(self):
        """Test handling of exceptions in custom validators."""

        class FaultyValidator(PromptValidator):
            def validate(self, prompt: str) -> tuple[bool, list[str]]:
                raise RuntimeError("Validator crashed!")

            def get_name(self) -> str:
                return "faulty_validator"

        self.generator.register_validator(FaultyValidator())

        # Should handle exception gracefully
        is_valid, errors = self.generator.validate_variant(self.base_variant)
        assert not is_valid
        assert any("faulty_validator" in error for error in errors)

    def test_performance_large_population(self):
        """Test performance with large population generation."""
        import time

        start_time = time.time()
        large_population = self.generator.generate_population(self.base_prompt, 100)
        end_time = time.time()

        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 5.0  # 5 seconds
        assert len(large_population) == 100
        assert all(isinstance(v, Variant) for v in large_population)

    def test_performance_complexity_analysis(self):
        """Test performance of complexity analysis on large texts."""
        import time

        # Create a long prompt
        long_prompt = "This is a test sentence. " * 1000  # 1000 sentences

        start_time = time.time()
        metrics = self.generator.analyze_prompt_complexity(long_prompt)
        end_time = time.time()

        # Should complete within reasonable time
        assert end_time - start_time < 2.0  # 2 seconds
        assert metrics.sentence_count == 1000
        assert metrics.word_count == 5000  # 5 words per sentence

    def test_memory_usage_mutation_history(self):
        """Test that mutation history doesn't grow unbounded."""
        variant = self.base_variant

        # Create many mutations to test history limit
        for i in range(150):  # More than the max history limit (100)
            variant = self.generator.mutate_variant(variant)

        # History should be limited
        history = self.generator.get_mutation_history(variant)
        assert len(history) <= 100  # Should respect the limit

    def test_edge_case_special_characters(self):
        """Test handling of prompts with special characters."""
        special_prompt = "Hello! 你好? émojis: 🎉🚀 @#$%^&*()_+ \n\t\r"

        population = self.generator.generate_population(special_prompt, 3)
        assert len(population) == 3

        # Analyze complexity with special characters
        metrics = self.generator.analyze_prompt_complexity(special_prompt)
        assert metrics.word_count > 0
        assert metrics.sentence_count > 0

    def test_edge_case_unicode_normalization(self):
        """Test Unicode handling and normalization."""
        # Different Unicode representations of the same character
        prompt1 = "café"  # NFC form
        prompt2 = "café"  # NFD form (different byte representation)

        metrics1 = self.generator.analyze_prompt_complexity(prompt1)
        metrics2 = self.generator.analyze_prompt_complexity(prompt2)

        # Should handle both forms gracefully
        assert metrics1.word_count > 0
        assert metrics2.word_count > 0

    def test_cross_platform_consistency(self):
        """Test that results are consistent across different line ending styles."""
        # Different line ending styles
        prompt_unix = "Line 1.\nLine 2.\nLine 3."
        prompt_windows = "Line 1.\r\nLine 2.\r\nLine 3."
        prompt_old_mac = "Line 1.\rLine 2.\rLine 3."

        metrics_unix = self.generator.analyze_prompt_complexity(prompt_unix)
        metrics_windows = self.generator.analyze_prompt_complexity(prompt_windows)
        metrics_old_mac = self.generator.analyze_prompt_complexity(prompt_old_mac)

        # Should produce similar results regardless of line endings
        assert (
            metrics_unix.sentence_count
            == metrics_windows.sentence_count
            == metrics_old_mac.sentence_count
        )
        assert (
            metrics_unix.word_count
            == metrics_windows.word_count
            == metrics_old_mac.word_count
        )

    def test_concurrent_generation_safety(self):
        """Test thread safety of generation operations."""
        import threading
        import time

        results = []
        errors = []

        def generate_variants():
            try:
                population = self.generator.generate_population(self.base_prompt, 10)
                results.append(len(population))
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_variants)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        assert len(errors) == 0, f"Concurrent generation had errors: {errors}"
        assert len(results) == 5
        assert all(result == 10 for result in results)
