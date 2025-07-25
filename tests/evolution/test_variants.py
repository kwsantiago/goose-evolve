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
from evolution.variants import VariantGenerator


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
