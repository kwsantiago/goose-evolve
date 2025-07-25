"""
Variant Generator implementation for Goose Evolve.
Creates prompt mutations and crossover operations for genetic algorithm evolution.
"""

import logging
import random
import re
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from evolution.interfaces import (
    DEFAULT_MUTATION_RATE,
    CrossoverType,
    PromptMutationType,
    Variant,
)
from evolution.interfaces import VariantGenerator as VariantGeneratorBase

logger = logging.getLogger(__name__)


class VariantGenerator(VariantGeneratorBase):
    """
    Generates agent variants through prompt mutations and crossover.
    Focuses on rule-based mutations without requiring LLM calls.
    """

    def __init__(self, mutation_rate: float = DEFAULT_MUTATION_RATE):
        self.mutation_rate = mutation_rate
        self.mutation_templates = self._initialize_mutation_templates()

        # Validation constraints
        self.validation_config: Dict[str, Any] = {
            "min_prompt_length": 10,
            "max_prompt_length": 5000,
            "max_sentences": 20,
            "forbidden_patterns": [
                r"<script.*?>",  # Script tags
                r"javascript:",  # JavaScript URLs
                r"eval\s*\(",  # Eval calls
                r"exec\s*\(",  # Exec calls
            ],
        }

        # Track generation statistics
        self.generation_stats = {
            "total_generated": 0,
            "mutations_applied": 0,
            "crossovers_performed": 0,
            "validation_failures": 0,
        }

    def generate_population(self, base_prompt: str, size: int) -> List[Variant]:
        """
        Generate initial population of variants from base prompt.

        Args:
            base_prompt: Starting prompt for mutations
            size: Number of variants to generate

        Returns:
            List of variant objects with mutations applied
        """
        population = []
        base_config = {"temperature": 0.7}  # Default configuration

        # Create base variant (no mutations)
        base_variant = Variant(
            id=uuid4(),
            parent_ids=[],
            generation=0,
            prompt=base_prompt,
            configuration=base_config,
            created_at=datetime.now(),
        )
        population.append(base_variant)

        # Generate mutated variants
        for i in range(size - 1):
            # Choose random mutation strategy
            mutation_type = random.choice(list(PromptMutationType))
            mutated_prompt = self.mutate_prompt(base_prompt, mutation_type)

            # Apply configuration mutations occasionally
            config = base_config.copy()
            if random.random() < self.mutation_rate:
                config = self._mutate_configuration(config)

            variant = Variant(
                id=uuid4(),
                parent_ids=[base_variant.id],
                generation=0,
                prompt=mutated_prompt,
                configuration=config,
                created_at=datetime.now(),
            )
            population.append(variant)

        self.generation_stats["total_generated"] += len(population)
        return population

    def mutate_prompt(self, prompt: str, mutation_type: PromptMutationType) -> str:
        """
        Apply specific mutation to prompt text.

        Args:
            prompt: Original prompt text
            mutation_type: Type of mutation to apply

        Returns:
            Mutated prompt string
        """
        try:
            if mutation_type == PromptMutationType.PARAPHRASE:
                return self._paraphrase_mutation(prompt)
            elif mutation_type == PromptMutationType.INSTRUCTION_ADD:
                return self._instruction_add_mutation(prompt)
            elif mutation_type == PromptMutationType.CONTEXT_EXPAND:
                return self._context_expand_mutation(prompt)
            elif mutation_type == PromptMutationType.COT_INJECTION:
                return self._cot_injection_mutation(prompt)
            elif mutation_type == PromptMutationType.TONE_SHIFT:
                return self._tone_shift_mutation(prompt)
            else:
                # Fallback to paraphrase
                return self._paraphrase_mutation(prompt)
        except Exception:
            # Return original prompt if mutation fails
            return prompt

    def crossover(
        self,
        parent1: Variant,
        parent2: Variant,
        crossover_type: CrossoverType = CrossoverType.UNIFORM,
    ) -> Tuple[Variant, Variant]:
        """
        Perform crossover between two parent variants.

        Args:
            parent1: First parent variant
            parent2: Second parent variant
            crossover_type: Type of crossover to perform

        Returns:
            Tuple of two offspring variants
        """
        if crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        else:
            # Fallback to uniform crossover
            return self._uniform_crossover(parent1, parent2)

    def mutate_variant(self, variant: Variant) -> Variant:
        """
        Create a mutated copy of a variant.

        Args:
            variant: Original variant to mutate

        Returns:
            New variant with mutations applied
        """
        # Choose random mutation type
        mutation_type = random.choice(list(PromptMutationType))
        mutated_prompt = self.mutate_prompt(variant.prompt, mutation_type)

        # Mutate configuration
        mutated_config = variant.configuration.copy()
        if random.random() < self.mutation_rate:
            mutated_config = self._mutate_configuration(mutated_config)

        new_variant = replace(
            variant,
            id=uuid4(),
            parent_ids=[variant.id],
            generation=variant.generation + 1,
            prompt=mutated_prompt,
            configuration=mutated_config,
            created_at=datetime.now(),
            fitness_score=None,
        )

        self.generation_stats["mutations_applied"] += 1
        return new_variant

    def _initialize_mutation_templates(self) -> Dict[str, Any]:
        """Initialize templates for different mutation types."""
        return {
            "instruction_prefixes": [
                "Please",
                "Make sure to",
                "Always",
                "Remember to",
                "Be sure to",
            ],
            "instruction_suffixes": [
                "Think step by step.",
                "Be thorough in your response.",
                "Provide detailed explanations.",
                "Consider multiple perspectives.",
                "Use examples when helpful.",
            ],
            "context_expanders": [
                "Given the context,",
                "Taking into account the situation,",
                "Considering the requirements,",
                "Based on the information provided,",
                "In this scenario,",
            ],
            "cot_triggers": [
                "Let's think step by step:",
                "Let me work through this:",
                "Breaking this down:",
                "Step by step:",
                "Let's analyze this:",
            ],
            "instruction_additions": [
                "Be precise and accurate.",
                "Focus on clarity and completeness.",
                "Provide actionable insights.",
                "Consider edge cases.",
                "Verify your reasoning.",
                "Double-check your work.",
                "Use concrete examples.",
                "Be systematic in your approach.",
            ],
            "context_expansions": [
                "In the current situation,",
                "Under these circumstances,",
                "With the given constraints,",
                "For this particular case,",
                "Taking all factors into account,",
                "In this specific context,",
                "Given these parameters,",
            ],
            "tone_modifiers": {
                "formal": [
                    "Please note that",
                    "It should be observed that",
                    "One must consider that",
                ],
                "casual": ["So basically", "Just keep in mind", "Here's the thing"],
                "technical": [
                    "From a technical standpoint",
                    "Algorithmically speaking",
                    "In terms of implementation",
                ],
                "helpful": [
                    "To help you better",
                    "For your convenience",
                    "To make this easier",
                ],
                "confident": ["Certainly", "Without doubt", "Definitely"],
            },
        }

    def _paraphrase_mutation(self, prompt: str) -> str:
        """
        Apply paraphrasing mutations to prompt.
        Simple rule-based transformations.
        """
        mutations = []

        # Add instruction prefix
        if random.random() < 0.3:
            prefix = random.choice(self.mutation_templates["instruction_prefixes"])
            mutations.append(f"{prefix} {prompt.lower()}")

        # Add instruction suffix
        if random.random() < 0.3:
            suffix = random.choice(self.mutation_templates["instruction_suffixes"])
            mutations.append(f"{prompt} {suffix}")

        # Add context expander
        if random.random() < 0.2:
            expander = random.choice(self.mutation_templates["context_expanders"])
            mutations.append(f"{expander} {prompt.lower()}")

        # Return random mutation or original
        if mutations:
            return random.choice(mutations)
        return prompt

    def _instruction_add_mutation(self, prompt: str) -> str:
        """
        Add specific instructions to improve prompt clarity.
        """
        instruction = random.choice(self.mutation_templates["instruction_additions"])

        # Add instruction at different positions
        position = random.choice(["start", "end", "middle"])

        if position == "start":
            return f"{instruction} {prompt}"
        elif position == "end":
            return f"{prompt} {instruction}"
        else:
            # Split prompt and insert in middle
            sentences = self._split_sentences(prompt)
            if len(sentences) > 1:
                mid_point = len(sentences) // 2
                sentences.insert(mid_point, instruction)
                return " ".join(sentences)
            else:
                return f"{prompt} {instruction}"

    def _context_expand_mutation(self, prompt: str) -> str:
        """
        Expand context to provide more situational awareness.
        """
        context_expansion = random.choice(self.mutation_templates["context_expansions"])

        # Add context expansion at the beginning
        return f"{context_expansion} {prompt.lower()}"

    def _cot_injection_mutation(self, prompt: str) -> str:
        """
        Inject chain-of-thought reasoning into prompt.
        """
        cot_trigger = random.choice(self.mutation_templates["cot_triggers"])

        # Add CoT trigger at beginning or end
        if random.random() < 0.5:
            return f"{cot_trigger} {prompt}"
        else:
            return f"{prompt} {cot_trigger}"

    def _tone_shift_mutation(self, prompt: str) -> str:
        """
        Shift the tone of the prompt using different modifiers.
        """
        # Choose random tone
        tone = random.choice(list(self.mutation_templates["tone_modifiers"].keys()))
        modifier = random.choice(self.mutation_templates["tone_modifiers"][tone])

        # Apply tone modifier at the beginning
        return f"{modifier}, {prompt.lower()}"

    def _uniform_crossover(
        self, parent1: Variant, parent2: Variant
    ) -> Tuple[Variant, Variant]:
        """
        Perform uniform crossover at sentence boundaries.
        """
        # Split prompts into sentences
        sentences1 = self._split_sentences(parent1.prompt)
        sentences2 = self._split_sentences(parent2.prompt)

        # Ensure we have sentences from both parents
        if not sentences1 or not sentences2:
            # Fallback: return mutated copies
            return (self.mutate_variant(parent1), self.mutate_variant(parent2))

        # Create offspring by mixing sentences
        offspring1_sentences = []
        offspring2_sentences = []

        max_length = max(len(sentences1), len(sentences2))

        for i in range(max_length):
            s1 = sentences1[i] if i < len(sentences1) else sentences1[-1]
            s2 = sentences2[i] if i < len(sentences2) else sentences2[-1]

            if random.random() < 0.5:
                offspring1_sentences.append(s1)
                offspring2_sentences.append(s2)
            else:
                offspring1_sentences.append(s2)
                offspring2_sentences.append(s1)

        # Combine configurations
        config1 = self._combine_configurations(
            parent1.configuration, parent2.configuration
        )
        config2 = self._combine_configurations(
            parent2.configuration, parent1.configuration
        )

        # Create offspring variants
        offspring1 = Variant(
            id=uuid4(),
            parent_ids=[parent1.id, parent2.id],
            generation=max(parent1.generation, parent2.generation) + 1,
            prompt=" ".join(offspring1_sentences),
            configuration=config1,
            created_at=datetime.now(),
        )

        offspring2 = Variant(
            id=uuid4(),
            parent_ids=[parent1.id, parent2.id],
            generation=max(parent1.generation, parent2.generation) + 1,
            prompt=" ".join(offspring2_sentences),
            configuration=config2,
            created_at=datetime.now(),
        )

        self.generation_stats["crossovers_performed"] += 1
        return (offspring1, offspring2)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _mutate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mutations to configuration parameters.
        """
        mutated = config.copy()

        # Mutate temperature
        if "temperature" in mutated:
            current_temp = mutated["temperature"]
            # Add gaussian noise
            noise = random.gauss(0, 0.1)
            new_temp = max(0.0, min(2.0, current_temp + noise))
            mutated["temperature"] = round(new_temp, 2)

        # Occasionally add new parameters
        if random.random() < 0.1:
            new_params = {
                "max_tokens": random.randint(100, 2000),
                "top_p": round(random.uniform(0.1, 1.0), 2),
                "frequency_penalty": round(random.uniform(0.0, 1.0), 2),
            }
            param_name = random.choice(list(new_params.keys()))
            mutated[param_name] = new_params[param_name]

        return mutated

    def _combine_configurations(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine two configurations for crossover.
        """
        combined = {}
        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            if key in config1 and key in config2:
                # Average numeric values, randomly choose others
                val1, val2 = config1[key], config2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    combined[key] = (val1 + val2) / 2
                else:
                    combined[key] = random.choice([val1, val2])
            elif key in config1:
                combined[key] = config1[key]
            else:
                combined[key] = config2[key]

        return combined

    def get_stats(self) -> Dict[str, int]:
        """Get generation statistics."""
        return self.generation_stats.copy()

    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            "total_generated": 0,
            "mutations_applied": 0,
            "crossovers_performed": 0,
            "validation_failures": 0,
        }

    def set_mutation_rate(self, rate: float):
        """
        Update the mutation rate for variant generation.

        Args:
            rate: New mutation rate (0.0 to 1.0)
        """
        if not isinstance(rate, (int, float)) or rate < 0.0 or rate > 1.0:
            raise ValueError(f"Mutation rate must be between 0.0 and 1.0, got {rate}")
        self.mutation_rate = rate

    def get_mutation_rate(self) -> float:
        """Get current mutation rate."""
        return self.mutation_rate

    def update_validation_config(self, config: Dict[str, Any]):
        """
        Update validation configuration parameters.

        Args:
            config: Dictionary with validation configuration updates
        """
        allowed_keys = {
            "min_prompt_length",
            "max_prompt_length",
            "max_sentences",
            "forbidden_patterns",
        }
        for key, value in config.items():
            if key in allowed_keys:
                self.validation_config[key] = value
            else:
                raise ValueError(f"Unknown validation config key: {key}")

    def get_validation_config(self) -> Dict[str, Any]:
        """Get current validation configuration."""
        return self.validation_config.copy()

    def validate_variant(self, variant: Variant) -> Tuple[bool, List[str]]:
        """
        Validate a variant for basic syntax and safety constraints.

        Args:
            variant: Variant to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check prompt length
        if len(variant.prompt) < self.validation_config["min_prompt_length"]:
            errors.append(
                f"Prompt too short: {len(variant.prompt)} < {self.validation_config['min_prompt_length']}"
            )

        if len(variant.prompt) > self.validation_config["max_prompt_length"]:
            errors.append(
                f"Prompt too long: {len(variant.prompt)} > {self.validation_config['max_prompt_length']}"
            )

        # Check sentence count
        sentences = self._split_sentences(variant.prompt)
        if len(sentences) > self.validation_config["max_sentences"]:
            errors.append(
                f"Too many sentences: {len(sentences)} > {self.validation_config['max_sentences']}"
            )

        # Check for forbidden patterns
        for pattern in self.validation_config["forbidden_patterns"]:
            if re.search(pattern, variant.prompt, re.IGNORECASE):
                errors.append(f"Forbidden pattern detected: {pattern}")

        # Validate configuration
        config_errors = self._validate_configuration(variant.configuration)
        errors.extend(config_errors)

        is_valid = len(errors) == 0
        if not is_valid:
            self.generation_stats["validation_failures"] += 1

        return is_valid, errors

    def _validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration parameters.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Check temperature range
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append(f"Invalid temperature: {temp} (must be 0-2)")

        # Check max_tokens
        if "max_tokens" in config:
            tokens = config["max_tokens"]
            if not isinstance(tokens, int) or tokens < 1 or tokens > 4000:
                errors.append(f"Invalid max_tokens: {tokens} (must be 1-4000)")

        # Check top_p
        if "top_p" in config:
            top_p = config["top_p"]
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                errors.append(f"Invalid top_p: {top_p} (must be 0-1)")

        # Check frequency_penalty
        if "frequency_penalty" in config:
            penalty = config["frequency_penalty"]
            if not isinstance(penalty, (int, float)) or penalty < 0 or penalty > 1:
                errors.append(f"Invalid frequency_penalty: {penalty} (must be 0-1)")

        return errors

    def filter_valid_variants(self, variants: List[Variant]) -> List[Variant]:
        """
        Filter a list of variants to return only valid ones.

        Args:
            variants: List of variants to filter

        Returns:
            List of valid variants
        """
        valid_variants = []

        for variant in variants:
            is_valid, errors = self.validate_variant(variant)
            if is_valid:
                valid_variants.append(variant)
            else:
                # Log validation failures for debugging
                logger.debug(f"Variant {variant.id} failed validation: {errors}")

        return valid_variants
