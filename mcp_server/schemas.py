"""
Pydantic schemas for MCP tool inputs and outputs.
Provides type validation and documentation for all MCP tools.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MutationType(str, Enum):
    """Available mutation strategies for prompt evolution."""

    PARAPHRASE = "paraphrase"
    INSTRUCTION_ADD = "instruction_add"
    CONTEXT_EXPAND = "context_expand"
    COT_INJECTION = "cot_injection"
    TONE_SHIFT = "tone_shift"


class TriggerType(str, Enum):
    """Types of evolution triggers."""

    MANUAL = "manual"
    THRESHOLD = "threshold"
    SCHEDULED = "scheduled"


# ============= Input Schemas =============


class StartEvolutionInput(BaseModel):
    """Input for starting an evolution cycle."""

    trigger_type: TriggerType = Field(
        default=TriggerType.MANUAL,
        description="Type of trigger: 'manual', 'threshold', or 'scheduled'",
    )
    metrics_snapshot: Dict[str, float] = Field(
        default_factory=dict, description="Current performance metrics"
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None, description="Override default evolution configuration"
    )


class GeneratePopulationInput(BaseModel):
    """Input for generating a population of variants."""

    base_prompt: str = Field(..., description="Base prompt to generate variants from")
    size: int = Field(
        default=50, ge=1, le=1000, description="Number of variants to generate"
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible generation"
    )


class MutatePromptInput(BaseModel):
    """Input for mutating a single prompt."""

    prompt: str = Field(..., description="Original prompt text to mutate")
    mutation_type: MutationType = Field(
        default=MutationType.PARAPHRASE, description="Type of mutation to apply"
    )


class CrossoverInput(BaseModel):
    """Input for crossover operation."""

    parent1_id: str = Field(..., description="UUID of first parent variant")
    parent2_id: str = Field(..., description="UUID of second parent variant")


class EvaluateVariantInput(BaseModel):
    """Input for evaluating a variant's fitness."""

    variant_id: str = Field(..., description="UUID of the variant to evaluate")
    test_results: Dict[str, Any] = Field(
        ...,
        description="Test results including success_rate, avg_response_time, error_count, resource_usage",
    )


class RegisterFitnessFunctionInput(BaseModel):
    """Input for registering a fitness function."""

    name: str = Field(..., description="Unique name for the fitness function")
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Weight in weighted sum calculation"
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description"
    )


class UpdateFitnessWeightsInput(BaseModel):
    """Input for updating fitness function weights."""

    weights: Dict[str, float] = Field(
        ..., description="Map of function names to new weights"
    )


class CheckSafetyInput(BaseModel):
    """Input for safety validation."""

    text: str = Field(..., description="Text to validate for safety")
    include_policy: bool = Field(
        default=True, description="Whether to check against default policy"
    )


class AddSafetyPatternInput(BaseModel):
    """Input for adding a custom safety pattern."""

    category: str = Field(
        ..., description="Pattern category (e.g., 'prompt_injection')"
    )
    pattern: str = Field(..., description="Regex pattern string to add")


class RecordMetricsInput(BaseModel):
    """Input for recording metrics."""

    task_id: str = Field(..., description="Unique task identifier")
    response_time: float = Field(..., ge=0, description="Response time in seconds")
    success: bool = Field(..., description="Whether task succeeded")
    token_usage: int = Field(default=0, ge=0, description="Tokens consumed")
    error_type: Optional[str] = Field(
        default=None, description="Type of error if failed"
    )
    agent_id: Optional[str] = Field(
        default=None, description="Agent identifier for multi-agent support"
    )


class GetMetricsWindowInput(BaseModel):
    """Input for getting metrics window."""

    window_minutes: int = Field(
        default=60, ge=1, le=10080, description="Time window in minutes (max 1 week)"
    )


class DetectAnomaliesInput(BaseModel):
    """Input for anomaly detection."""

    sensitivity: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Z-score threshold for anomalies"
    )
    window_minutes: Optional[int] = Field(
        default=None, description="Time window in minutes"
    )


# ============= Output Schemas =============


class VariantSummary(BaseModel):
    """Summary of a variant for list responses."""

    id: str = Field(..., description="Variant UUID")
    generation: int = Field(..., description="Generation number")
    prompt_preview: str = Field(..., description="First 100 chars of prompt")
    fitness_score: Optional[float] = Field(
        default=None, description="Fitness score if evaluated"
    )
    created_at: str = Field(..., description="ISO timestamp of creation")


class EvolutionStatusOutput(BaseModel):
    """Status of an evolution cycle."""

    cycle_id: str = Field(..., description="Cycle UUID")
    status: str = Field(..., description="Current status")
    generation: int = Field(..., description="Current generation number")
    population_size: int = Field(..., description="Current population size")
    best_fitness: Optional[float] = Field(
        default=None, description="Best fitness score"
    )
    avg_fitness: Optional[float] = Field(
        default=None, description="Average fitness score"
    )
    started_at: Optional[str] = Field(
        default=None, description="ISO timestamp of start"
    )
    elapsed_seconds: Optional[float] = Field(default=None, description="Elapsed time")


class FitnessBreakdown(BaseModel):
    """Detailed fitness score breakdown."""

    fitness_score: float = Field(..., description="Overall fitness score")
    individual_scores: Dict[str, float] = Field(
        ..., description="Scores from each fitness function"
    )
    normalized_scores: Dict[str, float] = Field(
        ..., description="Normalized scores (0-1)"
    )
    weights: Dict[str, float] = Field(..., description="Function weights used")
    explanations: Dict[str, str] = Field(..., description="Human-readable explanations")


class SafetyCheckOutput(BaseModel):
    """Result of a safety check."""

    is_safe: bool = Field(..., description="Whether text passed validation")
    violations: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of safety violations"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )


class MetricsWindowOutput(BaseModel):
    """Aggregated metrics for a time window."""

    window_minutes: int = Field(..., description="Time window size")
    total_requests: int = Field(..., description="Total requests in window")
    success_rate: float = Field(..., description="Success rate (0-1)")
    avg_response_time: float = Field(..., description="Average response time")
    p95_response_time: Optional[float] = Field(
        default=None, description="95th percentile response time"
    )
    total_tokens: int = Field(default=0, description="Total tokens used")


class TriggerCheckOutput(BaseModel):
    """Result of checking evolution trigger."""

    should_evolve: bool = Field(..., description="Whether thresholds are breached")
    trigger_event: Optional[Dict[str, Any]] = Field(
        default=None, description="Trigger details if applicable"
    )
    current_metrics: Dict[str, Any] = Field(..., description="Current metric values")
    thresholds: Dict[str, Any] = Field(..., description="Configured thresholds")


class ComplexityMetricsOutput(BaseModel):
    """Prompt complexity analysis."""

    sentence_count: int = Field(..., description="Number of sentences")
    word_count: int = Field(..., description="Number of words")
    avg_sentence_length: float = Field(..., description="Average sentence length")
    lexical_diversity: float = Field(..., description="Vocabulary diversity (0-1)")
    instruction_density: float = Field(
        default=0.0, description="Density of instruction keywords"
    )
    complexity_score: float = Field(default=0.0, description="Overall complexity score")
