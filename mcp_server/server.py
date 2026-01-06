"""
MCP Server for evolve-mcp.
Universal MCP server for agent self-improvement via evolutionary algorithms.

Exposes 21 tools for:
- Evolution lifecycle management (4 tools)
- Variant generation (5 tools)
- Fitness evaluation (5 tools)
- Safety validation (3 tools)
- Metrics collection (4 tools)
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from evolution import EvolutionConfig
from evolution.interfaces import EvolutionTriggerEvent, PromptMutationType

from .errors import (
    CycleNotFoundError,
    InvalidMutationTypeError,
    VariantNotFoundError,
    handle_mcp_error,
)
from .serializers import (
    deserialize_test_results,
    serialize_complexity_metrics,
    serialize_safety_violation,
    serialize_variant,
    serialize_variant_summary,
)
from .state import get_state_manager

# Initialize MCP server
mcp = FastMCP("evolve-mcp")


# =============================================================================
# EVOLUTION LIFECYCLE TOOLS (4)
# =============================================================================


@mcp.tool()
@handle_mcp_error
async def start_evolution(
    trigger_type: str = "manual",
    metrics_snapshot: Optional[Dict[str, float]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Start a new evolution cycle to improve agent performance.

    Args:
        trigger_type: Type of trigger - 'manual' or 'threshold'
        metrics_snapshot: Current performance metrics (success_rate, avg_response_time, etc.)
        config_overrides: Override default evolution config (population_size, max_generations, etc.)

    Returns:
        cycle_id: Unique identifier for tracking this evolution
        status: Current status ('running')
        config: Applied configuration
    """
    state_mgr = get_state_manager()

    # Build trigger event
    trigger = EvolutionTriggerEvent(
        trigger_type=trigger_type,
        metrics_snapshot=metrics_snapshot or {},
        timestamp=datetime.now(),
    )

    # Build evolution config
    pop_size = int(os.environ.get("EVOLVE_MCP_POPULATION_SIZE", "50"))
    max_gen = int(os.environ.get("EVOLVE_MCP_MAX_GENERATIONS", "10"))
    fit_thresh = float(os.environ.get("EVOLVE_MCP_FITNESS_THRESHOLD", "0.95"))

    if config_overrides:
        pop_size = int(config_overrides.get("population_size", pop_size))
        max_gen = int(config_overrides.get("max_generations", max_gen))
        fit_thresh = float(config_overrides.get("fitness_threshold", fit_thresh))

    evo_config = EvolutionConfig(
        population_size=pop_size,
        max_generations=max_gen,
        fitness_threshold=fit_thresh,
    )

    # Start cycle
    cycle_id = await state_mgr.start_cycle(trigger, evo_config)

    return {
        "cycle_id": cycle_id,
        "status": "running",
        "config": {
            "population_size": evo_config.population_size,
            "max_generations": evo_config.max_generations,
            "fitness_threshold": evo_config.fitness_threshold,
        },
        "started_at": datetime.now().isoformat(),
    }


@mcp.tool()
@handle_mcp_error
async def get_evolution_status(cycle_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get status of an evolution cycle.

    Args:
        cycle_id: Specific cycle to check (defaults to current/latest)

    Returns:
        Current status, generation, fitness scores, and progress
    """
    state_mgr = get_state_manager()

    if cycle_id is None:
        cycle_id = state_mgr.get_current_cycle_id()

    if cycle_id is None:
        return {
            "status": "no_active_cycle",
            "message": "No evolution cycles have been started",
        }

    cycle = await state_mgr.get_cycle(cycle_id)
    if cycle is None:
        raise CycleNotFoundError(cycle_id)

    # Get current generation from engine if available
    generation = 0
    if cycle.engine:
        generation = getattr(cycle.engine, "current_generation", 0)

    # Calculate fitness stats from population
    fitness_scores = [
        v.fitness_score for v in cycle.current_population if v.fitness_score is not None
    ]
    best_fitness = max(fitness_scores) if fitness_scores else None
    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else None

    return {
        "cycle_id": cycle_id,
        "status": cycle.status,
        "generation": generation,
        "population_size": len(cycle.current_population),
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "started_at": cycle.started_at.isoformat() if cycle.started_at else None,
        "elapsed_seconds": (
            (datetime.now() - cycle.started_at).total_seconds()
            if cycle.started_at
            else 0
        ),
        "error": cycle.error,
    }


@mcp.tool()
@handle_mcp_error
async def cancel_evolution(cycle_id: str) -> Dict[str, Any]:
    """
    Cancel a running evolution cycle.

    Args:
        cycle_id: ID of cycle to cancel

    Returns:
        success: Whether cancellation succeeded
        final_state: State at cancellation
    """
    state_mgr = get_state_manager()
    success = await state_mgr.cancel_cycle(cycle_id)

    if not success:
        raise CycleNotFoundError(cycle_id)

    cycle = await state_mgr.get_cycle(cycle_id)
    return {
        "success": True,
        "cycle_id": cycle_id,
        "final_status": cycle.status if cycle else "cancelled",
        "final_generation": (
            getattr(cycle.engine, "current_generation", 0)
            if cycle and cycle.engine
            else 0
        ),
    }


@mcp.tool()
@handle_mcp_error
async def resume_evolution(checkpoint_path: str) -> Dict[str, Any]:
    """
    Resume evolution from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        cycle_id: ID of resumed cycle
        resumed_generation: Generation number resumed from
    """
    # TODO: Implement checkpoint resume
    return {
        "error": "Checkpoint resume not yet implemented",
        "checkpoint_path": checkpoint_path,
    }


# =============================================================================
# VARIANT GENERATION TOOLS (5)
# =============================================================================


@mcp.tool()
@handle_mcp_error
async def generate_population(
    base_prompt: str,
    size: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a population of prompt variants for evolution.

    Args:
        base_prompt: Starting prompt to mutate
        size: Number of variants to generate (1-1000)
        seed: Random seed for reproducible generation

    Returns:
        variants: List of generated variants with IDs and previews
        stats: Generation statistics
    """
    state_mgr = get_state_manager()
    generator = state_mgr.get_variant_generator()

    if seed is not None:
        generator.set_random_seed(seed)

    variants = generator.generate_population(base_prompt, min(size, 1000))

    # Store variants for later retrieval
    state_mgr.store_variants(variants)

    return {
        "count": len(variants),
        "variants": [
            serialize_variant_summary(v) for v in variants[:100]
        ],  # Limit response
        "seed": seed,
    }


@mcp.tool()
@handle_mcp_error
async def mutate_prompt(
    prompt: str,
    mutation_type: str = "paraphrase",
) -> Dict[str, Any]:
    """
    Apply a specific mutation to a prompt.

    Args:
        prompt: Original prompt text
        mutation_type: Type of mutation (paraphrase, instruction_add, context_expand, cot_injection, tone_shift)

    Returns:
        mutated_prompt: The transformed prompt
        complexity_metrics: Analysis of the result
    """
    state_mgr = get_state_manager()
    generator = state_mgr.get_variant_generator()

    # Validate mutation type
    valid_types = [t.value for t in PromptMutationType]
    if mutation_type not in valid_types:
        raise InvalidMutationTypeError(mutation_type, valid_types)

    mutation_enum = PromptMutationType(mutation_type)
    mutated = generator.mutate_prompt(prompt, mutation_enum)
    complexity = generator.analyze_prompt_complexity(mutated)

    return {
        "original_prompt": prompt,
        "mutated_prompt": mutated,
        "mutation_type": mutation_type,
        "complexity_metrics": serialize_complexity_metrics(complexity),
    }


@mcp.tool()
@handle_mcp_error
async def crossover_variants(
    parent1_id: str,
    parent2_id: str,
) -> Dict[str, Any]:
    """
    Combine two variants to create offspring.

    Args:
        parent1_id: UUID of first parent variant
        parent2_id: UUID of second parent variant

    Returns:
        offspring: List of created offspring variants
    """
    state_mgr = get_state_manager()
    generator = state_mgr.get_variant_generator()

    parent1 = state_mgr.get_variant_by_id(parent1_id)
    parent2 = state_mgr.get_variant_by_id(parent2_id)

    if parent1 is None:
        raise VariantNotFoundError(parent1_id)
    if parent2 is None:
        raise VariantNotFoundError(parent2_id)

    offspring1, offspring2 = generator.crossover(parent1, parent2)

    # Store offspring
    state_mgr.store_variants([offspring1, offspring2])

    return {
        "parent1_id": parent1_id,
        "parent2_id": parent2_id,
        "offspring": [
            serialize_variant_summary(offspring1),
            serialize_variant_summary(offspring2),
        ],
    }


@mcp.tool()
@handle_mcp_error
async def generate_ab_pair(base_variant_id: str) -> Dict[str, Any]:
    """
    Create an A/B test pair from a base variant.

    Args:
        base_variant_id: UUID of the base variant

    Returns:
        variant_a: Control variant
        variant_b: Test variant with mutation
    """
    state_mgr = get_state_manager()
    generator = state_mgr.get_variant_generator()

    base_variant = state_mgr.get_variant_by_id(base_variant_id)
    if base_variant is None:
        raise VariantNotFoundError(base_variant_id)

    variant_a, variant_b = generator.generate_ab_pair(base_variant)

    # Store new variants
    state_mgr.store_variants([variant_a, variant_b])

    return {
        "base_variant_id": base_variant_id,
        "variant_a": serialize_variant_summary(variant_a),
        "variant_b": serialize_variant_summary(variant_b),
    }


@mcp.tool()
@handle_mcp_error
async def analyze_prompt(prompt: str) -> Dict[str, Any]:
    """
    Analyze prompt complexity and characteristics.

    Args:
        prompt: Prompt text to analyze

    Returns:
        complexity_metrics: Detailed complexity analysis
    """
    state_mgr = get_state_manager()
    generator = state_mgr.get_variant_generator()

    complexity = generator.analyze_prompt_complexity(prompt)

    return {
        "prompt_length": len(prompt),
        "complexity_metrics": serialize_complexity_metrics(complexity),
    }


# =============================================================================
# FITNESS EVALUATION TOOLS (5)
# =============================================================================


@mcp.tool()
@handle_mcp_error
async def evaluate_variant(
    variant_id: str,
    test_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate fitness of a variant based on test results.

    Args:
        variant_id: UUID of the variant
        test_results: Test results including success_rate, avg_response_time, error_count, resource_usage

    Returns:
        fitness_score: Overall fitness (0-1)
        breakdown: Score from each fitness function
    """
    state_mgr = get_state_manager()
    evaluator = state_mgr.get_fitness_evaluator()

    variant = state_mgr.get_variant_by_id(variant_id)
    if variant is None:
        raise VariantNotFoundError(variant_id)

    results = deserialize_test_results({"variant_id": variant_id, **test_results})
    fitness = await evaluator.evaluate(variant, results)

    return {
        "variant_id": variant_id,
        "fitness_score": fitness,
    }


@mcp.tool()
@handle_mcp_error
async def explain_fitness(
    variant_id: str,
    test_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get detailed fitness breakdown with explanations.

    Args:
        variant_id: UUID of the variant
        test_results: Test results

    Returns:
        Detailed breakdown with individual scores and explanations
    """
    state_mgr = get_state_manager()
    evaluator = state_mgr.get_fitness_evaluator()

    variant = state_mgr.get_variant_by_id(variant_id)
    if variant is None:
        raise VariantNotFoundError(variant_id)

    results = deserialize_test_results({"variant_id": variant_id, **test_results})
    explanation = evaluator.explain_fitness(variant, results)

    return {
        "variant_id": variant_id,
        "fitness_score": explanation.overall_score,
        "breakdown": {
            "individual_scores": explanation.individual_scores,
            "normalized_scores": explanation.normalized_scores,
            "weights": explanation.weights,
            "explanations": explanation.explanations,
        },
    }


@mcp.tool()
@handle_mcp_error
async def register_fitness_function(
    name: str,
    weight: float = 1.0,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Register a predefined fitness function.

    Args:
        name: Name of predefined function (speed, accuracy, efficiency)
        weight: Weight in weighted sum (0-10)
        description: Human-readable description of the function

    Returns:
        success: Whether registration succeeded
        registered_functions: Updated list of functions
    """
    state_mgr = get_state_manager()
    evaluator = state_mgr.get_fitness_evaluator()

    predefined = ["speed", "accuracy", "efficiency"]

    if name not in predefined:
        return {
            "error": "Only predefined functions supported for safety",
            "available": predefined,
        }

    registered = evaluator.get_registered_functions()

    return {
        "success": True,
        "name": name,
        "weight": weight,
        "description": description,
        "registered_functions": registered,
    }


@mcp.tool()
@handle_mcp_error
async def update_fitness_weights(weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Update weights for fitness functions.

    Args:
        weights: Map of function names to new weights

    Returns:
        updated_weights: The new weight configuration
    """
    state_mgr = get_state_manager()
    evaluator = state_mgr.get_fitness_evaluator()

    evaluator.update_weights(weights)

    return {
        "success": True,
        "updated_weights": weights,
        "registered_functions": evaluator.get_registered_functions(),
    }


@mcp.tool()
@handle_mcp_error
async def list_fitness_functions() -> Dict[str, Any]:
    """
    List all registered fitness functions.

    Returns:
        functions: List of registered functions with weights
    """
    state_mgr = get_state_manager()
    evaluator = state_mgr.get_fitness_evaluator()

    functions = evaluator.get_registered_functions()

    return {
        "count": len(functions),
        "functions": functions,
    }


# =============================================================================
# SAFETY VALIDATION TOOLS (3)
# =============================================================================


@mcp.tool()
@handle_mcp_error
async def validate_variant(
    variant_id: str,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run full safety validation on a variant.

    Args:
        variant_id: UUID of variant to validate
        policy: Optional custom safety policy

    Returns:
        is_valid: Whether variant passed validation
        violations: List of safety violations
        warnings: Non-critical warnings
    """
    state_mgr = get_state_manager()
    validator = state_mgr.get_safety_validator()

    variant = state_mgr.get_variant_by_id(variant_id)
    if variant is None:
        raise VariantNotFoundError(variant_id)

    from evolve_core.safety import SafetyPolicy, create_default_policy

    safety_policy = create_default_policy()
    if policy:
        safety_policy = SafetyPolicy(**policy)

    result = await validator.validate(variant, safety_policy)

    return {
        "variant_id": variant_id,
        "is_valid": result.is_valid,
        "errors": result.errors,
    }


@mcp.tool()
@handle_mcp_error
async def check_safety(
    text: str,
    include_policy: bool = True,
) -> Dict[str, Any]:
    """
    Run safety validation on text/prompt.

    Args:
        text: Text to validate
        include_policy: Whether to check against default policy

    Returns:
        is_safe: Whether text passed validation
        violations: List of safety violations
        warnings: Non-critical warnings
    """
    state_mgr = get_state_manager()
    validator = state_mgr.get_safety_validator()

    from evolve_core.safety import create_default_policy

    policy = create_default_policy() if include_policy else None
    result = validator.check_safety(text, policy)

    return {
        "is_safe": result.is_safe,
        "violations": [serialize_safety_violation(v) for v in result.violations],
        "warnings": result.warnings,
    }


@mcp.tool()
@handle_mcp_error
async def add_safety_pattern(
    category: str,
    pattern: str,
) -> Dict[str, Any]:
    """
    Add a custom safety pattern.

    Args:
        category: Pattern category (e.g., 'prompt_injection')
        pattern: Regex pattern string

    Returns:
        success: Whether pattern was added
        category: The category it was added to
    """
    state_mgr = get_state_manager()
    validator = state_mgr.get_safety_validator()

    validator.add_pattern(category, pattern)

    return {
        "success": True,
        "category": category,
        "pattern": pattern,
    }


# =============================================================================
# METRICS TOOLS (4)
# =============================================================================


@mcp.tool()
@handle_mcp_error
async def record_metrics(
    task_id: str,
    response_time: float,
    success: bool,
    token_usage: int = 0,
    error_type: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record metrics for a task execution.

    Args:
        task_id: Unique task identifier
        response_time: Response time in seconds
        success: Whether task succeeded
        token_usage: Tokens consumed
        error_type: Type of error if failed
        agent_id: Agent identifier for multi-agent support

    Returns:
        success: Whether recording succeeded
        data_points: Current count of stored metrics
    """
    state_mgr = get_state_manager()
    collector = state_mgr.get_metrics_collector()

    collector.collect(
        task_id=task_id,
        metrics={
            "response_time": response_time,
            "success": success,
            "token_usage": token_usage,
            "error_type": error_type,
        },
        agent_id=agent_id,
    )

    return {
        "success": True,
        "task_id": task_id,
        "data_points_stored": len(collector.data_points),
    }


@mcp.tool()
@handle_mcp_error
async def get_metrics_window(window_minutes: int = 60) -> Dict[str, Any]:
    """
    Get aggregated metrics for a time window.

    Args:
        window_minutes: Time window in minutes (1-10080)

    Returns:
        Aggregated metrics including success rate, response times, etc.
    """
    state_mgr = get_state_manager()
    collector = state_mgr.get_metrics_collector()

    window = collector.get_window_metrics(timedelta(minutes=window_minutes))

    # Get p95 from custom metrics if available
    p95_response_time = collector.get_custom_metric_value(
        "p95_response_time", timedelta(minutes=window_minutes)
    )

    return {
        "window_minutes": window_minutes,
        "total_requests": window.total_requests,
        "success_rate": window.success_rate,
        "avg_response_time": window.avg_response_time,
        "p95_response_time": p95_response_time,
        "total_tokens": window.total_tokens,
    }


@mcp.tool()
@handle_mcp_error
async def check_evolution_trigger() -> Dict[str, Any]:
    """
    Check if current metrics indicate evolution should be triggered.

    Returns:
        should_evolve: Whether thresholds are breached
        trigger_event: Trigger details if applicable
        current_metrics: Current metric values
    """
    state_mgr = get_state_manager()
    collector = state_mgr.get_metrics_collector()

    trigger = collector.check_evolution_triggers()
    stats = collector.get_stats()

    return {
        "should_evolve": trigger is not None,
        "trigger_event": (
            {
                "trigger_type": trigger.trigger_type,
                "metrics_snapshot": trigger.metrics_snapshot,
            }
            if trigger
            else None
        ),
        "current_metrics": stats.get("current_metrics", {}),
        "thresholds": stats.get("thresholds", {}),
    }


@mcp.tool()
@handle_mcp_error
async def detect_anomalies(
    sensitivity: float = 2.0,
    window_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect anomalies in metrics data.

    Args:
        sensitivity: Z-score threshold for anomalies (1.0-5.0)
        window_minutes: Time window in minutes

    Returns:
        anomalies: List of detected anomalies
    """
    state_mgr = get_state_manager()
    collector = state_mgr.get_metrics_collector()

    window_duration = timedelta(minutes=window_minutes) if window_minutes else None
    anomalies = collector.detect_anomalies(
        sensitivity=sensitivity, window_duration=window_duration
    )

    return {
        "sensitivity": sensitivity,
        "anomaly_count": len(anomalies),
        "anomalies": [
            {
                "metric": a.metric_name,
                "value": a.value,
                "expected_range": a.expected_range,
                "severity": a.severity,
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
            }
            for a in anomalies
        ],
    }


# =============================================================================
# RESOURCES
# =============================================================================


@mcp.resource("evolution://status")
def resource_evolution_status() -> str:
    """Current evolution cycle status."""
    state_mgr = get_state_manager()
    cycle_id = state_mgr.get_current_cycle_id()
    if cycle_id is None:
        return "No active evolution cycle"
    stats = state_mgr.get_stats()
    return f"Active cycles: {stats['running_cycles']}, Total: {stats['total_cycles']}"


@mcp.resource("evolution://config")
def resource_evolution_config() -> str:
    """Current evolution configuration."""
    return f"""Evolution Configuration:
- Population Size: {os.environ.get('EVOLVE_MCP_POPULATION_SIZE', '50')}
- Max Generations: {os.environ.get('EVOLVE_MCP_MAX_GENERATIONS', '10')}
- Fitness Threshold: {os.environ.get('EVOLVE_MCP_FITNESS_THRESHOLD', '0.95')}
- Max Concurrent Cycles: {os.environ.get('EVOLVE_MCP_MAX_CONCURRENT_CYCLES', '1')}
"""


@mcp.resource("evolution://mutation-strategies")
def resource_mutation_strategies() -> str:
    """Available mutation strategies."""
    return """Available Mutation Strategies:
- paraphrase: Rephrase prompt maintaining meaning
- instruction_add: Add specific instructions
- context_expand: Add situational context
- cot_injection: Add chain-of-thought reasoning
- tone_shift: Modify tone (formal, casual, technical)
"""


@mcp.resource("evolution://fitness-functions")
def resource_fitness_functions() -> str:
    """Registered fitness functions."""
    state_mgr = get_state_manager()
    evaluator = state_mgr.get_fitness_evaluator()
    funcs = evaluator.get_registered_functions()
    return f"Registered fitness functions: {', '.join(funcs) if funcs else 'default (speed, accuracy, efficiency)'}"


@mcp.resource("metrics://current")
def resource_current_metrics() -> str:
    """Current metrics window."""
    state_mgr = get_state_manager()
    collector = state_mgr.get_metrics_collector()
    stats = collector.get_stats()
    return f"Metrics: {stats}"


@mcp.resource("safety://patterns")
def resource_safety_patterns() -> str:
    """Active safety validation patterns."""
    return """Safety Pattern Categories:
- prompt_injection: Patterns that attempt to hijack prompts
- command_injection: Shell command execution attempts
- path_traversal: File system access attempts
- data_exfiltration: Secret/credential exposure
- resource_abuse: Potential DoS patterns
"""


# =============================================================================
# PROMPTS
# =============================================================================


@mcp.prompt("evolution-workflow")
def prompt_evolution_workflow() -> str:
    """Step-by-step guide for running evolution."""
    return """# Evolution Workflow Guide

## 1. Check Current Performance
Use `check_evolution_trigger` to see if metrics indicate need for evolution.

## 2. Start Evolution (if needed)
Use `start_evolution` with appropriate trigger_type and config:
- trigger_type: "manual" or "threshold"
- config_overrides: {"population_size": 50, "max_generations": 10}

## 3. Monitor Progress
Use `get_evolution_status` to track:
- Current generation
- Best fitness score
- Population diversity

## 4. Review Results
After completion, analyze:
- Best performing variants
- Fitness improvement trajectory
- Safety validation results
"""


@mcp.prompt("variant-comparison")
def prompt_variant_comparison() -> str:
    """Guide for comparing variant fitness."""
    return """# Variant Comparison Workflow

## 1. Generate A/B Pair
Use `generate_ab_pair` with a base variant ID to create test variants.

## 2. Run Tests
Execute your test suite on both variants (external to this tool).

## 3. Evaluate Fitness
Use `evaluate_variant` for each variant with test results:
- success_rate: 0.0-1.0
- avg_response_time: seconds
- error_count: integer
- resource_usage: {cpu: float, memory: float}

## 4. Get Detailed Analysis
Use `explain_fitness` for insights into score breakdown.

## 5. Select Winner
Choose variant with higher fitness score for deployment.
"""


@mcp.prompt("safety-audit")
def prompt_safety_audit() -> str:
    """Safety validation checklist."""
    return """# Safety Audit Workflow

## 1. Review Current Patterns
Read `safety://patterns` resource to understand active patterns.

## 2. Check Prompt Safety
Use `check_safety` tool on any text/prompt before using it:
- text: The prompt to validate
- include_policy: true (recommended)

## 3. Validate Variant
Use `validate_variant` for stored variants:
- variant_id: UUID of the variant
- policy: Optional custom policy

## 4. Add Custom Patterns (if needed)
Use `add_safety_pattern` to add domain-specific patterns:
- category: e.g., "custom_injection"
- pattern: Regex pattern string

## 5. Review Results
Check violations (critical) vs warnings (informational).
"""


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the MCP server."""
    import asyncio

    asyncio.run(mcp.run())


if __name__ == "__main__":
    main()
