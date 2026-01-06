"""
Serialization utilities for MCP responses.
Handles UUID, datetime, and dataclass conversion to JSON-safe formats.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List
from uuid import UUID


class EvolutionJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for evolution types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return super().default(obj)


def serialize_variant(variant: Any) -> Dict[str, Any]:
    """Convert Variant to MCP-safe dict.

    Args:
        variant: Variant dataclass instance

    Returns:
        JSON-serializable dictionary
    """
    return {
        "id": str(variant.id),
        "parent_ids": [str(pid) for pid in (variant.parent_ids or [])],
        "generation": variant.generation,
        "prompt": variant.prompt,
        "configuration": variant.configuration or {},
        "created_at": (
            variant.created_at.isoformat()
            if variant.created_at
            else datetime.now().isoformat()
        ),
        "fitness_score": variant.fitness_score,
    }


def serialize_variant_summary(variant: Any) -> Dict[str, Any]:
    """Convert Variant to summary format (less data).

    Args:
        variant: Variant dataclass instance

    Returns:
        JSON-serializable summary dictionary
    """
    prompt = variant.prompt or ""
    return {
        "id": str(variant.id),
        "generation": variant.generation,
        "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "fitness_score": variant.fitness_score,
        "created_at": (
            variant.created_at.isoformat()
            if variant.created_at
            else datetime.now().isoformat()
        ),
    }


def serialize_test_results(results: Any) -> Dict[str, Any]:
    """Convert TestResults to MCP-safe dict.

    Args:
        results: TestResults dataclass instance

    Returns:
        JSON-serializable dictionary
    """
    return {
        "variant_id": str(results.variant_id),
        "success_rate": results.success_rate,
        "avg_response_time": results.avg_response_time,
        "error_count": results.error_count,
        "resource_usage": results.resource_usage or {},
    }


def deserialize_test_results(data: Dict[str, Any]) -> Any:
    """Convert MCP dict to TestResults.

    Args:
        data: Dictionary from MCP request

    Returns:
        TestResults dataclass instance

    Raises:
        KeyError: If variant_id is missing from data
    """
    from evolution.interfaces import TestResults

    if "variant_id" not in data:
        raise KeyError("variant_id is required in test results data")

    return TestResults(
        variant_id=UUID(data["variant_id"]),
        success_rate=data.get("success_rate", 0.0),
        avg_response_time=data.get("avg_response_time", 0.0),
        error_count=data.get("error_count", 0),
        resource_usage=data.get("resource_usage", {}),
    )


def serialize_fitness_explanation(explanation: Any) -> Dict[str, Any]:
    """Convert FitnessExplanation to MCP-safe dict.

    Args:
        explanation: FitnessExplanation instance

    Returns:
        JSON-serializable dictionary
    """
    return {
        "individual_scores": explanation.individual_scores,
        "normalized_scores": explanation.normalized_scores,
        "weights": explanation.weights,
        "explanations": explanation.explanations,
        "overall_score": explanation.overall_score,
    }


def serialize_safety_violation(violation: Any) -> Dict[str, Any]:
    """Convert SafetyViolation to MCP-safe dict.

    Args:
        violation: SafetyViolation instance

    Returns:
        JSON-serializable dictionary
    """
    return {
        "rule_name": violation.rule_name,
        "severity": violation.severity,
        "description": violation.description,
        "matched_pattern": violation.matched_pattern,
        "location": violation.location,
    }


def serialize_complexity_metrics(metrics: Any) -> Dict[str, Any]:
    """Convert ComplexityMetrics to MCP-safe dict.

    Args:
        metrics: ComplexityMetrics instance

    Returns:
        JSON-serializable dictionary
    """
    return {
        "sentence_count": metrics.sentence_count,
        "word_count": metrics.word_count,
        "avg_sentence_length": metrics.avg_sentence_length,
        "lexical_diversity": metrics.lexical_diversity,
        "instruction_density": getattr(metrics, "instruction_density", 0.0),
        "complexity_score": getattr(metrics, "complexity_score", 0.0),
    }


def serialize_list(
    items: List[Any], serializer: Callable[[Any], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Serialize a list of items using the given serializer.

    Args:
        items: List of items to serialize
        serializer: Function to serialize each item

    Returns:
        List of serialized dictionaries
    """
    return [serializer(item) for item in items]
