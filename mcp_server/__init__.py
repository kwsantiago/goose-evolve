"""
MCP Server for evolve-mcp.
Universal MCP server for agent self-improvement via evolutionary algorithms.
"""

from .errors import (
    CycleAlreadyRunningError,
    CycleNotFoundError,
    InvalidConfigError,
    MCPEvolutionError,
    ValidationFailedError,
)
from .schemas import (
    EvaluateVariantInput,
    EvolutionStatusOutput,
    GeneratePopulationInput,
    MutationType,
    StartEvolutionInput,
    VariantSummary,
)
from .serializers import (
    EvolutionJSONEncoder,
    deserialize_test_results,
    serialize_test_results,
    serialize_variant,
)
from .state import EvolutionCycleState, EvolutionStateManager

__all__ = [
    # Errors
    "MCPEvolutionError",
    "CycleNotFoundError",
    "CycleAlreadyRunningError",
    "ValidationFailedError",
    "InvalidConfigError",
    # Schemas
    "MutationType",
    "StartEvolutionInput",
    "GeneratePopulationInput",
    "EvaluateVariantInput",
    "VariantSummary",
    "EvolutionStatusOutput",
    # Serializers
    "EvolutionJSONEncoder",
    "serialize_variant",
    "serialize_test_results",
    "deserialize_test_results",
    # State
    "EvolutionCycleState",
    "EvolutionStateManager",
]
