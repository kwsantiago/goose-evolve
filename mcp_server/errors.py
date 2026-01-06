"""
Error handling for MCP server.
Provides structured errors that translate to MCP-friendly responses.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional, cast


class MCPEvolutionError(Exception):
    """Base exception for MCP evolution errors."""

    def __init__(
        self,
        message: str,
        code: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_mcp_error(self) -> Dict[str, Any]:
        """Convert to MCP-friendly error response."""
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "details": self.details,
            }
        }


class CycleNotFoundError(MCPEvolutionError):
    """Raised when an evolution cycle is not found."""

    def __init__(self, cycle_id: str):
        super().__init__(
            f"Evolution cycle not found: {cycle_id}",
            "CYCLE_NOT_FOUND",
            {"cycle_id": cycle_id},
        )


class CycleAlreadyRunningError(MCPEvolutionError):
    """Raised when trying to start a cycle while one is already running."""

    def __init__(self, cycle_id: str):
        super().__init__(
            f"Evolution cycle already running: {cycle_id}",
            "CYCLE_ALREADY_RUNNING",
            {"cycle_id": cycle_id},
        )


class VariantNotFoundError(MCPEvolutionError):
    """Raised when a variant is not found."""

    def __init__(self, variant_id: str):
        super().__init__(
            f"Variant not found: {variant_id}",
            "VARIANT_NOT_FOUND",
            {"variant_id": variant_id},
        )


class ValidationFailedError(MCPEvolutionError):
    """Raised when safety validation fails."""

    def __init__(self, violations: list):
        super().__init__(
            "Safety validation failed",
            "VALIDATION_FAILED",
            {"violations": violations},
        )


class InvalidConfigError(MCPEvolutionError):
    """Raised when configuration is invalid."""

    def __init__(self, errors: list):
        super().__init__(
            "Invalid configuration",
            "INVALID_CONFIG",
            {"errors": errors},
        )


class InvalidMutationTypeError(MCPEvolutionError):
    """Raised when mutation type is invalid."""

    def __init__(self, mutation_type: str, valid_types: list):
        super().__init__(
            f"Invalid mutation type: {mutation_type}",
            "INVALID_MUTATION_TYPE",
            {"mutation_type": mutation_type, "valid_types": valid_types},
        )


class FitnessFunctionNotFoundError(MCPEvolutionError):
    """Raised when a fitness function is not found."""

    def __init__(self, name: str, available: list):
        super().__init__(
            f"Fitness function not found: {name}",
            "FITNESS_FUNCTION_NOT_FOUND",
            {"name": name, "available": available},
        )


def handle_mcp_error(func: Callable) -> Callable:
    """Decorator to convert exceptions to MCP-friendly responses.

    Usage:
        @handle_mcp_error
        async def my_tool(...):
            ...
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            return cast(Dict[str, Any], await func(*args, **kwargs))
        except MCPEvolutionError as e:
            return e.to_mcp_error()
        except Exception as e:
            return {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e),
                    "details": {"type": type(e).__name__},
                }
            }

    @wraps(func)
    def sync_wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            return cast(Dict[str, Any], func(*args, **kwargs))
        except MCPEvolutionError as e:
            return e.to_mcp_error()
        except Exception as e:
            return {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e),
                    "details": {"type": type(e).__name__},
                }
            }

    # Check if the function is async
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
