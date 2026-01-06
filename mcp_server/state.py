"""
State management for MCP server.
Manages evolution cycle lifecycle across MCP requests.
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from evolution import EvolutionConfig, EvolutionEngine, VariantGenerator
from evolution.fitness import FitnessEvaluator
from evolution.interfaces import (
    EvolutionTriggerEvent,
    TestResults,
    ValidationResult,
    Variant,
)
from evolve_core.safety import DefaultSafetyValidator, create_default_policy
from monitoring import MetricsCollector


class MockSandboxManager:
    """Mock sandbox manager for MCP server (sandbox not yet implemented)."""

    async def test_variant(self, variant: Variant) -> TestResults:
        """Return mock test results."""
        return TestResults(
            variant_id=variant.id,
            success_rate=0.9,
            avg_response_time=1.0,
            error_count=0,
            resource_usage={"memory": 100.0, "cpu": 50.0},
        )


@dataclass
class EvolutionCycleState:
    """State for a single evolution cycle."""

    cycle_id: UUID
    status: str = "pending"  # pending, running, paused, completed, cancelled, failed
    engine: Optional[EvolutionEngine] = None
    trigger: Optional[EvolutionTriggerEvent] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_population: List[Variant] = field(default_factory=list)
    fitness_history: List[Dict[str, Any]] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    error: Optional[str] = None


class EvolutionStateManager:
    """Manages lifecycle of evolution cycles across MCP requests."""

    def __init__(self):
        self._cycles: Dict[str, EvolutionCycleState] = {}
        self._current_cycle_id: Optional[str] = None
        self._lock = asyncio.Lock()

        # Shared component instances (lazy initialized)
        self._variant_generator: Optional[VariantGenerator] = None
        self._fitness_evaluator: Optional[FitnessEvaluator] = None
        self._safety_validator: Optional[DefaultSafetyValidator] = None
        self._metrics_collector: Optional[MetricsCollector] = None

        # Variant storage for cross-request access
        self._variants: Dict[str, Variant] = {}

        # Configuration
        self._max_concurrent_cycles = int(
            os.environ.get("EVOLVE_MCP_MAX_CONCURRENT_CYCLES", "1")
        )
        self._checkpoint_dir = os.environ.get(
            "EVOLVE_MCP_CHECKPOINT_DIR", ".evolve-mcp/checkpoints"
        )

    def get_variant_generator(self) -> VariantGenerator:
        """Get or create the variant generator."""
        if self._variant_generator is None:
            self._variant_generator = VariantGenerator()
        return self._variant_generator

    def get_fitness_evaluator(self) -> FitnessEvaluator:
        """Get or create the fitness evaluator."""
        if self._fitness_evaluator is None:
            self._fitness_evaluator = FitnessEvaluator()
        return self._fitness_evaluator

    def get_safety_validator(self) -> DefaultSafetyValidator:
        """Get or create the safety validator."""
        if self._safety_validator is None:
            self._safety_validator = DefaultSafetyValidator()
        return self._safety_validator

    def get_metrics_collector(self) -> MetricsCollector:
        """Get or create the metrics collector."""
        if self._metrics_collector is None:
            self._metrics_collector = MetricsCollector()
        return self._metrics_collector

    def has_running_cycle(self) -> bool:
        """Check if there's a currently running evolution cycle."""
        return any(c.status == "running" for c in self._cycles.values())

    def get_current_cycle_id(self) -> Optional[str]:
        """Get the current/latest cycle ID."""
        if self._current_cycle_id:
            return self._current_cycle_id
        # Return most recent cycle
        if self._cycles:
            return max(
                self._cycles.keys(),
                key=lambda k: self._cycles[k].started_at or datetime.min,
            )
        return None

    def get_running_count(self) -> int:
        """Get the number of currently running cycles."""
        return sum(1 for c in self._cycles.values() if c.status == "running")

    async def start_cycle(
        self, trigger: EvolutionTriggerEvent, config: EvolutionConfig
    ) -> str:
        """Start a new evolution cycle.

        Args:
            trigger: The trigger event that started evolution
            config: Evolution configuration

        Returns:
            cycle_id: Unique identifier for the new cycle

        Raises:
            CycleAlreadyRunningError: If max concurrent cycles reached
        """
        async with self._lock:
            if self.get_running_count() >= self._max_concurrent_cycles:
                running_id = next(
                    (k for k, v in self._cycles.items() if v.status == "running"),
                    "unknown",
                )
                from .errors import CycleAlreadyRunningError

                raise CycleAlreadyRunningError(running_id)

            cycle_id = str(uuid4())

            # Create engine
            engine = EvolutionEngine(
                variant_generator=self.get_variant_generator(),
                safety_validator=self.get_safety_validator(),
                sandbox_manager=MockSandboxManager(),
                fitness_evaluator=self.get_fitness_evaluator(),
                config=config,
            )

            # Create cycle state
            state = EvolutionCycleState(
                cycle_id=UUID(cycle_id),
                status="running",
                engine=engine,
                trigger=trigger,
                started_at=datetime.now(),
            )

            self._cycles[cycle_id] = state
            self._current_cycle_id = cycle_id

            # Start evolution in background
            async def run_evolution():
                try:
                    result = await engine.start_evolution_cycle(trigger)
                    state.status = "completed"
                    state.completed_at = datetime.now()
                    # Store winning variants
                    if result and hasattr(result, "best_variants"):
                        for v in result.best_variants:
                            self._variants[str(v.id)] = v
                except asyncio.CancelledError:
                    state.status = "cancelled"
                except Exception as e:
                    state.status = "failed"
                    state.error = str(e)

            state.task = asyncio.create_task(run_evolution())

            return cycle_id

    async def get_cycle(self, cycle_id: str) -> Optional[EvolutionCycleState]:
        """Get a cycle by ID."""
        return self._cycles.get(cycle_id)

    async def cancel_cycle(self, cycle_id: str) -> bool:
        """Cancel a running cycle.

        Args:
            cycle_id: ID of cycle to cancel

        Returns:
            True if cancelled, False if not found
        """
        async with self._lock:
            state = self._cycles.get(cycle_id)
            if state is None:
                return False

            if state.task and not state.task.done():
                state.task.cancel()
                try:
                    await state.task
                except asyncio.CancelledError:
                    pass

            state.status = "cancelled"
            state.completed_at = datetime.now()
            return True

    def store_variant(self, variant: Variant) -> None:
        """Store a variant for later retrieval."""
        self._variants[str(variant.id)] = variant

    def store_variants(self, variants: List[Variant]) -> None:
        """Store multiple variants."""
        for v in variants:
            self._variants[str(v.id)] = v

    def get_variant_by_id(self, variant_id: str) -> Optional[Variant]:
        """Get a variant by ID."""
        return self._variants.get(variant_id)

    def get_all_variants(self) -> List[Variant]:
        """Get all stored variants."""
        return list(self._variants.values())

    def clear_variants(self) -> None:
        """Clear all stored variants."""
        self._variants.clear()

    async def cleanup_completed(self, max_age_hours: int = 24) -> int:
        """Clean up old completed cycles.

        Args:
            max_age_hours: Maximum age in hours for completed cycles

        Returns:
            Number of cycles cleaned up
        """
        async with self._lock:
            cutoff = datetime.now()
            to_remove = []

            for cycle_id, state in self._cycles.items():
                if state.status in ("completed", "cancelled", "failed"):
                    if state.completed_at:
                        age_hours = (cutoff - state.completed_at).total_seconds() / 3600
                        if age_hours > max_age_hours:
                            to_remove.append(cycle_id)

            for cycle_id in to_remove:
                del self._cycles[cycle_id]

            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        return {
            "total_cycles": len(self._cycles),
            "running_cycles": self.get_running_count(),
            "stored_variants": len(self._variants),
            "max_concurrent": self._max_concurrent_cycles,
            "cycles_by_status": {
                status: sum(1 for c in self._cycles.values() if c.status == status)
                for status in ["pending", "running", "completed", "cancelled", "failed"]
            },
        }


# Global state manager instance
_state_manager: Optional[EvolutionStateManager] = None


def get_state_manager() -> EvolutionStateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = EvolutionStateManager()
    return _state_manager


def reset_state_manager() -> None:
    """Reset the global state manager (for testing)."""
    global _state_manager
    _state_manager = None
