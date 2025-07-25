"""Goose Evolve: Core Interface Definitions"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

# Enumerations (Core Only)


class EvolutionStrategy(Enum):
    """Supported strategies; start with GA."""

    GENETIC_ALGORITHM = "ga"
    EVOLUTION_STRATEGY = "es"  # P2 extension


class SelectionMethod(Enum):
    """Basic selection; tournament for MVP."""

    TOURNAMENT = "tournament"
    ELITISM = "elitism"


class PromptMutationType(Enum):
    """Prompt mutations; focus on basics."""

    PARAPHRASE = "paraphrase"
    INSTRUCTION_ADD = "instruction_add"
    CONTEXT_EXPAND = "context_expand"
    COT_INJECTION = "cot_injection"
    TONE_SHIFT = "tone_shift"


class CrossoverType(Enum):
    """Crossover; uniform for MVP."""

    UNIFORM = "uniform"


class DeploymentType(Enum):
    """Deployment; full/rollback for safety."""

    FULL = "full"
    ROLLBACK = "rollback"


# Data Classes (Essentials)


@dataclass
class Variant:
    """Agent variant; core fields only."""

    id: UUID
    parent_ids: List[UUID]
    generation: int
    prompt: str
    configuration: Dict[str, Any]
    created_at: datetime
    fitness_score: Optional[float] = None


@dataclass
class TestResults:
    """Sandbox results; basic metrics."""

    variant_id: UUID
    success_rate: float
    avg_response_time: float
    error_count: int
    resource_usage: Dict[str, float]


@dataclass
class SafetyPolicy:
    """Safety constraints; minimal policy."""

    max_resource_usage: Dict[str, float]
    forbidden_patterns: List[str]


@dataclass
class ValidationResult:
    """Validation output."""

    is_valid: bool
    errors: List[str]


@dataclass
class EvolutionTriggerEvent:
    """Trigger event; simple."""

    trigger_type: str  # "threshold", "manual"
    metrics_snapshot: Dict[str, float]
    timestamp: datetime
    trigger_reasons: Optional[List[str]] = None  # Optional list of trigger reasons


# Core Interfaces (Simplified Abstracts)


class EvolutionEngine(ABC):
    """Orchestrator; async for feasibility."""

    @abstractmethod
    async def start_evolution_cycle(
        self, trigger: EvolutionTriggerEvent
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def evaluate_population(
        self, population: List[Variant]
    ) -> List[Tuple[Variant, float]]:
        pass

    @abstractmethod
    def select_survivors(self, evaluated: List[Tuple[Variant, float]]) -> List[Variant]:
        pass


class FitnessEvaluator(ABC):
    """Evaluator; pluggable basics."""

    @abstractmethod
    async def evaluate(self, variant: Variant, results: TestResults) -> float:
        pass

    @abstractmethod
    def register_fitness_function(
        self, name: str, fn: Callable[[Variant, TestResults], float]
    ):
        pass


class VariantGenerator(ABC):
    """Generator; mutation-focused."""

    @abstractmethod
    def generate_population(self, base_prompt: str, size: int) -> List[Variant]:
        pass

    @abstractmethod
    def mutate_prompt(self, prompt: str, type: PromptMutationType) -> str:
        pass

    @abstractmethod
    def crossover(self, p1: Variant, p2: Variant) -> Tuple[Variant, Variant]:
        pass


class SafetyValidator(ABC):
    """Validator; core checks."""

    @abstractmethod
    async def validate(
        self, variant: Variant, policy: SafetyPolicy
    ) -> ValidationResult:
        pass


# Extension Points (Minimal)


class MutationOperator(ABC):
    """Custom mutator base."""

    @abstractmethod
    def mutate(self, value: Any) -> Any:
        pass


# Constants (Defaults for MVP)

DEFAULT_POPULATION_SIZE = 50
DEFAULT_MUTATION_RATE = 0.1
