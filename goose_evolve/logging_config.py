"""
Structured logging configuration for Goose Evolve.
Provides consistent logging across all components with JSON output support.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "variant_id"):
            log_data["variant_id"] = record.variant_id
        if hasattr(record, "generation"):
            log_data["generation"] = record.generation
        if hasattr(record, "fitness"):
            log_data["fitness"] = record.fitness
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "agent_id"):
            log_data["agent_id"] = record.agent_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """Human-readable log format for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        level = record.levelname[:4]

        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            level = f"{color}{level}{self.RESET}"

        msg = record.getMessage()

        # Add context if present
        extras = []
        if hasattr(record, "variant_id"):
            extras.append(f"variant={record.variant_id[:8]}")
        if hasattr(record, "generation"):
            extras.append(f"gen={record.generation}")
        if hasattr(record, "fitness"):
            extras.append(f"fitness={record.fitness:.4f}")
        if hasattr(record, "duration_ms"):
            extras.append(f"took={record.duration_ms}ms")

        extra_str = f" [{', '.join(extras)}]" if extras else ""

        return f"{ts} {level} {record.name}: {msg}{extra_str}"


class EvolutionLogger:
    """Specialized logger for evolution events with structured context."""

    def __init__(self, name: str = "goose_evolve"):
        self.logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """Set persistent context for all log messages."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()

    def _log(self, level: int, msg: str, **kwargs) -> None:
        """Log with context."""
        extra = {**self._context, **kwargs}
        self.logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    # Evolution-specific logging methods
    def generation_start(self, generation: int, population_size: int) -> None:
        self.info(
            f"Starting generation {generation}",
            event_type="generation_start",
            generation=generation,
            population_size=population_size,
        )

    def generation_complete(
        self, generation: int, best_fitness: float, avg_fitness: float, duration_ms: int
    ) -> None:
        self.info(
            f"Generation {generation} complete",
            event_type="generation_complete",
            generation=generation,
            fitness=best_fitness,
            avg_fitness=avg_fitness,
            duration_ms=duration_ms,
        )

    def variant_created(
        self, variant_id: str, parent_ids: list, mutation_type: str
    ) -> None:
        self.debug(
            f"Created variant via {mutation_type}",
            event_type="variant_created",
            variant_id=variant_id,
            parent_ids=parent_ids,
            mutation_type=mutation_type,
        )

    def variant_evaluated(
        self, variant_id: str, fitness: float, duration_ms: int
    ) -> None:
        self.debug(
            f"Evaluated variant",
            event_type="variant_evaluated",
            variant_id=variant_id,
            fitness=fitness,
            duration_ms=duration_ms,
        )

    def evolution_started(self, config: Dict[str, Any]) -> None:
        self.info(
            "Evolution cycle started",
            event_type="evolution_started",
            config=config,
        )

    def evolution_complete(
        self, generations: int, best_fitness: float, total_duration_ms: int
    ) -> None:
        self.info(
            f"Evolution complete after {generations} generations",
            event_type="evolution_complete",
            generations=generations,
            fitness=best_fitness,
            duration_ms=total_duration_ms,
        )

    def safety_violation(self, variant_id: str, violations: list) -> None:
        self.warning(
            f"Safety violation detected",
            event_type="safety_violation",
            variant_id=variant_id,
            violations=violations,
        )


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[Path] = None,
    use_colors: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON format for console output
        log_file: Optional file path for log output
        use_colors: Use colors in console output (ignored if json_output=True)
    """
    root_logger = logging.getLogger("goose_evolve")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    if json_output:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(HumanFormatter(use_colors=use_colors))
    root_logger.addHandler(console_handler)

    # File handler (always JSON for machine parsing)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Also configure evolution/monitoring loggers
    for name in ["evolution", "monitoring"]:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> EvolutionLogger:
    """Get a structured logger instance."""
    return EvolutionLogger(name)
