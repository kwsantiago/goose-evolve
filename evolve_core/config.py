"""
Centralized configuration management for evolve-mcp.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvolutionSettings:
    """Evolution algorithm settings."""

    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_count: int = 5
    tournament_size: int = 3
    max_generations: int = 100
    fitness_threshold: float = 0.95
    convergence_generations: int = 10
    convergence_threshold: float = 0.001


@dataclass
class SafetySettings:
    """Safety validation settings."""

    severity_threshold: str = "medium"
    max_prompt_length: int = 50000
    forbidden_patterns: List[str] = field(default_factory=list)
    enable_injection_detection: bool = True


@dataclass
class MetricsSettings:
    """Metrics collection settings."""

    window_size_minutes: int = 60
    retention_hours: int = 168  # 1 week
    anomaly_threshold: float = 2.0
    export_format: str = "json"


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    json_output: bool = False
    use_colors: bool = True
    log_file: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""

    evolution: EvolutionSettings = field(default_factory=EvolutionSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    metrics: MetricsSettings = field(default_factory=MetricsSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    state_dir: str = ".evolve-mcp"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            evolution=EvolutionSettings(**data.get("evolution", {})),
            safety=SafetySettings(**data.get("safety", {})),
            metrics=MetricsSettings(**data.get("metrics", {})),
            logging=LoggingSettings(**data.get("logging", {})),
            state_dir=data.get("state_dir", ".evolve-mcp"),
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to file."""
        if path is None:
            path = Path(self.state_dir) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_config(
    config_path: Optional[Path] = None,
    state_dir: Optional[Path] = None,
) -> Config:
    """Get configuration, loading from file if available.

    Priority:
    1. Explicit config_path
    2. Config in state_dir
    3. Config in current directory
    4. Environment variables
    5. Defaults
    """
    # Try loading from file
    paths_to_try = []

    if config_path:
        paths_to_try.append(config_path)
    if state_dir:
        paths_to_try.append(state_dir / "config.json")
    paths_to_try.extend(
        [
            Path(".evolve-mcp") / "config.json",
            Path("evolve-mcp.json"),
        ]
    )

    for path in paths_to_try:
        if path.exists():
            config = Config.load(path)
            # Apply environment overrides
            _apply_env_overrides(config)
            return config

    # Create default config with env overrides
    config = Config()
    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: Config) -> None:
    """Apply environment variable overrides to config."""
    env_mappings: Dict[str, tuple] = {
        "EVOLVE_MCP_POPULATION_SIZE": ("evolution", "population_size", int),
        "EVOLVE_MCP_MUTATION_RATE": ("evolution", "mutation_rate", float),
        "EVOLVE_MCP_MAX_GENERATIONS": ("evolution", "max_generations", int),
        "EVOLVE_MCP_LOG_LEVEL": ("logging", "level", str),
        "EVOLVE_MCP_LOG_JSON": (
            "logging",
            "json_output",
            lambda x: x.lower() == "true",
        ),
        "EVOLVE_MCP_STATE_DIR": (None, "state_dir", str),
    }

    for env_var, (section, key, converter) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                converted = converter(value)  # type: ignore[operator]
                if section:
                    setattr(getattr(config, section), key, converted)
                else:
                    setattr(config, key, converted)
            except (ValueError, TypeError):
                pass  # Ignore invalid env values
