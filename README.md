# evolve-mcp

Universal MCP server for agent self-improvement via evolutionary algorithms.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## What is evolve-mcp?

evolve-mcp is an MCP (Model Context Protocol) server that enables autonomous self-improvement for AI agents. It works with **Claude Code**, **Goose**, **ChatGPT**, and any MCP-compatible client.

Agents evolve by:
- Mutating and optimizing prompts using genetic algorithms
- Evaluating fitness through configurable metrics
- Validating safety before deployment
- Tracking performance over time

Inspired by Darwin Gödel Machine and AlphaEvolve, it focuses on openness, modularity, and user control with a local-first design.

## Features

- **21 MCP Tools** for complete evolution control
- **5 Mutation Strategies**: paraphrase, instruction_add, context_expand, cot_injection, tone_shift
- **Pluggable Fitness Functions** with weighted scoring
- **Safety Validation** with injection detection
- **Metrics Collection** with anomaly detection
- **Works with any MCP client**

## Quick Start

### Installation

```bash
git clone https://github.com/privkeyio/evolve-mcp.git
cd evolve-mcp
uv sync --all-extras
```

### Claude Code

```bash
claude mcp add evolve-mcp -- python -m mcp_server.server
```

### Goose

```yaml
# ~/.config/goose/config.yaml
extensions:
  - name: evolve-mcp
    type: mcp
    command: python -m mcp_server.server
```

## Usage Examples

### Start an Evolution Cycle

```python
# Use the start_evolution tool
{
  "trigger_type": "manual",
  "config_overrides": {
    "population_size": 50,
    "max_generations": 10
  }
}
```

### Mutate a Prompt

```python
# Use the mutate_prompt tool
{
  "prompt": "You are a helpful assistant",
  "mutation_type": "instruction_add"
}
```

### Check Safety

```python
# Use the check_safety tool
{
  "text": "Your prompt here",
  "include_policy": true
}
```

## MCP Tools Reference

### Evolution Lifecycle (4 tools)

| Tool | Description |
|------|-------------|
| `start_evolution` | Begin an evolution cycle |
| `get_evolution_status` | Check cycle progress |
| `cancel_evolution` | Stop a running cycle |
| `resume_evolution` | Resume from checkpoint |

### Variant Generation (5 tools)

| Tool | Description |
|------|-------------|
| `generate_population` | Create variant population |
| `mutate_prompt` | Apply specific mutation |
| `crossover_variants` | Combine two variants |
| `generate_ab_pair` | Create A/B test pair |
| `analyze_prompt` | Get complexity metrics |

### Fitness Evaluation (5 tools)

| Tool | Description |
|------|-------------|
| `evaluate_variant` | Calculate fitness score |
| `explain_fitness` | Detailed breakdown |
| `register_fitness_function` | Add custom metric |
| `update_fitness_weights` | Adjust weights |
| `list_fitness_functions` | List available |

### Safety Validation (3 tools)

| Tool | Description |
|------|-------------|
| `validate_variant` | Full safety check |
| `check_safety` | Quick text check |
| `add_safety_pattern` | Add custom pattern |

### Metrics (4 tools)

| Tool | Description |
|------|-------------|
| `record_metrics` | Log performance data |
| `get_metrics_window` | Aggregated metrics |
| `check_evolution_trigger` | Should evolve? |
| `detect_anomalies` | Find anomalies |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVOLVE_MCP_POPULATION_SIZE` | `50` | Default population size |
| `EVOLVE_MCP_MAX_GENERATIONS` | `10` | Default max generations |
| `EVOLVE_MCP_FITNESS_THRESHOLD` | `0.95` | Early stop threshold |
| `EVOLVE_MCP_MAX_CONCURRENT_CYCLES` | `1` | Parallel evolution limit |
| `EVOLVE_MCP_CHECKPOINT_DIR` | `.evolve-mcp/checkpoints` | State storage |

## Architecture

```text
evolve-mcp/
├── mcp_server/          # MCP integration layer
│   ├── server.py        # FastMCP server with 21 tools
│   ├── state.py         # Cycle state management
│   ├── schemas.py       # Pydantic models
│   ├── serializers.py   # JSON serialization
│   └── errors.py        # Error handling
├── evolution/           # Core evolution engine
│   ├── engine.py        # Genetic algorithm orchestration
│   ├── variants.py      # Mutation & crossover
│   ├── fitness.py       # Fitness evaluation
│   └── interfaces.py    # Abstract interfaces
├── evolve_core/         # Infrastructure
│   ├── safety.py        # Safety validation
│   ├── config.py        # Configuration
│   └── logging_config.py
└── monitoring/          # Metrics collection
    └── metrics.py
```

## Current Status

### Production-Ready Components
- **Evolution Engine** - Full genetic algorithm orchestration with state persistence
- **Variant Generator** - 5 mutation strategies with deterministic mode
- **Fitness Evaluator** - Pareto optimization, parallel evaluation
- **Metrics Collector** - Multi-agent support, anomaly detection
- **Safety Validator** - Injection detection, policy enforcement
- **MCP Server** - 21 tools, 6 resources, 3 prompts

### Test Coverage
- **173+ tests passing** including integration tests
- **~95% coverage** for core components

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type check
uv run mypy evolution monitoring evolve_core mcp_server
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

AGPL-3.0 License - see [LICENSE](LICENSE) for details.

## Support

- GitHub Issues: Report bugs or suggest features
- See [architecture.md](docs/architecture.md) for detailed system design

## Citation

```bibtex
@software{evolve-mcp,
  title = {evolve-mcp: Universal MCP Server for Agent Self-Improvement},
  author = {PrivKey LLC},
  year = {2025},
  url = {https://github.com/privkeyio/evolve-mcp}
}
```
