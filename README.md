# Goose Evolve

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/kwsantiago/goose-evolve.svg?style=social)](https://github.com/kwsantiago/goose-evolve/stargazers)

## What is Goose Evolve?

Goose Evolve is an open-source MCP extension for [Goose AI](https://goose.ai) that enables autonomous self-improvement. Agents evolve by rewriting prompts, MCP configs, and code via evolutionary algorithms and sandboxed testing. Inspired by Darwin Gödel Machine and AlphaEvolve, it focuses on openness, modularity, and user control.

## Current Status

**4 out of 12 core components implemented** - Basic evolution infrastructure is working, but safety/testing and MCP integration are not yet implemented.

### What's Working
- [x] **Evolution Engine** - Genetic algorithm orchestration with tournament selection
- [x] **Variant Generator** - 5 mutation strategies (paraphrase, instruction_add, context_expand, cot_injection, tone_shift) + crossover
- [x] **Metrics Collector** - Performance monitoring with sliding windows and trigger detection
- [x] **Fitness Evaluator** - Multi-dimensional scoring (speed, accuracy, efficiency) with plugin support

### What's Missing
- [ ] **Safety Validator** - Pre-test validation and injection detection
- [ ] **Sandbox Manager** - Docker-based isolated testing
- [ ] **MCP Integration** - No Goose connection yet
- [ ] **Deployment Manager** - User approval workflow
- [ ] **CLI Interface** - Command-line control
- [ ] **Logging & Telemetry** - Observability infrastructure
- [ ] **Documentation** - Architecture and API docs
- [ ] **Hot-Reload** - Dynamic prompt updates

## Installation

### Prerequisites
- Python 3.9 or higher

### Install from source

```bash
git clone https://github.com/kwsantiago/goose-evolve.git
cd goose-evolve
pip install -e .
```

## Quick Start

**Note**: This example uses mocked components for Safety Validator and Sandbox Manager which are not yet implemented.

```python
import asyncio
from datetime import datetime

from evolution import EvolutionEngine, FitnessEvaluator, VariantGenerator
from evolution.interfaces import EvolutionTriggerEvent, TestResults, ValidationResult


async def main():
    # IMPLEMENTED: These components are fully functional
    variant_generator = VariantGenerator()
    fitness_evaluator = FitnessEvaluator()
    
    # NOT IMPLEMENTED: Using mocks for missing components
    class MockSafetyValidator:
        async def validate(self, variant):
            return ValidationResult(is_valid=True, errors=[])
    
    class MockSandboxManager:
        async def test_variant(self, variant):
            return TestResults(
                variant_id=variant.id,
                success_rate=0.9,
                avg_response_time=1.2,
                error_count=0,
                resource_usage={'memory': 100.0, 'cpu': 50.0}
            )
    
    # Create evolution engine with mix of real and mocked components
    engine = EvolutionEngine(
        variant_generator=variant_generator,
        safety_validator=MockSafetyValidator(),
        sandbox_manager=MockSandboxManager(),
        fitness_evaluator=fitness_evaluator
    )
    
    # Create a trigger event
    trigger = EvolutionTriggerEvent(
        trigger_type="threshold",
        metrics_snapshot={'success_rate': 0.75, 'avg_response_time': 1.5},
        timestamp=datetime.now()
    )
    
    # Run evolution cycle
    result = await engine.start_evolution_cycle(trigger)
    print(f"Evolution completed: {result['best_fitness']}")


# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

## How It Works

The evolution process follows these steps:

1. **Monitor** [x] - MetricsCollector tracks performance with sliding windows
2. **Trigger** [x] - Threshold detection generates evolution events
3. **Generate** [x] - VariantGenerator creates mutations using 5 strategies
4. **Validate** [ ] - SafetyValidator not implemented (mocked in tests)
5. **Test** [ ] - SandboxManager not implemented (mocked in tests)
6. **Evaluate** [x] - FitnessEvaluator scores on speed/accuracy/efficiency
7. **Deploy** [ ] - No deployment mechanism or MCP integration yet

## Development Phases

### Phase 1: Core Infrastructure (4/4 Complete)
- [x] **Evolution Engine** - Full genetic algorithm implementation with tournament selection
- [x] **Variant Generator** - 5 mutation types (paraphrase, instruction_add, context_expand, cot_injection, tone_shift) + crossover
- [x] **Metrics Collector** - Performance monitoring with 2.26% overhead (under 5% target)
- [x] **Fitness Evaluator** - Multi-dimensional scoring (speed, accuracy, efficiency) with plugin support

### Phase 2: Safety & Testing (0/2 Complete)
- [ ] **Safety Validator** - Pre-test validation and injection detection (not started)
- [ ] **Sandbox Manager** - Docker-based isolated testing environment (not started)

### Phase 3: MCP Integration (0/2 Complete)  
- [ ] **Prompt Hot-Reload** - Dynamic prompt updates for Goose agents (not started)
- [ ] **Deployment Manager** - User approval workflow and variant deployment (not started)

### Phase 4: MVP Polish (0/3 Complete)
- [ ] **CLI Interface** - Command-line control and status display (not started)
- [ ] **Logging & Telemetry** - Observability infrastructure and audit logs (not started)  
- [ ] **Core Documentation** - Architecture guides and API reference (not started)

### Test Coverage
- **86 tests passing** (0 failures)
- **4 modules tested**: engine, variants, fitness, metrics
- **Coverage**: ~95% for implemented components


## Roadmap

### Immediate Priorities (Q3 2025)
1. **Safety Validator** - Implement pre-test validation
2. **Docker Sandbox** - Build isolated testing environment
3. **MCP Bridge** - Connect to Goose agents
4. **CLI** - Basic command-line interface

### Next Quarter (Q4 2025)
- User approval workflow
- Logging and telemetry
- Documentation
- Integration testing

## Contributing

We need help with:
- **Safety Validator** implementation
- **Docker Sandbox** setup
- **MCP Integration** for Goose agents
- **CLI Development**
- Custom fitness functions
- Documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
MIT License. See LICENSE for details.

## Contact
- GitHub Issues: Report bugs or suggest features.
- Join Goose Discord for discussions.

