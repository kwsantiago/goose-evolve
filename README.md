# Goose Evolve

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/kwsantiago/goose-evolve.svg?style=social)](https://github.com/kwsantiago/goose-evolve/stargazers)

## Overview

Goose Evolve is an open-source MCP extension for [Goose AI](https://goose.ai) that enables autonomous self-improvement. Agents evolve by rewriting prompts, MCP configs, and code via evolutionary algorithms and sandboxed testing. Inspired by Darwin Gödel Machine and AlphaEvolve, it focuses on openness, modularity, and user control.

### Key Features (Planned)
- **Self-Improvement Loop:** Monitor performance, generate variants, test, and deploy.
- **Multi-Modal Support:** Voice/emotion, sketches, IoT tools.
- **Safety-First:** Sandbox isolation, validation, and rollback.
- **Extensible:** Custom fitness functions and mutators.

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

```python
import asyncio
from datetime import datetime

from evolution import EvolutionEngine, FitnessEvaluator, VariantGenerator
from evolution.interfaces import EvolutionTriggerEvent, TestResults, ValidationResult


async def main():
    # Create required components
    variant_generator = VariantGenerator()
    fitness_evaluator = FitnessEvaluator()
    
    # Given MVP status, we'll use mock safety validator and sandbox manager
    # In production, these would be implemented properly
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
    
    # Create evolution engine
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

## Development Status

### What Works Now
- [X] Evolution engine with genetic algorithms
- [X] Prompt mutations and crossover  
- [X] Fitness evaluation (speed, accuracy, efficiency)
- [X] Performance monitoring and trigger detection

### Currently Mocked (Need Implementation)
- Safety validator (basic validation exists in variant generator)
- Sandbox manager (Docker-based testing)
- MCP integration (no Goose connection yet)
- User approval workflow

### Next Steps
- Build proper safety validator with injection detection
- Implement Docker sandbox for isolated variant testing
- Add MCP integration for Goose agent prompt hot-reload  
- Create CLI for evolution control and approvals

### Later
- Configuration evolution beyond prompts
- Multi-modal support
- Advanced optimization strategies

## Contributing
We welcome contributions! See CONTRIBUTING.md for guidelines. Focus on custom mutators or fitness functions.

## License
MIT License. See LICENSE for details.

## Contact
- GitHub Issues: Report bugs or suggest features.
- Join Goose Discord for discussions.

