# Contributing to evolve-mcp

Thank you for your interest in contributing to evolve-mcp! We welcome contributions that improve autonomous agent evolution through genetic algorithms.

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### Fork and Clone

```bash
git clone https://github.com/privkeyio/evolve-mcp.git
cd evolve-mcp
git checkout -b your-feature-branch
```

### Development Setup

```bash
# Install in development mode with all dependencies
uv sync --all-extras

# Run tests to verify setup
uv run pytest

# Check code formatting
uv run black --check .
uv run isort --check-only .
uv run mypy evolution/ monitoring/
```

## Types of Contributions

### 1. Custom Mutators
Extend the mutation system to create new prompt transformation strategies.

Example: Tone Adjustment Mutator
```python
from evolution.interfaces import PromptMutationType
from evolution.variants import VariantGenerator

class ToneAdjustMutator(VariantGenerator):
    def mutate_prompt(self, prompt: str, mutation_type: PromptMutationType) -> str:
        if mutation_type == PromptMutationType.TONE_SHIFT:
            return "Please respond in a friendly tone: " + prompt
        return super().mutate_prompt(prompt, mutation_type)
```

### 2. Fitness Functions
Develop new performance metrics to evaluate agent effectiveness.

Example: Response Speed Evaluator
```python
from evolution.interfaces import TestResults
from evolution.fitness import FitnessEvaluator

async def response_speed_fitness(variant, results: TestResults) -> float:
    """Custom fitness function that prioritizes faster response times."""
    if results.avg_response_time <= 0:
        return 0.0
    return 1.0 / results.avg_response_time

# Register the custom fitness function
evaluator = FitnessEvaluator()
evaluator.register_fitness_function("response_speed", response_speed_fitness, weight=0.3)
```

### 3. Safety Validators
Implement validation rules to ensure variant safety.

Example: Resource Limit Validator
```python
from evolution.interfaces import ValidationResult, Variant

class ResourceLimitValidator:
    async def validate(self, variant: Variant) -> ValidationResult:
        """Validate that variant doesn't exceed resource limits."""
        errors = []
        
        # Check prompt length
        if len(variant.prompt) > 10000:
            errors.append("Prompt exceeds maximum length")
        
        # Check configuration limits
        if variant.configuration.get('max_tokens', 0) > 4000:
            errors.append("max_tokens exceeds limit")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
```

## Code Style and Standards

### Formatting
- Use Python 3.12+ features where appropriate
- Follow PEP 8 formatting guidelines
- Run `black .` for automatic formatting
- Use `flake8` for linting

### Testing
- Write unit tests with `pytest`
- Aim for 80% test coverage minimum
- Include integration tests for new components
- Run tests: `pytest tests/`

### Documentation
- Document all functions with clear docstrings
- Use type hints consistently
- Update README.md for user-facing changes
- Add examples in `examples/` directory

## Pull Request Process

### 1. Prepare Your Changes
```bash
# Make your changes
git add .
git commit -m "Add ToneAdjustMutator for friendly prompts"
git push origin your-feature-branch
```

### 2. Create Pull Request
- Open a PR against the `main` branch
- Provide a clear description of changes
- Link to related issues (e.g., "Fixes #123")
- Ensure all CI checks pass

### 3. Review Process
- PRs are typically reviewed within 48 hours
- Address feedback promptly
- Maintain a clean commit history
- Be responsive to reviewer suggestions

## Non-Code Contributions

### Documentation
- Improve README.md clarity
- Create tutorials in `docs/`
- Update architecture documentation
- Write usage examples

### Issue Reporting
- Report bugs with reproduction steps
- Suggest new features with use cases
- Provide clear problem descriptions
- Include system information when relevant

### Community Support
- Help answer questions in issues
- Share usage experiences
- Provide feedback on new features
- Participate in design discussions

## Contribution Guidelines

### Be Respectful
- Provide constructive feedback
- Be patient with review processes
- Help maintain a welcoming community

### Stay Aligned
Contributions should enhance:
- **Openness**: Keep the system transparent and accessible
- **Modularity**: Maintain clean, extensible interfaces  
- **Safety**: Prioritize secure evolution practices
- **Performance**: Optimize for efficiency and scalability

### Scope Management
- Focus on small, targeted changes
- Break large features into smaller PRs
- Discuss major changes in issues first
- Maintain backward compatibility when possible

## Key Areas for Contribution

### High Priority
- **Custom Mutators**: New prompt transformation strategies
- **Fitness Functions**: Novel performance evaluation metrics
- **Safety Validators**: Enhanced security and validation rules
- **Documentation**: Usage guides and API documentation

### Medium Priority
- **Evolution Strategies**: Alternative optimization algorithms
- **MCP Integration**: Enhanced Goose agent compatibility
- **Performance Optimization**: Speed and memory improvements
- **Testing**: Expanded test coverage and scenarios

### Future Enhancements
- **Multi-modal Support**: Voice, vision, and IoT capabilities
- **Distributed Evolution**: Multi-node processing support
- **Advanced Analytics**: Deep performance insights
- **UI/Dashboard**: Visual evolution monitoring

## License

All contributions are licensed under the MIT License. By contributing, you agree to license your work under the same terms.

## Questions and Support

- **GitHub Issues**: For bugs and feature requests
- **Goose Discord**: For discussions and community support
- **Email**: For private inquiries (see maintainer profiles)

Happy evolving!

---

