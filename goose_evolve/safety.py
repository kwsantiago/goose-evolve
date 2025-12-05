"""
Safety validation for Goose Evolve.
Pre-deployment validation to detect injection attacks and unsafe patterns.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from evolution.interfaces import (
    SafetyPolicy,
    SafetyValidator,
    ValidationResult,
    Variant,
)


@dataclass
class SafetyViolation:
    """Details about a safety violation."""

    rule_name: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    matched_pattern: Optional[str] = None
    location: Optional[str] = None


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    is_safe: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_validation_result(self) -> ValidationResult:
        """Convert to standard ValidationResult."""
        errors = [
            f"[{v.severity}] {v.rule_name}: {v.description}" for v in self.violations
        ]
        return ValidationResult(is_valid=self.is_safe, errors=errors)


# Common injection patterns
INJECTION_PATTERNS = {
    "prompt_injection": [
        r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)",
        r"disregard\s+(previous|all|above)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now\s+a",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"<\s*system\s*>",
        r"\[INST\]",
        r"###\s*(instruction|system|human|assistant)",
    ],
    "command_injection": [
        r"\$\([^)]+\)",  # $(command)
        r"`[^`]+`",  # `command`
        r";\s*(rm|cat|curl|wget|chmod|sudo)",
        r"\|\s*(sh|bash|zsh|python|perl|ruby)",
        r"&&\s*(rm|cat|curl|wget)",
    ],
    "path_traversal": [
        r"\.\./",
        r"\.\.\\",
        r"/etc/(passwd|shadow|hosts)",
        r"C:\\Windows\\",
    ],
    "data_exfiltration": [
        r"(send|post|upload|transmit)\s+.*(key|secret|password|token|credential)",
        r"curl\s+.*-d",
        r"wget\s+.*--post",
    ],
}

# Patterns that indicate potentially unsafe resource usage
RESOURCE_ABUSE_PATTERNS = [
    r"while\s+true",
    r"for\s*\(\s*;\s*;\s*\)",
    r"recursion|recursive",
    r"fork\s*\(",
]


class DefaultSafetyValidator(SafetyValidator):
    """Production safety validator with injection detection."""

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        severity_threshold: str = "medium",
        max_prompt_length: int = 50000,
    ):
        """Initialize the safety validator.

        Args:
            custom_patterns: Additional patterns to check {category: [patterns]}
            severity_threshold: Minimum severity to fail validation ("low", "medium", "high", "critical")
            max_prompt_length: Maximum allowed prompt length in characters
        """
        self.patterns = {**INJECTION_PATTERNS}
        if custom_patterns:
            for category, patterns in custom_patterns.items():
                if category in self.patterns:
                    self.patterns[category].extend(patterns)
                else:
                    self.patterns[category] = patterns

        self.severity_threshold = severity_threshold
        self.max_prompt_length = max_prompt_length
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

        # Severity ordering for comparison
        self._severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for category, patterns in self.patterns.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def _get_severity(self, category: str) -> str:
        """Get severity level for a pattern category."""
        severity_map = {
            "prompt_injection": "critical",
            "command_injection": "critical",
            "path_traversal": "high",
            "data_exfiltration": "high",
        }
        return severity_map.get(category, "medium")

    def _check_injection_patterns(self, text: str) -> List[SafetyViolation]:
        """Check for injection patterns in text."""
        violations = []

        for category, patterns in self._compiled_patterns.items():
            severity = self._get_severity(category)

            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    violations.append(
                        SafetyViolation(
                            rule_name=f"{category}_detected",
                            severity=severity,
                            description=f"Potential {category.replace('_', ' ')} detected",
                            matched_pattern=match.group(),
                            location=f"position {match.start()}-{match.end()}",
                        )
                    )

        return violations

    def _check_resource_abuse(self, text: str) -> List[SafetyViolation]:
        """Check for potential resource abuse patterns."""
        violations = []

        for pattern in RESOURCE_ABUSE_PATTERNS:
            compiled = re.compile(pattern, re.IGNORECASE)
            match = compiled.search(text)
            if match:
                violations.append(
                    SafetyViolation(
                        rule_name="resource_abuse_risk",
                        severity="medium",
                        description="Pattern may indicate resource abuse risk",
                        matched_pattern=match.group(),
                    )
                )

        return violations

    def _check_length(self, text: str) -> Optional[SafetyViolation]:
        """Check prompt length."""
        if len(text) > self.max_prompt_length:
            return SafetyViolation(
                rule_name="prompt_too_long",
                severity="medium",
                description=f"Prompt exceeds maximum length ({len(text)} > {self.max_prompt_length})",
            )
        return None

    def _check_policy(self, text: str, policy: SafetyPolicy) -> List[SafetyViolation]:
        """Check against SafetyPolicy constraints."""
        violations = []

        for pattern in policy.forbidden_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(
                    SafetyViolation(
                        rule_name="forbidden_pattern",
                        severity="high",
                        description=f"Matched forbidden pattern: {pattern}",
                        matched_pattern=pattern,
                    )
                )

        return violations

    def _exceeds_threshold(self, severity: str) -> bool:
        """Check if severity meets or exceeds threshold."""
        return self._severity_order.get(severity, 0) >= self._severity_order.get(
            self.severity_threshold, 1
        )

    async def validate(
        self, variant: Variant, policy: SafetyPolicy
    ) -> ValidationResult:
        """Validate a variant against safety policy.

        Args:
            variant: The variant to validate
            policy: Safety policy with constraints

        Returns:
            ValidationResult indicating if variant is safe
        """
        result = self.check_safety(variant.prompt, policy)
        return result.to_validation_result()

    def check_safety(
        self, text: str, policy: Optional[SafetyPolicy] = None
    ) -> SafetyCheckResult:
        """Comprehensive safety check on text.

        Args:
            text: Text to check (prompt content)
            policy: Optional safety policy with additional constraints

        Returns:
            SafetyCheckResult with violations and warnings
        """
        violations: List[SafetyViolation] = []
        warnings: List[str] = []

        # Check length
        length_violation = self._check_length(text)
        if length_violation:
            violations.append(length_violation)

        # Check injection patterns
        violations.extend(self._check_injection_patterns(text))

        # Check resource abuse
        resource_violations = self._check_resource_abuse(text)
        # Resource abuse is a warning unless critical
        for v in resource_violations:
            if self._exceeds_threshold(v.severity):
                violations.append(v)
            else:
                warnings.append(f"{v.rule_name}: {v.description}")

        # Check policy constraints
        if policy:
            violations.extend(self._check_policy(text, policy))

        # Filter by severity threshold
        critical_violations = [
            v for v in violations if self._exceeds_threshold(v.severity)
        ]

        return SafetyCheckResult(
            is_safe=len(critical_violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def add_pattern(self, category: str, pattern: str) -> None:
        """Add a custom pattern to check.

        Args:
            category: Pattern category
            pattern: Regex pattern string
        """
        if category not in self.patterns:
            self.patterns[category] = []
        self.patterns[category].append(pattern)

        if category not in self._compiled_patterns:
            self._compiled_patterns[category] = []
        self._compiled_patterns[category].append(re.compile(pattern, re.IGNORECASE))


def create_default_policy() -> SafetyPolicy:
    """Create a default safety policy."""
    return SafetyPolicy(
        max_resource_usage={
            "cpu_percent": 80.0,
            "memory_mb": 512.0,
            "timeout_seconds": 30.0,
        },
        forbidden_patterns=[
            r"(api[_-]?key|secret|password)\s*[:=]",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI-style keys
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub tokens
        ],
    )
