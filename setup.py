"""Setup script for Goose Evolve"""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="goose-evolve",
    version="0.1.0",
    description="Self-improvement MCP extension for Goose agents",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "pytest-mock", "black", "isort", "mypy"],
    },
)
