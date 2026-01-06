"""
CLI interface for evolve-mcp.
Provides commands for monitoring and controlling evolution cycles.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from evolution import EvolutionConfig, EvolutionEngine, VariantGenerator
from evolution.fitness import FitnessEvaluator
from monitoring import MetricsCollector


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="evolve-mcp",
        description="Universal MCP server for agent self-improvement",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show evolution status")
    status_parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".evolve-mcp"),
        help="State directory path",
    )

    # History command
    history_parser = subparsers.add_parser("history", help="Show evolution history")
    history_parser.add_argument(
        "--limit", "-n", type=int, default=10, help="Number of entries to show"
    )
    history_parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".evolve-mcp"),
        help="State directory path",
    )

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Show collected metrics")
    metrics_parser.add_argument(
        "--format",
        choices=["json", "csv", "table"],
        default="table",
        help="Output format",
    )
    metrics_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Time window in minutes",
    )
    metrics_parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".evolve-mcp"),
        help="State directory path",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export metrics data")
    export_parser.add_argument("output", type=Path, help="Output file path")
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "prometheus"],
        default="json",
        help="Export format",
    )
    export_parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".evolve-mcp"),
        help="State directory path",
    )

    # Init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize evolve-mcp in current directory"
    )
    init_parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".evolve-mcp"),
        help="State directory path",
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Show or set configuration")
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )
    config_parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".evolve-mcp"),
        help="State directory path",
    )

    return parser


def cmd_status(args: argparse.Namespace) -> int:
    """Show evolution status."""
    state_dir = args.state_dir

    if not state_dir.exists():
        print("No evolution state found. Run 'evolve-mcp init' first.")
        return 1

    state_file = state_dir / "state.json"
    if not state_file.exists():
        print("Status: Not started")
        print("No evolution cycles have been run yet.")
        return 0

    with open(state_file) as f:
        state = json.load(f)

    print(f"Status: {state.get('status', 'unknown')}")
    print(f"Current generation: {state.get('generation', 0)}")
    print(f"Population size: {state.get('population_size', 0)}")
    print(f"Best fitness: {state.get('best_fitness', 'N/A')}")

    if state.get("last_updated"):
        print(f"Last updated: {state.get('last_updated')}")

    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Show evolution history."""
    state_dir = args.state_dir
    history_file = state_dir / "history.json"

    if not history_file.exists():
        print("No evolution history found.")
        return 0

    with open(history_file) as f:
        history = json.load(f)

    entries = history.get("entries", [])[-args.limit :]

    if not entries:
        print("No history entries.")
        return 0

    print(f"{'Generation':<12} {'Best Fitness':<14} {'Avg Fitness':<14} {'Timestamp'}")
    print("-" * 60)

    for entry in entries:
        gen = entry.get("generation", "?")
        best = entry.get("best_fitness", "N/A")
        avg = entry.get("avg_fitness", "N/A")
        ts = entry.get("timestamp", "N/A")

        if isinstance(best, float):
            best = f"{best:.4f}"
        if isinstance(avg, float):
            avg = f"{avg:.4f}"

        print(f"{gen:<12} {best:<14} {avg:<14} {ts}")

    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """Show collected metrics."""
    state_dir = args.state_dir
    metrics_file = state_dir / "metrics.json"

    if not metrics_file.exists():
        print("No metrics data found.")
        return 0

    with open(metrics_file) as f:
        metrics = json.load(f)

    if args.format == "json":
        print(json.dumps(metrics, indent=2))
    elif args.format == "csv":
        if metrics.get("data"):
            keys = list(metrics["data"][0].keys()) if metrics["data"] else []
            print(",".join(keys))
            for row in metrics["data"]:
                print(",".join(str(row.get(k, "")) for k in keys))
    else:  # table
        summary = metrics.get("summary", {})
        print("Metrics Summary")
        print("-" * 40)
        print(f"Total requests: {summary.get('total_requests', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.2%}")
        print(f"Avg response time: {summary.get('avg_response_time', 0):.3f}s")
        print(f"Total tokens: {summary.get('total_tokens', 0)}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export metrics data."""
    state_dir = args.state_dir
    metrics_file = state_dir / "metrics.json"

    if not metrics_file.exists():
        print("No metrics data to export.")
        return 1

    with open(metrics_file) as f:
        metrics = json.load(f)

    output_path = args.output

    if args.format == "json":
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    elif args.format == "csv":
        import csv

        data = metrics.get("data", [])
        if data:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    elif args.format == "prometheus":
        lines = []
        summary = metrics.get("summary", {})
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                lines.append(f"evolve_mcp_{key} {value}")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    print(f"Exported to {output_path}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize evolve-mcp in current directory."""
    state_dir = args.state_dir

    if state_dir.exists():
        print(f"Already initialized at {state_dir}")
        return 0

    state_dir.mkdir(parents=True)

    # Create default config
    config = {
        "population_size": 50,
        "mutation_rate": 0.1,
        "elite_count": 5,
        "max_generations": 100,
        "fitness_threshold": 0.95,
        "created_at": datetime.now().isoformat(),
    }

    with open(state_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create empty state
    state = {
        "status": "initialized",
        "generation": 0,
        "population_size": 0,
        "best_fitness": None,
        "last_updated": datetime.now().isoformat(),
    }

    with open(state_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    # Create empty history
    with open(state_dir / "history.json", "w") as f:
        json.dump({"entries": []}, f)

    # Create empty metrics
    with open(state_dir / "metrics.json", "w") as f:
        json.dump({"summary": {}, "data": []}, f)

    print(f"Initialized evolve-mcp at {state_dir}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show or set configuration."""
    state_dir = args.state_dir
    config_file = state_dir / "config.json"

    if not config_file.exists():
        print("No configuration found. Run 'evolve-mcp init' first.")
        return 1

    with open(config_file) as f:
        config = json.load(f)

    print("Current Configuration")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key}: {value}")

    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "status": cmd_status,
        "history": cmd_history,
        "metrics": cmd_metrics,
        "export": cmd_export,
        "init": cmd_init,
        "config": cmd_config,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
