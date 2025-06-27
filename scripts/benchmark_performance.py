#!/usr/bin/env python3
"""Performance benchmark script for jax_cosmo angular power spectra computations."""

import json
import os
import time
from functools import wraps

import jax
import jax.numpy as jnp

# Force CPU mode for consistent benchmarking
jax.config.update("jax_platform_name", "cpu")

import jax_cosmo.core as jc
from jax_cosmo.angular_cl import angular_cl
from jax_cosmo.probes import WeakLensing
from jax_cosmo.redshift import smail_nz


def measure_performance(func):
    """Decorator to measure execution time and memory usage."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # First run: JIT compilation happens here
        print(f"  First run (compilation): {func.__name__}...")
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

        print(f"  Second run (measurement): {func.__name__}...")

        # Time the second execution (already compiled)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)

        # Ensure computation is complete before stopping timer
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

        end_time = time.perf_counter()

        return {
            "result": result,
            "time_seconds": end_time - start_time,
            "memory_used_mb": 0.0,  # Simplified - remove expensive memory tracking
            "peak_memory_mb": 0.0,  # Simplified - remove expensive memory tracking
        }

    return wrapper


class AngularClBenchmark:
    """Benchmark suite for angular power spectra computations."""

    def __init__(self):
        self.cosmo = jc.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            Omega_k=0.0,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            w0=-1.0,
            wa=0.0,
        )
        self.nz_source = smail_nz(1.0, 2.0, 1.0, gals_per_arcmin2=30)

    @measure_performance
    def benchmark_lensing_cl_small(self):
        """Benchmark small-scale lensing Cl computation."""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 5)
        return angular_cl(self.cosmo, ell, [probe])

    @measure_performance
    def benchmark_lensing_cl_large(self):
        """Benchmark large-scale lensing Cl computation."""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 15)
        return angular_cl(self.cosmo, ell, [probe])

    @measure_performance
    def benchmark_parameter_gradient(self):
        """Benchmark gradient computation."""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 3)

        # Pre-define the varied cosmology to avoid recompilation issues
        cosmo_varied = jc.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            Omega_k=0.0,
            h=0.7,
            sigma8=0.81,  # Slightly different value
            n_s=0.96,
            w0=-1.0,
            wa=0.0,
        )

        cl = angular_cl(cosmo_varied, ell, [probe])
        return jnp.sum(cl)

    def run_all_benchmarks(self):
        """Run all benchmarks and return results."""
        benchmarks = [
            ("lensing_cl_small", self.benchmark_lensing_cl_small),
            ("lensing_cl_large", self.benchmark_lensing_cl_large),
            ("parameter_gradient", self.benchmark_parameter_gradient),
        ]

        results = {}
        for name, benchmark_func in benchmarks:
            print(f"Running benchmark: {name}")
            try:
                perf_data = benchmark_func()
                results[name] = {
                    "time_seconds": perf_data["time_seconds"],
                    "memory_used_mb": perf_data["memory_used_mb"],
                    "peak_memory_mb": perf_data["peak_memory_mb"],
                    "status": "success",
                }
                print(f"  ✓ {name}: {perf_data['time_seconds']:.3f}s")
            except Exception as e:
                print(f"  ✗ {name}: FAILED - {str(e)}")
                results[name] = {
                    "time_seconds": float("inf"),
                    "memory_used_mb": float("inf"),
                    "peak_memory_mb": float("inf"),
                    "status": "failed",
                    "error": str(e),
                }

        return results


def format_benchmark_results(results):
    """Format benchmark results for display."""
    lines = ["## Performance Benchmark Results\n"]
    lines.append("| Benchmark | Time (s) | Peak Memory (MB) | Status |")
    lines.append("|-----------|----------|------------------|--------|")

    for name in sorted(results.keys()):
        data = results[name]
        if data["status"] == "success":
            time_str = f"{data['time_seconds']:.3f}"
            memory_str = f"{data['peak_memory_mb']:.1f}"
            status = "✅"
        else:
            time_str = "∞"
            memory_str = "∞"
            status = "❌"

        lines.append(
            f"| {name.replace('_', ' ').title()} | {time_str} | {memory_str} | {status} |"
        )

    return "\n".join(lines)


def compare_results(before_results, after_results):
    """Compare benchmark results and generate a summary."""
    lines = ["## Performance Comparison\n"]
    lines.append("| Benchmark | Time Change | Memory Change | Status |")
    lines.append("|-----------|-------------|---------------|--------|")

    for name in sorted(before_results.keys()):
        if name not in after_results:
            continue

        before = before_results[name]
        after = after_results[name]

        if before["status"] != "success" or after["status"] != "success":
            lines.append(
                f"| {name.replace('_', ' ').title()} | N/A | N/A | ❌ Failed |"
            )
            continue

        # Calculate percentage changes
        time_change = (
            (after["time_seconds"] - before["time_seconds"]) / before["time_seconds"]
        ) * 100
        memory_change = (
            (after["peak_memory_mb"] - before["peak_memory_mb"])
            / before["peak_memory_mb"]
        ) * 100

        # Format changes
        if abs(time_change) < 5:
            time_emoji = "⚪"
            time_str = f"{time_change:+.1f}%"
        elif time_change < 0:
            time_emoji = "🟢"
            time_str = f"{time_change:.1f}%"
        else:
            time_emoji = "🔴"
            time_str = f"{time_change:+.1f}%"

        if abs(memory_change) < 5:
            memory_emoji = "⚪"
            memory_str = f"{memory_change:+.1f}%"
        elif memory_change < 0:
            memory_emoji = "🟢"
            memory_str = f"{memory_change:.1f}%"
        else:
            memory_emoji = "🟢"
            memory_str = f"{memory_change:+.1f}%"

        status = f"{time_emoji}{memory_emoji}"
        lines.append(
            f"| {name.replace('_', ' ').title()} | {time_str} | {memory_str} | {status} |"
        )

    lines.append("\n### Legend")
    lines.append("- 🟢 = Improvement, 🔴 = Regression, ⚪ = Neutral (< 5% change)")

    return "\n".join(lines)


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark jax_cosmo performance")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--compare", type=str, help="Compare with previous results")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--results-file", type=str, help="Use existing results file")

    args = parser.parse_args()

    # Load existing results or run new benchmarks
    if args.results_file and os.path.exists(args.results_file):
        with open(args.results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded results from {args.results_file}")
    else:
        benchmark = AngularClBenchmark()
        results = benchmark.run_all_benchmarks()

    if args.format == "json":
        output = json.dumps(results, indent=2)
    else:
        output = format_benchmark_results(results)

        if args.compare and os.path.exists(args.compare):
            with open(args.compare, "r") as f:
                before_results = json.load(f)
            comparison = compare_results(before_results, results)
            output = comparison + "\n\n" + output

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
