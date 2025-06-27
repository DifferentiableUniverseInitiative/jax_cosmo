#!/usr/bin/env python3
"""
Performance benchmark script for jax_cosmo angular power spectra computations.

This script benchmarks the key computations used in tests to track performance
changes across different versions of the code.
"""

import json
import os
import time
import tracemalloc
from functools import wraps
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import psutil

# Force CPU mode for consistent benchmarking
jax.config.update("jax_platform_name", "cpu")

import jax_cosmo.core as jc
from jax_cosmo.angular_cl import angular_cl
from jax_cosmo.probes import NumberCounts, WeakLensing
from jax_cosmo.redshift import delta_nz, smail_nz


def measure_performance(func):
    """Decorator to measure execution time and memory usage."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Warm up JIT compilation
        try:
            func(*args, **kwargs)
        except:
            pass

        # Start memory tracing
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time the execution
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        # Measure memory
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "result": result,
            "time_seconds": end_time - start_time,
            "memory_used_mb": memory_after - memory_before,
            "peak_memory_mb": peak / 1024 / 1024,
        }

    return wrapper


class AngularClBenchmark:
    """Benchmark suite for angular power spectra computations."""

    def __init__(self):
        # Standard cosmology for all tests
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

        # Standard redshift distributions
        self.nz_source = smail_nz(1.0, 2.0, 1.0, gals_per_arcmin2=30)
        self.nz_lens = smail_nz(0.5, 1.5, 0.8, gals_per_arcmin2=10)

    @measure_performance
    def benchmark_lensing_cl_small(self):
        """Benchmark small-scale lensing Cl computation (similar to test_lensing_cl)"""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 20)  # 20 ell values
        return angular_cl(self.cosmo, ell, [probe], npoints=64)

    @measure_performance
    def benchmark_lensing_cl_large(self):
        """Benchmark large-scale lensing Cl computation"""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 100)  # 100 ell values
        return angular_cl(self.cosmo, ell, [probe], npoints=128)

    @measure_performance
    def benchmark_clustering_cl(self):
        """Benchmark clustering Cl computation (similar to test_clustering_cl)"""
        probe = NumberCounts([self.nz_lens], bias=2.0)
        ell = jnp.logspace(1, 3, 50)
        return angular_cl(self.cosmo, ell, [probe], npoints=64)

    @measure_performance
    def benchmark_cross_correlation(self):
        """Benchmark cross-correlation between lensing and clustering"""
        lensing_probe = WeakLensing([self.nz_source])
        clustering_probe = NumberCounts([self.nz_lens], bias=2.0)
        ell = jnp.logspace(1, 3, 50)
        return angular_cl(
            self.cosmo, ell, [lensing_probe, clustering_probe], npoints=64
        )

    @measure_performance
    def benchmark_multi_bin_lensing(self):
        """Benchmark multi-bin lensing analysis"""
        nz1 = smail_nz(1.0, 2.0, 0.8, gals_per_arcmin2=15)
        nz2 = smail_nz(1.5, 2.0, 1.2, gals_per_arcmin2=15)
        probe = WeakLensing([nz1, nz2])
        ell = jnp.logspace(1, 3, 50)
        return angular_cl(self.cosmo, ell, [probe], npoints=64)

    @measure_performance
    def benchmark_high_precision(self):
        """Benchmark high-precision computation with many sample points"""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 30)
        return angular_cl(self.cosmo, ell, [probe], npoints=256)

    @measure_performance
    def benchmark_parameter_gradient(self):
        """Benchmark gradient computation w.r.t. cosmological parameters"""
        probe = WeakLensing([self.nz_source])
        ell = jnp.logspace(1, 3, 20)

        def compute_cl(sigma8):
            cosmo_varied = jc.Cosmology(
                Omega_c=0.25,
                Omega_b=0.05,
                Omega_k=0.0,
                h=0.7,
                sigma8=sigma8,
                n_s=0.96,
                w0=-1.0,
                wa=0.0,
            )
            cl = angular_cl(cosmo_varied, ell, [probe], npoints=64)
            return jnp.sum(cl)  # Sum for scalar output

        grad_func = jax.grad(compute_cl)
        return grad_func(0.8)

    def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks and return results."""
        benchmarks = [
            ("lensing_cl_small", self.benchmark_lensing_cl_small),
            ("lensing_cl_large", self.benchmark_lensing_cl_large),
            ("multi_bin_lensing", self.benchmark_multi_bin_lensing),
            ("high_precision", self.benchmark_high_precision),
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


def format_benchmark_results(results: Dict[str, Dict[str, float]]) -> str:
    """Format benchmark results for display."""
    lines = []
    lines.append("## Performance Benchmark Results\n")

    # Table header
    lines.append("| Benchmark | Time (s) | Peak Memory (MB) | Status |")
    lines.append("|-----------|----------|------------------|--------|")

    # Sort by benchmark name for consistent output
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


def compare_results(before_results: Dict, after_results: Dict) -> str:
    """Compare benchmark results and generate a summary."""
    lines = []
    lines.append("## Performance Comparison\n")

    # Table header
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

        # Format changes with colors
        if abs(time_change) < 5:  # Less than 5% change is neutral
            time_emoji = "⚪"
            time_str = f"{time_change:+.1f}%"
        elif time_change < 0:  # Faster is good
            time_emoji = "🟢"
            time_str = f"{time_change:.1f}%"
        else:  # Slower is bad
            time_emoji = "🔴"
            time_str = f"{time_change:+.1f}%"

        if abs(memory_change) < 5:  # Less than 5% change is neutral
            memory_emoji = "⚪"
            memory_str = f"{memory_change:+.1f}%"
        elif memory_change < 0:  # Less memory is good
            memory_emoji = "🟢"
            memory_str = f"{memory_change:.1f}%"
        else:  # More memory is bad
            memory_emoji = "🔴"
            memory_str = f"{memory_change:+.1f}%"

        status = f"{time_emoji}{memory_emoji}"
        lines.append(
            f"| {name.replace('_', ' ').title()} | {time_str} | {memory_str} | {status} |"
        )

    # Add summary
    lines.append("\n### Legend")
    lines.append("- 🟢 = Improvement (faster/less memory)")
    lines.append("- 🔴 = Regression (slower/more memory)")
    lines.append("- ⚪ = Neutral (< 5% change)")
    lines.append("- Time Change: Negative = faster, Positive = slower")
    lines.append("- Memory Change: Negative = less memory, Positive = more memory")

    return "\n".join(lines)


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark jax_cosmo performance")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument(
        "--compare", type=str, help="Compare with previous results file"
    )
    parser.add_argument(
        "--format", choices=["json", "markdown"], default="json", help="Output format"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Use existing results file instead of running benchmarks",
    )

    args = parser.parse_args()

    # Load existing results or run new benchmarks
    if args.results_file and os.path.exists(args.results_file):
        with open(args.results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded results from {args.results_file}")
    else:
        # Run benchmarks
        benchmark = AngularClBenchmark()
        results = benchmark.run_all_benchmarks()

    if args.format == "json":
        output = json.dumps(results, indent=2)
    else:
        output = format_benchmark_results(results)

        # Add comparison if requested
        if args.compare and os.path.exists(args.compare):
            with open(args.compare, "r") as f:
                before_results = json.load(f)
            comparison = compare_results(before_results, results)
            output = comparison + "\n\n" + output

    # Save or print results
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
