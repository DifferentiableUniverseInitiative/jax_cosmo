# Performance Benchmark System

This directory contains a comprehensive performance benchmarking system for `jax_cosmo` that automatically runs on pull requests to track performance changes.

## Overview

The benchmark system consists of:

1. **`benchmark_performance.py`** - Core benchmarking script
2. **`.github/workflows/performance-benchmark.yml`** - GitHub Actions workflow
3. **Automated PR comments** - Performance comparison reports

## What Gets Benchmarked

The system benchmarks key `angular_cl` computations that represent typical usage patterns:

- **`lensing_cl_small`** - Small-scale weak lensing power spectra (20 ell values)
- **`lensing_cl_large`** - Large-scale weak lensing power spectra (100 ell values) 
- **`multi_bin_lensing`** - Multi-bin weak lensing analysis
- **`high_precision`** - High-precision computation (256 sample points)
- **`parameter_gradient`** - Gradient computation w.r.t. cosmological parameters

Each benchmark measures:
- **Execution time** (seconds)
- **Peak memory usage** (MB)
- **Memory efficiency** 

## How It Works

### Automatic PR Benchmarking

When you open or update a pull request that modifies Python files:

1. **Baseline benchmark** runs on the target branch (usually `master`)
2. **PR benchmark** runs on your changes
3. **Comparison report** gets posted as a comment on the PR
4. **Performance regression check** warns about significant slowdowns (>20%)

### Manual Benchmarking

You can run benchmarks manually:

```bash
# Install dependencies
pip install psutil

# Run all benchmarks
python benchmark_performance.py --format markdown

# Save results to file  
python benchmark_performance.py --output results.json

# Compare with baseline
python benchmark_performance.py \
  --results-file new_results.json \
  --compare baseline_results.json \
  --format markdown
```

## Example Output

The benchmark generates reports like this:

```markdown
## Performance Comparison

| Benchmark | Time Change | Memory Change | Status |
|-----------|-------------|---------------|--------|
| Lensing Cl Small | -5.2% | +2.1% | 🟢⚪ |
| Lensing Cl Large | +1.3% | -3.8% | ⚪🟢 |
| Multi Bin Lensing | -12.7% | -8.4% | 🟢🟢 |

### Legend
- 🟢 = Improvement (faster/less memory)
- 🔴 = Regression (slower/more memory)  
- ⚪ = Neutral (< 5% change)
```

## Performance Optimization Tips

Based on the benchmarks, here are common optimization strategies:

### For Speed Improvements:
- **Reduce integration points** (`npoints`) where accuracy permits
- **Optimize spline operations** using vectorized functions
- **Cache expensive computations** like power spectra
- **Use efficient gradient rules** for integration bounds

### For Memory Efficiency:
- **Vectorize operations** to reduce temporary arrays
- **Stream computations** for large ell arrays
- **Optimize array shapes** to minimize copies
- **Use in-place operations** where possible

## Interpreting Results

### Time Performance:
- **< 5% change**: Normal variation, no action needed
- **5-20% regression**: Consider if change is worth the cost
- **> 20% regression**: Significant - optimization recommended

### Memory Performance:
- **Memory increases**: Check for array growth or inefficient operations
- **Memory decreases**: Good! Often indicates better vectorization

### Gradient Performance:
- Important for parameter inference applications
- Custom autodiff rules can provide 2-10x speedups
- Memory efficiency crucial for large parameter spaces

## Contributing Performance Improvements

When submitting performance improvements:

1. **Run benchmarks locally** before submitting
2. **Document expected changes** in the PR description
3. **Check the automated report** confirms improvements
4. **Consider accuracy trade-offs** if applicable

The automated system will help track whether your changes actually improve performance in practice!

## Troubleshooting

**Benchmark fails to run:**
- Ensure all dependencies are installed: `pip install psutil`
- Check that `jax_cosmo` is properly installed: `pip install -e .`

**High variance in results:**
- Benchmarks include JIT warmup to reduce variance
- CPU-only mode used for consistency
- Multiple runs may be needed for noisy environments

**Memory measurements:**
- Peak memory includes Python overhead
- Relative changes more reliable than absolute values
- Memory tracing has some overhead itself

---

This performance tracking helps ensure that `jax_cosmo` remains fast and memory-efficient as new features are added! 🚀 