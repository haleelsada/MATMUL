# High-Performance General Matrix Multiplication Implementation

A highly optimized C++ implementation of General Matrix Multiplication (GEMM) using SIMD instructions and multi-threading. This code demonstrates advanced optimization techniques for dense linear algebra computations.

## Features

- **Adaptive SIMD Support**: Automatically uses the best available instruction set
  - AVX-512 with 8×8 microkernels for maximum performance
  - AVX2 with 4×4 microkernels as fallback
- **Cache-Aware Blocking**: Configurable tiling parameters (MC, KC, NC) for optimal cache utilization
- **Parallel Execution**: OpenMP-based multi-threading with dynamic scheduling
- **Power-of-Two Padding**: Automatic stride padding to avoid cache-set conflicts
- **FMA Instructions**: Fused multiply-add operations for improved throughput

## Algorithm Details

The implementation uses a three-level tiled (blocked) GEMM algorithm:

```
C = C + A × B
```

Where matrices are processed in blocks to maximize data reuse in L1/L2/L3 caches. The innermost computation uses hand-optimized SIMD microkernels that operate on register-sized tiles.

### Microkernel Sizes
- **AVX-512**: 8×8 tiles (8 doubles per 512-bit register)
- **AVX2**: 4×4 tiles (4 doubles per 256-bit register)

## Build Instructions

```bash
make
```

## Usage
For baseline python testing
```bash
./run.sh baseline 1000
```

For optimized code
```bash
./run.sh optimized 1000 2 128 64 256
```
output for both will be similar to this
```
❯ ./run.sh baseline 1000
Running baseline python implementation
N=1000 P=8 time=0.460784 seconds
```
```
❯ ./run.sh optimized 1000 2 128 64 256
Running optimized binary
N=1000 threads=2 Blocking: MC=128 KC=64 NC=256 LD=1000
avx2 supported
Time(s)=0.033267  GFLOPs=60.120281  Checksum=4950.575779
```

### Parameters
- `N`: Matrix dimension (computes N×N matrices)
- `num_threads`: Number of OpenMP threads
- `MC`: (optional) Row blocking size (default: 128)
- `KC`: (optional) Inner dimension blocking size (default: 64)
- `NC`: (optional) Column blocking size (default: 256)


## Performance Tips

1. **Thread Count**: Set to the number of physical cores for best performance
2. **Matrix Size**: Larger matrices (N ≥ 2048) show better GFLOPS due to amortized overhead
3. **Blocking Parameters**: Tune MC, KC, NC based on your CPU's cache hierarchy:
   - KC should fit in L1 cache
   - MC×KC should fit in L2 cache
   - KC×NC should fit in L3 cache

## Output

The program reports:
- Matrix dimension and threading configuration
- Blocking parameters and stride (LD)
- Execution time in seconds
- Performance in GFLOPS (billions of floating-point operations per second)
- Result checksum for verification

## Technical Highlights

### Power-of-Two Padding
When N is a power of two, the code adds padding to the leading dimension to prevent cache-set conflicts that can severely degrade performance.

### Memory Alignment
All matrices are 64-byte aligned using `posix_memalign` for optimal SIMD load/store operations.

### Edge Case Handling
Partial tiles at matrix boundaries are handled by a scalar fallback routine to ensure correctness.

## Requirements

- C++11 or later
- OpenMP support
- CPU with AVX2 or AVX-512 support
- GCC, Clang, or ICC compiler

## Performance Expectations

On modern hardware (e.g., Intel Xeon Scalable or AMD EPYC):
- AVX-512 version: 80-95% of theoretical peak FLOPS
- AVX2 version: 70-85% of theoretical peak FLOPS

Actual performance depends on matrix size, cache hierarchy, memory bandwidth, and CPU architecture.

## License

This code is provided as-is for educational and research purposes.
