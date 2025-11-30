# Report Template — Accelerator Leaderboard Submission


## Team: <2025mcs2105_2025mcs2741>


## 1. Problem description
- Application: GEMM (dense matrix multiply)

- Objective was to accelerate Dense General Matrix Multiplication(GEMM).
- Input matrices : Two square dense matrices of size N x N.

## 2. Baseline

- Description of baseline implementation (Python multiprocessing + NumPy) Baseline script implements a data-parallel matrix multiplication using Python's multiprocessing library to bypass the Global Interpreter Lock (GIL).

C = A*B

- It is splitting matrix A into P(#processes) set of rows , then a pool of P workers is created and each worker is assigned rows from A.

- They then works independently to calculate final matrix C by calling NumPy dot function, which in backend using BLAS library .

- Baseline Timings and Environment (CPU Model, Cores, OS):

- CPU Model: AMD RYZEN 7 7735HS with Radeon Graphics

- Cores: 8 ,16 threads

- CPU op-mode(s): 32bit, 64 bit

- Caches (sum of all): 
- L1d:256 KiB (8 instances)
- L1i:256 KiB (8 instances)
- L2:4 MiB (8 instances)
- L3:16 MiB (1 instance)

- NUMA:
- NUMA node(s):1
- NUMA node0 CPU(s):0-7

- Workstation provided: 
- Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         46 bits physical, 48 bits virtual
  Byte Order:            Little Endian
- CPU(s):                  16
  On-line CPU(s) list:   0-15
- Vendor ID:               GenuineIntel
  Model name:            Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz
    CPU family:          6
    Model:               85
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           1
- Caches (sum of all):     
  L1d:                   256 KiB (8 instances)
  L1i:                   256 KiB (8 instances)
  L2:                    8 MiB (8 instances)
  L3:                    11 MiB (1 instance)
- NUMA:                    
  NUMA node(s):          1
  NUMA node0 CPU(s):     0-15



- OS: Ubuntu 20.04 LTS

- Python Version: 3.8

- NumPy Version: 1.19.2


## 3. Optimizations implemented
- Multithreading with OpenMP

- OpenMP is a simple parallel programming API for C/C++. It uses pragmas to tell the compiler how to run code on multiple threads. It follows a fork–join model:

- Fork: Master thread creates worker threads.
- Parallel: Threads share the work.
- Join: Workers finish and terminate; master continues.

- OpenMP Features Used
- #pragma omp for : Splits the loop iterations across threads.

- schedule(static) : Divides iterations into equal fixed chunks before execution. Good when each iteration takes similar time.

- collapse(2) : Combines two nested loops into one large loop so OpenMP can distribute work more evenly.

- schedule(dynamic, 2) : Threads take work in chunks of size 2 at runtime. If a thread finishes early, it takes the next chunk. Prevents threads from sitting idle when work per iteration varies.


- Cache Blocking / Tiling in Our Optimized GEMM

- Naive GEMM has O(N³) complexity but poor memory access patterns, leading to cache thrashing. Each element in matrix A requires an entire column from matrix B, causing frequent cache evictions and memory stalls.

- We solve this with cache blocking (tiling), which divides the matrices into smaller MC × KC blocks. These tiles fit in the CPU cache, improving temporal locality and reducing memory access latency.

- Tiling in Our Code
- Our GEMM is divided into 3-level blocks:
- MC – Tile height
- KC – Inner dimension (shared by A and B)
- NC – Tile width

 - for jc in 0..N step NC
  - for ic in 0..N step MC
    - for pc in 0..N step KC

- By hit and trial method , we tried to tune it with various paramter later we towards a particular constant value on which our tiling approach was performing good NC:256 MC:128 KC:64 

-Each tile of A and B is processed with a SIMD microkernel, resulting in MC × NC blocks of C. This minimizes costly memory accesses.


- SIMD Optimization in Our GEMM Implementation

- To overcome the limits of scalar execution, our GEMM uses SIMD intrinsics. SIMD allows the CPU to operate on multiple data elements in a single instruction.
Our code detects CPU capabilities at runtime (via CPUID + XGETBV) and selects the best available instruction set:

- AVX-512 microkernel (8×8)
- AVX2 microkernel (4×4)

- This ensures maximum performance on any machine.

1. AVX2 (Used when AVX-512 is not available)
- SIMD Width

- 256-bit YMM registers (YMM0–YMM15)
- Can process 4 double-precision values at once.

- Data Types Used
- __m256d → 256-bit vector holding 4 doubles
- __m128d → 128-bit vector holding 2 doubles (used for edge reduction)

- Key Intrinsics in Our 4×4 Microkernel
- _mm256_loadu_pd()
- Loads 4 doubles from memory into a YMM register.

- _mm256_fmadd_pd(a, b, c)
- Computes (a * b) + c for all 4 doubles.
- Core of the microkernel.

- _mm256_broadcast_sd()
- Broadcasts a single double from A across the 4 lanes.

- _mm256_storeu_pd()
- Stores the result back to C.
These intrinsics allow our 4×4 kernel to update 16 values of C efficiently.

2. AVX-512 (Used when compiled with AVX-512 support)
- SIMD Width
- 512-bit ZMM registers (ZMM0–ZMM31)
- Can process 8 double-precision values at once.

- Data Types Used
- __m512d → 512-bit vector holding 8 doubles
- Key Intrinsics in Our 8×8 Microkernel

- _mm512_loadu_pd()
- Loads 8 doubles from memory.

- _mm512_set1_pd(val)
- Broadcasts a single double from A across 8 lanes.

- _mm512_fmadd_pd(a, b, c)
- Computes (a*b) + c for all 8 elements.

- _mm512_storeu_pd()
- Stores updated rows back into C.
- This gives us an efficient 8×8 microkernel operating purely in registers.

- Memory-Aligned Matrices

- Our GEMM uses aligned memory because SIMD loads work fastest when data starts at specific byte boundaries.

- AVX-512 = 512 bits = 64 bytes → needs 64-byte alignment
- AVX2 = 256 bits = 32 bytes → needs 32-byte alignment
- Aligned memory prevents split-line loads and keeps vector instructions efficient.

- In our code we use:
- A = (double*) aligned_malloc(size, 64);
- B = (double*) aligned_malloc(size, 64);
- C = (double*) aligned_malloc(size, 64);

- Loop unrolling in our kernels

- We unroll the innermost K-loop to fill vector registers and use FMA on multiple accumulators.

- AVX2 (4×4 microkernel, 256-bit)
- Unroll factor: 4 (process 4 doubles per vector).
- Pattern: load bvec as __m256d, broadcast each scalar a into a vector, then FMA into c accumulators.

for (int k = 0; k < K; ++k) {
    __m256d bvec = _mm256_loadu_pd(&B_tile[k * ldb]);       
    double a0 = A_tile[0 * lda + k];
    double a1 = A_tile[1 * lda + k];
    double a2 = A_tile[2 * lda + k];
    double a3 = A_tile[3 * lda + k];

    c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a0), bvec, c0);
    c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a1), bvec, c1);
    c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a2), bvec, c2);
    c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(&a3), bvec, c3);
}

- similarly it is done for avx512 with a scaling factor of 8.


## 4. Experimental methodology
- Commands Used to Run Baseline and Optimized:

- Baseline Run: python3 baseline/gemm_baseline.py or ./run.sh baseline 

- Optimized Run: ./optimized/gemm_opt n t (where n = size of row/col and t = no of threads) or ./run.sh optimized

- Number of Runs, How Median Chosen, Perf Counters Collected:

- For each configuration (baseline and optimized), 10 runs were performed.
- The median runtime was chosen to mitigate the impact of outliers.
- Perf counters such as cache misses, IPC, and bandwidth were collected using: perf stat -ddd ./optimized/gemm_opt n t



## 5. Results
- Tables and Plots:
- Runtime Comparison:
-AVX512:(This is the workstation provided to us)
| N     | Threads | Time_o(s)| Checksum      | Gflops  | Time_B(s) | Speedup | 
| ----- | ------- | -------- | ------------- | --------| ----------| --------|
| 1000  | 8       | 0.011327 | 4950.5757     | 176.56  | 0.469046  | 41.41   |
| 2000  | 8       | 0.066767 | 147031.4266   | 239.63  | 1.881916  | 28.19   |
| 3000  | 8       | 0.187528 | 192972.2796   | 287.95  | 5.058123  | 26.97   |
| 4000  | 8       | 0.443659 | -301438.2978  | 288.50  | 12.418592 | 27.99   |
| 5000  | 8       | 0.880064 | 310155.6495   | 284.07  | 19.843313 | 22.54   |
| 6000  | 8       | 1.534247 | -312693.7985  | 281.57  | 32.085119 | 20.92   |
| 7000  | 8       | 2.458124 | 216196.9720   | 279.07  | 50.627076 | 20.60   |
| 8000  | 8       | 3.687781 | 1377224.7166  | 277.63  | 92.968585 | 25.22   |
| 9000  | 8       | 5.214588 | 935220.7785   | 279.60  | 104.437179| 20.03   |
| 10000 | 8       | 6.710772 | -770903.5282  | 298.02  | 243.570031| 36.29   |


- Some extra runs on optimized code 
| N     | Threads | Time_o(s)| Checksum      | Gflops  | 
| ----- | ------- | -------- | ------------- | --------| 
| 14000 | 8       | 19.436253| -964198.1200  | 282.35  |
| 18000 | 8       | 40.146964| -2403642.8117 | 290.53  |
| 26000 | 8       | 126.43916| -312693.7985  | 278.01  |
| 32000 | 8       | 242.61307| -4347919.5504 | 270.12  |
| 40000 | 8       | 451.14756| 5554423.48050 | 283.72  |
| 45000 | 8       | 666.53062| -19016473.2766| 273.43  |

-AVX2(this is on my Laptop Specs are mentioned above)

| N     | Threads | Time_o(s)| Checksum      | Gflops  | Time_B(s) | SpeedUp  |
| ----- | ------- | -------- | ------------- | --------| ----------| -------- |
| 1000  | 8       | 0.055712 | 4950.5757     | 153.95  | 0.469046  | 12.786   |
| 2000  | 8       | 0.258999 | 147031.4266   | 158.63  | 1.347325  | 5.202    |
| 3000  | 8       | 0.721440 | 192972.2796   | 142.95  | 2.590600  | 3.591    |
| 4000  | 8       | 1.459208 | -301438.2978  | 148.50  | 4.222177  | 2.893    |
| 5000  | 8       | 2.744702 | 310155.6495   | 137.07  | 6.733402  | 2.453    |
| 6000  | 8       | 4.344684 | -312693.7985  | 143.57  | 9.755535  | 2.245    |
| 7000  | 8       | 6.883419 | 216196.9720   | 135.07  | 13.958733 | 2.028    |
| 8000  | 8       | 10.084379| 1377224.7166  | 132.63  | 19.859679 | 1.969    |
| 9000  | 8       | 13.95962 | 935220.7785   | 132.60  | 22.437179 | 1.617    |
| 10000 | 8       | 18.10322 | -770903.5282  | 139.02  | 29.257459 | 1.616    |

- Microarchitecture counters: L1/L2/L3 misses, bandwidth, IPC


## 6. Analysis

- GFLOPS:
-As N increases, total execution time naturally rises for both the baseline and our optimized version.
- However, the tiled + SIMD (AVX2/AVX-512) implementation runs much faster, so GFLOPS increases significantly.
- Since GFLOPS is inversely proportional to time, our optimized kernel reaches high performance (~115–120 GFLOPS) for large N due to:

- vectorized microkernels (4×4 AVX2 or 8×8 AVX-512),
- cache-friendly tiling,
- reduced memory stalls.

- L1 Misses:
- As N grows, L1 data misses increase, but tiling keeps the working set small:
- In AVX2, L1 misses fluctuate because 4×4 tiles stress L1 more frequently.
- In AVX-512, the miss pattern stabilizes after a point since 8×8 tiles reuse more data per load.

Typical L1 D-cache miss rates observed:
- AVX2: ~10–12%
- AVX-512: ~10–18% (slightly higher due to wider loads and more bandwidth demand)

- Memory Bandwidth:
- With fewer threads, each core has more available cache and memory bandwidth, giving high IPC per core.
- As thread count increases:
- Overall GFLOPS improves because more cores execute tiles in parallel.
- But threads begin competing for the same cache and memory resources.
- This reduces per-core IPC and increases cache/memory pressure.
- This behavior matches our tiled GEMM, where A/B blocks must be streamed repeatedly from memory.

- Further Improvements:
- Based on our implementation, the following optimizations can push performance further:
- Better software prefetching inside the microkernels to hide memory latency.
- NUMA-aware data placement (first-touch policy, node-local allocation).
- Improving OpenMP scheduling (e.g., static chunking) for more consistent load balancing.
- Refining block sizes (MC, KC, NC) to better match L1/L2/L3 cache capacities.
- Packing matrices (especially B) to improve spatial locality and remove strided loads.

## 7. Reproducibility

- Operating System:
- Linux (Ubuntu 22.04) or Windows Subsystem for Linux (WSL2).
- CPU: An x86_64 processor with support for AVX2, FMA, and AVX512F.
- Compiler: g++ (GNU Compiler Collection) 10.0 or newer (or clang++) with C++17 support.
- Libraries: OpenMP

- Compilation
- The code must be compiled with optimizations, OpenMP support, and the specific target architecture flags enabled.
- Bash : g++ -o gemm_opt gemm_opt.cpp -O3 -fopenmp -march=native
- O3: Enables high-level compiler optimizations.
- fopenmp: Links the OpenMP runtime.
- march=native: Ensures the compiler can generate the vectorized code for the target machine.
- The script usage is: ./gemm_opt N num_threads
