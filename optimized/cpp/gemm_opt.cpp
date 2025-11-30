// gemm_power_of_two_fix.cpp
#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>
using namespace std;

// Runtime SIMD check 
// void check_simd_support_print() {
//     unsigned a,b,c,d;
//     bool sse=0,avx=0,avx2=0,avx512=0;

//     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1));
//     sse = (d >> 25) & 1;
//     if ((c & (1<<27)) && (c & (1<<28))) {
//         unsigned x0a,x0d;
//         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
//         if ((x0a & 6) == 6) avx = 1;
//     }

//     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(7));
//     if (avx && (b & (1<<5))) avx2 = 1;
//     if ((b & (1<<16))) {
//         unsigned x0a,x0d;
//         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
//         if ((x0a & 0xE0) == 0xE0) avx512 = 1;
//     }

//     cout << "Detected SIMD support → ";
//     cout << "SSE=" << sse << " AVX=" << avx << " AVX2=" << avx2 << " AVX512=" << avx512 << "\n";
//     if (avx512) cout << "→ Best available SIMD: AVX-512\n";
//     else if (avx2) cout << "→ Best available SIMD: AVX2\n";
//     else if (avx) cout << "→ Best available SIMD: AVX\n";
//     else cout << "→ Fallback: SSE or Scalar\n";
// }

// Returns true if OS + CPU supports full AVX-512 state (checks XGETBV & CPUID)
// bool check_avx512_runtime() {
//     unsigned a,b,c,d;
//     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(1));
//     bool avx = false;
//     if ((c & (1<<27)) && (c & (1<<28))) {
//         unsigned x0a,x0d;
//         __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
//         if ((x0a & 6) == 6) avx = true;
//     }
//     __asm__ volatile("cpuid":"=a"(a),"=b"(b),"=c"(c),"=d"(d):"a"(7));
//     bool avx512_cpuid = (b & (1<<16));
//     if (!avx || !avx512_cpuid) return false;
//     unsigned x0a,x0d;
//     __asm__ volatile("xgetbv":"=a"(x0a),"=d"(x0d):"c"(0));
//     return ((x0a & 0xE0) == 0xE0);
// }

// Memory helpers
static inline void* aligned_malloc(size_t bytes, size_t align = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0) return nullptr;
    return p;
}
static inline void aligned_free(void* p) { free(p); }

static inline void scalar_small_generic(const double* __restrict__ A_tile,const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc, int K, int M, int N)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += A_tile[i * lda + k] * B_tile[k * ldb + j];
            C_tile[i * ldc + j] += acc;
        }
}

// AVX-512 8x8 microkernel & tiled GEMM (compiled only if AVX-512 enabled)
#ifdef __AVX512F__
namespace avx512 {
    constexpr int MR = 8;
    constexpr int NR = 8;

    static inline void microkernel8x8(const double* __restrict__ A_tile, const double* __restrict__ B_tile, double* __restrict__ C_tile,int lda, int ldb, int ldc, int K)
    {
        __m512d c0 = _mm512_loadu_pd(&C_tile[0 * ldc]);
        __m512d c1 = _mm512_loadu_pd(&C_tile[1 * ldc]);
        __m512d c2 = _mm512_loadu_pd(&C_tile[2 * ldc]);
        __m512d c3 = _mm512_loadu_pd(&C_tile[3 * ldc]);
        __m512d c4 = _mm512_loadu_pd(&C_tile[4 * ldc]);
        __m512d c5 = _mm512_loadu_pd(&C_tile[5 * ldc]);
        __m512d c6 = _mm512_loadu_pd(&C_tile[6 * ldc]);
        __m512d c7 = _mm512_loadu_pd(&C_tile[7 * ldc]);

        for (int k = 0; k < K; ++k) {
            __m512d bvec = _mm512_loadu_pd(&B_tile[k * ldb]);
            double a0 = A_tile[0 * lda + k];
            double a1 = A_tile[1 * lda + k];
            double a2 = A_tile[2 * lda + k];
            double a3 = A_tile[3 * lda + k];
            double a4 = A_tile[4 * lda + k];
            double a5 = A_tile[5 * lda + k];
            double a6 = A_tile[6 * lda + k];
            double a7 = A_tile[7 * lda + k];

            c0 = _mm512_fmadd_pd(_mm512_set1_pd(a0), bvec, c0);
            c1 = _mm512_fmadd_pd(_mm512_set1_pd(a1), bvec, c1);
            c2 = _mm512_fmadd_pd(_mm512_set1_pd(a2), bvec, c2);
            c3 = _mm512_fmadd_pd(_mm512_set1_pd(a3), bvec, c3);
            c4 = _mm512_fmadd_pd(_mm512_set1_pd(a4), bvec, c4);
            c5 = _mm512_fmadd_pd(_mm512_set1_pd(a5), bvec, c5);
            c6 = _mm512_fmadd_pd(_mm512_set1_pd(a6), bvec, c6);
            c7 = _mm512_fmadd_pd(_mm512_set1_pd(a7), bvec, c7);
        }

        _mm512_storeu_pd(&C_tile[0 * ldc], c0);
        _mm512_storeu_pd(&C_tile[1 * ldc], c1);
        _mm512_storeu_pd(&C_tile[2 * ldc], c2);
        _mm512_storeu_pd(&C_tile[3 * ldc], c3);
        _mm512_storeu_pd(&C_tile[4 * ldc], c4);
        _mm512_storeu_pd(&C_tile[5 * ldc], c5);
        _mm512_storeu_pd(&C_tile[6 * ldc], c6);
        _mm512_storeu_pd(&C_tile[7 * ldc], c7);
    }

    // now accept lda/ldb/ldc so we can use padded strides
    void gemm_tiled_avx512(const double* A, const double* B, double* C,int N, int MC, int KC, int NC, int lda, int ldb, int ldc)
    {
    #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC) {
            for (int ic = 0; ic < N; ic += MC) {
                for (int pc = 0; pc < N; pc += KC) {
                    int jMax = min(jc + NC, N);
                    int iMax = min(ic + MC, N);
                    int pMax = min(pc + KC, N);
                    int klen = pMax - pc;

                    for (int j = jc; j < jMax; j += NR) {
                        int j_rem = min(NR, jMax - j);
                        for (int i = ic; i < iMax; i += MR) {
                            int i_rem = min(MR, iMax - i);
                            double* Cblk = &C[i * ldc + j];
                            const double* Ablk = &A[i * lda + pc];
                            const double* Bblk = &B[pc * ldb + j];

                            if (i_rem == MR && j_rem == NR)
                                microkernel8x8(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
                            else
                                scalar_small_generic(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
                        }
                    }
                }
            }
        }
    }
} // namespace avx512
#endif // __AVX512F__

// AVX2 4x4 microkernel & tiled GEMM (compiled when AVX-512 not enabled)
#ifndef __AVX512F__
namespace avx2 {
    constexpr int MR = 4;
    constexpr int NR = 4;

    static inline void microkernel4x4(const double* __restrict__ A_tile,const double* __restrict__ B_tile,double* __restrict__ C_tile,int lda, int ldb, int ldc,int K)
    {
        __m256d c0 = _mm256_loadu_pd(&C_tile[0 * ldc]);
        __m256d c1 = _mm256_loadu_pd(&C_tile[1 * ldc]);
        __m256d c2 = _mm256_loadu_pd(&C_tile[2 * ldc]);
        __m256d c3 = _mm256_loadu_pd(&C_tile[3 * ldc]);

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

        _mm256_storeu_pd(&C_tile[0 * ldc], c0);
        _mm256_storeu_pd(&C_tile[1 * ldc], c1);
        _mm256_storeu_pd(&C_tile[2 * ldc], c2);
        _mm256_storeu_pd(&C_tile[3 * ldc], c3);
    }

    // Accept lda/ldb/ldc too
    void gemm_tiled_4x4(const double* A, const double* B, double* C,int N, int MC, int KC, int NC,int lda, int ldb, int ldc)
    {
    #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int jc = 0; jc < N; jc += NC) {
            for (int ic = 0; ic < N; ic += MC) {
                for (int pc = 0; pc < N; pc += KC) {
                    int jMax = min(jc + NC, N);
                    int iMax = min(ic + MC, N);
                    int pMax = min(pc + KC, N);
                    int klen = pMax - pc;

                    for (int j = jc; j < jMax; j += NR) {
                        int j_rem = min(NR, jMax - j);
                        for (int i = ic; i < iMax; i += MR) {
                            int i_rem = min(MR, iMax - i);

                            double* Cblk = &C[i * ldc + j];
                            const double* Ablk = &A[i * lda + pc];
                            const double* Bblk = &B[pc * ldb + j];

                            if (i_rem == MR && j_rem == NR)
                                microkernel4x4(Ablk, Bblk, Cblk, lda, ldb, ldc, klen);
                            else
                                scalar_small_generic(Ablk, Bblk, Cblk, lda, ldb, ldc, klen, i_rem, j_rem);
                        }
                    }
                }
            }
        }
    }
} // namespace avx2
#endif // !__AVX512F__

// Helper: detect power-of-two
static inline bool is_power_of_two(unsigned x) { return x && ((x & (x - 1)) == 0); }

// Main program: run compiled kernel only
int main(int argc, char** argv)
{
    // check_simd_support_print();

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads [MC KC NC]\n";
        cerr << "If compiled with AVX-512 this runs AVX-512 8x8 kernel; otherwise AVX2 4x4 fallback.\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    // choose non-power-of-two default blocking
    int MC = (argc > 3) ? atoi(argv[3]) : 128; // was 128
    int KC = (argc > 4) ? atoi(argv[4]) : 64;  // was 64
    int NC = (argc > 5) ? atoi(argv[5]) : 256; // was 256

    // If N is a power-of-two, add small padding to leading dimension to avoid cache-set conflicts.
    int pad = is_power_of_two((unsigned)N) ? 16 : 0;
    int LD = N + pad; // padded leading dimension (stride)

    omp_set_dynamic(0);
    omp_set_num_threads(T);

    cout << fixed << setprecision(6);
// #ifdef __AVX512F__
//     cout << "Compiled with AVX-512 support: running AVX-512 8x8 kernel\n";
//     if (!check_avx512_runtime()) {
//         cerr << "Runtime check failed: AVX-512 not available/enabled. Aborting.\n";
//         return 2;
//     }
// #else
//     cout << "Compiled without AVX-512: running AVX2 4x4 kernel fallback\n";
// #endif

    cout << "N="<<N<<" threads="<<T<<" Blocking: MC="<<MC<<" KC="<<KC<<" NC="<<NC<<" LD="<<LD<<"\n";

    // allocate with padded stride: allocate N rows * LD elements per row
    size_t A_elems = (size_t)N * (size_t)LD;
    size_t B_elems = (size_t)N * (size_t)LD;
    size_t C_elems = (size_t)N * (size_t)LD;

    double *A = (double*)aligned_malloc(sizeof(double) * A_elems);
    double *B = (double*)aligned_malloc(sizeof(double) * B_elems);
    double *C = (double*)aligned_malloc(sizeof(double) * C_elems);
    if (!A || !B || !C) { cerr<<"Allocation failed\n"; return 1; }

    // initialize only the N x N logical matrix cells; padded columns remain untouched (but allocated)
    mt19937_64 rng(12345);
    normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * LD + j] = dist(rng);
            B[i * LD + j] = dist(rng);
        }
        // optionally zero pad tail elements for safety
        for (int j = N; j < LD; ++j) {
            A[i * LD + j] = 0.0;
            B[i * LD + j] = 0.0;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < LD; ++j)
            C[i * LD + j] = 0.0;

    double t0 = omp_get_wtime();
    #ifdef __AVX512F__
    cout<<"avx512 supported\n";
    avx512::gemm_tiled_avx512(A,B,C,N,MC,KC,NC, LD, LD, LD);

    #else
    cout<<"avx2 supported\n";
    avx2::gemm_tiled_4x4(A,B,C,N,MC,KC,NC, LD, LD, LD);
    
    #endif
    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;
    double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed * 1e9);

    double checksum = 0.0;
    #pragma omp parallel for reduction(+ : checksum)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i*LD + j];

    
    cout << "Time(s)=" << elapsed << "  GFLOPs=" << gflops << "  Checksum=" << checksum << "\n";

    aligned_free(A); aligned_free(B); aligned_free(C);

    return 0;
}