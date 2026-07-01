// ============================================================
// SML Phase 1: Microcircuit Forward Function (First Implementation)
//
// Bukan benchmark — ini kode SML pertama yang akan jadi
// inner loop inference. Diukur dan dibandingkan terhadap
// prediksi Phase 0.
//
// Keputusan Phase 1:
//   - INT8 weights (spec v0.1 saran: "Mulai INT8 di Fase 1,
//     eksperimen INT4 di Fase 5")
//   - FP32 activations (tidak ada loss dari quantization sisi input)
//   - ReLU activation (GELU di Phase 2 setelah correctness validated)
//   - Single mikrosirkit forward (multi-core di Phase 2)
//
// Build: gcc -O3 -march=native -mavx2 -mfma phase1_microcircuit.c \
//          -o phase1_microcircuit -lm
// Run:   ./phase1_microcircuit
// ============================================================

#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================
// Dimensi (sesuai spec v0.1)
// ============================================================
#define D_IN   384   // input dimension per microcircuit
#define D_OUT  256   // output dimension per microcircuit
#define ROW_BYTES (D_IN * sizeof(int8_t))  // 384 bytes per row

// ============================================================
// Struct mikrosirkit — versi Phase 1 (INT8)
// Ukuran aktual ~98 KB; muat di L2 (256 KB) dengan banyak ruang sisa.
// Phase 5 akan upgrade ke INT4 (~50 KB), spec target 192 KB.
// ============================================================
typedef struct {
    alignas(64) int8_t W[D_OUT][D_IN];   // 256 × 384 = 96 KB
    alignas(64) float  scales[D_OUT];    // 1 KB — per-row scale
    alignas(64) float  bias[D_OUT];      // 1 KB — per-row bias
} Microcircuit;

// Compile-time size check
_Static_assert(sizeof(((Microcircuit*)0)->W) == 98304, "W size mismatch");

// ============================================================
// Helpers: timing
// ============================================================
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ============================================================
// Horizontal sum: __m256 → float
// ============================================================
static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum128 = _mm_add_ps(hi, lo);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ============================================================
// Activation: ReLU (Phase 1). GELU di Phase 2.
// ============================================================
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// ============================================================
// FORWARD FUNCTION — HOT PATH
//
// Untuk setiap dari 256 baris W:
//   1. Dot product 384-dim antara baris W (INT8) dan input a (FP32)
//   2. Dequantize: scale × dot
//   3. Tambahkan bias
//   4. Apply ReLU
//
// Strategi AVX2:
//   - 4 accumulator paralel untuk break dependency chain
//   - Process 32 INT8 elements per iteration (1 cache line worth)
//   - Convert INT8 → FP32 inline (sign-extend via _mm256_cvtepi8_epi32)
// ============================================================
void microcircuit_forward(const Microcircuit* M,
                          const float* a,        // aligned 32B, size D_IN
                          float* y) {            // aligned 32B, size D_OUT
    for (int row = 0; row < D_OUT; ++row) {
        const int8_t* W_row = M->W[row];

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // Process 384 elements: 12 iterations × 32 elements
        for (int col = 0; col < D_IN; col += 32) {
            // Load 32 INT8 weights (1 cache line worth)
            __m256i w_int8 = _mm256_load_si256(
                (const __m256i*)(W_row + col));

            // Split into 4 vectors of 8 INT32 each, then to FP32
            __m128i w_lo = _mm256_castsi256_si128(w_int8);            // bytes 0-15
            __m128i w_hi = _mm256_extracti128_si256(w_int8, 1);       // bytes 16-31

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_srli_si128(w_lo, 8)));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_srli_si128(w_hi, 8)));

            // Load 32 FP32 activations (4 vectors)
            __m256 a0 = _mm256_load_ps(a + col +  0);
            __m256 a1 = _mm256_load_ps(a + col +  8);
            __m256 a2 = _mm256_load_ps(a + col + 16);
            __m256 a3 = _mm256_load_ps(a + col + 24);

            // FMA — 4 independent chains
            acc0 = _mm256_fmadd_ps(w0, a0, acc0);
            acc1 = _mm256_fmadd_ps(w1, a1, acc1);
            acc2 = _mm256_fmadd_ps(w2, a2, acc2);
            acc3 = _mm256_fmadd_ps(w3, a3, acc3);
        }

        // Combine 4 accumulators
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        float dot = hsum_avx2(acc);

        // Dequantize, bias, activation
        y[row] = relu(dot * M->scales[row] + M->bias[row]);
    }
}

// ============================================================
// REFERENCE: Scalar version untuk correctness check
// Slow tapi obviously correct. Used sebagai oracle.
// ============================================================
void microcircuit_forward_scalar(const Microcircuit* M,
                                  const float* a,
                                  float* y) {
    for (int row = 0; row < D_OUT; ++row) {
        float dot = 0.0f;
        for (int col = 0; col < D_IN; ++col) {
            dot += (float)M->W[row][col] * a[col];
        }
        y[row] = relu(dot * M->scales[row] + M->bias[row]);
    }
}

// ============================================================
// Initialization helpers
// ============================================================
static int64_t lcg_state = 1;
static int8_t rand_int8(void) {
    lcg_state = lcg_state * 6364136223846793005LL + 1442695040888963407LL;
    return (int8_t)((lcg_state >> 33) & 0xFF);
}
static float rand_float(void) {
    lcg_state = lcg_state * 6364136223846793005LL + 1442695040888963407LL;
    uint32_t bits = (uint32_t)(lcg_state >> 32);
    return ((float)(bits & 0xFFFFFF) / (float)0xFFFFFF) * 2.0f - 1.0f;
}

static void init_microcircuit(Microcircuit* M) {
    for (int r = 0; r < D_OUT; ++r) {
        for (int c = 0; c < D_IN; ++c) {
            M->W[r][c] = rand_int8();
        }
        // Scale dipilih supaya output reasonable
        // dot ~ sum of (INT8 ~ ±64) * (FP32 ~ ±1) over 384 elem
        // ~ sqrt(384) * 64 * 1 ≈ 1250 (random walk variance)
        // scale 0.001 supaya output ~ ±1.25
        M->scales[r] = 0.001f;
        M->bias[r] = rand_float() * 0.1f;
    }
}

static void init_input(float* a) {
    for (int i = 0; i < D_IN; ++i) {
        a[i] = rand_float();
    }
}

// ============================================================
// Correctness test: AVX2 vs scalar reference
// ============================================================
static int verify_correctness(const Microcircuit* M, const float* a) {
    alignas(32) float y_avx2[D_OUT];
    alignas(32) float y_scalar[D_OUT];

    microcircuit_forward(M, a, y_avx2);
    microcircuit_forward_scalar(M, a, y_scalar);

    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < D_OUT; ++i) {
        float diff = fabsf(y_avx2[i] - y_scalar[i]);
        if (diff > max_diff) max_diff = diff;
        // Toleransi FP accumulation order: relative 1e-4
        float ref = fabsf(y_scalar[i]) + 1e-6f;
        if (diff / ref > 1e-4f) errors++;
    }
    printf("Correctness:\n");
    printf("  Max absolute diff vs scalar: %.6e\n", max_diff);
    printf("  Mismatched outputs (rel > 1e-4): %d / %d\n", errors, D_OUT);
    return errors == 0;
}

// ============================================================
// Output sanity check: no NaN/Inf, reasonable magnitude
// ============================================================
static void check_output_sanity(const float* y) {
    int nans = 0, infs = 0, zeros = 0;
    float min_v = 1e9f, max_v = -1e9f, sum_abs = 0.0f;
    for (int i = 0; i < D_OUT; ++i) {
        if (isnan(y[i])) nans++;
        else if (isinf(y[i])) infs++;
        else if (y[i] == 0.0f) zeros++;
        else {
            if (y[i] < min_v) min_v = y[i];
            if (y[i] > max_v) max_v = y[i];
            sum_abs += fabsf(y[i]);
        }
    }
    int finite = D_OUT - nans - infs;
    printf("Output stats:\n");
    printf("  NaN: %d, Inf: %d, Zero: %d, Finite non-zero: %d\n",
           nans, infs, zeros, finite - zeros);
    if (finite - zeros > 0) {
        printf("  Range: [%.4f, %.4f]\n", min_v, max_v);
        printf("  Mean |y|: %.4f\n", sum_abs / (finite - zeros));
    }
}

// ============================================================
// Latency benchmark
// ============================================================
static void benchmark_latency(const Microcircuit* M, const float* a) {
    alignas(32) float y[D_OUT];

    // Warmup — get into L2
    for (int i = 0; i < 100; ++i) {
        microcircuit_forward(M, a, y);
    }

    // Measure
    const int iterations = 100000;
    double t0 = now_sec();
    for (int i = 0; i < iterations; ++i) {
        microcircuit_forward(M, a, y);
    }
    double t1 = now_sec();

    double total_us = (t1 - t0) * 1e6;
    double per_call_us = total_us / iterations;

    // FLOPs analysis
    // Per forward: 256 rows × 384 cols × 2 (MAC) = 196608 FLOP
    //            + 256 × (mul + add + relu) ≈ 768 small ops
    double flops_per_call = 196608.0;
    double gflops = (flops_per_call / per_call_us) / 1e3;

    printf("Latency benchmark:\n");
    printf("  Iterations           : %d\n", iterations);
    printf("  Total time           : %.2f ms\n", total_us / 1000.0);
    printf("  Per microcircuit fwd : %.2f µs\n", per_call_us);
    printf("  Effective throughput : %.1f GFLOPS\n", gflops);
    printf("\n");

    // Compare to Phase 0 predictions
    printf("Phase 0 prediction comparison:\n");
    printf("  Theoretical compute  : %.2f µs (196608 FLOP / 66.8 GFLOPS)\n",
           196608.0 / 66.8 / 1e3 * 1e6 / 1e6);  // µs
    printf("  L2 memory load floor : %.2f µs (96 KB / 45.4 GB/s)\n",
           (96.0 * 1024.0 / (45.4 * 1e9)) * 1e6);
    printf("  Actual (warm L2)     : %.2f µs\n", per_call_us);

    double compute_us = 196608.0 / 66.8e9 * 1e6;
    double memory_us = (96.0 * 1024.0 / (45.4 * 1e9)) * 1e6;
    double predicted_us = (compute_us > memory_us) ? compute_us : memory_us;
    double overhead = per_call_us - predicted_us;
    printf("  Overhead (conversion+activation+loop): %.2f µs (%.0f%%)\n",
           overhead, 100.0 * overhead / predicted_us);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 1: Microcircuit Forward (INT8)\n");
    printf("================================================\n");
    printf("Microcircuit size: %zu bytes (%.1f KB)\n",
           sizeof(Microcircuit),
           sizeof(Microcircuit) / 1024.0);
    printf("  (fits di L2 256 KB dengan margin %zu KB)\n\n",
           (256 * 1024 - sizeof(Microcircuit)) / 1024);

    // Allocate aligned
    Microcircuit* M;
    if (posix_memalign((void**)&M, 64, sizeof(Microcircuit)) != 0) {
        fprintf(stderr, "Alloc failed\n");
        return 1;
    }
    float* a;
    if (posix_memalign((void**)&a, 32, D_IN * sizeof(float)) != 0) {
        fprintf(stderr, "Alloc failed\n");
        return 1;
    }

    // Initialize
    init_microcircuit(M);
    init_input(a);

    // Run tests
    int ok = verify_correctness(M, a);
    printf("\n");

    if (!ok) {
        printf("FAIL — output AVX2 tidak match scalar reference.\n");
        printf("       Tidak lanjut benchmark sampai correctness fixed.\n");
        return 1;
    }

    alignas(32) float y[D_OUT];
    microcircuit_forward(M, a, y);
    check_output_sanity(y);
    printf("\n");

    benchmark_latency(M, a);

    printf("\n================================================\n");
    printf("  Phase 1 status\n");
    printf("================================================\n");
    printf("Jika latensi < 6 µs dan correctness PASS:\n");
    printf("  → Phase 1 selesai. Next: Phase 2 multi-core dispatch.\n");
    printf("Jika latensi > 10 µs:\n");
    printf("  → Investigate: profile dengan perf, cek assembly.\n");

    free(a);
    free(M);
    return 0;
}
