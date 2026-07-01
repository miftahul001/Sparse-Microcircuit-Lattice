// ============================================================================
// sml_mvp.c — Sparse Microcircuit Lattice, MVP Implementation
// ============================================================================
//
// Single-file C library yang mengimplementasikan SML sesuai
// SML-architecture-spec-v0.2.md.
//
// USAGE MODES:
//
//   1. Sebagai LIBRARY (dari kode Anda):
//        gcc -O3 -march=native -mavx2 -mfma -mpopcnt \
//            -c sml_mvp.c -o sml_mvp.o
//        gcc your_code.c sml_mvp.o -lm -lpthread
//
//      Public API di section [PUBLIC API] di bawah.
//
//   2. Sebagai STANDALONE (dengan test main):
//        gcc -O3 -march=native -mavx2 -mfma -mpopcnt \
//            -DSML_STANDALONE sml_mvp.c -o sml_mvp -lm -lpthread
//        ./sml_mvp
//
// SPEC REFERENCE: SML-architecture-spec-v0.2.md
// HARDWARE TARGET: Intel i5-8500T @ 2.1 GHz, AVX2+FMA+POPCNT, 32GB DDR4
//
// LICENSE: TBD (placeholder)
// AUTHOR: Miftahul Munir + Claude
// DATE: 1 Juli 2026
// ============================================================================

#define _GNU_SOURCE

// ============================================================================
// [PUBLIC API HEADER-STYLE]
// Kalau Anda pakai file ini sebagai library, extract section ini ke sml.h
// ============================================================================
#include <stdint.h>
#include <stddef.h>

// Compile-time constants (bisa di-override via -D flag)
#ifndef SML_D_STATE
#define SML_D_STATE 256   // Dimensi state vector
#endif
#ifndef SML_K_ROUTE
#define SML_K_ROUTE 6     // Top-K microcircuit per inference
#endif
#ifndef SML_NUM_HYPERPLANES
#define SML_NUM_HYPERPLANES 32   // SimHash bit width
#endif
#ifndef SML_MAX_STEPS
#define SML_MAX_STEPS 50
#endif
#ifndef SML_CONV_EPSILON
#define SML_CONV_EPSILON 0.01f   // Relative L2 change threshold
#endif

// Forward declarations (opaque handle recommended untuk future compat)
typedef struct SMLContext SMLContext;

// --- Lifecycle ---

// Create SML context dengan N microcircuits.
// weights_seed digunakan untuk reproducible init.
// Return NULL kalau gagal alloc.
SMLContext* sml_create(int num_microcircuits, uint32_t weights_seed);

// Destroy context, free semua memory.
void sml_destroy(SMLContext* ctx);

// --- Inference ---

typedef struct {
    int num_steps;         // Steps yang di-take (max SML_MAX_STEPS)
    int converged;         // 1 = converge, 0 = hit MAX_STEPS
    float final_delta;     // Rel L2 change di step terakhir
    float final_norm;      // ‖output‖
    double latency_us;     // Wall clock inference latency
} SMLInferenceResult;

// Run inference. Input dan output harus 32-byte aligned, size SML_D_STATE floats.
// normalize_between_steps: 1 = jaga state di unit L2 sphere (recommended).
// Return result stats.
SMLInferenceResult sml_inference(
    SMLContext* ctx,
    const float* input,
    float* output,
    int normalize_between_steps
);

// --- Introspection (opsional) ---

// Dapatkan routing decision untuk input tertentu (untuk debugging/analysis).
// out_mc_ids harus punya slot minimal SML_K_ROUTE.
void sml_route_only(
    SMLContext* ctx,
    const float* input,
    int* out_mc_ids
);

// Cek apakah CPU support AVX2 + FMA + POPCNT.
// Return 1 kalau OK, 0 kalau tidak.
int sml_check_cpu_support(void);

// ============================================================================
// [IMPLEMENTATION]
// ============================================================================

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <stdalign.h>

// ----------------------------------------------------------------------------
// Type: Microcircuit
// ----------------------------------------------------------------------------
typedef struct {
    // Weight matrix INT8, square D_STATE × D_STATE
    alignas(64) int8_t W[SML_D_STATE][SML_D_STATE];
    // Per-row scale + bias untuk dequantisasi + activation
    alignas(64) float scales[SML_D_STATE];
    alignas(64) float bias[SML_D_STATE];
    // Routing metadata
    alignas(64) uint32_t self_hash;
    alignas(32) float resonance_vec[SML_D_STATE];
    // Total ~67 KB per microcircuit
} Microcircuit;

typedef struct {
    uint32_t hash;
    uint32_t mc_index;
} LSHEntry;

typedef struct {
    int idx;
    int dist;
} Candidate;

// ----------------------------------------------------------------------------
// Struct: SMLContext (opaque handle)
// ----------------------------------------------------------------------------
struct SMLContext {
    int num_microcircuits;

    // Storage untuk semua microcircuits (pointer array untuk aligned alloc)
    Microcircuit** mcs;

    // Routing hyperplanes (fixed setelah init)
    alignas(32) float hyperplanes[SML_NUM_HYPERPLANES][SML_D_STATE];

    // LSH index (linear scan MVP)
    LSHEntry* lsh_entries;

    // Internal RNG state (untuk deterministic init)
    uint32_t rng_state;
};

// ----------------------------------------------------------------------------
// Internal: RNG (xorshift32)
// ----------------------------------------------------------------------------
static inline uint32_t rng_next(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rng_float(uint32_t* state) {
    return ((float)rng_next(state) / (float)UINT32_MAX) * 2.0f - 1.0f;
}

static float rng_gaussian(uint32_t* state) {
    float u1 = ((float)rng_next(state) / (float)UINT32_MAX);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float u2 = ((float)rng_next(state) / (float)UINT32_MAX);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static int8_t rng_int8(uint32_t* state) {
    return (int8_t)(rng_next(state) & 0xFF);
}

// ----------------------------------------------------------------------------
// Internal: AVX2 helpers
// ----------------------------------------------------------------------------
static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(hi, lo);
    __m128 shuf = _mm_movehdup_ps(s);
    __m128 sums = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static inline float relu_scalar(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ----------------------------------------------------------------------------
// Internal: L2 norm, distance, normalize (AVX2)
// ----------------------------------------------------------------------------
static float l2_norm_sq(const float* v) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    for (int i = 0; i < SML_D_STATE; i += 32) {
        __m256 v0 = _mm256_load_ps(&v[i +  0]);
        __m256 v1 = _mm256_load_ps(&v[i +  8]);
        __m256 v2 = _mm256_load_ps(&v[i + 16]);
        __m256 v3 = _mm256_load_ps(&v[i + 24]);
        s0 = _mm256_fmadd_ps(v0, v0, s0);
        s1 = _mm256_fmadd_ps(v1, v1, s1);
        s2 = _mm256_fmadd_ps(v2, v2, s2);
        s3 = _mm256_fmadd_ps(v3, v3, s3);
    }
    return hsum_avx2(_mm256_add_ps(_mm256_add_ps(s0, s1),
                                    _mm256_add_ps(s2, s3)));
}

static float l2_distance_sq(const float* a, const float* b) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    for (int i = 0; i < SML_D_STATE; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_load_ps(&a[i +  0]),
                                   _mm256_load_ps(&b[i +  0]));
        __m256 d1 = _mm256_sub_ps(_mm256_load_ps(&a[i +  8]),
                                   _mm256_load_ps(&b[i +  8]));
        __m256 d2 = _mm256_sub_ps(_mm256_load_ps(&a[i + 16]),
                                   _mm256_load_ps(&b[i + 16]));
        __m256 d3 = _mm256_sub_ps(_mm256_load_ps(&a[i + 24]),
                                   _mm256_load_ps(&b[i + 24]));
        s0 = _mm256_fmadd_ps(d0, d0, s0);
        s1 = _mm256_fmadd_ps(d1, d1, s1);
        s2 = _mm256_fmadd_ps(d2, d2, s2);
        s3 = _mm256_fmadd_ps(d3, d3, s3);
    }
    return hsum_avx2(_mm256_add_ps(_mm256_add_ps(s0, s1),
                                    _mm256_add_ps(s2, s3)));
}

static void normalize_unit(float* v) {
    float norm_sq = l2_norm_sq(v);
    if (norm_sq < 1e-16f) return;
    float inv_norm = 1.0f / sqrtf(norm_sq);
    __m256 inv = _mm256_set1_ps(inv_norm);
    for (int i = 0; i < SML_D_STATE; i += 8) {
        _mm256_store_ps(&v[i], _mm256_mul_ps(_mm256_load_ps(&v[i]), inv));
    }
}

// ----------------------------------------------------------------------------
// Internal: SimHash AVX2
// ----------------------------------------------------------------------------
static uint32_t simhash(const SMLContext* ctx, const float* a) {
    uint32_t sig = 0;
    for (int h = 0; h < SML_NUM_HYPERPLANES; ++h) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const float* hp = ctx->hyperplanes[h];
        for (int i = 0; i < SML_D_STATE; i += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i]),
                                    _mm256_load_ps(&a[i]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i+8]),
                                    _mm256_load_ps(&a[i+8]), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i+16]),
                                    _mm256_load_ps(&a[i+16]), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i+24]),
                                    _mm256_load_ps(&a[i+24]), acc3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        sig |= (hsum_avx2(acc) > 0.0f ? 1u : 0u) << h;
    }
    return sig;
}

// ----------------------------------------------------------------------------
// Internal: Route (linear scan, top-K)
// ----------------------------------------------------------------------------
static void route(const SMLContext* ctx, const float* query, int* out_ids) {
    uint32_t q_hash = simhash(ctx, query);

    Candidate best[SML_K_ROUTE];
    for (int i = 0; i < SML_K_ROUTE; ++i) {
        best[i].idx = -1;
        best[i].dist = INT_MAX;
    }

    for (int i = 0; i < ctx->num_microcircuits; ++i) {
        int dist = __builtin_popcount(q_hash ^ ctx->lsh_entries[i].hash);
        if (dist < best[SML_K_ROUTE - 1].dist) {
            int j = SML_K_ROUTE - 1;
            while (j > 0 && best[j - 1].dist > dist) {
                best[j] = best[j - 1];
                j--;
            }
            best[j].idx = (int)ctx->lsh_entries[i].mc_index;
            best[j].dist = dist;
        }
    }
    for (int i = 0; i < SML_K_ROUTE; ++i) out_ids[i] = best[i].idx;
}

// ----------------------------------------------------------------------------
// Internal: Microcircuit forward (D_STATE × D_STATE INT8, ReLU)
// ----------------------------------------------------------------------------
static void microcircuit_forward(const Microcircuit* M,
                                  const float* a, float* y) {
    for (int row = 0; row < SML_D_STATE; ++row) {
        const int8_t* W_row = M->W[row];
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int col = 0; col < SML_D_STATE; col += 32) {
            __m256i w_int8 = _mm256_load_si256(
                (const __m256i*)(W_row + col));
            __m128i w_lo = _mm256_castsi256_si128(w_int8);
            __m128i w_hi = _mm256_extracti128_si256(w_int8, 1);

            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_srli_si128(w_lo, 8)));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_srli_si128(w_hi, 8)));

            __m256 a0 = _mm256_load_ps(a + col + 0);
            __m256 a1 = _mm256_load_ps(a + col + 8);
            __m256 a2 = _mm256_load_ps(a + col + 16);
            __m256 a3 = _mm256_load_ps(a + col + 24);

            acc0 = _mm256_fmadd_ps(w0, a0, acc0);
            acc1 = _mm256_fmadd_ps(w1, a1, acc1);
            acc2 = _mm256_fmadd_ps(w2, a2, acc2);
            acc3 = _mm256_fmadd_ps(w3, a3, acc3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        float dot = hsum_avx2(acc);
        y[row] = relu_scalar(dot * M->scales[row] + M->bias[row]);
    }
}

// ----------------------------------------------------------------------------
// Internal: Energy-weighted fusion
// ----------------------------------------------------------------------------
static void fuse_energy_weighted(const float outputs[SML_K_ROUTE][SML_D_STATE],
                                  float* result) {
    float norms_sq[SML_K_ROUTE];
    float total = 0.0f;
    for (int i = 0; i < SML_K_ROUTE; ++i) {
        norms_sq[i] = l2_norm_sq(outputs[i]);
        total += norms_sq[i];
    }

    // Fallback: kalau semua zero, uniform average
    if (total < 1e-12f) {
        for (int j = 0; j < SML_D_STATE; ++j) {
            float s = 0.0f;
            for (int i = 0; i < SML_K_ROUTE; ++i) s += outputs[i][j];
            result[j] = s / SML_K_ROUTE;
        }
        return;
    }

    float inv_total = 1.0f / total;
    __m256 w_vecs[SML_K_ROUTE];
    for (int i = 0; i < SML_K_ROUTE; ++i) {
        w_vecs[i] = _mm256_set1_ps(norms_sq[i] * inv_total);
    }
    for (int j = 0; j < SML_D_STATE; j += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < SML_K_ROUTE; ++i) {
            sum = _mm256_fmadd_ps(w_vecs[i],
                                    _mm256_load_ps(&outputs[i][j]), sum);
        }
        _mm256_store_ps(&result[j], sum);
    }
}

// ----------------------------------------------------------------------------
// Internal: Init microcircuit (calibrated scale + structured bias)
// Sesuai spec v0.2 Section 5, 7
// ----------------------------------------------------------------------------
static void init_microcircuit(Microcircuit* M, uint32_t* rng) {
    long sum_abs_w = 0;
    for (int r = 0; r < SML_D_STATE; ++r) {
        for (int c = 0; c < SML_D_STATE; ++c) {
            int8_t w = rng_int8(rng);
            M->W[r][c] = w;
            sum_abs_w += (w < 0) ? -w : w;
        }
    }
    float mean_abs_w = (float)sum_abs_w / (SML_D_STATE * SML_D_STATE);
    // Calibrated scale (Section 5, spec v0.2):
    //   Untuk unit-norm input, dot ~ mean_abs_w
    //   Target output magnitude ~ 1/sqrt(D)
    //   → scale = 1 / (sqrt(D) × mean_abs_w)
    float scale_val = 1.0f / (sqrtf((float)SML_D_STATE) * mean_abs_w);
    float bias_val = 0.05f / sqrtf((float)SML_D_STATE);

    for (int r = 0; r < SML_D_STATE; ++r) {
        M->scales[r] = scale_val;
        M->bias[r] = bias_val;
    }
    for (int i = 0; i < SML_D_STATE; ++i) {
        M->resonance_vec[i] = rng_float(rng);
    }
    // self_hash akan di-compute setelah hyperplanes ready (di sml_create)
    M->self_hash = 0;
}

// ============================================================================
// [PUBLIC API IMPLEMENTATION]
// ============================================================================

int sml_check_cpu_support(void) {
    // Compile-time check via __builtin_cpu_supports (GCC/Clang)
    // Runtime check untuk graceful failure
    if (!__builtin_cpu_supports("avx2")) return 0;
    if (!__builtin_cpu_supports("fma")) return 0;
    if (!__builtin_cpu_supports("popcnt")) return 0;
    return 1;
}

SMLContext* sml_create(int num_microcircuits, uint32_t weights_seed) {
    if (num_microcircuits <= 0) return NULL;
    if (!sml_check_cpu_support()) {
        fprintf(stderr, "SML: CPU tidak support AVX2/FMA/POPCNT\n");
        return NULL;
    }

    SMLContext* ctx = calloc(1, sizeof(SMLContext));
    if (!ctx) return NULL;

    ctx->num_microcircuits = num_microcircuits;
    ctx->rng_state = weights_seed ? weights_seed : 42;

    // Init hyperplanes (random Gaussian)
    for (int h = 0; h < SML_NUM_HYPERPLANES; ++h) {
        for (int i = 0; i < SML_D_STATE; ++i) {
            ctx->hyperplanes[h][i] = rng_gaussian(&ctx->rng_state);
        }
    }

    // Allocate microcircuits
    ctx->mcs = calloc(num_microcircuits, sizeof(Microcircuit*));
    if (!ctx->mcs) { free(ctx); return NULL; }

    for (int i = 0; i < num_microcircuits; ++i) {
        if (posix_memalign((void**)&ctx->mcs[i], 64,
                            sizeof(Microcircuit)) != 0) {
            for (int j = 0; j < i; ++j) free(ctx->mcs[j]);
            free(ctx->mcs);
            free(ctx);
            return NULL;
        }
        init_microcircuit(ctx->mcs[i], &ctx->rng_state);
        // Compute self_hash setelah hyperplanes ready
        ctx->mcs[i]->self_hash = simhash(ctx, ctx->mcs[i]->resonance_vec);
    }

    // Build LSH index
    ctx->lsh_entries = malloc(num_microcircuits * sizeof(LSHEntry));
    if (!ctx->lsh_entries) {
        for (int i = 0; i < num_microcircuits; ++i) free(ctx->mcs[i]);
        free(ctx->mcs);
        free(ctx);
        return NULL;
    }
    for (int i = 0; i < num_microcircuits; ++i) {
        ctx->lsh_entries[i].hash = ctx->mcs[i]->self_hash;
        ctx->lsh_entries[i].mc_index = (uint32_t)i;
    }

    return ctx;
}

void sml_destroy(SMLContext* ctx) {
    if (!ctx) return;
    if (ctx->lsh_entries) free(ctx->lsh_entries);
    if (ctx->mcs) {
        for (int i = 0; i < ctx->num_microcircuits; ++i) {
            if (ctx->mcs[i]) free(ctx->mcs[i]);
        }
        free(ctx->mcs);
    }
    free(ctx);
}

void sml_route_only(SMLContext* ctx, const float* input, int* out_mc_ids) {
    // Normalize a temporary copy untuk consistent routing
    alignas(32) float normalized[SML_D_STATE];
    memcpy(normalized, input, SML_D_STATE * sizeof(float));
    normalize_unit(normalized);
    route(ctx, normalized, out_mc_ids);
}

SMLInferenceResult sml_inference(
    SMLContext* ctx,
    const float* input,
    float* output,
    int normalize_between_steps
) {
    SMLInferenceResult result = {0};
    double t_start = now_sec();

    alignas(32) float a_cur[SML_D_STATE];
    alignas(32) float a_new[SML_D_STATE];
    alignas(32) float outs[SML_K_ROUTE][SML_D_STATE];

    // Setup: normalize input, freeze routing
    memcpy(a_cur, input, SML_D_STATE * sizeof(float));
    normalize_unit(a_cur);

    int mc_ids[SML_K_ROUTE];
    route(ctx, a_cur, mc_ids);

    // Cascade loop
    for (int step = 0; step < SML_MAX_STEPS; ++step) {
        // 6× sequential forward (kandidat untuk paralel di Phase 4 v3)
        for (int i = 0; i < SML_K_ROUTE; ++i) {
            microcircuit_forward(ctx->mcs[mc_ids[i]], a_cur, outs[i]);
        }

        // Fusion
        fuse_energy_weighted(outs, a_new);

        // Optional normalize
        if (normalize_between_steps) normalize_unit(a_new);

        // Convergence check
        float diff_sq = l2_distance_sq(a_cur, a_new);
        float cur_norm_sq = l2_norm_sq(a_cur);
        float rel_delta = sqrtf(diff_sq / (cur_norm_sq + 1e-8f));

        result.num_steps = step + 1;
        result.final_delta = rel_delta;

        if (rel_delta < SML_CONV_EPSILON) {
            result.converged = 1;
            memcpy(a_cur, a_new, SML_D_STATE * sizeof(float));
            break;
        }

        memcpy(a_cur, a_new, SML_D_STATE * sizeof(float));
    }

    result.final_norm = sqrtf(l2_norm_sq(a_cur));
    memcpy(output, a_cur, SML_D_STATE * sizeof(float));
    result.latency_us = (now_sec() - t_start) * 1e6;
    return result;
}

// ============================================================================
// [STANDALONE TEST HARNESS]
// Build dengan -DSML_STANDALONE untuk enable test main
// ============================================================================
#ifdef SML_STANDALONE

#include <stdio.h>

static void print_banner(void) {
    printf("================================================================\n");
    printf("  SML MVP — Standalone Test\n");
    printf("  Spec: SML-architecture-spec-v0.2.md\n");
    printf("  D_STATE=%d, K_ROUTE=%d, HYPERPLANES=%d, MAX_STEPS=%d\n",
           SML_D_STATE, SML_K_ROUTE, SML_NUM_HYPERPLANES, SML_MAX_STEPS);
    printf("================================================================\n\n");
}

// LCG untuk generate test input yang reproducible
static uint32_t input_rng = 12345;
static float test_input_gen(void) {
    input_rng = input_rng * 1103515245u + 12345u;
    return ((float)input_rng / (float)UINT32_MAX) * 2.0f - 1.0f;
}

static void test_cpu_support(void) {
    printf("--- CPU Support Check ---\n");
    if (sml_check_cpu_support()) {
        printf("  ✓ AVX2 + FMA + POPCNT available\n\n");
    } else {
        printf("  ✗ CPU tidak support required features. Abort.\n");
        exit(1);
    }
}

static void test_lifecycle(void) {
    printf("--- Lifecycle: create → destroy ---\n");
    double t0 = now_sec();
    SMLContext* ctx = sml_create(1000, 42);
    double t_create = now_sec() - t0;
    if (!ctx) { printf("  FAIL: sml_create returned NULL\n"); exit(1); }
    printf("  ✓ Create 1000 mc: %.2f ms\n", t_create * 1000);
    printf("    Memory: ~%.1f MB\n",
           (double)sizeof(Microcircuit) * 1000 / (1024*1024));
    sml_destroy(ctx);
    printf("  ✓ Destroy complete\n\n");
}

static void test_single_inference(SMLContext* ctx) {
    printf("--- Single Inference (with normalize between steps) ---\n");
    alignas(32) float input[SML_D_STATE], output[SML_D_STATE];
    for (int i = 0; i < SML_D_STATE; ++i) input[i] = test_input_gen();

    SMLInferenceResult r = sml_inference(ctx, input, output, 1);

    // Sanity checks
    int nans = 0, infs = 0;
    for (int i = 0; i < SML_D_STATE; ++i) {
        if (isnan(output[i])) nans++;
        else if (isinf(output[i])) infs++;
    }

    printf("  Steps: %d\n", r.num_steps);
    printf("  Converged: %s\n", r.converged ? "YES" : "NO");
    printf("  Final delta: %.4f (threshold %.4f)\n",
           r.final_delta, SML_CONV_EPSILON);
    printf("  Final norm: %.4f\n", r.final_norm);
    printf("  Latency: %.1f µs\n", r.latency_us);
    printf("  Output sanity: NaN=%d, Inf=%d\n", nans, infs);
    if (r.converged && nans == 0 && infs == 0) {
        printf("  ✓ PASS\n\n");
    } else {
        printf("  ✗ FAIL\n\n");
    }
}

static void test_batch_inference(SMLContext* ctx) {
    printf("--- Batch Inference: 100 trials ---\n");
    alignas(32) float input[SML_D_STATE], output[SML_D_STATE];

    int converged_count = 0;
    long total_steps = 0;
    double total_latency = 0;
    double min_latency = 1e18, max_latency = 0;

    for (int trial = 0; trial < 100; ++trial) {
        for (int i = 0; i < SML_D_STATE; ++i) input[i] = test_input_gen();
        SMLInferenceResult r = sml_inference(ctx, input, output, 1);
        if (r.converged) converged_count++;
        total_steps += r.num_steps;
        total_latency += r.latency_us;
        if (r.latency_us < min_latency) min_latency = r.latency_us;
        if (r.latency_us > max_latency) max_latency = r.latency_us;
    }

    printf("  Converged: %d / 100 (%.0f%%)\n",
           converged_count, converged_count * 1.0);
    printf("  Mean steps: %.1f\n", total_steps / 100.0);
    printf("  Latency: mean %.1f µs, min %.1f, max %.1f\n",
           total_latency / 100.0, min_latency, max_latency);
    printf("  Throughput: %.0f inference/sec\n",
           1e6 / (total_latency / 100.0));

    if (converged_count == 100) printf("  ✓ PASS\n\n");
    else printf("  ⚠ %d didn't converge\n\n", 100 - converged_count);
}

static void test_determinism(SMLContext* ctx) {
    printf("--- Determinism Check ---\n");
    alignas(32) float input[SML_D_STATE], out1[SML_D_STATE], out2[SML_D_STATE];
    for (int i = 0; i < SML_D_STATE; ++i) input[i] = test_input_gen();

    SMLInferenceResult r1 = sml_inference(ctx, input, out1, 1);
    SMLInferenceResult r2 = sml_inference(ctx, input, out2, 1);

    float max_diff = 0;
    for (int i = 0; i < SML_D_STATE; ++i) {
        float d = fabsf(out1[i] - out2[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("  Run 1: %d steps, delta %.6f\n", r1.num_steps, r1.final_delta);
    printf("  Run 2: %d steps, delta %.6f\n", r2.num_steps, r2.final_delta);
    printf("  Max output diff: %.6e\n", max_diff);
    if (max_diff < 1e-6f && r1.num_steps == r2.num_steps) {
        printf("  ✓ Deterministic\n\n");
    } else {
        printf("  ✗ Non-deterministic!\n\n");
    }
}

static void test_content_addressing(SMLContext* ctx) {
    printf("--- Content-Addressed Routing ---\n");
    printf("  (Note: dengan random weights, signal lemah — expected)\n");

    int similar_close_count = 0, different_far_count = 0;

    for (int trial = 0; trial < 20; ++trial) {
        alignas(32) float q1[SML_D_STATE], q2[SML_D_STATE], q3[SML_D_STATE];
        alignas(32) float o1[SML_D_STATE], o2[SML_D_STATE], o3[SML_D_STATE];

        for (int i = 0; i < SML_D_STATE; ++i) {
            q1[i] = test_input_gen();
            q2[i] = q1[i] + test_input_gen() * 0.05f;  // small perturbation
            q3[i] = test_input_gen();
        }

        // Compare routing decisions (more direct signal than output)
        int ids1[SML_K_ROUTE], ids2[SML_K_ROUTE], ids3[SML_K_ROUTE];
        sml_route_only(ctx, q1, ids1);
        sml_route_only(ctx, q2, ids2);
        sml_route_only(ctx, q3, ids3);

        int overlap_12 = 0, overlap_13 = 0;
        for (int i = 0; i < SML_K_ROUTE; ++i)
            for (int j = 0; j < SML_K_ROUTE; ++j) {
                if (ids1[i] == ids2[j]) overlap_12++;
                if (ids1[i] == ids3[j]) overlap_13++;
            }

        if (overlap_12 > overlap_13) similar_close_count++;
        if (overlap_13 < overlap_12) different_far_count++;

        // Also run inference untuk timing
        sml_inference(ctx, q1, o1, 1);
        sml_inference(ctx, q2, o2, 1);
        sml_inference(ctx, q3, o3, 1);
    }

    printf("  Routing overlap similar (q1-q2) > different (q1-q3): %d/20\n",
           similar_close_count);
    if (similar_close_count >= 15) {
        printf("  ✓ Strong content-addressed routing\n\n");
    } else if (similar_close_count >= 10) {
        printf("  ~ Moderate content-addressed routing\n\n");
    } else {
        printf("  ? Weak signal (may need training)\n\n");
    }
}

static void test_latency_breakdown(SMLContext* ctx) {
    printf("--- Latency Breakdown Estimate ---\n");

    // Timing dari sml_inference (route + cascade)
    alignas(32) float input[SML_D_STATE], output[SML_D_STATE];
    for (int i = 0; i < SML_D_STATE; ++i) input[i] = test_input_gen();

    SMLInferenceResult r = sml_inference(ctx, input, output, 1);
    double per_step = (r.num_steps > 0) ?
                        r.latency_us / r.num_steps : 0;

    printf("  Total inference: %.1f µs (%d steps)\n",
           r.latency_us, r.num_steps);
    printf("  Per cascade step: %.1f µs\n", per_step);
    printf("  Route (once, ~O(N) = %d ops): ~%.1f µs (embedded in first step)\n",
           ctx->num_microcircuits,
           ctx->num_microcircuits * 0.0025);  // ~2.5 ns per popcount
    printf("  Cascade steps: %d × %.1f µs\n\n",
           r.num_steps, per_step);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    print_banner();

    test_cpu_support();
    test_lifecycle();

    printf("--- Creating context untuk main tests (1000 mc, seed 42) ---\n");
    SMLContext* ctx = sml_create(1000, 42);
    if (!ctx) { printf("Failed\n"); return 1; }
    printf("  ✓ Ready\n\n");

    test_single_inference(ctx);
    test_batch_inference(ctx);
    test_determinism(ctx);
    test_content_addressing(ctx);
    test_latency_breakdown(ctx);

    sml_destroy(ctx);

    printf("================================================================\n");
    printf("  MVP Standalone Test complete.\n");
    printf("================================================================\n");
    printf("Usage sebagai library:\n");
    printf("  gcc -O3 -march=native -mavx2 -mfma -mpopcnt \\\n");
    printf("      -c sml_mvp.c -o sml_mvp.o\n");
    printf("  gcc your_code.c sml_mvp.o -lm -lpthread\n");
    printf("\n");
    printf("Extract sml.h dari section [PUBLIC API HEADER-STYLE] di atas.\n");
    return 0;
}

#endif  // SML_STANDALONE
