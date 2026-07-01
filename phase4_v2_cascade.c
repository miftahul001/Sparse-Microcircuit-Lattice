// ============================================================
// SML Phase 4 v2: Cascade with Better Init + Routing Diagnostic
//
// TUJUAN: Isolasi apakah cascade non-convergence disebabkan oleh:
//   (a) Weight/scale init tidak tepat
//   (b) Routing discontinuity (routing berubah setiap step)
//   (c) Kombinasi keduanya
//
// EKSPERIMEN:
//   Config A: dynamic routing + better init
//   Config B: FIXED routing + better init (route sekali di step 0)
//
// FIXES vs v1:
//   1. Input dinormalisasi ke unit L2 norm
//   2. Weight scale calibrated: scales[r] = 1.0 / sqrt(D_STATE * mean_|W|²)
//   3. Bias structured (fixed offset, bukan random)
//   4. Optional: fixed routing mode
//   5. Trajectory logging untuk deep dive
//
// Build: gcc -O3 -march=native -mavx2 -mfma -mpopcnt \
//          phase4_v2_cascade.c -o phase4_v2_cascade -lm
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
#include <limits.h>

// ============================================================
// Config
// ============================================================
#define D_STATE            256
#define NUM_MICROCIRCUITS  1000
#define K_ROUTE            6
#define NUM_HYPERPLANES    32
#define ROUTE_RAND_PROB    20
#define MAX_STEPS          50
#define CONV_EPSILON       0.01f

// Routing modes
typedef enum {
    ROUTING_DYNAMIC = 0,   // Route setiap step (Phase 4 v1)
    ROUTING_FIXED = 1      // Route sekali dari input awal, freeze
} RoutingMode;

// ============================================================
// Structs (sama dengan v1)
// ============================================================
typedef struct {
    alignas(64) int8_t W[D_STATE][D_STATE];
    alignas(64) float  scales[D_STATE];
    alignas(64) float  bias[D_STATE];
    alignas(64) uint32_t self_hash;
    alignas(32) float resonance_vec[D_STATE];
} Microcircuit;

typedef struct { uint32_t hash; uint32_t mc_index; } LSHEntry;
typedef struct { LSHEntry* entries; int count; } LSHLinearIndex;
typedef struct { int idx; int dist; } Candidate;

typedef struct {
    int num_steps;
    int converged;
    float final_delta;
    float max_delta;
    float final_norm;      // NEW: ‖a_final‖
    float initial_norm;    // NEW: ‖a_initial‖
} CascadeResult;

// ============================================================
// Globals
// ============================================================
static alignas(32) float g_hyperplanes[NUM_HYPERPLANES][D_STATE];
static uint32_t g_rng_state = 42;

// ============================================================
// RNG (sama)
// ============================================================
static inline uint32_t xorshift32(void) {
    uint32_t x = g_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng_state = x;
    return x;
}
static float rand_float(void) {
    return ((float)xorshift32() / (float)UINT32_MAX) * 2.0f - 1.0f;
}
static float rand_gaussian(void) {
    float u1 = ((float)xorshift32() / (float)UINT32_MAX);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float u2 = ((float)xorshift32() / (float)UINT32_MAX);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}
static int8_t rand_int8(void) { return (int8_t)(xorshift32() & 0xFF); }

// ============================================================
// Helpers
// ============================================================
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
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
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

// ============================================================
// L2 helpers (sama)
// ============================================================
static inline float l2_norm_sq_avx2(const float* v) {
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < D_STATE; i += 8) {
        __m256 x = _mm256_load_ps(&v[i]);
        s = _mm256_fmadd_ps(x, x, s);
    }
    return hsum_avx2(s);
}
static inline float l2_norm_avx2(const float* v) {
    return sqrtf(l2_norm_sq_avx2(v));
}
static inline float l2_distance_sq_avx2(const float* a, const float* b) {
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < D_STATE; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_load_ps(&a[i]),
                                  _mm256_load_ps(&b[i]));
        s = _mm256_fmadd_ps(d, d, s);
    }
    return hsum_avx2(s);
}

// NEW: normalize ke unit L2 norm (in-place)
static inline void normalize_unit(float* v) {
    float norm = l2_norm_avx2(v);
    if (norm < 1e-8f) return;
    __m256 inv = _mm256_set1_ps(1.0f / norm);
    for (int i = 0; i < D_STATE; i += 8) {
        __m256 x = _mm256_load_ps(&v[i]);
        _mm256_store_ps(&v[i], _mm256_mul_ps(x, inv));
    }
}

// ============================================================
// SimHash (sama)
// ============================================================
static uint32_t simhash_avx2(const float* a) {
    uint32_t sig = 0;
    for (int h = 0; h < NUM_HYPERPLANES; ++h) {
        __m256 acc = _mm256_setzero_ps();
        const float* hp = g_hyperplanes[h];
        for (int i = 0; i < D_STATE; i += 8) {
            acc = _mm256_fmadd_ps(_mm256_load_ps(&hp[i]),
                                    _mm256_load_ps(&a[i]), acc);
        }
        sig |= (hsum_avx2(acc) > 0.0f ? 1u : 0u) << h;
    }
    return sig;
}

// ============================================================
// Route (linear scan)
// ============================================================
static void route_linear(const LSHLinearIndex* idx, const float* query,
                          int* out_ids, int k, int enable_random) {
    uint32_t q_hash = simhash_avx2(query);
    int extended = k + 2;
    Candidate best[K_ROUTE + 2];
    for (int i = 0; i < extended; ++i) {
        best[i].idx = -1; best[i].dist = INT_MAX;
    }
    for (int i = 0; i < idx->count; ++i) {
        int dist = __builtin_popcount(q_hash ^ idx->entries[i].hash);
        if (dist < best[extended - 1].dist) {
            int j = extended - 1;
            while (j > 0 && best[j - 1].dist > dist) {
                best[j] = best[j - 1]; j--;
            }
            best[j].idx = (int)idx->entries[i].mc_index;
            best[j].dist = dist;
        }
    }
    for (int i = 0; i < k; ++i) out_ids[i] = best[i].idx;
    if (enable_random && best[k].idx != -1) {
        if ((xorshift32() % 100) < ROUTE_RAND_PROB) {
            int slot = xorshift32() % k;
            int rep = k + (xorshift32() & 1);
            if (rep < extended && best[rep].idx != -1) {
                out_ids[slot] = best[rep].idx;
            }
        }
    }
}

// ============================================================
// Microcircuit forward (sama v1)
// ============================================================
static void microcircuit_forward(const Microcircuit* M,
                                  const float* a, float* y) {
    for (int row = 0; row < D_STATE; ++row) {
        const int8_t* W_row = M->W[row];
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        for (int col = 0; col < D_STATE; col += 32) {
            __m256i w_int8 = _mm256_load_si256((const __m256i*)(W_row + col));
            __m128i w_lo = _mm256_castsi256_si128(w_int8);
            __m128i w_hi = _mm256_extracti128_si256(w_int8, 1);
            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_lo, 8)));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w_hi, 8)));
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
        y[row] = relu(dot * M->scales[row] + M->bias[row]);
    }
}

// ============================================================
// Fusion (sama)
// ============================================================
static void fuse_energy_weighted(const float outputs[K_ROUTE][D_STATE],
                                  float* result) {
    float norms_sq[K_ROUTE];
    float total = 0.0f;
    for (int i = 0; i < K_ROUTE; ++i) {
        norms_sq[i] = l2_norm_sq_avx2(outputs[i]);
        total += norms_sq[i];
    }
    if (total < 1e-12f) {
        for (int j = 0; j < D_STATE; ++j) {
            float s = 0.0f;
            for (int i = 0; i < K_ROUTE; ++i) s += outputs[i][j];
            result[j] = s / K_ROUTE;
        }
        return;
    }
    float inv_total = 1.0f / total;
    __m256 w_vecs[K_ROUTE];
    for (int i = 0; i < K_ROUTE; ++i)
        w_vecs[i] = _mm256_set1_ps(norms_sq[i] * inv_total);
    for (int j = 0; j < D_STATE; j += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < K_ROUTE; ++i)
            sum = _mm256_fmadd_ps(w_vecs[i], _mm256_load_ps(&outputs[i][j]), sum);
        _mm256_store_ps(&result[j], sum);
    }
}

// ============================================================
// CASCADE — supports both routing modes
// ============================================================
static CascadeResult cascade_inference_v2(
    const LSHLinearIndex* idx, Microcircuit** mcs,
    const float* input, float* output,
    RoutingMode routing_mode,
    int normalize_between_steps,
    int verbose_log
) {
    CascadeResult r = {0};

    alignas(32) float a_cur[D_STATE];
    alignas(32) float a_new[D_STATE];
    alignas(32) float outs[K_ROUTE][D_STATE];

    memcpy(a_cur, input, D_STATE * sizeof(float));
    normalize_unit(a_cur);  // Input normalization
    r.initial_norm = l2_norm_avx2(a_cur);

    // Fixed routing: route sekali di awal
    int fixed_mc_ids[K_ROUTE];
    if (routing_mode == ROUTING_FIXED) {
        route_linear(idx, a_cur, fixed_mc_ids, K_ROUTE, 0);
        if (verbose_log) {
            printf("    Fixed routing: mc [");
            for (int i = 0; i < K_ROUTE; ++i)
                printf("%d%s", fixed_mc_ids[i], i < K_ROUTE-1 ? "," : "");
            printf("]\n");
        }
    }

    if (verbose_log) {
        printf("    Step 0: norm=%.4f\n", r.initial_norm);
    }

    for (int step = 0; step < MAX_STEPS; ++step) {
        int mc_ids[K_ROUTE];

        if (routing_mode == ROUTING_FIXED) {
            memcpy(mc_ids, fixed_mc_ids, K_ROUTE * sizeof(int));
        } else {
            route_linear(idx, a_cur, mc_ids, K_ROUTE, 0);
        }

        // Forward 6 mikrosirkit
        for (int i = 0; i < K_ROUTE; ++i)
            microcircuit_forward(mcs[mc_ids[i]], a_cur, outs[i]);

        fuse_energy_weighted(outs, a_new);

        // Normalize between steps (opsi)
        if (normalize_between_steps) normalize_unit(a_new);

        // Convergence check
        float diff_sq = l2_distance_sq_avx2(a_cur, a_new);
        float cur_norm_sq = l2_norm_sq_avx2(a_cur);
        float rel_delta = sqrtf(diff_sq / (cur_norm_sq + 1e-8f));
        float new_norm = l2_norm_avx2(a_new);

        r.num_steps = step + 1;
        r.final_delta = rel_delta;
        r.final_norm = new_norm;
        if (rel_delta > r.max_delta) r.max_delta = rel_delta;

        if (verbose_log) {
            printf("    Step %2d: norm=%.4f, delta=%.4f, mc[0]=%d\n",
                   step + 1, new_norm, rel_delta, mc_ids[0]);
        }

        if (rel_delta < CONV_EPSILON) {
            r.converged = 1;
            memcpy(a_cur, a_new, D_STATE * sizeof(float));
            break;
        }
        memcpy(a_cur, a_new, D_STATE * sizeof(float));
    }

    memcpy(output, a_cur, D_STATE * sizeof(float));
    return r;
}

// ============================================================
// Init (dengan CALIBRATED scale)
//
// Idea: pilih scale supaya output magnitude ~= input magnitude.
// dot product magnitude ~ sqrt(D_STATE) × mean(|w|) × mean(|a|)
//                       = sqrt(256) × 64 × (1/sqrt(D_STATE))
//                       = sqrt(256) × 64 × 1/sqrt(256)
//                       = 64
// So scale = 1/64 ≈ 0.015 supaya scaled ~1
// ============================================================
static void init_hyperplanes(void) {
    for (int h = 0; h < NUM_HYPERPLANES; ++h)
        for (int i = 0; i < D_STATE; ++i)
            g_hyperplanes[h][i] = rand_gaussian();
}

static void init_microcircuit_v2(Microcircuit* M) {
    // Compute mean |W| untuk calibration
    long sum_abs_w = 0;
    for (int r = 0; r < D_STATE; ++r) {
        for (int c = 0; c < D_STATE; ++c) {
            int8_t w = rand_int8();
            M->W[r][c] = w;
            sum_abs_w += (w < 0) ? -w : w;
        }
    }
    float mean_abs_w = (float)sum_abs_w / (D_STATE * D_STATE);

    // Calibrated scale:
    // Input unit L2 → mean |a| ~ 1/sqrt(D_STATE)
    // Dot ~ sqrt(D_STATE) × mean_abs_w × (1/sqrt(D_STATE)) = mean_abs_w
    // So scaled ~ mean_abs_w × scale
    // Want scaled ~ 1/sqrt(D_STATE) (unit-norm output)
    // → scale = 1 / (sqrt(D_STATE) × mean_abs_w)
    float scale_val = 1.0f / (sqrtf((float)D_STATE) * mean_abs_w);

    for (int r = 0; r < D_STATE; ++r) {
        M->scales[r] = scale_val;
        // Structured bias: sedikit positif untuk avoid full ReLU dead
        M->bias[r] = 0.05f / sqrtf((float)D_STATE);
    }
    for (int i = 0; i < D_STATE; ++i) M->resonance_vec[i] = rand_float();
    M->self_hash = simhash_avx2(M->resonance_vec);
}

static void index_build(LSHLinearIndex* idx, Microcircuit** mcs, int n) {
    idx->count = n;
    idx->entries = malloc(n * sizeof(LSHEntry));
    for (int i = 0; i < n; ++i) {
        idx->entries[i].hash = mcs[i]->self_hash;
        idx->entries[i].mc_index = (uint32_t)i;
    }
}

// ============================================================
// Tests
// ============================================================
static void test_trajectory_log(const LSHLinearIndex* idx, Microcircuit** mcs,
                                  RoutingMode mode, const char* label,
                                  int normalize_between) {
    printf("--- Trajectory: %s (normalize_between=%d) ---\n",
           label, normalize_between);
    alignas(32) float input[D_STATE], output[D_STATE];
    for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();

    CascadeResult r = cascade_inference_v2(idx, mcs, input, output,
                                            mode, normalize_between, 1);
    printf("  Result: %d steps, converged=%s, final_delta=%.4f, final_norm=%.4f\n\n",
           r.num_steps, r.converged ? "YES" : "NO",
           r.final_delta, r.final_norm);
}

static void test_convergence_stats(const LSHLinearIndex* idx, Microcircuit** mcs,
                                     RoutingMode mode, const char* label,
                                     int normalize_between) {
    int converged = 0;
    long total_steps = 0;
    float total_delta = 0, total_norm = 0;
    float min_delta = 1e9, max_delta = 0;

    alignas(32) float input[D_STATE], output[D_STATE];
    for (int trial = 0; trial < 100; ++trial) {
        for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();
        CascadeResult r = cascade_inference_v2(idx, mcs, input, output,
                                                mode, normalize_between, 0);
        if (r.converged) converged++;
        total_steps += r.num_steps;
        total_delta += r.final_delta;
        total_norm += r.final_norm;
        if (r.final_delta < min_delta) min_delta = r.final_delta;
        if (r.final_delta > max_delta) max_delta = r.final_delta;
    }

    printf("  %-30s: converged %3d/100, mean_steps %.1f, mean_delta %.4f "
           "(min %.4f, max %.4f), mean_norm %.4f\n",
           label, converged, total_steps / 100.0, total_delta / 100.0,
           min_delta, max_delta, total_norm / 100.0);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 4 v2: Cascade Diagnostic\n");
    printf("================================================\n\n");

    init_hyperplanes();

    printf("Building %d microcircuits (calibrated scale)...\n", NUM_MICROCIRCUITS);
    Microcircuit** mcs = malloc(NUM_MICROCIRCUITS * sizeof(Microcircuit*));
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        posix_memalign((void**)&mcs[i], 64, sizeof(Microcircuit));
        init_microcircuit_v2(mcs[i]);
    }
    printf("  Sample mc[0]: scale=%.6f, bias=%.6f (calibrated)\n",
           mcs[0]->scales[0], mcs[0]->bias[0]);
    printf("  (Phase 4 v1: scale=0.001, bias=random(-0.1, 0.1))\n\n");

    LSHLinearIndex idx;
    index_build(&idx, mcs, NUM_MICROCIRCUITS);

    // Trajectory logs untuk 4 kombinasi
    printf("=== Trajectory examples (1 random input) ===\n");
    g_rng_state = 100;
    test_trajectory_log(&idx, mcs, ROUTING_DYNAMIC,
                        "DYNAMIC routing, no norm between", 0);
    g_rng_state = 100;
    test_trajectory_log(&idx, mcs, ROUTING_DYNAMIC,
                        "DYNAMIC routing + norm between",   1);
    g_rng_state = 100;
    test_trajectory_log(&idx, mcs, ROUTING_FIXED,
                        "FIXED routing, no norm between",   0);
    g_rng_state = 100;
    test_trajectory_log(&idx, mcs, ROUTING_FIXED,
                        "FIXED routing + norm between",     1);

    // Convergence stats untuk 4 kombinasi
    printf("=== Convergence stats (100 trials each) ===\n");
    test_convergence_stats(&idx, mcs, ROUTING_DYNAMIC,
                            "Dynamic + no norm",  0);
    test_convergence_stats(&idx, mcs, ROUTING_DYNAMIC,
                            "Dynamic + norm",     1);
    test_convergence_stats(&idx, mcs, ROUTING_FIXED,
                            "Fixed + no norm",    0);
    test_convergence_stats(&idx, mcs, ROUTING_FIXED,
                            "Fixed + norm",       1);
    printf("\n");

    // Cleanup
    free(idx.entries);
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) free(mcs[i]);
    free(mcs);

    printf("================================================\n");
    printf("  Interpretasi\n");
    printf("================================================\n");
    printf("Bandingkan konfigurasi:\n");
    printf("  - Kalau FIXED converge but DYNAMIC tidak\n");
    printf("    → Routing discontinuity confirmed as root cause\n");
    printf("    → Design: route per-inference, bukan per-step\n\n");
    printf("  - Kalau NORMALIZE membantu significantly\n");
    printf("    → Magnitude drift was contributing factor\n\n");
    printf("  - Kalau semua masih 0%% converge\n");
    printf("    → Architecture butuh training untuk stable dynamics\n");
    printf("    → Phase 6 (training) adalah keharusan, bukan opsional\n");
    return 0;
}
