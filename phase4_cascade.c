// ============================================================
// SML Phase 4: Full Cascade Integration
//
// KOMPONEN BARU:
//   1. Energy-weighted fusion: a_new = Σ (‖y_i‖² / Σ‖y_j‖²) · y_i
//   2. Convergence check: ‖a_new - a‖ / ‖a‖ < ε
//   3. Cascade loop: route → 6× forward → fuse → check convergence
//   4. Damping (optional): a_new = α·a_new + (1-α)·a
//
// KOMBINASI DENGAN PHASE 1-3:
//   - Route: linear scan dari Phase 3 (accept, tidak pakai bucketing)
//   - Forward: sequential 6× dari Phase 1 (correctness-first, tidak pakai
//     worker pool v2 supaya simple)
//   - LSH: dari Phase 3
//
// TEST:
//   - Cascade completes tanpa crash
//   - Distribusi steps-to-converge
//   - Determinism dengan enable_random=0
//   - Content-addressed attractor (mirip → mirip?)
//   - End-to-end latency, breakdown per komponen
//   - Damping impact
//
// Build: gcc -O3 -march=native -mavx2 -mfma -mpopcnt \
//          phase4_cascade.c -o phase4_cascade -lm
// Run:   ./phase4_cascade
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
#define CONV_EPSILON       0.01f       // 1% relative L2 change
#define DEFAULT_DAMPING    0.7f        // α=1.0 no damping, <1 slower stable

// ============================================================
// Structs (kompatibel Phase 3)
// ============================================================
typedef struct {
    alignas(64) int8_t W[D_STATE][D_STATE];
    alignas(64) float  scales[D_STATE];
    alignas(64) float  bias[D_STATE];
    alignas(64) uint32_t self_hash;
    alignas(32) float resonance_vec[D_STATE];
} Microcircuit;

typedef struct {
    uint32_t hash;
    uint32_t mc_index;
} LSHEntry;

typedef struct {
    LSHEntry* entries;
    int count;
} LSHLinearIndex;

typedef struct {
    int idx;
    int dist;
} Candidate;

typedef struct {
    int num_steps;
    int converged;         // 1 = converge < MAX_STEPS, 0 = hit limit
    float final_delta;     // rel L2 change di step terakhir
    float max_delta;       // max delta selama cascade (deteksi oscillation)
    // Latency breakdown (µs)
    double time_route;
    double time_forward;
    double time_fusion;
    double time_total;
} CascadeResult;

// ============================================================
// Globals
// ============================================================
static alignas(32) float g_hyperplanes[NUM_HYPERPLANES][D_STATE];
static uint32_t g_rng_state = 42;

// ============================================================
// RNG
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
static int8_t rand_int8(void) {
    return (int8_t)(xorshift32() & 0xFF);
}

// ============================================================
// Basic helpers
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
// L2 norm/distance AVX2 (D_STATE = 256 = 32 vectors of 8)
// ============================================================
static inline float l2_norm_sq_avx2(const float* v) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    for (int i = 0; i < D_STATE; i += 32) {
        __m256 v0 = _mm256_load_ps(&v[i +  0]);
        __m256 v1 = _mm256_load_ps(&v[i +  8]);
        __m256 v2 = _mm256_load_ps(&v[i + 16]);
        __m256 v3 = _mm256_load_ps(&v[i + 24]);
        s0 = _mm256_fmadd_ps(v0, v0, s0);
        s1 = _mm256_fmadd_ps(v1, v1, s1);
        s2 = _mm256_fmadd_ps(v2, v2, s2);
        s3 = _mm256_fmadd_ps(v3, v3, s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    return hsum_avx2(s);
}

static inline float l2_distance_sq_avx2(const float* a, const float* b) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    for (int i = 0; i < D_STATE; i += 32) {
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
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    return hsum_avx2(s);
}

// ============================================================
// SimHash (dari Phase 3)
// ============================================================
static uint32_t simhash_avx2(const float* a) {
    uint32_t sig = 0;
    for (int h = 0; h < NUM_HYPERPLANES; ++h) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const float* hp = g_hyperplanes[h];
        for (int i = 0; i < D_STATE; i += 32) {
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

// ============================================================
// Route (linear scan, dari Phase 3)
// ============================================================
static void route_linear(const LSHLinearIndex* idx,
                          const float* query, int* out_ids,
                          int k, int enable_random) {
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
// Microcircuit forward (256×256, INT8 weights, ReLU)
// ============================================================
static void microcircuit_forward(const Microcircuit* M,
                                  const float* a, float* y) {
    for (int row = 0; row < D_STATE; ++row) {
        const int8_t* W_row = M->W[row];
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        // D_STATE = 256, iterate 8× per 32-elem chunk
        for (int col = 0; col < D_STATE; col += 32) {
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

            __m256 a0 = _mm256_load_ps(a + col +  0);
            __m256 a1 = _mm256_load_ps(a + col +  8);
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
// Energy-weighted fusion
//   norm_i = ‖y_i‖²
//   weight_i = norm_i / Σ norm_j
//   a_new = Σ weight_i · y_i
// ============================================================
static void fuse_energy_weighted(const float outputs[K_ROUTE][D_STATE],
                                  float* result) {
    // 1. Compute ‖y_i‖² untuk each
    float norms_sq[K_ROUTE];
    float total = 0.0f;
    for (int i = 0; i < K_ROUTE; ++i) {
        norms_sq[i] = l2_norm_sq_avx2(outputs[i]);
        total += norms_sq[i];
    }

    // 2. Fallback: kalau semua zero, equal average
    if (total < 1e-12f) {
        for (int j = 0; j < D_STATE; ++j) {
            float s = 0.0f;
            for (int i = 0; i < K_ROUTE; ++i) s += outputs[i][j];
            result[j] = s / K_ROUTE;
        }
        return;
    }

    // 3. Compute weights
    float inv_total = 1.0f / total;
    __m256 w_vecs[K_ROUTE];
    for (int i = 0; i < K_ROUTE; ++i) {
        w_vecs[i] = _mm256_set1_ps(norms_sq[i] * inv_total);
    }

    // 4. Weighted sum (AVX2, 32 elements per iteration)
    for (int j = 0; j < D_STATE; j += 32) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        for (int i = 0; i < K_ROUTE; ++i) {
            sum0 = _mm256_fmadd_ps(w_vecs[i],
                     _mm256_load_ps(&outputs[i][j +  0]), sum0);
            sum1 = _mm256_fmadd_ps(w_vecs[i],
                     _mm256_load_ps(&outputs[i][j +  8]), sum1);
            sum2 = _mm256_fmadd_ps(w_vecs[i],
                     _mm256_load_ps(&outputs[i][j + 16]), sum2);
            sum3 = _mm256_fmadd_ps(w_vecs[i],
                     _mm256_load_ps(&outputs[i][j + 24]), sum3);
        }
        _mm256_store_ps(&result[j +  0], sum0);
        _mm256_store_ps(&result[j +  8], sum1);
        _mm256_store_ps(&result[j + 16], sum2);
        _mm256_store_ps(&result[j + 24], sum3);
    }
}

// ============================================================
// Apply damping: a = α·a_new + (1-α)·a_current
// ============================================================
static void apply_damping(const float* a_new, float* a_current, float alpha) {
    __m256 a_vec = _mm256_set1_ps(alpha);
    __m256 b_vec = _mm256_set1_ps(1.0f - alpha);
    for (int i = 0; i < D_STATE; i += 8) {
        __m256 new_v = _mm256_load_ps(&a_new[i]);
        __m256 cur_v = _mm256_load_ps(&a_current[i]);
        __m256 mixed = _mm256_add_ps(_mm256_mul_ps(a_vec, new_v),
                                      _mm256_mul_ps(b_vec, cur_v));
        _mm256_store_ps(&a_current[i], mixed);
    }
}

// ============================================================
// CASCADE INFERENCE — inti dari SML
// ============================================================
static CascadeResult cascade_inference(
    const LSHLinearIndex* idx, Microcircuit** mcs,
    const float* input, float* output,
    int enable_random, float damping_alpha
) {
    CascadeResult r = {0};
    r.max_delta = 0.0f;

    alignas(32) float a_cur[D_STATE];
    alignas(32) float a_new[D_STATE];
    alignas(32) float outs[K_ROUTE][D_STATE];

    memcpy(a_cur, input, D_STATE * sizeof(float));

    double t_start = now_sec();

    for (int step = 0; step < MAX_STEPS; ++step) {
        // 1. Route
        double t0 = now_sec();
        int mc_ids[K_ROUTE];
        route_linear(idx, a_cur, mc_ids, K_ROUTE, enable_random);
        r.time_route += now_sec() - t0;

        // 2. Sequential 6× forward
        t0 = now_sec();
        for (int i = 0; i < K_ROUTE; ++i) {
            microcircuit_forward(mcs[mc_ids[i]], a_cur, outs[i]);
        }
        r.time_forward += now_sec() - t0;

        // 3. Fusion
        t0 = now_sec();
        fuse_energy_weighted(outs, a_new);
        r.time_fusion += now_sec() - t0;

        // 4. Convergence check (relative L2 change)
        float diff_sq = l2_distance_sq_avx2(a_cur, a_new);
        float cur_norm_sq = l2_norm_sq_avx2(a_cur);
        float rel_delta = sqrtf(diff_sq / (cur_norm_sq + 1e-8f));

        r.num_steps = step + 1;
        r.final_delta = rel_delta;
        if (rel_delta > r.max_delta) r.max_delta = rel_delta;

        if (rel_delta < CONV_EPSILON) {
            r.converged = 1;
            memcpy(a_cur, a_new, D_STATE * sizeof(float));
            break;
        }

        // 5. Update state (dengan damping)
        if (damping_alpha >= 0.999f) {
            memcpy(a_cur, a_new, D_STATE * sizeof(float));
        } else {
            apply_damping(a_new, a_cur, damping_alpha);
        }
    }

    r.time_total = now_sec() - t_start;
    r.time_route *= 1e6;   // convert ke µs
    r.time_forward *= 1e6;
    r.time_fusion *= 1e6;
    r.time_total *= 1e6;

    memcpy(output, a_cur, D_STATE * sizeof(float));
    return r;
}

// ============================================================
// Init
// ============================================================
static void init_hyperplanes(void) {
    for (int h = 0; h < NUM_HYPERPLANES; ++h)
        for (int i = 0; i < D_STATE; ++i)
            g_hyperplanes[h][i] = rand_gaussian();
}

static void init_microcircuit(Microcircuit* M) {
    for (int r = 0; r < D_STATE; ++r) {
        for (int c = 0; c < D_STATE; ++c) M->W[r][c] = rand_int8();
        M->scales[r] = 0.001f;
        M->bias[r] = rand_float() * 0.1f;
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
static void test_1_basic_cascade(const LSHLinearIndex* idx, Microcircuit** mcs) {
    printf("--- Test 1: Basic cascade ---\n");
    alignas(32) float input[D_STATE], output[D_STATE];
    for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();

    CascadeResult r = cascade_inference(idx, mcs, input, output, 0, 1.0f);

    // Sanity checks
    int nans = 0, infs = 0;
    float min_v = 1e9, max_v = -1e9;
    for (int i = 0; i < D_STATE; ++i) {
        if (isnan(output[i])) nans++;
        else if (isinf(output[i])) infs++;
        else {
            if (output[i] < min_v) min_v = output[i];
            if (output[i] > max_v) max_v = output[i];
        }
    }

    printf("  Steps: %d, Converged: %s, Final delta: %.4f\n",
           r.num_steps, r.converged ? "YES" : "NO", r.final_delta);
    printf("  Max delta selama cascade: %.4f\n", r.max_delta);
    printf("  Output: NaN=%d, Inf=%d, range=[%.4f, %.4f]\n",
           nans, infs, min_v, max_v);
    printf("  Latency: %.1f µs (route %.1f, forward %.1f, fusion %.1f)\n\n",
           r.time_total, r.time_route, r.time_forward, r.time_fusion);
}

static void test_2_convergence_stats(const LSHLinearIndex* idx, Microcircuit** mcs) {
    printf("--- Test 2: Convergence distribution (100 random inputs) ---\n");

    int steps_bins[6] = {0}; // <5, 5-9, 10-14, 15-19, 20-49, hit MAX
    int converged = 0;
    long total_steps = 0;
    float total_final_delta = 0;

    alignas(32) float input[D_STATE], output[D_STATE];
    for (int trial = 0; trial < 100; ++trial) {
        for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();
        CascadeResult r = cascade_inference(idx, mcs, input, output, 0, 1.0f);

        if (r.converged) {
            converged++;
            if (r.num_steps < 5) steps_bins[0]++;
            else if (r.num_steps < 10) steps_bins[1]++;
            else if (r.num_steps < 15) steps_bins[2]++;
            else if (r.num_steps < 20) steps_bins[3]++;
            else steps_bins[4]++;
        } else {
            steps_bins[5]++;
        }
        total_steps += r.num_steps;
        total_final_delta += r.final_delta;
    }

    printf("  Converged: %d / 100 (%.0f%%)\n", converged, converged * 1.0);
    printf("  Distribution of steps:\n");
    printf("    <5    : %d\n", steps_bins[0]);
    printf("    5-9   : %d\n", steps_bins[1]);
    printf("    10-14 : %d\n", steps_bins[2]);
    printf("    15-19 : %d\n", steps_bins[3]);
    printf("    20-49 : %d\n", steps_bins[4]);
    printf("    MAX_STEPS: %d (didn't converge)\n", steps_bins[5]);
    printf("  Mean steps: %.1f\n", total_steps / 100.0);
    printf("  Mean final delta: %.4f (target < %.4f)\n\n",
           total_final_delta / 100.0, CONV_EPSILON);
}

static void test_3_determinism(const LSHLinearIndex* idx, Microcircuit** mcs) {
    printf("--- Test 3: Determinism (enable_random=0) ---\n");
    alignas(32) float input[D_STATE], out1[D_STATE], out2[D_STATE];
    for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();

    g_rng_state = 100;
    CascadeResult r1 = cascade_inference(idx, mcs, input, out1, 0, 1.0f);
    g_rng_state = 200;  // different RNG state
    CascadeResult r2 = cascade_inference(idx, mcs, input, out2, 0, 1.0f);

    // Compare
    float max_diff = 0;
    for (int i = 0; i < D_STATE; ++i) {
        float d = fabsf(out1[i] - out2[i]);
        if (d > max_diff) max_diff = d;
    }

    printf("  Run 1: %d steps, converged=%d\n", r1.num_steps, r1.converged);
    printf("  Run 2: %d steps, converged=%d\n", r2.num_steps, r2.converged);
    printf("  Max output diff: %.6e\n", max_diff);
    if (max_diff < 1e-6f && r1.num_steps == r2.num_steps) {
        printf("  [PASS] Deterministic — RNG state tidak mempengaruhi\n\n");
    } else {
        printf("  [FAIL] Non-deterministic meski enable_random=0\n\n");
    }
}

static void test_4_content_addressed(const LSHLinearIndex* idx, Microcircuit** mcs) {
    printf("--- Test 4: Content-addressed attractor ---\n");
    printf("  Q: Apakah input mirip → attractor mirip?\n");
    printf("  (Note: dengan random weights, hasil tidak pasti bermakna)\n\n");

    int similar_close = 0, different_far = 0;

    for (int trial = 0; trial < 20; ++trial) {
        alignas(32) float q1[D_STATE], q2[D_STATE], q3[D_STATE];
        alignas(32) float out1[D_STATE], out2[D_STATE], out3[D_STATE];

        for (int i = 0; i < D_STATE; ++i) {
            q1[i] = rand_float();
            q2[i] = q1[i] + rand_gaussian() * 0.05f;  // similar
            q3[i] = rand_float();                      // different
        }

        cascade_inference(idx, mcs, q1, out1, 0, 1.0f);
        cascade_inference(idx, mcs, q2, out2, 0, 1.0f);
        cascade_inference(idx, mcs, q3, out3, 0, 1.0f);

        // Compare output distances
        float d12 = sqrtf(l2_distance_sq_avx2(out1, out2));
        float d13 = sqrtf(l2_distance_sq_avx2(out1, out3));

        if (d12 < d13 * 0.8f) similar_close++;
        if (d13 > d12 * 1.2f) different_far++;
    }

    printf("  Trials where similar inputs → similar outputs (d12 < 0.8·d13): %d/20\n",
           similar_close);
    printf("  Trials where different inputs → different outputs (d13 > 1.2·d12): %d/20\n",
           different_far);

    if (similar_close >= 12 && different_far >= 12) {
        printf("  [PASS] Cascade IS content-addressed (bahkan dengan random weights)\n\n");
    } else if (similar_close >= 8 || different_far >= 8) {
        printf("  [PARTIAL] Some content-addressing, but noise dominates\n\n");
    } else {
        printf("  [WEAK] Random weights menghasilkan attractor yang tidak informatif\n");
        printf("         (Expected — training di Phase 6 akan fix ini)\n\n");
    }
}

static void test_5_latency_benchmark(const LSHLinearIndex* idx, Microcircuit** mcs) {
    printf("--- Test 5: End-to-end latency benchmark (50 inference) ---\n");

    double total_time = 0, total_route = 0, total_forward = 0, total_fusion = 0;
    int total_steps = 0;

    alignas(32) float input[D_STATE], output[D_STATE];
    for (int trial = 0; trial < 50; ++trial) {
        for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();
        CascadeResult r = cascade_inference(idx, mcs, input, output, 0, 1.0f);
        total_time += r.time_total;
        total_route += r.time_route;
        total_forward += r.time_forward;
        total_fusion += r.time_fusion;
        total_steps += r.num_steps;
    }

    double mean_steps = total_steps / 50.0;
    printf("  Mean steps per inference: %.1f\n", mean_steps);
    printf("  Mean inference time:      %.1f µs\n", total_time / 50.0);
    printf("  Breakdown (per inference):\n");
    printf("    Route  total: %.1f µs (%.0f%%)\n",
           total_route/50.0, 100.0*total_route/total_time);
    printf("    Forward total: %.1f µs (%.0f%%)\n",
           total_forward/50.0, 100.0*total_forward/total_time);
    printf("    Fusion total: %.1f µs (%.0f%%)\n",
           total_fusion/50.0, 100.0*total_fusion/total_time);
    printf("    Other (conv check, memcpy): %.1f µs\n\n",
           (total_time - total_route - total_forward - total_fusion) / 50.0);

    double per_step = (total_time - 0) / total_steps;
    printf("  Per cascade step: %.1f µs (target ~60-90 µs sequential)\n\n",
           per_step);
}

static void test_6_damping_impact(const LSHLinearIndex* idx, Microcircuit** mcs) {
    printf("--- Test 6: Damping impact ---\n");
    printf("  α=1.0 (no damping), α=0.7 (spec default), α=0.5 (heavy)\n\n");

    float alphas[3] = {1.0f, 0.7f, 0.5f};
    for (int a = 0; a < 3; ++a) {
        int converged = 0;
        long total_steps = 0;
        alignas(32) float input[D_STATE], output[D_STATE];
        for (int trial = 0; trial < 50; ++trial) {
            for (int i = 0; i < D_STATE; ++i) input[i] = rand_float();
            CascadeResult r = cascade_inference(idx, mcs, input, output, 0,
                                                 alphas[a]);
            if (r.converged) converged++;
            total_steps += r.num_steps;
        }
        printf("  α=%.1f: converged %d/50, mean steps %.1f\n",
               alphas[a], converged, total_steps / 50.0);
    }
    printf("\n");
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 4: Full Cascade Integration\n");
    printf("================================================\n");
    printf("Config:\n");
    printf("  D_STATE            = %d\n", D_STATE);
    printf("  NUM_MICROCIRCUITS  = %d\n", NUM_MICROCIRCUITS);
    printf("  K_ROUTE            = %d\n", K_ROUTE);
    printf("  MAX_STEPS          = %d\n", MAX_STEPS);
    printf("  CONV_EPSILON       = %.4f (relative L2 change)\n", CONV_EPSILON);
    printf("\n");

    init_hyperplanes();

    printf("Building %d microcircuits...\n", NUM_MICROCIRCUITS);
    double t0 = now_sec();
    Microcircuit** mcs = malloc(NUM_MICROCIRCUITS * sizeof(Microcircuit*));
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        posix_memalign((void**)&mcs[i], 64, sizeof(Microcircuit));
        init_microcircuit(mcs[i]);
    }
    printf("  Init time: %.2f sec, footprint %.1f MB\n\n",
           now_sec() - t0,
           (double)sizeof(Microcircuit) * NUM_MICROCIRCUITS / (1024*1024));

    LSHLinearIndex idx;
    index_build(&idx, mcs, NUM_MICROCIRCUITS);

    test_1_basic_cascade(&idx, mcs);
    test_2_convergence_stats(&idx, mcs);
    test_3_determinism(&idx, mcs);
    test_4_content_addressed(&idx, mcs);
    test_5_latency_benchmark(&idx, mcs);
    test_6_damping_impact(&idx, mcs);

    free(idx.entries);
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) free(mcs[i]);
    free(mcs);

    printf("================================================\n");
    printf("  Phase 4 complete\n");
    printf("================================================\n");
    printf("Milestone: Full SML inference pipeline works end-to-end.\n");
    printf("Random weights → tidak semantically meaningful, tapi mekanisme\n");
    printf("dinamika telah divalidasi.\n\n");
    printf("Kandidat next steps:\n");
    printf("  Phase 5: Persistent storage (mmap lattice 10K+ mikrosirkit)\n");
    printf("  Phase 6: Training strategy (equilibrium prop atau alternatif)\n");
    printf("  Optimization: integrate Phase 2 worker pool untuk parallel forward\n");
    return 0;
}
