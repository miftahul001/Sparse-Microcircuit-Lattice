// ============================================================
// SML MVP: Sparse Microcircuit Lattice — Single-File Library
//
// Production-ready consolidated implementation dari SML architecture,
// mengikuti spec v0.2 (post-Phase 4 empirical validation).
//
// Features:
//   - Fixed routing per inference (validated Phase 4 v2: 100% konvergensi)
//   - 256×256 microcircuits, INT8 weights
//   - AVX2 optimized: forward, SimHash, L2 norm, fusion
//   - Energy-weighted fusion + optional state normalization
//   - Cascade inference dengan convergence detection
//   - Sequential 6× forward (parallel dispatch di future work)
//
// Compile sebagai library:
//   gcc -O3 -march=native -mavx2 -mfma -mpopcnt -c sml_mvp.c
//
// Compile dengan test main:
//   gcc -O3 -march=native -mavx2 -mfma -mpopcnt -DSML_STANDALONE \
//       sml_mvp.c -o sml_mvp -lm
//   ./sml_mvp
//
// Requirements:
//   - x86_64 CPU dengan AVX2 + FMA + POPCNT (Intel Haswell+ / AMD Excavator+)
//   - Linux dengan glibc (posix_memalign, clock_gettime)
//   - No external dependencies selain libc + libm
//
// Version: 1.0 (MVP)
// Spec:    SML-architecture-spec-v0.2.md
// ============================================================

#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

// ============================================================
// COMPILE-TIME CONFIGURATION
// Ubah nilai ini dan recompile untuk menyesuaikan
// ============================================================

// State dimension. Harus kelipatan 32 (AVX2 constraint).
#define SML_D_STATE           256

// Route arity — berapa microcircuit yang dipilih per inference
#define SML_K_ROUTE           6

// SimHash width — jumlah hyperplane, hasilkan bit signature width
#define SML_NUM_HYPERPLANES   32

// Cascade iteration limit (safety cap)
#define SML_MAX_STEPS         50

// Convergence threshold (relative L2 change antar step)
#define SML_CONV_EPSILON      0.01f

// Route randomness probability (0-100%). Set 0 untuk fully deterministic.
#define SML_ROUTE_RAND_PROB   0

// Normalize state antar cascade step (1 = recommended, 0 = no)
#define SML_NORMALIZE_STATE   1

// ============================================================
// PUBLIC API — Struct definitions
// ============================================================

/**
 * Satu microcircuit: unit komputasi atomik SML.
 * Ukuran ~67 KB, aligned untuk L2 residence.
 */
typedef struct {
    alignas(64) int8_t   W[SML_D_STATE][SML_D_STATE];   // 64 KB weights
    alignas(64) float    scales[SML_D_STATE];           // 1 KB dequant scales
    alignas(64) float    bias[SML_D_STATE];             // 1 KB activation offset
    alignas(64) uint32_t self_hash;                     // 4 B — SimHash(resonance_vec)
    alignas(32) float    resonance_vec[SML_D_STATE];    // 1 KB — routing signature
} SMLMicrocircuit;

/**
 * LSH routing table entry (sorted by hash saat lattice created).
 */
typedef struct {
    uint32_t hash;
    uint32_t mc_index;
} SMLLSHEntry;

/**
 * The lattice: complete SML instance.
 * User bisa inspect fields untuk debugging, tapi lifecycle di-manage
 * lewat sml_create/sml_destroy.
 */
typedef struct {
    SMLMicrocircuit** microcircuits;   // Array of aligned pointers
    int               num_microcircuits;
    SMLLSHEntry*      lsh_entries;      // For routing lookup
    alignas(32) float hyperplanes[SML_NUM_HYPERPLANES][SML_D_STATE];
    uint32_t          rng_state;        // Not thread-safe
} SMLLattice;

/**
 * Result dari satu inference call.
 * Berisi timing + convergence stats untuk debugging/monitoring.
 */
typedef struct {
    int    num_steps;                    // Cascade steps taken
    int    converged;                    // 1 = converged, 0 = hit MAX_STEPS
    float  final_delta;                  // Final relative L2 change
    float  max_delta;                    // Max delta observed during cascade
    int    route_ids[SML_K_ROUTE];       // Microcircuit IDs yang dipilih
    double time_us;                      // Total inference wall time (µs)
} SMLResult;

// ============================================================
// PUBLIC API — Function declarations
// ============================================================

/**
 * Create lattice baru dengan random init.
 *
 * num_microcircuits: berapa mc (recommended 100-10000; scale tested)
 * seed: RNG seed untuk reproducible init (0 → default 0xDEADBEEF)
 *
 * Return: allocated SMLLattice, atau NULL jika allocation gagal.
 * Caller wajib call sml_destroy() untuk cleanup.
 */
extern SMLLattice* sml_create(int num_microcircuits, uint32_t seed);

/**
 * Destroy lattice, free semua memory.
 * Safe dipanggil dengan NULL.
 */
extern void sml_destroy(SMLLattice* lat);

/**
 * Run inference: input → route → cascade → output.
 *
 * lat:    lattice yang di-create dari sml_create
 * input:  D_STATE floats (unaligned OK, akan di-copy internal)
 * output: D_STATE floats (unaligned OK, hasil fixed point)
 *
 * Return: SMLResult dengan stats. Kalau result.converged==0,
 *         cascade hit MAX_STEPS (masih return valid output tapi
 *         mungkin bukan fixed point sesungguhnya).
 */
extern SMLResult sml_inference(SMLLattice* lat,
                                const float* input,
                                float* output);

// ============================================================
// INTERNAL — RNG (xorshift32)
// ============================================================

static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static float rng_float(uint32_t* state) {
    return ((float)xorshift32(state) / (float)UINT32_MAX) * 2.0f - 1.0f;
}

static float rng_gaussian(uint32_t* state) {
    float u1 = ((float)xorshift32(state) / (float)UINT32_MAX);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float u2 = ((float)xorshift32(state) / (float)UINT32_MAX);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static int8_t rng_int8(uint32_t* state) {
    return (int8_t)(xorshift32(state) & 0xFF);
}

// ============================================================
// INTERNAL — AVX2 helpers
// ============================================================

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Horizontal sum: __m256 → float
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

// L2 norm squared: Σ v_i²
static inline float l2_norm_sq(const float* v) {
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
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1),
                              _mm256_add_ps(s2, s3));
    return hsum_avx2(s);
}

// L2 distance squared: Σ (a_i - b_i)²
static inline float l2_distance_sq(const float* a, const float* b) {
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
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1),
                              _mm256_add_ps(s2, s3));
    return hsum_avx2(s);
}

// Normalize vector ke unit L2 norm (in-place)
static inline void normalize_unit(float* v) {
    float norm = sqrtf(l2_norm_sq(v));
    if (norm < 1e-8f) return;
    __m256 inv = _mm256_set1_ps(1.0f / norm);
    for (int i = 0; i < SML_D_STATE; i += 8) {
        _mm256_store_ps(&v[i], _mm256_mul_ps(_mm256_load_ps(&v[i]), inv));
    }
}

static inline float sml_relu(float x) { return x > 0.0f ? x : 0.0f; }

// ============================================================
// INTERNAL — SimHash 32-bit
//
// Untuk setiap hyperplane h_i:
//   sig_i = (h_i · v > 0) ? 1 : 0
// signature = concat semua bit
// ============================================================

static uint32_t simhash_avx2(const float hyperplanes[SML_NUM_HYPERPLANES][SML_D_STATE],
                              const float* v) {
    uint32_t sig = 0;
    for (int h = 0; h < SML_NUM_HYPERPLANES; ++h) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const float* hp = hyperplanes[h];
        for (int i = 0; i < SML_D_STATE; i += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i +  0]),
                                    _mm256_load_ps(&v[i +  0]), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i +  8]),
                                    _mm256_load_ps(&v[i +  8]), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i + 16]),
                                    _mm256_load_ps(&v[i + 16]), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_load_ps(&hp[i + 24]),
                                    _mm256_load_ps(&v[i + 24]), acc3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        sig |= (hsum_avx2(acc) > 0.0f ? 1u : 0u) << h;
    }
    return sig;
}

// ============================================================
// INTERNAL — Route: linear scan, top-K by Hamming distance
//
// Fixed routing (per spec v0.2 Section 6): dipanggil SEKALI di awal
// inference, hasilnya di-freeze untuk seluruh cascade.
// ============================================================

typedef struct { int idx; int dist; } Candidate;

static void route_linear(const SMLLattice* lat,
                          const float* query,
                          int* out_ids,
                          uint32_t* rng_state) {
    uint32_t q_hash = simhash_avx2(lat->hyperplanes, query);

    // Track top (K + 2) untuk enable randomness swap
    const int extended = SML_K_ROUTE + 2;
    Candidate best[SML_K_ROUTE + 2];
    for (int i = 0; i < extended; ++i) {
        best[i].idx = -1;
        best[i].dist = INT_MAX;
    }

    // Linear scan (cukup untuk N < 10K; spec Section 6.2)
    for (int i = 0; i < lat->num_microcircuits; ++i) {
        int dist = __builtin_popcount(q_hash ^ lat->lsh_entries[i].hash);
        if (dist < best[extended - 1].dist) {
            // Insertion sort
            int j = extended - 1;
            while (j > 0 && best[j - 1].dist > dist) {
                best[j] = best[j - 1];
                j--;
            }
            best[j].idx = (int)lat->lsh_entries[i].mc_index;
            best[j].dist = dist;
        }
    }

    // Copy top-K ke output
    for (int i = 0; i < SML_K_ROUTE; ++i) out_ids[i] = best[i].idx;

    // Optional: slight randomness (Phase 3a design decision)
    if (SML_ROUTE_RAND_PROB > 0 && best[SML_K_ROUTE].idx != -1) {
        if ((xorshift32(rng_state) % 100) < SML_ROUTE_RAND_PROB) {
            int slot = xorshift32(rng_state) % SML_K_ROUTE;
            int rep = SML_K_ROUTE + (xorshift32(rng_state) & 1);
            if (rep < extended && best[rep].idx != -1) {
                out_ids[slot] = best[rep].idx;
            }
        }
    }
}

// ============================================================
// INTERNAL — Microcircuit forward pass
//
// y = ReLU(dequant(W) · a + bias)
//
// AVX2 strategy: 4 parallel accumulator per row untuk ILP.
// INT8 → FP32 inline conversion (unavoidable overhead di Skylake).
// Bottleneck: Port 5 (shuffle/convert), bukan compute (Phase 1 diagnosis).
// ============================================================

static void microcircuit_forward(const SMLMicrocircuit* M,
                                  const float* a, float* y) {
    for (int row = 0; row < SML_D_STATE; ++row) {
        const int8_t* W_row = M->W[row];
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int col = 0; col < SML_D_STATE; col += 32) {
            // Load 32 INT8 weights (1 cache line)
            __m256i w_int8 = _mm256_load_si256(
                (const __m256i*)(W_row + col));
            __m128i w_lo = _mm256_castsi256_si128(w_int8);
            __m128i w_hi = _mm256_extracti128_si256(w_int8, 1);

            // INT8 → INT32 → FP32 (4 vectors of 8 elements)
            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_lo));
            __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_srli_si128(w_lo, 8)));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w_hi));
            __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            _mm_srli_si128(w_hi, 8)));

            // Load 32 FP32 activations
            __m256 a0 = _mm256_load_ps(a + col +  0);
            __m256 a1 = _mm256_load_ps(a + col +  8);
            __m256 a2 = _mm256_load_ps(a + col + 16);
            __m256 a3 = _mm256_load_ps(a + col + 24);

            // FMA
            acc0 = _mm256_fmadd_ps(w0, a0, acc0);
            acc1 = _mm256_fmadd_ps(w1, a1, acc1);
            acc2 = _mm256_fmadd_ps(w2, a2, acc2);
            acc3 = _mm256_fmadd_ps(w3, a3, acc3);
        }

        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        float dot = hsum_avx2(acc);
        y[row] = sml_relu(dot * M->scales[row] + M->bias[row]);
    }
}

// ============================================================
// INTERNAL — Energy-weighted fusion
//
// w_i = ‖y_i‖² / Σ ‖y_j‖²
// result = Σ w_i · y_i
//
// Semantik: microcircuit yang lebih "confident" (magnitude tinggi)
// berkontribusi lebih. Tanpa learned parameter.
// ============================================================

static void fuse_energy_weighted(const float outputs[SML_K_ROUTE][SML_D_STATE],
                                  float* result) {
    // 1. Compute ‖y_i‖² per output
    float norms_sq[SML_K_ROUTE];
    float total = 0.0f;
    for (int i = 0; i < SML_K_ROUTE; ++i) {
        norms_sq[i] = l2_norm_sq(outputs[i]);
        total += norms_sq[i];
    }

    // 2. Fallback: kalau semua zero, uniform average
    if (total < 1e-12f) {
        for (int j = 0; j < SML_D_STATE; ++j) {
            float s = 0.0f;
            for (int i = 0; i < SML_K_ROUTE; ++i) s += outputs[i][j];
            result[j] = s / SML_K_ROUTE;
        }
        return;
    }

    // 3. Compute weights, broadcast ke AVX2 register
    float inv_total = 1.0f / total;
    __m256 w_vecs[SML_K_ROUTE];
    for (int i = 0; i < SML_K_ROUTE; ++i) {
        w_vecs[i] = _mm256_set1_ps(norms_sq[i] * inv_total);
    }

    // 4. Weighted sum, AVX2 32-elem per iteration
    for (int j = 0; j < SML_D_STATE; j += 32) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        for (int i = 0; i < SML_K_ROUTE; ++i) {
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
// INTERNAL — Microcircuit initialization
//
// Kalibrasi (spec v0.2 Section 5.1):
//   - W: random INT8 [-128, 127]
//   - scales[r] = 1 / (sqrt(D) × mean|W|) → output magnitude ~ input magnitude
//   - bias[r] = 0.05/sqrt(D) → small positive, hindari ReLU dead zone
//   - resonance_vec: random uniform [-1, 1]
//   - self_hash: SimHash(resonance_vec)
// ============================================================

static void init_microcircuit(SMLMicrocircuit* M,
                                const float hyperplanes[SML_NUM_HYPERPLANES][SML_D_STATE],
                                uint32_t* rng_state) {
    // 1. Init weights INT8, track mean absolute value
    long sum_abs_w = 0;
    for (int r = 0; r < SML_D_STATE; ++r) {
        for (int c = 0; c < SML_D_STATE; ++c) {
            int8_t w = rng_int8(rng_state);
            M->W[r][c] = w;
            sum_abs_w += (w < 0) ? -w : w;
        }
    }
    float mean_abs_w = (float)sum_abs_w / (SML_D_STATE * SML_D_STATE);

    // 2. Calibrated scale + structured bias
    float scale_val = 1.0f / (sqrtf((float)SML_D_STATE) * mean_abs_w);
    float bias_val = 0.05f / sqrtf((float)SML_D_STATE);
    for (int r = 0; r < SML_D_STATE; ++r) {
        M->scales[r] = scale_val;
        M->bias[r] = bias_val;
    }

    // 3. Resonance vector + self_hash
    for (int i = 0; i < SML_D_STATE; ++i) {
        M->resonance_vec[i] = rng_float(rng_state);
    }
    M->self_hash = simhash_avx2(hyperplanes, M->resonance_vec);
}

// ============================================================
// PUBLIC API — Implementations
// ============================================================

SMLLattice* sml_create(int num_microcircuits, uint32_t seed) {
    if (num_microcircuits <= 0) return NULL;
    if (seed == 0) seed = 0xDEADBEEF;  // xorshift32 tidak bisa jalan dari 0

    SMLLattice* lat = calloc(1, sizeof(SMLLattice));
    if (!lat) return NULL;

    lat->num_microcircuits = num_microcircuits;
    lat->rng_state = seed;

    // 1. Init hyperplanes untuk SimHash (Gaussian sampled)
    for (int h = 0; h < SML_NUM_HYPERPLANES; ++h) {
        for (int i = 0; i < SML_D_STATE; ++i) {
            lat->hyperplanes[h][i] = rng_gaussian(&lat->rng_state);
        }
    }

    // 2. Allocate microcircuit array
    lat->microcircuits = calloc(num_microcircuits, sizeof(SMLMicrocircuit*));
    if (!lat->microcircuits) {
        free(lat);
        return NULL;
    }

    // 3. Init each microcircuit (each aligned 64-byte)
    for (int i = 0; i < num_microcircuits; ++i) {
        SMLMicrocircuit* mc;
        if (posix_memalign((void**)&mc, 64, sizeof(SMLMicrocircuit)) != 0) {
            // Cleanup partial
            for (int j = 0; j < i; ++j) free(lat->microcircuits[j]);
            free(lat->microcircuits);
            free(lat);
            return NULL;
        }
        init_microcircuit(mc, lat->hyperplanes, &lat->rng_state);
        lat->microcircuits[i] = mc;
    }

    // 4. Build LSH index (unsorted; linear scan tidak butuh sorted)
    lat->lsh_entries = calloc(num_microcircuits, sizeof(SMLLSHEntry));
    if (!lat->lsh_entries) {
        for (int i = 0; i < num_microcircuits; ++i) free(lat->microcircuits[i]);
        free(lat->microcircuits);
        free(lat);
        return NULL;
    }
    for (int i = 0; i < num_microcircuits; ++i) {
        lat->lsh_entries[i].hash = lat->microcircuits[i]->self_hash;
        lat->lsh_entries[i].mc_index = (uint32_t)i;
    }

    return lat;
}

void sml_destroy(SMLLattice* lat) {
    if (!lat) return;
    if (lat->microcircuits) {
        for (int i = 0; i < lat->num_microcircuits; ++i) {
            free(lat->microcircuits[i]);
        }
        free(lat->microcircuits);
    }
    free(lat->lsh_entries);
    free(lat);
}

SMLResult sml_inference(SMLLattice* lat, const float* input, float* output) {
    SMLResult result = {0};

    // Cascade buffers (stack-allocated, aligned untuk AVX2)
    alignas(32) float a_cur[SML_D_STATE];
    alignas(32) float a_new[SML_D_STATE];
    alignas(32) float outs[SML_K_ROUTE][SML_D_STATE];

    // 1. Normalize input ke unit L2
    memcpy(a_cur, input, SML_D_STATE * sizeof(float));
    normalize_unit(a_cur);

    double t_start = now_sec();

    // 2. Route ONCE (fixed routing per inference)
    route_linear(lat, a_cur, result.route_ids, &lat->rng_state);

    // 3. Cascade loop
    for (int step = 0; step < SML_MAX_STEPS; ++step) {
        // 3a. Forward semua 6 microcircuit (sequential; parallel di future)
        for (int i = 0; i < SML_K_ROUTE; ++i) {
            microcircuit_forward(lat->microcircuits[result.route_ids[i]],
                                 a_cur, outs[i]);
        }

        // 3b. Energy-weighted fusion
        fuse_energy_weighted(outs, a_new);

        // 3c. Optional: normalize state
        if (SML_NORMALIZE_STATE) normalize_unit(a_new);

        // 3d. Convergence check
        float diff_sq = l2_distance_sq(a_cur, a_new);
        float cur_norm_sq = l2_norm_sq(a_cur);
        float rel_delta = sqrtf(diff_sq / (cur_norm_sq + 1e-8f));

        result.num_steps = step + 1;
        result.final_delta = rel_delta;
        if (rel_delta > result.max_delta) result.max_delta = rel_delta;

        if (rel_delta < SML_CONV_EPSILON) {
            result.converged = 1;
            memcpy(a_cur, a_new, SML_D_STATE * sizeof(float));
            break;
        }
        memcpy(a_cur, a_new, SML_D_STATE * sizeof(float));
    }

    // 4. Copy final state ke output
    memcpy(output, a_cur, SML_D_STATE * sizeof(float));

    result.time_us = (now_sec() - t_start) * 1e6;
    return result;
}

// ============================================================
// OPTIONAL: Test main
// Compile dengan -DSML_STANDALONE untuk enable
// ============================================================

#ifdef SML_STANDALONE

static void print_result(const char* label, const SMLResult* r) {
    printf("  %s:\n", label);
    printf("    Steps: %d, Converged: %s\n",
           r->num_steps, r->converged ? "YES" : "NO");
    printf("    Delta: final=%.4f, max=%.4f\n", r->final_delta, r->max_delta);
    printf("    Route: [");
    for (int i = 0; i < SML_K_ROUTE; ++i) {
        printf("%d%s", r->route_ids[i], i < SML_K_ROUTE - 1 ? "," : "");
    }
    printf("]\n");
    printf("    Time: %.1f µs\n", r->time_us);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    printf("================================================\n");
    printf("  SML MVP Test Standalone\n");
    printf("================================================\n\n");

    printf("Config:\n");
    printf("  D_STATE          = %d\n", SML_D_STATE);
    printf("  K_ROUTE          = %d\n", SML_K_ROUTE);
    printf("  MAX_STEPS        = %d\n", SML_MAX_STEPS);
    printf("  CONV_EPSILON     = %.4f\n", SML_CONV_EPSILON);
    printf("  NORMALIZE_STATE  = %d\n", SML_NORMALIZE_STATE);
    printf("  ROUTE_RAND_PROB  = %d%%\n", SML_ROUTE_RAND_PROB);
    printf("  Microcircuit size = %.1f KB\n\n",
           sizeof(SMLMicrocircuit) / 1024.0);

    // 1. Create lattice
    const int N = 1000;
    printf("Creating lattice N=%d... ", N);
    fflush(stdout);
    double t0 = now_sec();
    SMLLattice* lat = sml_create(N, 42);
    if (!lat) {
        fprintf(stderr, "FAILED\n");
        return 1;
    }
    printf("OK (%.2f sec, footprint %.1f MB)\n\n",
           now_sec() - t0,
           (double)sizeof(SMLMicrocircuit) * N / (1024*1024));

    // 2. Run 3 sanity inferences dengan input berbeda
    printf("Sanity inferences:\n");

    alignas(32) float input1[SML_D_STATE], output1[SML_D_STATE];
    alignas(32) float input2[SML_D_STATE], output2[SML_D_STATE];
    alignas(32) float input3[SML_D_STATE], output3[SML_D_STATE];

    uint32_t seed_state = 100;
    for (int i = 0; i < SML_D_STATE; ++i) {
        input1[i] = rng_float(&seed_state);
        input2[i] = input1[i] + rng_gaussian(&seed_state) * 0.05f;  // similar
        input3[i] = rng_float(&seed_state);                          // different
    }

    SMLResult r1 = sml_inference(lat, input1, output1);
    SMLResult r2 = sml_inference(lat, input2, output2);
    SMLResult r3 = sml_inference(lat, input3, output3);

    print_result("Input 1 (random)", &r1);
    print_result("Input 2 (similar to 1)", &r2);
    print_result("Input 3 (random, different)", &r3);

    // Check if similar inputs → same route (spec v0.2 Section 6)
    int overlap_12 = 0, overlap_13 = 0;
    for (int i = 0; i < SML_K_ROUTE; ++i)
        for (int j = 0; j < SML_K_ROUTE; ++j) {
            if (r1.route_ids[i] == r2.route_ids[j]) overlap_12++;
            if (r1.route_ids[i] == r3.route_ids[j]) overlap_13++;
        }
    printf("\nRoute overlap:\n");
    printf("  Input 1 vs 2 (similar):   %d/%d\n", overlap_12, SML_K_ROUTE);
    printf("  Input 1 vs 3 (different): %d/%d\n", overlap_13, SML_K_ROUTE);

    // 3. Benchmark: 100 inferences, mean latency
    printf("\nBenchmark: 100 inferences...\n");
    alignas(32) float bench_in[SML_D_STATE], bench_out[SML_D_STATE];
    double total_time = 0;
    int total_steps = 0, total_converged = 0;

    for (int trial = 0; trial < 100; ++trial) {
        for (int i = 0; i < SML_D_STATE; ++i)
            bench_in[i] = rng_float(&seed_state);
        SMLResult r = sml_inference(lat, bench_in, bench_out);
        total_time += r.time_us;
        total_steps += r.num_steps;
        if (r.converged) total_converged++;
    }

    printf("  Converged:  %d / 100\n", total_converged);
    printf("  Mean steps: %.1f\n", total_steps / 100.0);
    printf("  Mean time:  %.1f µs\n", total_time / 100.0);
    printf("  Throughput: %.0f inference/sec\n\n",
           1e6 * 100.0 / total_time);

    // 4. Cleanup
    sml_destroy(lat);

    printf("================================================\n");
    printf("  SML MVP works end-to-end.\n");
    printf("  Untuk build sebagai library:\n");
    printf("    gcc -O3 -march=native -mavx2 -mfma -mpopcnt -c sml_mvp.c\n");
    printf("================================================\n");

    return 0;
}

#endif  // SML_STANDALONE
