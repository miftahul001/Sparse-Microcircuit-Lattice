// ============================================================
// SML Phase 3a: LSH Routing (First Implementation)
//
// Untuk pertama kalinya, SML mulai jadi neural network beneran:
// tidak hardcoded 6-of-6, tapi 6-of-N via content-addressed routing.
//
// Komponen:
//   1. Microcircuit dimensi 256×256 (revised dari Phase 1-2)
//   2. SimHash AVX2: 32 hyperplane, 256-dim → 32-bit signature
//   3. LSH index: sorted array of (hash, mc_id)
//   4. Top-K router: pilih 6 mikrosirkit terdekat by Hamming distance
//   5. Slight randomness untuk eksplorasi
//
// Untuk N=100 mikrosirkit di Phase 3a, linear scan sudah cukup cepat
// (~100 popcount = ~200 ns). Bucketing di Phase 3b saat N naik.
//
// Build: gcc -O3 -march=native -mavx2 -mfma -mpopcnt \
//          phase3a_lsh_routing.c -o phase3a_lsh_routing -lm
// Run:   ./phase3a_lsh_routing
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
#define D_STATE            256    // state & mc I/O dimensi (revised)
#define NUM_MICROCIRCUITS  100    // Phase 3a: kecil untuk test
#define K_ROUTE            6      // pilih top-6 mikrosirkit per query
#define NUM_HYPERPLANES    32     // SimHash 32-bit
#define ROUTE_RAND_PROB    20     // 20% probabilitas slight randomness

// ============================================================
// Microcircuit (revised: square 256×256)
// ============================================================
typedef struct {
    alignas(64) int8_t W[D_STATE][D_STATE];    // 64 KB
    alignas(64) float  scales[D_STATE];        // 1 KB
    alignas(64) float  bias[D_STATE];          // 1 KB
    // Metadata untuk routing:
    alignas(64) uint32_t self_hash;            // SimHash 32-bit dari resonance_vec
    alignas(32) float resonance_vec[D_STATE];  // 1 KB — signature vector
} Microcircuit;

// Total ~68 KB per mikrosirkit, fits comfortably di L2 256 KB

// ============================================================
// LSH Index: sorted array of (hash, mc_index)
// ============================================================
typedef struct {
    uint32_t hash;
    uint32_t mc_index;
} LSHEntry;

typedef struct {
    LSHEntry* entries;   // sorted by hash ascending
    int count;
} LSHIndex;

static int lsh_entry_cmp(const void* a, const void* b) {
    uint32_t ha = ((const LSHEntry*)a)->hash;
    uint32_t hb = ((const LSHEntry*)b)->hash;
    if (ha < hb) return -1;
    if (ha > hb) return 1;
    return 0;
}

// ============================================================
// Global hyperplanes untuk SimHash (loaded once)
// ============================================================
static alignas(32) float g_hyperplanes[NUM_HYPERPLANES][D_STATE];

// ============================================================
// RNG: xorshift32, lebih baik dari rand() dan reproducible
// ============================================================
static uint32_t g_rng_state = 42;

static inline uint32_t xorshift32(void) {
    uint32_t x = g_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng_state = x;
    return x;
}

// Uniform in [-1, 1]
static float rand_float(void) {
    return ((float)xorshift32() / (float)UINT32_MAX) * 2.0f - 1.0f;
}

// Approximate Gaussian via Box-Muller (untuk hyperplane init)
static float rand_gaussian(void) {
    float u1 = ((float)xorshift32() / (float)UINT32_MAX);
    if (u1 < 1e-7f) u1 = 1e-7f;  // avoid log(0)
    float u2 = ((float)xorshift32() / (float)UINT32_MAX);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static int8_t rand_int8(void) {
    return (int8_t)(xorshift32() & 0xFF);
}

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
    __m128 sum128 = _mm_add_ps(hi, lo);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ============================================================
// SimHash AVX2: 256-dim FP32 → 32-bit signature
//
// Untuk setiap hyperplane h:
//   dot = <hyperplane_h, a>
//   bit_h = (dot > 0)
// signature = concat(bit_0, ..., bit_31)
//
// Target: ~500 ns
// ============================================================
static uint32_t simhash_avx2(const float* a) {
    uint32_t sig = 0;
    for (int h = 0; h < NUM_HYPERPLANES; ++h) {
        // 4 parallel accumulator untuk ILP
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        const float* hp = g_hyperplanes[h];
        for (int i = 0; i < D_STATE; i += 32) {
            __m256 hp0 = _mm256_load_ps(&hp[i +  0]);
            __m256 hp1 = _mm256_load_ps(&hp[i +  8]);
            __m256 hp2 = _mm256_load_ps(&hp[i + 16]);
            __m256 hp3 = _mm256_load_ps(&hp[i + 24]);

            __m256 a0 = _mm256_load_ps(&a[i +  0]);
            __m256 a1 = _mm256_load_ps(&a[i +  8]);
            __m256 a2 = _mm256_load_ps(&a[i + 16]);
            __m256 a3 = _mm256_load_ps(&a[i + 24]);

            acc0 = _mm256_fmadd_ps(hp0, a0, acc0);
            acc1 = _mm256_fmadd_ps(hp1, a1, acc1);
            acc2 = _mm256_fmadd_ps(hp2, a2, acc2);
            acc3 = _mm256_fmadd_ps(hp3, a3, acc3);
        }

        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                    _mm256_add_ps(acc2, acc3));
        float dot = hsum_avx2(acc);
        sig |= (dot > 0.0f ? 1u : 0u) << h;
    }
    return sig;
}

// ============================================================
// Route: query vector → top-K microcircuit indices
//
// 1. Compute SimHash(query)
// 2. Scan LSH index, compute Hamming distance untuk each entry
// 3. Track top-K+2 candidates by min distance
// 4. Copy top-K ke output
// 5. Kalau enable_random: dengan prob 20%, swap salah satu dengan best[K] or best[K+1]
// ============================================================
typedef struct {
    int idx;
    int dist;
} Candidate;

static void route(const LSHIndex* idx,
                  const float* query,
                  int* out_ids,
                  int k,
                  int enable_random) {
    uint32_t q_hash = simhash_avx2(query);

    // Track top (K+2) candidates
    int extended = k + 2;
    Candidate best[K_ROUTE + 2];
    for (int i = 0; i < extended; ++i) {
        best[i].idx = -1;
        best[i].dist = INT_MAX;
    }

    // Linear scan (N=100, ~200 ns total)
    for (int i = 0; i < idx->count; ++i) {
        int dist = __builtin_popcount(q_hash ^ idx->entries[i].hash);

        if (dist < best[extended - 1].dist) {
            // Insertion sort ke best[]
            int j = extended - 1;
            while (j > 0 && best[j - 1].dist > dist) {
                best[j] = best[j - 1];
                j--;
            }
            best[j].idx = (int)idx->entries[i].mc_index;
            best[j].dist = dist;
        }
    }

    // Copy top-K
    for (int i = 0; i < k; ++i) out_ids[i] = best[i].idx;

    // Slight randomness
    // Prob 20%: swap salah satu slot dengan best[K] or best[K+1]
    if (enable_random && best[k].idx != -1) {
        if ((xorshift32() % 100) < ROUTE_RAND_PROB) {
            int swap_slot = xorshift32() % k;
            int replacement = k + (xorshift32() & 1);  // K or K+1
            if (replacement < extended && best[replacement].idx != -1) {
                out_ids[swap_slot] = best[replacement].idx;
            }
        }
    }
}

// ============================================================
// Init
// ============================================================
static void init_hyperplanes(void) {
    for (int h = 0; h < NUM_HYPERPLANES; ++h) {
        for (int i = 0; i < D_STATE; ++i) {
            g_hyperplanes[h][i] = rand_gaussian();
        }
    }
}

static void init_microcircuit(Microcircuit* M) {
    for (int r = 0; r < D_STATE; ++r) {
        for (int c = 0; c < D_STATE; ++c) M->W[r][c] = rand_int8();
        M->scales[r] = 0.001f;
        M->bias[r] = rand_float() * 0.1f;
    }
    // Resonance vector: random (later akan di-set dari real signatures)
    for (int i = 0; i < D_STATE; ++i) M->resonance_vec[i] = rand_float();
    // self_hash = SimHash dari resonance_vec
    M->self_hash = simhash_avx2(M->resonance_vec);
}

static void lsh_index_build(LSHIndex* idx, Microcircuit** mcs, int n) {
    idx->count = n;
    idx->entries = malloc(n * sizeof(LSHEntry));
    for (int i = 0; i < n; ++i) {
        idx->entries[i].hash = mcs[i]->self_hash;
        idx->entries[i].mc_index = (uint32_t)i;
    }
    qsort(idx->entries, n, sizeof(LSHEntry), lsh_entry_cmp);
}

// ============================================================
// Tests
// ============================================================
static void test_1_simhash_benchmark(void) {
    printf("--- Test 1: SimHash microbenchmark ---\n");
    alignas(32) float a[D_STATE];
    for (int i = 0; i < D_STATE; ++i) a[i] = rand_float();

    // Warmup
    volatile uint32_t sink = 0;
    for (int i = 0; i < 1000; ++i) sink += simhash_avx2(a);

    int iters = 1000000;
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) sink += simhash_avx2(a);
    double t1 = now_sec();
    (void)sink;

    double ns_per_call = (t1 - t0) * 1e9 / iters;
    printf("  Per SimHash call: %.1f ns\n", ns_per_call);
    printf("  (Target: <500 ns)\n\n");
}

static void test_2_route_benchmark(const LSHIndex* idx) {
    printf("--- Test 2: Route microbenchmark ---\n");
    alignas(32) float a[D_STATE];
    for (int i = 0; i < D_STATE; ++i) a[i] = rand_float();

    int out_ids[K_ROUTE];

    // Warmup
    for (int i = 0; i < 1000; ++i) route(idx, a, out_ids, K_ROUTE, 0);

    int iters = 1000000;
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) route(idx, a, out_ids, K_ROUTE, 0);
    double t1 = now_sec();

    double ns_per_call = (t1 - t0) * 1e9 / iters;
    printf("  Per route call (no randomness): %.1f ns\n", ns_per_call);
    printf("  (Target: <2000 ns)\n");
    printf("  Breakdown: SimHash + %d Hamming distances + top-6 insertion sort\n\n",
           NUM_MICROCIRCUITS);
}

static void test_3_similarity(const LSHIndex* idx) {
    printf("--- Test 3: Similarity coherence ---\n");
    printf("  Query mirip → route output overlap tinggi\n");
    printf("  Query berbeda → route output overlap rendah\n\n");

    alignas(32) float q1[D_STATE], q2[D_STATE], q3[D_STATE];
    for (int i = 0; i < D_STATE; ++i) {
        q1[i] = rand_float();
        q2[i] = q1[i] + rand_gaussian() * 0.05f;   // q1 + noise kecil
        q3[i] = rand_float();                        // totally different
    }

    int ids1[K_ROUTE], ids2[K_ROUTE], ids3[K_ROUTE];
    route(idx, q1, ids1, K_ROUTE, 0);
    route(idx, q2, ids2, K_ROUTE, 0);
    route(idx, q3, ids3, K_ROUTE, 0);

    // Count overlap
    int overlap_12 = 0, overlap_13 = 0;
    for (int i = 0; i < K_ROUTE; ++i) {
        for (int j = 0; j < K_ROUTE; ++j) {
            if (ids1[i] == ids2[j]) overlap_12++;
            if (ids1[i] == ids3[j]) overlap_13++;
        }
    }

    printf("  q1 vs q2 (mirip):     overlap = %d/%d microcircuit\n",
           overlap_12, K_ROUTE);
    printf("  q1 vs q3 (berbeda):   overlap = %d/%d microcircuit\n",
           overlap_13, K_ROUTE);

    if (overlap_12 >= K_ROUTE - 2 && overlap_13 <= 2) {
        printf("  [PASS] Routing coherent — mirip menghasilkan output mirip\n\n");
    } else if (overlap_12 > overlap_13) {
        printf("  [OK]   Trend benar tapi coherence lemah\n\n");
    } else {
        printf("  [FAIL] Routing tidak coherent — investigate SimHash quality\n\n");
    }
}

static void test_4_distribution(const LSHIndex* idx) {
    printf("--- Test 4: Distribution uniformity ---\n");
    printf("  10.000 random query, cek apakah semua mikrosirkit ke-select\n");
    printf("  Ratio min/max selection count harus dalam ~2x untuk uniform\n\n");

    int counts_no_rand[NUM_MICROCIRCUITS] = {0};
    int counts_rand[NUM_MICROCIRCUITS] = {0};

    alignas(32) float q[D_STATE];
    int ids[K_ROUTE];

    // Deterministic routing
    for (int trial = 0; trial < 10000; ++trial) {
        for (int i = 0; i < D_STATE; ++i) q[i] = rand_float();
        route(idx, q, ids, K_ROUTE, 0);
        for (int i = 0; i < K_ROUTE; ++i) counts_no_rand[ids[i]]++;
    }

    // With randomness
    for (int trial = 0; trial < 10000; ++trial) {
        for (int i = 0; i < D_STATE; ++i) q[i] = rand_float();
        route(idx, q, ids, K_ROUTE, 1);
        for (int i = 0; i < K_ROUTE; ++i) counts_rand[ids[i]]++;
    }

    // Statistics
    int min_no = INT_MAX, max_no = 0, never_no = 0;
    int min_r = INT_MAX, max_r = 0, never_r = 0;
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        if (counts_no_rand[i] == 0) never_no++;
        else {
            if (counts_no_rand[i] < min_no) min_no = counts_no_rand[i];
            if (counts_no_rand[i] > max_no) max_no = counts_no_rand[i];
        }
        if (counts_rand[i] == 0) never_r++;
        else {
            if (counts_rand[i] < min_r) min_r = counts_rand[i];
            if (counts_rand[i] > max_r) max_r = counts_rand[i];
        }
    }

    printf("  Deterministic routing (60.000 total selections):\n");
    printf("    Never selected: %d mikrosirkit\n", never_no);
    printf("    Min/Max count:  %d / %d  (ratio %.2fx)\n",
           min_no, max_no, (float)max_no / (min_no > 0 ? min_no : 1));

    printf("  With slight randomness:\n");
    printf("    Never selected: %d mikrosirkit\n", never_r);
    printf("    Min/Max count:  %d / %d  (ratio %.2fx)\n",
           min_r, max_r, (float)max_r / (min_r > 0 ? min_r : 1));

    if (never_r == 0) {
        printf("  [PASS] Randomness memastikan semua mikrosirkit ter-explore\n\n");
    } else {
        printf("  [WARN] %d mikrosirkit tidak pernah dipilih meski dengan randomness\n\n",
               never_r);
    }
}

static void test_5_hash_diversity(Microcircuit** mcs) {
    printf("--- Test 5: Hash diversity ---\n");
    printf("  Cek self_hash mikrosirkit unique dan tersebar\n\n");

    int duplicates = 0;
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        for (int j = i + 1; j < NUM_MICROCIRCUITS; ++j) {
            if (mcs[i]->self_hash == mcs[j]->self_hash) duplicates++;
        }
    }

    // Compute mean pairwise Hamming distance
    long total_dist = 0, pair_count = 0;
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        for (int j = i + 1; j < NUM_MICROCIRCUITS; ++j) {
            total_dist += __builtin_popcount(mcs[i]->self_hash
                                              ^ mcs[j]->self_hash);
            pair_count++;
        }
    }
    double mean_dist = (double)total_dist / pair_count;

    printf("  Duplicate hashes: %d (expected ~0)\n", duplicates);
    printf("  Mean pairwise Hamming distance: %.2f / 32 bits\n", mean_dist);
    printf("  (Untuk random 32-bit hashes, expected ~16)\n\n");
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 3a: LSH Routing\n");
    printf("================================================\n");
    printf("Config:\n");
    printf("  D_STATE          = %d\n", D_STATE);
    printf("  NUM_MICROCIRCUITS= %d\n", NUM_MICROCIRCUITS);
    printf("  K_ROUTE          = %d\n", K_ROUTE);
    printf("  NUM_HYPERPLANES  = %d (SimHash bit width)\n", NUM_HYPERPLANES);
    printf("  ROUTE_RAND_PROB  = %d%%\n\n", ROUTE_RAND_PROB);

    printf("Memory footprint:\n");
    printf("  Per microcircuit: %.1f KB\n", sizeof(Microcircuit) / 1024.0);
    printf("  Total lattice:    %.1f MB\n\n",
           (double)sizeof(Microcircuit) * NUM_MICROCIRCUITS / (1024*1024));

    // Init global hyperplanes
    init_hyperplanes();

    // Alokasi mikrosirkit
    Microcircuit** mcs = malloc(NUM_MICROCIRCUITS * sizeof(Microcircuit*));
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        posix_memalign((void**)&mcs[i], 64, sizeof(Microcircuit));
        init_microcircuit(mcs[i]);
    }

    // Build LSH index
    LSHIndex idx;
    lsh_index_build(&idx, mcs, NUM_MICROCIRCUITS);

    // Run tests
    test_5_hash_diversity(mcs);
    test_1_simhash_benchmark();
    test_2_route_benchmark(&idx);
    test_3_similarity(&idx);
    test_4_distribution(&idx);

    // Cleanup
    free(idx.entries);
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) free(mcs[i]);
    free(mcs);

    printf("================================================\n");
    printf("  Phase 3a complete\n");
    printf("================================================\n");
    printf("Kalau semua PASS, next:\n");
    printf("  Phase 3b: Scale ke 1000 mikrosirkit + LSH bucketing\n");
    printf("  Phase 4:  Integrasi routing + dispatch + fusion (full cascade)\n");
    return 0;
}
