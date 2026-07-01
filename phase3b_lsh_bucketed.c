// ============================================================
// SML Phase 3b: LSH Bucketing + Scale to 1000 Microcircuits
//
// Perbedaan dari Phase 3a:
//   1. NUM_MICROCIRCUITS: 100 → 1000
//   2. Tambah LSH Bucketed Index (Option B: prefix-sorted array)
//   3. Head-to-head: linear scan vs bucketed
//   4. Fix DCE bug di microbenchmark (Phase 3a Test 2 = 0.0 ns)
//   5. Test quality preservation (recall) dari bucketing
//
// Design bucketing:
//   - BUCKET_BITS = 8 → 256 bucket
//   - Bucket index = high 8 bits dari hash
//   - Probe: self bucket + 8 Hamming-1 neighbor bucket = 9 bucket total
//   - Rata2 candidates per query: (1000/256) × 9 ≈ 35 candidate
//
// Build: gcc -O3 -march=native -mavx2 -mfma -mpopcnt \
//          phase3b_lsh_bucketed.c -o phase3b_lsh_bucketed -lm
// Run:   ./phase3b_lsh_bucketed
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
#define NUM_MICROCIRCUITS  1000    // scaled up 10x
#define K_ROUTE            6
#define NUM_HYPERPLANES    32
#define ROUTE_RAND_PROB    20

// Bucketing config
#define BUCKET_BITS        8
#define NUM_BUCKETS        (1 << BUCKET_BITS)
#define BUCKET_MASK        (NUM_BUCKETS - 1)

// ============================================================
// Structs
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

// Linear scan index (Phase 3a style, untuk baseline)
typedef struct {
    LSHEntry* entries;
    int count;
} LSHLinearIndex;

// Bucketed index (Phase 3b innovation)
typedef struct {
    LSHEntry* entries;                   // sorted by (bucket, hash)
    int count;
    int bucket_start[NUM_BUCKETS + 1];   // bucket_start[i]..[i+1] = bucket i
} LSHBucketIndex;

typedef struct {
    int idx;
    int dist;
} Candidate;

// ============================================================
// Global hyperplanes
// ============================================================
static alignas(32) float g_hyperplanes[NUM_HYPERPLANES][D_STATE];

// ============================================================
// RNG (xorshift32)
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
// SimHash AVX2 (sama dengan Phase 3a)
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
// Bucket helpers
// ============================================================
static inline int get_bucket(uint32_t hash) {
    return (int)(hash >> (32 - BUCKET_BITS));  // high 8 bits
}

// ============================================================
// Index builders
// ============================================================
static int cmp_by_hash(const void* a, const void* b) {
    uint32_t ha = ((const LSHEntry*)a)->hash;
    uint32_t hb = ((const LSHEntry*)b)->hash;
    return (ha < hb) ? -1 : (ha > hb) ? 1 : 0;
}

static int cmp_by_bucket_then_hash(const void* a, const void* b) {
    uint32_t ha = ((const LSHEntry*)a)->hash;
    uint32_t hb = ((const LSHEntry*)b)->hash;
    int ba = get_bucket(ha);
    int bb = get_bucket(hb);
    if (ba != bb) return ba - bb;
    return (ha < hb) ? -1 : (ha > hb) ? 1 : 0;
}

static void linear_index_build(LSHLinearIndex* idx, Microcircuit** mcs, int n) {
    idx->count = n;
    idx->entries = malloc(n * sizeof(LSHEntry));
    for (int i = 0; i < n; ++i) {
        idx->entries[i].hash = mcs[i]->self_hash;
        idx->entries[i].mc_index = (uint32_t)i;
    }
    qsort(idx->entries, n, sizeof(LSHEntry), cmp_by_hash);
}

static void bucket_index_build(LSHBucketIndex* idx, Microcircuit** mcs, int n) {
    idx->count = n;
    idx->entries = malloc(n * sizeof(LSHEntry));
    for (int i = 0; i < n; ++i) {
        idx->entries[i].hash = mcs[i]->self_hash;
        idx->entries[i].mc_index = (uint32_t)i;
    }
    qsort(idx->entries, n, sizeof(LSHEntry), cmp_by_bucket_then_hash);

    // Populate bucket_start
    for (int b = 0; b <= NUM_BUCKETS; ++b) idx->bucket_start[b] = n;

    for (int i = 0; i < n; ++i) {
        int b = get_bucket(idx->entries[i].hash);
        if (idx->bucket_start[b] > i) idx->bucket_start[b] = i;
    }

    // Fill empty buckets: point to next non-empty bucket start
    int next_start = n;
    for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
        if (idx->bucket_start[b] == n) idx->bucket_start[b] = next_start;
        else next_start = idx->bucket_start[b];
    }
    idx->bucket_start[NUM_BUCKETS] = n;
}

// ============================================================
// Helper: insert candidate into sorted best[] array
// ============================================================
static inline void insert_candidate(Candidate* best, int extended,
                                     int mc_idx, int dist) {
    if (dist >= best[extended - 1].dist) return;
    int j = extended - 1;
    while (j > 0 && best[j - 1].dist > dist) {
        best[j] = best[j - 1];
        j--;
    }
    best[j].idx = mc_idx;
    best[j].dist = dist;
}

// ============================================================
// Route: LINEAR scan (baseline)
// ============================================================
static void route_linear(const LSHLinearIndex* idx,
                          const float* query,
                          int* out_ids,
                          int k,
                          int enable_random) {
    uint32_t q_hash = simhash_avx2(query);

    int extended = k + 2;
    Candidate best[K_ROUTE + 2];
    for (int i = 0; i < extended; ++i) {
        best[i].idx = -1;
        best[i].dist = INT_MAX;
    }

    for (int i = 0; i < idx->count; ++i) {
        int dist = __builtin_popcount(q_hash ^ idx->entries[i].hash);
        insert_candidate(best, extended, (int)idx->entries[i].mc_index, dist);
    }

    for (int i = 0; i < k; ++i) out_ids[i] = best[i].idx;

    if (enable_random && best[k].idx != -1) {
        if ((xorshift32() % 100) < ROUTE_RAND_PROB) {
            int swap_slot = xorshift32() % k;
            int replacement = k + (xorshift32() & 1);
            if (replacement < extended && best[replacement].idx != -1) {
                out_ids[swap_slot] = best[replacement].idx;
            }
        }
    }
}

// ============================================================
// Route: BUCKETED (self + 8 Hamming-1 neighbors)
// ============================================================
static void route_bucketed(const LSHBucketIndex* idx,
                            const float* query,
                            int* out_ids,
                            int k,
                            int enable_random) {
    uint32_t q_hash = simhash_avx2(query);
    int q_bucket = get_bucket(q_hash);

    int extended = k + 2;
    Candidate best[K_ROUTE + 2];
    for (int i = 0; i < extended; ++i) {
        best[i].idx = -1;
        best[i].dist = INT_MAX;
    }

    // Scan self bucket + 8 Hamming-1 buckets
    // Bucket dinyatakan by (q_bucket XOR bit_mask)
    // bit_masks: {0, 1, 2, 4, 8, 16, 32, 64, 128}
    for (int m = 0; m < BUCKET_BITS + 1; ++m) {
        int probe_bucket = (m == 0) ? q_bucket : (q_bucket ^ (1 << (m - 1)));
        int start = idx->bucket_start[probe_bucket];
        int end = idx->bucket_start[probe_bucket + 1];
        for (int i = start; i < end; ++i) {
            int dist = __builtin_popcount(q_hash ^ idx->entries[i].hash);
            insert_candidate(best, extended,
                             (int)idx->entries[i].mc_index, dist);
        }
    }

    for (int i = 0; i < k; ++i) out_ids[i] = best[i].idx;

    if (enable_random && best[k].idx != -1) {
        if ((xorshift32() % 100) < ROUTE_RAND_PROB) {
            int swap_slot = xorshift32() % k;
            int replacement = k + (xorshift32() & 1);
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
    for (int i = 0; i < D_STATE; ++i) M->resonance_vec[i] = rand_float();
    M->self_hash = simhash_avx2(M->resonance_vec);
}

// ============================================================
// Tests
// ============================================================
static void test_bucket_distribution(const LSHBucketIndex* idx) {
    printf("--- Test 1: Bucket distribution ---\n");
    printf("  256 buckets, %d entries, expected mean %.1f per bucket\n\n",
           idx->count, (double)idx->count / NUM_BUCKETS);

    int empty = 0, min_size = INT_MAX, max_size = 0;
    long total_size = 0;
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        int size = idx->bucket_start[b + 1] - idx->bucket_start[b];
        if (size == 0) empty++;
        else {
            if (size < min_size) min_size = size;
            if (size > max_size) max_size = size;
        }
        total_size += size;
    }

    printf("  Empty buckets: %d / %d\n", empty, NUM_BUCKETS);
    printf("  Non-empty: min %d, max %d, mean %.2f\n",
           min_size, max_size,
           (double)total_size / (NUM_BUCKETS - empty));
    printf("  Total entries: %ld (expected %d)\n\n", total_size, idx->count);
}

static void test_simhash_latency(void) {
    printf("--- Test 2: SimHash microbenchmark ---\n");

    // Pre-generate multiple queries untuk hindari trivial caching
    const int NQ = 64;
    alignas(32) float queries[64][D_STATE];
    for (int q = 0; q < NQ; ++q)
        for (int i = 0; i < D_STATE; ++i) queries[q][i] = rand_float();

    volatile uint32_t sink = 0;
    for (int i = 0; i < 1000; ++i) sink += simhash_avx2(queries[i & 63]);

    int iters = 1000000;
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) sink += simhash_avx2(queries[i & 63]);
    double t1 = now_sec();
    (void)sink;

    printf("  Per SimHash call: %.1f ns (target <500 ns)\n\n",
           (t1 - t0) * 1e9 / iters);
}

static void test_route_latency(const LSHLinearIndex* lin,
                                const LSHBucketIndex* buck) {
    printf("--- Test 3: Route latency comparison (FIXED benchmark) ---\n");

    const int NQ = 64;
    alignas(32) float queries[64][D_STATE];
    for (int q = 0; q < NQ; ++q)
        for (int i = 0; i < D_STATE; ++i) queries[q][i] = rand_float();

    int out_ids[K_ROUTE];
    volatile int sink = 0;

    // Warmup
    for (int i = 0; i < 1000; ++i) {
        route_linear(lin, queries[i & 63], out_ids, K_ROUTE, 0);
        sink ^= out_ids[0];
    }

    // Linear scan
    int iters = 100000;
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        route_linear(lin, queries[i & 63], out_ids, K_ROUTE, 0);
        sink ^= out_ids[0] ^ out_ids[K_ROUTE - 1];  // force use of output
    }
    double t1 = now_sec();
    double linear_ns = (t1 - t0) * 1e9 / iters;

    // Bucketed
    for (int i = 0; i < 1000; ++i) {
        route_bucketed(buck, queries[i & 63], out_ids, K_ROUTE, 0);
        sink ^= out_ids[0];
    }
    t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        route_bucketed(buck, queries[i & 63], out_ids, K_ROUTE, 0);
        sink ^= out_ids[0] ^ out_ids[K_ROUTE - 1];
    }
    t1 = now_sec();
    double bucketed_ns = (t1 - t0) * 1e9 / iters;
    (void)sink;

    printf("  Linear scan (1000 popcount): %.1f ns per route\n", linear_ns);
    printf("  Bucketed (probe 9 buckets):  %.1f ns per route\n", bucketed_ns);
    printf("  Speedup: %.2fx\n", linear_ns / bucketed_ns);
    printf("  (Target: bucketed <2000 ns, speedup >1.5x)\n\n");
}

static void test_quality_preservation(const LSHLinearIndex* lin,
                                       const LSHBucketIndex* buck) {
    printf("--- Test 4: Bucketing quality (recall vs linear) ---\n");
    printf("  Untuk 1000 query random, compare top-K bucketed vs linear\n");
    printf("  Bucketed harusnya menemukan mayoritas nearest neighbors\n\n");

    int total_overlap = 0;
    int perfect_matches = 0;
    int significant_miss = 0;  // overlap < K/2

    for (int trial = 0; trial < 1000; ++trial) {
        alignas(32) float q[D_STATE];
        for (int i = 0; i < D_STATE; ++i) q[i] = rand_float();

        int lin_ids[K_ROUTE], buck_ids[K_ROUTE];
        route_linear(lin, q, lin_ids, K_ROUTE, 0);
        route_bucketed(buck, q, buck_ids, K_ROUTE, 0);

        int overlap = 0;
        for (int i = 0; i < K_ROUTE; ++i)
            for (int j = 0; j < K_ROUTE; ++j)
                if (lin_ids[i] == buck_ids[j]) overlap++;

        total_overlap += overlap;
        if (overlap == K_ROUTE) perfect_matches++;
        if (overlap < K_ROUTE / 2) significant_miss++;
    }

    double mean_overlap = (double)total_overlap / 1000;
    double recall = mean_overlap / K_ROUTE;

    printf("  Mean overlap:      %.2f / %d (%.0f%% recall)\n",
           mean_overlap, K_ROUTE, recall * 100);
    printf("  Perfect matches:   %d / 1000 (%.1f%%)\n",
           perfect_matches, perfect_matches / 10.0);
    printf("  Significant miss:  %d / 1000 (%.1f%%)\n",
           significant_miss, significant_miss / 10.0);

    if (recall > 0.85) {
        printf("  [PASS] Bucketing preserves quality (>85%% recall)\n\n");
    } else if (recall > 0.7) {
        printf("  [MARGINAL] Recall %.0f%% — mungkin perlu probe Hamming-2 juga\n\n",
               recall * 100);
    } else {
        printf("  [FAIL] Recall <70%%, bucketing kehilangan terlalu banyak candidate\n\n");
    }
}

static void test_similarity_at_scale(const LSHBucketIndex* idx) {
    printf("--- Test 5: Similarity coherence at scale (1000 mc) ---\n");

    int strong_coherent = 0, weak_coherent = 0, incoherent = 0;

    for (int trial = 0; trial < 100; ++trial) {
        alignas(32) float q1[D_STATE], q2[D_STATE], q3[D_STATE];
        for (int i = 0; i < D_STATE; ++i) {
            q1[i] = rand_float();
            q2[i] = q1[i] + rand_gaussian() * 0.05f;
            q3[i] = rand_float();
        }

        int ids1[K_ROUTE], ids2[K_ROUTE], ids3[K_ROUTE];
        route_bucketed(idx, q1, ids1, K_ROUTE, 0);
        route_bucketed(idx, q2, ids2, K_ROUTE, 0);
        route_bucketed(idx, q3, ids3, K_ROUTE, 0);

        int overlap_12 = 0, overlap_13 = 0;
        for (int i = 0; i < K_ROUTE; ++i)
            for (int j = 0; j < K_ROUTE; ++j) {
                if (ids1[i] == ids2[j]) overlap_12++;
                if (ids1[i] == ids3[j]) overlap_13++;
            }

        if (overlap_12 >= 4 && overlap_13 <= 1) strong_coherent++;
        else if (overlap_12 > overlap_13) weak_coherent++;
        else incoherent++;
    }

    printf("  100 trials:\n");
    printf("    Strong coherent  (mirip≥4, beda≤1): %d\n", strong_coherent);
    printf("    Weak coherent   (mirip > beda):    %d\n", weak_coherent);
    printf("    Incoherent      (mirip ≤ beda):    %d\n\n", incoherent);

    if (strong_coherent >= 70) {
        printf("  [PASS] Content-addressed routing scales dengan baik\n\n");
    } else {
        printf("  [WARN] Coherence menurun di skala besar\n\n");
    }
}

static void test_distribution_1000(const LSHBucketIndex* idx) {
    printf("--- Test 6: Distribution uniformity (1000 mc, 10000 query) ---\n");

    int* counts_det = calloc(NUM_MICROCIRCUITS, sizeof(int));
    int* counts_rand = calloc(NUM_MICROCIRCUITS, sizeof(int));

    alignas(32) float q[D_STATE];
    int ids[K_ROUTE];

    for (int trial = 0; trial < 10000; ++trial) {
        for (int i = 0; i < D_STATE; ++i) q[i] = rand_float();
        route_bucketed(idx, q, ids, K_ROUTE, 0);
        for (int i = 0; i < K_ROUTE; ++i) counts_det[ids[i]]++;
    }
    for (int trial = 0; trial < 10000; ++trial) {
        for (int i = 0; i < D_STATE; ++i) q[i] = rand_float();
        route_bucketed(idx, q, ids, K_ROUTE, 1);
        for (int i = 0; i < K_ROUTE; ++i) counts_rand[ids[i]]++;
    }

    int never_det = 0, never_rand = 0;
    int min_det = INT_MAX, max_det = 0;
    int min_rand = INT_MAX, max_rand = 0;
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        if (counts_det[i] == 0) never_det++;
        else {
            if (counts_det[i] < min_det) min_det = counts_det[i];
            if (counts_det[i] > max_det) max_det = counts_det[i];
        }
        if (counts_rand[i] == 0) never_rand++;
        else {
            if (counts_rand[i] < min_rand) min_rand = counts_rand[i];
            if (counts_rand[i] > max_rand) max_rand = counts_rand[i];
        }
    }

    printf("  Deterministic: never=%d, min=%d, max=%d (ratio %.2fx)\n",
           never_det, min_det, max_det,
           (float)max_det / (min_det > 0 ? min_det : 1));
    printf("  Randomness:    never=%d, min=%d, max=%d (ratio %.2fx)\n\n",
           never_rand, min_rand, max_rand,
           (float)max_rand / (min_rand > 0 ? min_rand : 1));

    free(counts_det);
    free(counts_rand);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 3b: LSH Bucketing @ 1000 microcircuits\n");
    printf("================================================\n");
    printf("Config:\n");
    printf("  NUM_MICROCIRCUITS = %d\n", NUM_MICROCIRCUITS);
    printf("  BUCKET_BITS       = %d (%d buckets)\n",
           BUCKET_BITS, NUM_BUCKETS);
    printf("  Total lattice     = %.1f MB\n\n",
           (double)sizeof(Microcircuit) * NUM_MICROCIRCUITS / (1024*1024));

    init_hyperplanes();

    printf("Building %d microcircuits...\n", NUM_MICROCIRCUITS);
    double t0 = now_sec();
    Microcircuit** mcs = malloc(NUM_MICROCIRCUITS * sizeof(Microcircuit*));
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) {
        posix_memalign((void**)&mcs[i], 64, sizeof(Microcircuit));
        init_microcircuit(mcs[i]);
    }
    double t1 = now_sec();
    printf("  Init time: %.2f sec (mostly random weight generation)\n\n",
           t1 - t0);

    // Build both indices
    LSHLinearIndex lin;
    LSHBucketIndex buck;
    linear_index_build(&lin, mcs, NUM_MICROCIRCUITS);
    bucket_index_build(&buck, mcs, NUM_MICROCIRCUITS);

    // Tests
    test_bucket_distribution(&buck);
    test_simhash_latency();
    test_route_latency(&lin, &buck);
    test_quality_preservation(&lin, &buck);
    test_similarity_at_scale(&buck);
    test_distribution_1000(&buck);

    // Cleanup
    free(lin.entries);
    free(buck.entries);
    for (int i = 0; i < NUM_MICROCIRCUITS; ++i) free(mcs[i]);
    free(mcs);

    printf("================================================\n");
    printf("  Phase 3b complete\n");
    printf("================================================\n");
    printf("Next: Phase 4 — Full cascade integration\n");
    printf("  Routing + dispatch + fusion + fixed-point convergence\n");
    return 0;
}
