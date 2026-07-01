// ============================================================
// SML Phase 2 (v2): Multi-Core Dispatch dengan Blocking Sync
//
// FIX: v1 menggunakan spin-wait yang konsumsi 100% CPU per worker
// bahkan saat idle. Di bawah quota --cpus="4.8", 7 thread spinning
// = starvation + container hang.
//
// v2 menggunakan pthread_barrier_t. Worker block (sleep) saat
// menunggu kerja, tidak konsumsi quota. Hardware-longevity friendly.
//
// Trade-off: dispatch latency naik dari ~50ns (spin) ke ~1-3µs
// (futex syscall via barrier). Untuk workload 14µs/microcircuit,
// overhead barrier = 10-20% — masih acceptable.
//
// Build: gcc -O3 -march=native -mavx2 -mfma phase2_dispatch_v2.c \
//          -o phase2_dispatch_v2 -lpthread -lm
// Run:   ./phase2_dispatch_v2
// ============================================================

#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define D_IN       384
#define D_OUT      256
#define NUM_WORKERS 6

// ============================================================
// Microcircuit (copy dari Phase 1)
// ============================================================
typedef struct {
    alignas(64) int8_t W[D_OUT][D_IN];
    alignas(64) float  scales[D_OUT];
    alignas(64) float  bias[D_OUT];
} Microcircuit;

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

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

void microcircuit_forward(const Microcircuit* M,
                          const float* a, float* y) {
    for (int row = 0; row < D_OUT; ++row) {
        const int8_t* W_row = M->W[row];
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int col = 0; col < D_IN; col += 32) {
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
// Pool dengan barrier sync
//
// PROTOCOL:
//   1. Main set work fields untuk semua worker
//   2. Main + workers semua hit start_barrier → barrier complete → semua release
//   3. Workers process (main blocks di end_barrier)
//   4. Workers selesai hit end_barrier → barrier complete → semua release
//   5. Main return dari dispatch_step
//   6. Workers loop back, block di start_barrier untuk dispatch berikutnya
//
// Total NUM_WORKERS+1 thread di barrier (worker × 6 + main × 1 = 7)
// ============================================================
typedef struct {
    int core_id;
    pthread_t thread;
    const Microcircuit* mc;
    const float* a;
    float* y;
} Worker;

typedef struct {
    Worker workers[NUM_WORKERS];
    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;
    _Atomic int should_exit;
} WorkerPool;

// Pool pointer untuk worker_main (akses barrier + exit flag)
static WorkerPool* g_pool = NULL;

static void* worker_main(void* arg) {
    Worker* w = (Worker*)arg;

    // Pin ke core spesifik
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    while (1) {
        // BLOCK sampai main set work + signal via barrier.
        // Saat block, thread di kernel sleep state, tidak konsumsi quota.
        pthread_barrier_wait(&g_pool->start_barrier);

        // Cek shutdown
        if (atomic_load_explicit(&g_pool->should_exit,
                                  memory_order_acquire)) {
            return NULL;
        }

        // Lakukan kerja
        microcircuit_forward(w->mc, w->a, w->y);

        // Signal selesai + block sampai main release
        pthread_barrier_wait(&g_pool->end_barrier);
    }
}

static int pool_init(WorkerPool* pool) {
    g_pool = pool;
    atomic_init(&pool->should_exit, 0);

    // Barrier untuk NUM_WORKERS + 1 (main) thread
    if (pthread_barrier_init(&pool->start_barrier, NULL,
                              NUM_WORKERS + 1) != 0) return -1;
    if (pthread_barrier_init(&pool->end_barrier, NULL,
                              NUM_WORKERS + 1) != 0) return -1;

    for (int i = 0; i < NUM_WORKERS; ++i) {
        pool->workers[i].core_id = i;
        pool->workers[i].mc = NULL;
        pool->workers[i].a = NULL;
        pool->workers[i].y = NULL;

        if (pthread_create(&pool->workers[i].thread, NULL,
                           worker_main, &pool->workers[i]) != 0) {
            fprintf(stderr, "pthread_create gagal worker %d\n", i);
            return -1;
        }
    }
    return 0;
}

static void pool_destroy(WorkerPool* pool) {
    // Signal shutdown
    atomic_store_explicit(&pool->should_exit, 1,
                          memory_order_release);
    // Wake all workers via start_barrier (mereka cek flag lalu exit)
    pthread_barrier_wait(&pool->start_barrier);

    for (int i = 0; i < NUM_WORKERS; ++i) {
        pthread_join(pool->workers[i].thread, NULL);
    }
    pthread_barrier_destroy(&pool->start_barrier);
    pthread_barrier_destroy(&pool->end_barrier);
    g_pool = NULL;
}

static void dispatch_step(WorkerPool* pool,
                          const Microcircuit* mcs[NUM_WORKERS],
                          const float* a,
                          float* ys[NUM_WORKERS]) {
    // Assign work
    for (int i = 0; i < NUM_WORKERS; ++i) {
        pool->workers[i].mc = mcs[i];
        pool->workers[i].a = a;
        pool->workers[i].y = ys[i];
    }
    // Wake all workers
    pthread_barrier_wait(&pool->start_barrier);
    // Workers process in parallel (main sleeps here)
    // Wait for all to finish
    pthread_barrier_wait(&pool->end_barrier);
}

// ============================================================
// Init helpers (sama dengan Phase 1)
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
        for (int c = 0; c < D_IN; ++c) M->W[r][c] = rand_int8();
        M->scales[r] = 0.001f;
        M->bias[r] = rand_float() * 0.1f;
    }
}
static void init_input(float* a) {
    for (int i = 0; i < D_IN; ++i) a[i] = rand_float();
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 2 v2: Multi-Core (Blocking Sync)\n");
    printf("================================================\n\n");

    // Allocate
    Microcircuit* mcs_storage[NUM_WORKERS];
    const Microcircuit* mcs[NUM_WORKERS];
    float* ys[NUM_WORKERS];
    for (int i = 0; i < NUM_WORKERS; ++i) {
        posix_memalign((void**)&mcs_storage[i], 64, sizeof(Microcircuit));
        init_microcircuit(mcs_storage[i]);
        mcs[i] = mcs_storage[i];
        posix_memalign((void**)&ys[i], 32, D_OUT * sizeof(float));
    }
    float* a;
    posix_memalign((void**)&a, 32, D_IN * sizeof(float));
    init_input(a);

    // BENCHMARK 1: Single-core BEFORE creating workers
    // (workers belum spawn, tidak ada kontensi)
    printf("Benchmark 1: Single-core (NO workers running)\n");
    {
        alignas(32) float y[D_OUT];
        for (int i = 0; i < 100; ++i) microcircuit_forward(mcs[0], a, y);
        double t0 = now_sec();
        for (int i = 0; i < 100000; ++i)
            microcircuit_forward(mcs[0], a, y);
        double t1 = now_sec();
        double per_call = (t1 - t0) * 1e6 / 100000;
        printf("  Per forward: %.2f µs\n", per_call);
    }
    printf("\n");

    // Init pool (workers sekarang spawn, tapi block di barrier)
    WorkerPool pool;
    if (pool_init(&pool) != 0) {
        fprintf(stderr, "Pool init failed\n");
        return 1;
    }
    usleep(10000);  // workers settle

    // Correctness
    {
        // Reference
        float* ys_ref[NUM_WORKERS];
        for (int i = 0; i < NUM_WORKERS; ++i) {
            posix_memalign((void**)&ys_ref[i], 32, D_OUT * sizeof(float));
            microcircuit_forward(mcs[i], a, ys_ref[i]);
        }
        // Parallel
        dispatch_step(&pool, mcs, a, ys);
        // Compare
        float max_diff = 0.0f; int errors = 0;
        for (int w = 0; w < NUM_WORKERS; ++w) {
            for (int i = 0; i < D_OUT; ++i) {
                float diff = fabsf(ys[w][i] - ys_ref[w][i]);
                if (diff > max_diff) max_diff = diff;
                if (diff > 1e-5f) errors++;
            }
        }
        printf("Dispatch correctness:\n");
        printf("  Max diff: %.6e, mismatches: %d/%d\n\n",
               max_diff, errors, NUM_WORKERS * D_OUT);
        for (int i = 0; i < NUM_WORKERS; ++i) free(ys_ref[i]);
        if (errors > 0) {
            fprintf(stderr, "Correctness FAIL, abort\n");
            pool_destroy(&pool);
            return 1;
        }
    }

    // BENCHMARK 2: Single-core WITH workers spawned (but sleeping)
    // Ini test: apakah blocking workers benar-benar tidak konsumsi quota?
    printf("Benchmark 2: Single-core (workers spawned, blocking)\n");
    {
        alignas(32) float y[D_OUT];
        for (int i = 0; i < 100; ++i) microcircuit_forward(mcs[0], a, y);
        double t0 = now_sec();
        for (int i = 0; i < 100000; ++i)
            microcircuit_forward(mcs[0], a, y);
        double t1 = now_sec();
        double per_call = (t1 - t0) * 1e6 / 100000;
        printf("  Per forward: %.2f µs\n", per_call);
        printf("  (Expected: sama dengan Benchmark 1 kalau workers benar-benar sleep)\n");
    }
    printf("\n");

    // BENCHMARK 3: 6-core dispatch
    printf("Benchmark 3: 6-core dispatch\n");
    // Iterations lebih rendah dulu untuk hindari hang risk
    int iters = 10000;
    for (int i = 0; i < 100; ++i) dispatch_step(&pool, mcs, a, ys);
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) dispatch_step(&pool, mcs, a, ys);
    double t1 = now_sec();
    double per_dispatch = (t1 - t0) * 1e6 / iters;
    printf("  Per dispatch (6 fwd paralel): %.2f µs\n", per_dispatch);
    printf("\n");

    // Analisis
    // Pakai Benchmark 1 sebagai true single-core baseline
    double t_single_clean = 0;  // akan diisi dari output Benchmark 1
    // For analysis, re-measure clean single-core one more time after pool exists
    {
        alignas(32) float y[D_OUT];
        double tt0 = now_sec();
        for (int i = 0; i < 100000; ++i)
            microcircuit_forward(mcs[0], a, y);
        double tt1 = now_sec();
        t_single_clean = (tt1 - tt0) * 1e6 / 100000;
    }

    printf("================================================\n");
    printf("  Analisis Scaling\n");
    printf("================================================\n");
    printf("Single-core (final measurement):   %.2f µs\n", t_single_clean);
    printf("6-core dispatch:                   %.2f µs\n", per_dispatch);
    printf("Sequential (6 × single):           %.2f µs\n", 6 * t_single_clean);
    double speedup = (6.0 * t_single_clean) / per_dispatch;
    double efficiency = speedup / NUM_WORKERS * 100.0;
    double overhead = per_dispatch - t_single_clean;
    printf("Speedup:                           %.2fx (ideal: 6x)\n", speedup);
    printf("Efficiency:                        %.1f%%\n", efficiency);
    printf("Dispatch overhead (barrier sync):  %.2f µs\n", overhead);
    printf("\n");

    if (efficiency > 70.0) {
        printf("[GOOD] Scaling acceptable untuk SML.\n");
    } else if (efficiency > 50.0) {
        printf("[MARGINAL] Throttling kemungkinan masih ada karena --cpus quota.\n");
        printf("           Test ulang dengan --cpus=6 untuk konfirmasi.\n");
    } else {
        printf("[POOR] Investigate: barrier overhead vs work duration ratio.\n");
    }

    pool_destroy(&pool);
    free(a);
    for (int i = 0; i < NUM_WORKERS; ++i) {
        free(ys[i]);
        free(mcs_storage[i]);
    }
    return 0;
}
