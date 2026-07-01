// ============================================================
// SML Phase 2: Multi-Core Dispatch
//
// Tujuan: 6 worker thread paralel, masing-masing forward
// satu mikrosirkit di core berbeda. Ukur scaling efficiency
// dan dispatch overhead.
//
// Dispatch mechanism: atomic int + spin-wait dengan PAUSE.
// Rationale: untuk workload 14 µs, semaphore syscall (~2 µs)
// akan dominasi. Atomic spin = ~50 ns dispatch latency.
//
// Build: gcc -O3 -march=native -mavx2 -mfma phase2_dispatch.c \
//          -o phase2_dispatch -lpthread -lm
// Run:   ./phase2_dispatch
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

// ============================================================
// Konstanta — dari spec & Phase 1
// ============================================================
#define D_IN       384
#define D_OUT      256
#define NUM_CORES  6

// Worker state
#define STATE_IDLE     0
#define STATE_WORKING  1
#define STATE_DONE     2

// ============================================================
// Microcircuit struct (copy dari Phase 1 — INT8 weights)
// ============================================================
typedef struct {
    alignas(64) int8_t W[D_OUT][D_IN];
    alignas(64) float  scales[D_OUT];
    alignas(64) float  bias[D_OUT];
} Microcircuit;

// ============================================================
// Helpers (copy dari Phase 1)
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

// ============================================================
// Microcircuit forward (copy dari Phase 1, tidak berubah)
// ============================================================
void microcircuit_forward(const Microcircuit* M,
                          const float* a,
                          float* y) {
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
// Worker — satu per core, persistent thread
// State machine: IDLE → WORKING → DONE → IDLE → ...
//
// CRITICAL: state field di-pad ke cache line sendiri untuk
// avoid false sharing antar worker. Tanpa padding, 6 atomic
// di 6 core akan bouncing cache line yang sama → bencana.
// ============================================================
typedef struct {
    // Cache line 1: state (hot, di-poll terus)
    alignas(64) _Atomic int state;
    char _pad_state[60];

    // Cache line 2: work assignment (cold setelah dispatch)
    alignas(64) const Microcircuit* mc;
    const float* a;
    float* y;
    char _pad_work[40];

    // Cache line 3: metadata (cold)
    alignas(64) int core_id;
    pthread_t thread;
    _Atomic int should_exit;
} Worker;

// Compile-time check: tiap Worker exclusive di cache line-nya
_Static_assert(sizeof(Worker) >= 192, "Worker harus span beberapa cache line");

// ============================================================
// Worker main loop — spin-wait dengan PAUSE
// ============================================================
static void* worker_main(void* arg) {
    Worker* w = (Worker*)arg;

    // Pin ke core spesifik
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(),
                                sizeof(cpuset), &cpuset) != 0) {
        fprintf(stderr, "Warning: gagal pin worker %d ke core %d\n",
                w->core_id, w->core_id);
    }

    while (!atomic_load_explicit(&w->should_exit, memory_order_acquire)) {
        // Spin sampai state = WORKING
        // PAUSE: hint ke CPU untuk hemat power & reduce memory order violation
        while (atomic_load_explicit(&w->state, memory_order_acquire)
               != STATE_WORKING) {
            __builtin_ia32_pause();
            if (atomic_load_explicit(&w->should_exit,
                                     memory_order_acquire)) {
                return NULL;
            }
        }

        // Lakukan kerja (data sudah di-set sebelum dispatcher
        // memodifikasi state)
        microcircuit_forward(w->mc, w->a, w->y);

        // Signal selesai
        atomic_store_explicit(&w->state, STATE_DONE,
                              memory_order_release);
    }

    return NULL;
}

// ============================================================
// Worker pool
// ============================================================
typedef struct {
    Worker workers[NUM_CORES];
} WorkerPool;

static int pool_init(WorkerPool* pool) {
    for (int i = 0; i < NUM_CORES; ++i) {
        memset(&pool->workers[i], 0, sizeof(Worker));
        pool->workers[i].core_id = i;
        atomic_init(&pool->workers[i].state, STATE_IDLE);
        atomic_init(&pool->workers[i].should_exit, 0);

        if (pthread_create(&pool->workers[i].thread, NULL,
                           worker_main, &pool->workers[i]) != 0) {
            fprintf(stderr, "pthread_create failed for worker %d\n", i);
            return -1;
        }
    }
    return 0;
}

static void pool_destroy(WorkerPool* pool) {
    for (int i = 0; i < NUM_CORES; ++i) {
        atomic_store_explicit(&pool->workers[i].should_exit, 1,
                              memory_order_release);
    }
    for (int i = 0; i < NUM_CORES; ++i) {
        pthread_join(pool->workers[i].thread, NULL);
    }
}

// ============================================================
// Dispatch: assign 6 mikrosirkit ke 6 worker, tunggu selesai
// ============================================================
static void dispatch_step(WorkerPool* pool,
                          const Microcircuit* mcs[NUM_CORES],
                          const float* a,
                          float* ys[NUM_CORES]) {
    // Set work assignment + trigger workers
    // Note: work fields ditulis SEBELUM state, dengan release semantics
    for (int i = 0; i < NUM_CORES; ++i) {
        pool->workers[i].mc = mcs[i];
        pool->workers[i].a  = a;
        pool->workers[i].y  = ys[i];
        atomic_store_explicit(&pool->workers[i].state, STATE_WORKING,
                              memory_order_release);
    }

    // Spin-wait sampai semua selesai
    for (int i = 0; i < NUM_CORES; ++i) {
        while (atomic_load_explicit(&pool->workers[i].state,
                                    memory_order_acquire) != STATE_DONE) {
            __builtin_ia32_pause();
        }
        // Reset ke IDLE untuk dispatch berikutnya
        atomic_store_explicit(&pool->workers[i].state, STATE_IDLE,
                              memory_order_relaxed);
    }
}

// ============================================================
// Initialization
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
// Benchmarks
// ============================================================
static double benchmark_single_core(const Microcircuit* M,
                                     const float* a, float* y,
                                     int iterations) {
    // Warmup
    for (int i = 0; i < 100; ++i) microcircuit_forward(M, a, y);

    double t0 = now_sec();
    for (int i = 0; i < iterations; ++i) microcircuit_forward(M, a, y);
    double t1 = now_sec();

    return (t1 - t0) * 1e6 / iterations;  // µs per call
}

static double benchmark_six_core(WorkerPool* pool,
                                  const Microcircuit* mcs[NUM_CORES],
                                  const float* a,
                                  float* ys[NUM_CORES],
                                  int iterations) {
    // Warmup
    for (int i = 0; i < 100; ++i) dispatch_step(pool, mcs, a, ys);

    double t0 = now_sec();
    for (int i = 0; i < iterations; ++i) dispatch_step(pool, mcs, a, ys);
    double t1 = now_sec();

    return (t1 - t0) * 1e6 / iterations;  // µs per dispatch
}

// Measure pure dispatch overhead: send work but worker does nothing
static double benchmark_empty_dispatch(WorkerPool* pool, int iterations) {
    // Temporarily redefine "work" — use null pointers, worker won't actually run
    // Actually, we can't easily make worker do nothing without modifying.
    // Alternative: measure dispatch + reset cycle by checking time AFTER
    // signaling but BEFORE waiting. But we want full round-trip.
    //
    // For now, use real work but compare to single-core baseline.
    (void)pool; (void)iterations;
    return 0.0;
}

// ============================================================
// Correctness check: output dari 6-core dispatch harus identik
// dengan 6× single-core sequential
// ============================================================
static int verify_dispatch_correctness(WorkerPool* pool,
                                        const Microcircuit* mcs[NUM_CORES],
                                        const float* a,
                                        float* ys_parallel[NUM_CORES]) {
    // Reference: sequential
    float* ys_ref[NUM_CORES];
    for (int i = 0; i < NUM_CORES; ++i) {
        posix_memalign((void**)&ys_ref[i], 32, D_OUT * sizeof(float));
        microcircuit_forward(mcs[i], a, ys_ref[i]);
    }

    // Parallel
    dispatch_step(pool, mcs, a, ys_parallel);

    // Compare
    int errors = 0;
    float max_diff = 0.0f;
    for (int w = 0; w < NUM_CORES; ++w) {
        for (int i = 0; i < D_OUT; ++i) {
            float diff = fabsf(ys_parallel[w][i] - ys_ref[w][i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-5f) errors++;
        }
    }

    for (int i = 0; i < NUM_CORES; ++i) free(ys_ref[i]);

    printf("Dispatch correctness:\n");
    printf("  Max diff vs sequential: %.6e\n", max_diff);
    printf("  Mismatches: %d / %d\n", errors, NUM_CORES * D_OUT);
    return errors == 0;
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 2: Multi-Core Dispatch\n");
    printf("================================================\n\n");

    // Allocate 6 mikrosirkit (semua identik untuk simplicity benchmark)
    // Real SML akan punya 120K, tapi 6 cukup untuk test dispatch
    Microcircuit* mcs_storage[NUM_CORES];
    const Microcircuit* mcs[NUM_CORES];
    float* ys[NUM_CORES];

    for (int i = 0; i < NUM_CORES; ++i) {
        posix_memalign((void**)&mcs_storage[i], 64, sizeof(Microcircuit));
        init_microcircuit(mcs_storage[i]);
        mcs[i] = mcs_storage[i];

        posix_memalign((void**)&ys[i], 32, D_OUT * sizeof(float));
    }

    float* a;
    posix_memalign((void**)&a, 32, D_IN * sizeof(float));
    init_input(a);

    // Init worker pool
    WorkerPool pool;
    if (pool_init(&pool) != 0) {
        fprintf(stderr, "Pool init failed\n");
        return 1;
    }

    // Allow workers to settle (pin, ready)
    usleep(10000);  // 10ms

    // Verify correctness
    if (!verify_dispatch_correctness(&pool, mcs, a, ys)) {
        fprintf(stderr, "Correctness FAILED, abort\n");
        pool_destroy(&pool);
        return 1;
    }
    printf("\n");

    // -----------------------------------------------------------
    // Benchmark 1: Single-core baseline (reproduce Phase 1)
    // -----------------------------------------------------------
    printf("Benchmark 1: Single-core baseline\n");
    double t_single = benchmark_single_core(mcs[0], a, ys[0], 100000);
    printf("  Per forward (single core, no thread): %.2f µs\n", t_single);
    printf("\n");

    // -----------------------------------------------------------
    // Benchmark 2: 6-core dispatch
    // -----------------------------------------------------------
    printf("Benchmark 2: 6-core dispatch\n");
    double t_six = benchmark_six_core(&pool, mcs, a, ys, 100000);
    printf("  Per dispatch step (6 fwd paralel): %.2f µs\n", t_six);
    printf("\n");

    // -----------------------------------------------------------
    // Analisis scaling
    // -----------------------------------------------------------
    double speedup = (6.0 * t_single) / t_six;
    double efficiency = speedup / NUM_CORES * 100.0;
    double dispatch_overhead = t_six - t_single;

    printf("================================================\n");
    printf("  Scaling Analysis\n");
    printf("================================================\n");
    printf("Sequential (6 × single):     %.2f µs\n", 6 * t_single);
    printf("Parallel (6-core dispatch):  %.2f µs\n", t_six);
    printf("Speedup:                     %.2fx (ideal: 6x)\n", speedup);
    printf("Efficiency:                  %.1f%%\n", efficiency);
    printf("Dispatch overhead:           %.2f µs\n", dispatch_overhead);
    printf("\n");

    printf("Implikasi untuk SML inference:\n");
    if (efficiency > 85.0) {
        printf("  [EXCELLENT] Scaling near-ideal.\n");
        printf("  Cascade step warm: ~%.0f µs (matches t_six)\n", t_six);
        printf("  Next: Phase 3 (LSH routing)\n");
    } else if (efficiency > 65.0) {
        printf("  [GOOD] Scaling acceptable.\n");
        printf("  Dispatch overhead %.1f µs adalah ~%.0f%% dari compute.\n",
               dispatch_overhead, 100.0 * dispatch_overhead / t_single);
        printf("  Cascade step warm: ~%.0f µs\n", t_six);
        printf("  Next: Phase 3 atau optimize dispatch dulu\n");
    } else {
        printf("  [POOR] Scaling buruk. Investigate:\n");
        printf("  - Apakah workers benar-benar pinned? (cek top -H)\n");
        printf("  - False sharing di Worker struct?\n");
        printf("  - Spin-wait overhead vs work duration?\n");
    }

    // Cleanup
    pool_destroy(&pool);
    free(a);
    for (int i = 0; i < NUM_CORES; ++i) {
        free(ys[i]);
        free(mcs_storage[i]);
    }
    return 0;
}
