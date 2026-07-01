// ============================================================
// SML Phase 2 (v3): Hybrid Spin + Futex Dispatch
//
// v1: Pure spin — hang di bawah --cpus=4.8 karena 100% CPU
// v2: pthread_barrier — bekerja, tapi overhead 19 µs (mahal untuk 14 µs work)
// v3: Hybrid — spin briefly (fast dispatch), fallback futex (hemat quota)
//
// Untuk continuous inference (dispatch <10-20 µs interval): worker spinning,
//   dispatch latency ~100 ns. Ini path fast.
// Untuk idle >20 µs: worker sleep via futex, dispatch mahal (~5 µs wake).
//
// Build: gcc -O3 -march=native -mavx2 -mfma phase2_dispatch_v3.c \
//          -o phase2_dispatch_v3 -lpthread -lm
// Run:   ./phase2_dispatch_v3
// ============================================================

#define _GNU_SOURCE
#include <immintrin.h>
#include <linux/futex.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define D_IN        384
#define D_OUT       256
#define NUM_WORKERS 6

// Spin duration before fallback to futex sleep
// 500 iterasi × ~5 ns per PAUSE = ~2.5 µs spin window
// Cukup untuk cover 1 dispatch cycle jika dispatch datang cepat
#define SPIN_ITERATIONS 500

// ============================================================
// Futex helpers
// ============================================================
static inline int futex_wait(int* addr, int expected) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, expected, NULL, NULL, 0);
}
static inline int futex_wake(int* addr, int n_wake) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, n_wake, NULL, NULL, 0);
}

// ============================================================
// Microcircuit (copy)
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
// Sync design: generation counter + per-worker DONE flag
//
// Main increments `dispatch_gen`, wake all sleepers via 1 futex_wake.
// Workers spin-check `dispatch_gen`, fallback futex_wait if idle too long.
// Workers signal completion via atomic increment `completion_count`.
// Main spin-checks `completion_count`, no futex needed (brief window).
// ============================================================
typedef struct {
    // Per-worker context, aligned to own cache line
    alignas(64) int core_id;
    pthread_t thread;
    const Microcircuit* mc;
    const float* a;
    float* y;
    int last_seen_generation;
    struct WorkerPool_* pool;  // back-ref
    char _pad[16];  // pad to cache line
} Worker;

typedef struct WorkerPool_ {
    // Hot atomic: generation counter (touched by all)
    alignas(64) _Atomic int dispatch_gen;
    char _pad1[60];

    // Hot atomic: completion counter
    alignas(64) _Atomic int completion_count;
    char _pad2[60];

    // Shutdown flag
    alignas(64) _Atomic int should_exit;
    char _pad3[60];

    // Workers
    Worker workers[NUM_WORKERS];
} WorkerPool;

// ============================================================
// Worker main loop — HYBRID sync
// ============================================================
static void* worker_main(void* arg) {
    Worker* w = (Worker*)arg;
    WorkerPool* pool = w->pool;

    // Pin to core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(w->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    _Atomic int* gen_addr = &pool->dispatch_gen;
    w->last_seen_generation = 0;

    while (!atomic_load_explicit(&pool->should_exit,
                                  memory_order_acquire)) {
        // Wait for new generation (= new work)
        int cur_gen;

        // Phase 1: brief spin
        int i;
        for (i = 0; i < SPIN_ITERATIONS; ++i) {
            cur_gen = atomic_load_explicit(gen_addr,
                                            memory_order_acquire);
            if (cur_gen != w->last_seen_generation) goto has_work;
            __builtin_ia32_pause();
        }

        // Phase 2: futex sleep (idle > spin window)
        // Loop untuk handle spurious wakeup & should_exit
        while (1) {
            cur_gen = atomic_load_explicit(gen_addr,
                                            memory_order_acquire);
            if (cur_gen != w->last_seen_generation) goto has_work;
            if (atomic_load_explicit(&pool->should_exit,
                                      memory_order_acquire)) {
                return NULL;
            }
            // FUTEX_WAIT: atomically check gen == last_seen, sleep if so.
            // Cast _Atomic int* to int* — layout compatible for futex.
            futex_wait((int*)gen_addr, w->last_seen_generation);
        }

    has_work:
        w->last_seen_generation = cur_gen;

        // Do the work
        microcircuit_forward(w->mc, w->a, w->y);

        // Signal completion (atomic increment)
        atomic_fetch_add_explicit(&pool->completion_count, 1,
                                   memory_order_release);
    }
    return NULL;
}

// ============================================================
// Pool init/destroy
// ============================================================
static int pool_init(WorkerPool* pool) {
    atomic_init(&pool->dispatch_gen, 0);
    atomic_init(&pool->completion_count, 0);
    atomic_init(&pool->should_exit, 0);

    for (int i = 0; i < NUM_WORKERS; ++i) {
        pool->workers[i].core_id = i;
        pool->workers[i].pool = pool;
        pool->workers[i].mc = NULL;
        pool->workers[i].a = NULL;
        pool->workers[i].y = NULL;

        if (pthread_create(&pool->workers[i].thread, NULL,
                           worker_main, &pool->workers[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

static void pool_destroy(WorkerPool* pool) {
    atomic_store_explicit(&pool->should_exit, 1,
                          memory_order_release);
    // Wake any sleepers
    futex_wake((int*)&pool->dispatch_gen, INT32_MAX);

    for (int i = 0; i < NUM_WORKERS; ++i) {
        pthread_join(pool->workers[i].thread, NULL);
    }
}

// ============================================================
// Dispatch: set work + increment gen + wake sleepers + spin-wait done
// ============================================================
static void dispatch_step(WorkerPool* pool,
                          const Microcircuit* mcs[NUM_WORKERS],
                          const float* a,
                          float* ys[NUM_WORKERS]) {
    // 1. Set work fields untuk semua worker
    for (int i = 0; i < NUM_WORKERS; ++i) {
        pool->workers[i].mc = mcs[i];
        pool->workers[i].a  = a;
        pool->workers[i].y  = ys[i];
    }

    // 2. Reset completion counter
    atomic_store_explicit(&pool->completion_count, 0,
                          memory_order_relaxed);

    // 3. Increment generation (dengan release fence → workers see updated work)
    atomic_fetch_add_explicit(&pool->dispatch_gen, 1,
                              memory_order_release);

    // 4. Wake any sleeping workers (no-op untuk spinning workers)
    // Single syscall wakes all waiters on the address
    futex_wake((int*)&pool->dispatch_gen, NUM_WORKERS);

    // 5. Spin-wait for all workers done
    // Brief window (~14 µs), spin OK
    while (atomic_load_explicit(&pool->completion_count,
                                 memory_order_acquire) < NUM_WORKERS) {
        __builtin_ia32_pause();
    }
}

// ============================================================
// Init helpers
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
    printf("  SML Phase 2 v3: Hybrid Spin+Futex Dispatch\n");
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

    // BENCHMARK 1: Single-core baseline
    printf("Benchmark 1: Single-core baseline (no pool)\n");
    {
        alignas(32) float y[D_OUT];
        for (int i = 0; i < 100; ++i) microcircuit_forward(mcs[0], a, y);
        double t0 = now_sec();
        for (int i = 0; i < 100000; ++i)
            microcircuit_forward(mcs[0], a, y);
        double t1 = now_sec();
        double per_call = (t1 - t0) * 1e6 / 100000;
        printf("  Per forward: %.2f µs\n\n", per_call);
    }

    // Init pool
    WorkerPool* pool;
    posix_memalign((void**)&pool, 64, sizeof(WorkerPool));
    if (pool_init(pool) != 0) return 1;
    usleep(20000);  // workers settle di spin/sleep

    // Correctness
    {
        float* ys_ref[NUM_WORKERS];
        for (int i = 0; i < NUM_WORKERS; ++i) {
            posix_memalign((void**)&ys_ref[i], 32, D_OUT * sizeof(float));
            microcircuit_forward(mcs[i], a, ys_ref[i]);
        }
        dispatch_step(pool, mcs, a, ys);
        int errors = 0; float max_diff = 0;
        for (int w = 0; w < NUM_WORKERS; ++w) {
            for (int i = 0; i < D_OUT; ++i) {
                float diff = fabsf(ys[w][i] - ys_ref[w][i]);
                if (diff > max_diff) max_diff = diff;
                if (diff > 1e-5f) errors++;
            }
        }
        printf("Correctness: max diff %.6e, mismatches %d/%d\n\n",
               max_diff, errors, NUM_WORKERS * D_OUT);
        for (int i = 0; i < NUM_WORKERS; ++i) free(ys_ref[i]);
        if (errors > 0) { pool_destroy(pool); return 1; }
    }

    // BENCHMARK 2: Continuous dispatch (workers stay in spin phase)
    printf("Benchmark 2: 6-core continuous dispatch (hot path)\n");
    printf("  Workers stay in spin phase (dispatch < spin window ~2.5 µs)\n");
    for (int i = 0; i < 1000; ++i) dispatch_step(pool, mcs, a, ys);  // warmup
    {
        int iters = 100000;
        double t0 = now_sec();
        for (int i = 0; i < iters; ++i) dispatch_step(pool, mcs, a, ys);
        double t1 = now_sec();
        double per_dispatch = (t1 - t0) * 1e6 / iters;
        printf("  Per dispatch: %.2f µs\n\n", per_dispatch);
    }

    // BENCHMARK 3: Dispatch with delay (force workers to sleep between)
    printf("Benchmark 3: 6-core dispatch with 100 µs sleep (cold path)\n");
    printf("  Workers force ke futex sleep antar dispatch\n");
    {
        int iters = 1000;
        double t0 = now_sec();
        for (int i = 0; i < iters; ++i) {
            dispatch_step(pool, mcs, a, ys);
            usleep(100);  // 100 µs sleep → workers definitely sleep via futex
        }
        double t1 = now_sec();
        double per_iter = (t1 - t0) * 1e6 / iters;
        // Subtract the 100 µs sleep untuk dapat pure dispatch cost
        printf("  Per dispatch (subtract 100µs sleep): %.2f µs\n\n",
               per_iter - 100.0);
    }

    // Final single-core baseline
    double t_single;
    {
        alignas(32) float y[D_OUT];
        double t0 = now_sec();
        for (int i = 0; i < 100000; ++i)
            microcircuit_forward(mcs[0], a, y);
        double t1 = now_sec();
        t_single = (t0 == t1) ? 14.0 : (t1 - t0) * 1e6 / 100000;
    }

    printf("================================================\n");
    printf("  Perbandingan Design\n");
    printf("================================================\n");
    printf("v2 (pthread_barrier):       ~33 µs dispatch (efficiency 42%%)\n");
    printf("v3 (hybrid, hot path):      lihat Benchmark 2\n");
    printf("v3 (hybrid, cold path):     lihat Benchmark 3\n");
    printf("Single-core baseline:       %.2f µs\n", t_single);
    printf("\n");
    printf("Untuk SML inference kontinu, path yang relevan: Benchmark 2\n");
    printf("Karena cascade step tiap ~15-30 µs, worker selalu dalam spin.\n");

    pool_destroy(pool);
    free(a);
    for (int i = 0; i < NUM_WORKERS; ++i) {
        free(ys[i]);
        free(mcs_storage[i]);
    }
    free(pool);
    return 0;
}
