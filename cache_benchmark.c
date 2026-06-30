// ============================================================
// SML Phase 0: Memory & Cache Bandwidth Benchmark
//
// Mengukur bandwidth pada setiap level hierarki memory dengan
// access pattern yang mirror workload SML aktual.
//
// 5 test:
//   1. L2 bandwidth (single-core)    — kritis: mikrosirkit 192KB
//   2. L3 bandwidth (single-core)    — beberapa mikrosirkit hot
//   3. RAM bandwidth (single-core)   — cold microcircuit load
//   4. RAM bandwidth (6-core agg)    — cascade cold start, WORST CASE
//   5. L2 bandwidth (6-core agg)     — steady state inference, BEST CASE
//
// Build: gcc -O3 -march=native -mavx2 cache_benchmark.c -o cache_benchmark -lpthread
// Run:   ./cache_benchmark
// ============================================================

#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MICROCIRCUIT_SIZE   (192 * 1024)         // 192 KB — sesuai spec
#define L3_FIT              (4 * 1024 * 1024)    // 4 MB — fits in L3 (9MB shared)
#define RAM_SIZE            (256 * 1024 * 1024)  // 256 MB — well beyond L3
#define NUM_CORES           6

static double elapsed_sec(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
}

// ============================================================
// Load function: stream-read seluruh buffer dengan 8 parallel
// AVX2 streams. Return checksum supaya tidak di-DCE.
//
// Pattern: 8 stream paralel, 32 bytes each = 256 bytes per iter
// Ini matches SML's microcircuit row access (256 baris dari 384 elem)
// ============================================================
static inline __m256i stream_read_avx2(const uint8_t* buf, size_t size) {
    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    __m256i s2 = _mm256_setzero_si256();
    __m256i s3 = _mm256_setzero_si256();
    __m256i s4 = _mm256_setzero_si256();
    __m256i s5 = _mm256_setzero_si256();
    __m256i s6 = _mm256_setzero_si256();
    __m256i s7 = _mm256_setzero_si256();

    const __m256i* p = (const __m256i*)buf;
    size_t n = size / 256;  // 8 × 32 bytes per iter

    for (size_t i = 0; i < n; ++i) {
        s0 = _mm256_xor_si256(s0, _mm256_load_si256(p + 0));
        s1 = _mm256_xor_si256(s1, _mm256_load_si256(p + 1));
        s2 = _mm256_xor_si256(s2, _mm256_load_si256(p + 2));
        s3 = _mm256_xor_si256(s3, _mm256_load_si256(p + 3));
        s4 = _mm256_xor_si256(s4, _mm256_load_si256(p + 4));
        s5 = _mm256_xor_si256(s5, _mm256_load_si256(p + 5));
        s6 = _mm256_xor_si256(s6, _mm256_load_si256(p + 6));
        s7 = _mm256_xor_si256(s7, _mm256_load_si256(p + 7));
        p += 8;
    }

    __m256i a = _mm256_xor_si256(s0, s1);
    __m256i b = _mm256_xor_si256(s2, s3);
    __m256i c = _mm256_xor_si256(s4, s5);
    __m256i d = _mm256_xor_si256(s6, s7);
    return _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d));
}

// Sink untuk prevent DCE
static volatile uint64_t g_sink = 0;

static void consume(__m256i v) {
    uint64_t r[4];
    _mm256_storeu_si256((__m256i*)r, v);
    g_sink ^= r[0] ^ r[1] ^ r[2] ^ r[3];
}

// ============================================================
// Single-core bandwidth test
// Loop N kali, return GB/s
// ============================================================
static double measure_bandwidth_single(uint8_t* buf, size_t size, int passes) {
    // Warmup — bring into cache
    for (int i = 0; i < 5; ++i) consume(stream_read_avx2(buf, size));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < passes; ++i) {
        consume(stream_read_avx2(buf, size));
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = elapsed_sec(t0, t1);

    double bytes = (double)size * passes;
    return bytes / sec / 1e9;  // GB/s
}

// ============================================================
// Worker thread untuk aggregate test
// ============================================================
typedef struct {
    int core_id;
    uint8_t* buf;
    size_t size;
    int passes;
    double bandwidth;
    pthread_barrier_t* start_barrier;
    pthread_barrier_t* end_barrier;
} worker_arg_t;

static void* worker(void* arg_) {
    worker_arg_t* arg = (worker_arg_t*)arg_;

    // Pin to specific core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(arg->core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    // Warmup
    for (int i = 0; i < 5; ++i) consume(stream_read_avx2(arg->buf, arg->size));

    // Wait for all threads ready
    pthread_barrier_wait(arg->start_barrier);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < arg->passes; ++i) {
        consume(stream_read_avx2(arg->buf, arg->size));
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    pthread_barrier_wait(arg->end_barrier);

    double sec = elapsed_sec(t0, t1);
    double bytes = (double)arg->size * arg->passes;
    arg->bandwidth = bytes / sec / 1e9;

    return NULL;
}

// ============================================================
// Aggregate bandwidth: NUM_CORES threads, each with own buffer
// Mirror SML 6-mikrosirkit-per-cascade-step access pattern
// ============================================================
static double measure_bandwidth_aggregate(size_t per_thread_size, int passes) {
    pthread_t threads[NUM_CORES];
    worker_arg_t args[NUM_CORES];
    pthread_barrier_t start_barrier, end_barrier;

    pthread_barrier_init(&start_barrier, NULL, NUM_CORES);
    pthread_barrier_init(&end_barrier, NULL, NUM_CORES);

    // Allocate separate buffers per thread (avoid sharing/false sharing)
    for (int i = 0; i < NUM_CORES; ++i) {
        if (posix_memalign((void**)&args[i].buf, 64, per_thread_size) != 0) {
            fprintf(stderr, "alloc failed\n");
            exit(1);
        }
        memset(args[i].buf, 0xA5 ^ i, per_thread_size);
        args[i].core_id = i;
        args[i].size = per_thread_size;
        args[i].passes = passes;
        args[i].bandwidth = 0.0;
        args[i].start_barrier = &start_barrier;
        args[i].end_barrier = &end_barrier;
    }

    for (int i = 0; i < NUM_CORES; ++i) {
        pthread_create(&threads[i], NULL, worker, &args[i]);
    }

    double total = 0.0;
    for (int i = 0; i < NUM_CORES; ++i) {
        pthread_join(threads[i], NULL);
        total += args[i].bandwidth;
        free(args[i].buf);
    }

    pthread_barrier_destroy(&start_barrier);
    pthread_barrier_destroy(&end_barrier);

    return total;
}

// ============================================================
// Helpers untuk implikasi SML
// ============================================================
static void print_implication(const char* label, double bw_gbps) {
    double us_per_microcircuit = (MICROCIRCUIT_SIZE / 1024.0 / 1024.0 / 1024.0) / bw_gbps * 1e6;
    printf("  %s: %.1f GB/s  →  %.1f µs per mikrosirkit (192 KB)\n",
           label, bw_gbps, us_per_microcircuit);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 0: Memory Bandwidth Benchmark\n");
    printf("================================================\n\n");

    // ----------------------------------------------------------
    // Test 1: L2 bandwidth (single core)
    // ----------------------------------------------------------
    printf("Test 1: L2 bandwidth (single core, 192 KB working set)\n");
    printf("  Buffer = ukuran satu mikrosirkit. Fits dalam L2 (256 KB).\n");
    printf("  Mengukur: kecepatan stream weight dari L2 ke register.\n");
    {
        uint8_t* buf;
        posix_memalign((void**)&buf, 64, MICROCIRCUIT_SIZE);
        memset(buf, 0xA5, MICROCIRCUIT_SIZE);

        // ~20K passes untuk 100ms minimum
        double bw = measure_bandwidth_single(buf, MICROCIRCUIT_SIZE, 30000);
        print_implication("L2 single-core", bw);
        free(buf);
    }
    printf("\n");

    // ----------------------------------------------------------
    // Test 2: L3 bandwidth (single core)
    // ----------------------------------------------------------
    printf("Test 2: L3 bandwidth (single core, 4 MB working set)\n");
    printf("  Buffer > L2 tapi < L3 (9 MB). Data spill ke L3.\n");
    printf("  Mengukur: penalti kalau mikrosirkit tidak hot di L2.\n");
    {
        uint8_t* buf;
        posix_memalign((void**)&buf, 64, L3_FIT);
        memset(buf, 0xA5, L3_FIT);

        double bw = measure_bandwidth_single(buf, L3_FIT, 1500);
        print_implication("L3 single-core", bw);
        free(buf);
    }
    printf("\n");

    // ----------------------------------------------------------
    // Test 3: RAM bandwidth (single core)
    // ----------------------------------------------------------
    printf("Test 3: RAM bandwidth (single core, 256 MB working set)\n");
    printf("  Buffer >> L3. Setiap read kena DRAM.\n");
    printf("  Mengukur: cold cascade — pertama kali load mikrosirkit.\n");
    {
        uint8_t* buf;
        posix_memalign((void**)&buf, 64, RAM_SIZE);
        // Touch every page untuk hindari demand-paging selama test
        for (size_t i = 0; i < RAM_SIZE; i += 4096) buf[i] = 0xA5;

        double bw = measure_bandwidth_single(buf, RAM_SIZE, 50);
        print_implication("RAM single-core", bw);
        free(buf);
    }
    printf("\n");

    // ----------------------------------------------------------
    // Test 4: RAM aggregate bandwidth (6 cores parallel)
    // ----------------------------------------------------------
    printf("Test 4: RAM aggregate (6 cores parallel, 256 MB each)\n");
    printf("  6 thread, masing-masing read buffer 256 MB sendiri-sendiri.\n");
    printf("  Mengukur: WORST CASE SML — semua 6 core load cold microcircuit.\n");
    {
        double total = measure_bandwidth_aggregate(RAM_SIZE, 50);
        printf("  Aggregate RAM: %.1f GB/s\n", total);
        double cascade_step_us = (6.0 * MICROCIRCUIT_SIZE / 1024.0 / 1024.0 / 1024.0)
                                 / total * 1e6;
        printf("  → Cold cascade step (6 × 192 KB): %.1f µs\n", cascade_step_us);
    }
    printf("\n");

    // ----------------------------------------------------------
    // Test 5: L2 aggregate bandwidth (6 cores parallel)
    // ----------------------------------------------------------
    printf("Test 5: L2 aggregate (6 cores parallel, 192 KB each)\n");
    printf("  6 thread, masing-masing buffer fits di L2-nya sendiri.\n");
    printf("  Mengukur: STEADY STATE SML — mikrosirkit hot di L2.\n");
    {
        double total = measure_bandwidth_aggregate(MICROCIRCUIT_SIZE, 30000);
        printf("  Aggregate L2: %.1f GB/s\n", total);
        double warm_step_us = (6.0 * MICROCIRCUIT_SIZE / 1024.0 / 1024.0 / 1024.0)
                              / total * 1e6;
        printf("  → Warm cascade step (6 × 192 KB from L2): %.1f µs\n", warm_step_us);
    }
    printf("\n");

    // ----------------------------------------------------------
    // Analisis & implikasi
    // ----------------------------------------------------------
    printf("================================================\n");
    printf("  Implikasi untuk SML Inference Loop\n");
    printf("================================================\n");
    printf("Satu langkah cascade = 6 mikrosirkit forward paralel.\n");
    printf("Compute saja (66.8 GFLOPS × 6 core, 100K FLOP/mikrosirkit): ~1.5 µs\n");
    printf("Memory bottleneck akan mendominasi total step time.\n");
    printf("\n");
    printf("Skenario realistic:\n");
    printf("  - Cascade step 1 (cold)   : pakai angka Test 4\n");
    printf("  - Cascade step 2-N (warm) : pakai angka Test 5\n");
    printf("  - Total per inferensi     : 1 cold + (~5) warm = sum\n");
    printf("\n");
    printf("Volatile sink (DCE prevention): 0x%lx\n", g_sink);

    return 0;
}
