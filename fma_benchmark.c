// ============================================================
// SML Phase 0: FMA Throughput Benchmark (Corrected)
//
// Mengukur 4 hal:
//   1. Latency-bound throughput  (dependency chain)
//   2. Peak throughput            (8 independent accumulators)
//   3. Sustained throughput       (deteksi thermal throttling)
//   4. CPU frequency selama test  (deteksi throttle)
//
// Build: gcc -O3 -march=native -mavx2 -mfma fma_benchmark.c -o fma_benchmark
// Run:   ./fma_benchmark
// ============================================================

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

// Untuk baca CPU frequency aktual
static double read_cpu_mhz(void) {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0.0;
    char line[256];
    double mhz = 0.0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                mhz = atof(colon + 1);
                break;
            }
        }
    }
    fclose(f);
    return mhz;
}

static double elapsed_sec(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
}

// ============================================================
// Test 1: Latency-bound (single dependency chain)
// Mengukur batas bawah — apa yang terjadi kalau kode kita
// punya data dependency yang ketat
// ============================================================
static double test_latency_bound(long iterations) {
    __m256 a = _mm256_set1_ps(1.0001f);
    __m256 b = _mm256_set1_ps(0.9999f);
    __m256 c = _mm256_setzero_ps();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (long i = 0; i < iterations; ++i) {
        c = _mm256_fmadd_ps(a, b, c);  // depends on previous c
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = elapsed_sec(t0, t1);

    // Prevent dead code elimination
    float r[8]; _mm256_storeu_ps(r, c);
    if (r[0] == -42424242.0f) printf("impossible\n");

    return (iterations * 4.0 * 16.0) / sec / 1e9;  // GFLOPS
}

// ============================================================
// Test 2: Peak throughput (8 independent accumulators)
// Mengukur batas atas — best case skenario
// SML inner loop akan ada di antara test 1 dan test 2
// ============================================================
static double test_peak_throughput(long iterations) {
    __m256 a = _mm256_set1_ps(1.0001f);
    __m256 b = _mm256_set1_ps(0.9999f);
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (long i = 0; i < iterations; ++i) {
        c0 = _mm256_fmadd_ps(a, b, c0);
        c1 = _mm256_fmadd_ps(a, b, c1);
        c2 = _mm256_fmadd_ps(a, b, c2);
        c3 = _mm256_fmadd_ps(a, b, c3);
        c4 = _mm256_fmadd_ps(a, b, c4);
        c5 = _mm256_fmadd_ps(a, b, c5);
        c6 = _mm256_fmadd_ps(a, b, c6);
        c7 = _mm256_fmadd_ps(a, b, c7);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = elapsed_sec(t0, t1);

    // Prevent dead code elimination — sum all accumulators
    __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(c0, c1),
                                              _mm256_add_ps(c2, c3)),
                                _mm256_add_ps(_mm256_add_ps(c4, c5),
                                              _mm256_add_ps(c6, c7)));
    float r[8]; _mm256_storeu_ps(r, sum);
    if (r[0] == -42424242.0f) printf("impossible\n");

    return (iterations * 8.0 * 16.0) / sec / 1e9;  // GFLOPS
}

// ============================================================
// Test 3: Sustained throughput (long duration)
// Cek apakah CPU T-series throttle setelah beberapa detik
// SML inferensi akan run kontinu, jadi sustained yang penting
// ============================================================
static void test_sustained(int duration_sec) {
    __m256 a = _mm256_set1_ps(1.0001f);
    __m256 b = _mm256_set1_ps(0.9999f);
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();

    printf("    Sample tiap detik (GFLOPS):  ");
    fflush(stdout);

    double total_gflops = 0.0;
    int samples = 0;

    for (int sec = 0; sec < duration_sec; ++sec) {
        const long iters_per_sec = 100000000L;  // tune sampai ~1 sec
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (long i = 0; i < iters_per_sec; ++i) {
            c0 = _mm256_fmadd_ps(a, b, c0);
            c1 = _mm256_fmadd_ps(a, b, c1);
            c2 = _mm256_fmadd_ps(a, b, c2);
            c3 = _mm256_fmadd_ps(a, b, c3);
            c4 = _mm256_fmadd_ps(a, b, c4);
            c5 = _mm256_fmadd_ps(a, b, c5);
            c6 = _mm256_fmadd_ps(a, b, c6);
            c7 = _mm256_fmadd_ps(a, b, c7);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = elapsed_sec(t0, t1);
        double gflops = (iters_per_sec * 8.0 * 16.0) / elapsed / 1e9;

        printf("%.1f ", gflops);
        fflush(stdout);
        total_gflops += gflops;
        samples++;
    }
    printf("\n");

    // Prevent DCE
    __m256 sum = _mm256_add_ps(c0, c7);
    float r[8]; _mm256_storeu_ps(r, sum);
    if (r[0] == -42424242.0f) printf("impossible\n");

    printf("    Average sustained: %.2f GFLOPS\n", total_gflops / samples);

    double final_mhz = read_cpu_mhz();
    printf("    CPU MHz setelah test: %.0f\n", final_mhz);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("================================================\n");
    printf("  SML Phase 0: FMA Throughput Benchmark\n");
    printf("================================================\n\n");

    double initial_mhz = read_cpu_mhz();
    printf("CPU MHz saat start: %.0f\n\n", initial_mhz);

    // --- Test 1: Latency-bound ---
    printf("Test 1: Latency-bound (single accumulator)\n");
    printf("  Mengukur worst case dengan dependency chain ketat.\n");
    printf("  Expected: ~8 GFLOPS (4-cycle FMA latency / 2.1 GHz)\n");
    double gf1 = test_latency_bound(250000000L);
    printf("  Hasil: %.2f GFLOPS\n\n", gf1);

    // --- Test 2: Peak throughput ---
    printf("Test 2: Peak throughput (8 independent accumulators)\n");
    printf("  Mengukur best case dengan ILP penuh.\n");
    printf("  Theoretical peak: 2 FMA/cycle x 16 FLOP x 2.1 GHz = 67 GFLOPS\n");
    printf("  Realistic Skylake target: 40-60 GFLOPS sustained\n");
    double gf2 = test_peak_throughput(125000000L);
    printf("  Hasil: %.2f GFLOPS\n\n", gf2);

    // --- Test 3: Sustained ---
    printf("Test 3: Sustained throughput (10 detik continuous)\n");
    printf("  Deteksi thermal throttling pada CPU T-series.\n");
    test_sustained(10);
    printf("\n");

    // --- Analisis ---
    printf("================================================\n");
    printf("  Analisis\n");
    printf("================================================\n");

    double speedup = gf2 / gf1;
    printf("Speedup (peak / latency-bound): %.1fx\n", speedup);
    if (speedup > 4.0) {
        printf("  -> ILP works dengan baik, CPU sehat\n");
    } else {
        printf("  -> ILP terbatas, mungkin ada bottleneck\n");
    }

    printf("\nImplikasi untuk SML spec:\n");
    if (gf2 > 40.0) {
        printf("  [PASS] Peak >40 GFLOPS single-core.\n");
        printf("         Target estimasi spec (24 GFLOPS) tercapai dengan margin.\n");
        printf("         Lanjut ke Fase 0 berikutnya: cache bandwidth benchmark.\n");
    } else if (gf2 > 20.0) {
        printf("  [MARGINAL] Peak %.1f GFLOPS — di bawah ekspektasi.\n", gf2);
        printf("             Mungkin perlu konservatif di estimasi performance SML.\n");
        printf("             Revise spec ke target 15 GFLOPS per core sustainable.\n");
    } else {
        printf("  [FAIL] Peak <20 GFLOPS — ada masalah.\n");
        printf("         Investigasi: governor performance? Container limit?\n");
        printf("         Thermal? Coba run di luar Docker untuk compare.\n");
    }

    return 0;
}
