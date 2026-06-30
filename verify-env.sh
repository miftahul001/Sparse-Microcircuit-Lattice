#!/usr/bin/env bash
# ============================================================
# SML Environment Verification
# Jalankan ini SEGERA setelah container pertama kali masuk.
# Confirms AVX2, GCC, cache info, tools — semua yang spec butuhkan.
# ============================================================

set -uo pipefail  # tidak set -e karena beberapa cek bisa fail tanpa fatal

PASS="\033[32m✓\033[0m"
FAIL="\033[31m✗\033[0m"
WARN="\033[33m!\033[0m"

echo "================================================"
echo "  SML Environment Verification"
echo "================================================"
echo

# ============================================================
# 1. CPU & SIMD capability
# ============================================================
echo "--- CPU & SIMD ---"
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -n1 | cut -d: -f2 | xargs)
echo "CPU Model       : ${CPU_MODEL}"

CORES_AVAILABLE=$(nproc)
echo "Cores tersedia  : ${CORES_AVAILABLE}"

if grep -q "avx2" /proc/cpuinfo; then
    echo -e "${PASS} AVX2 available (critical untuk SML inner loop)"
else
    echo -e "${FAIL} AVX2 TIDAK ADA — desain SML tidak bisa lanjut!"
fi

if grep -q "fma" /proc/cpuinfo; then
    echo -e "${PASS} FMA available (perlu untuk _mm256_fmadd_ps)"
else
    echo -e "${FAIL} FMA TIDAK ADA — performa akan drop ~2x"
fi

if grep -q "avx512" /proc/cpuinfo; then
    echo -e "${WARN} AVX-512 detected — bisa upgrade desain nanti"
fi

echo

# ============================================================
# 2. Cache hierarchy — kritis untuk desain mikrosirkit 192KB
# ============================================================
echo "--- Cache Hierarchy ---"
if command -v lscpu &> /dev/null; then
    lscpu | grep -E "(L1d|L1i|L2|L3) cache"
else
    echo "Manual check:"
    cat /sys/devices/system/cpu/cpu0/cache/index*/size 2>/dev/null | head -4
fi
echo

# ============================================================
# 3. Memory & resource limits
# ============================================================
echo "--- Memory ---"
MEM_TOTAL_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
MEM_TOTAL_GB=$(echo "scale=1; ${MEM_TOTAL_KB}/1024/1024" | bc)
echo "Memory total    : ${MEM_TOTAL_GB} GB"

# Cgroup memory limit (container limit)
if [ -f /sys/fs/cgroup/memory.max ]; then
    MEM_LIMIT=$(cat /sys/fs/cgroup/memory.max)
    if [ "${MEM_LIMIT}" = "max" ]; then
        echo -e "${WARN} Memory limit  : tidak diset (--memory flag?)"
    else
        MEM_LIMIT_GB=$(echo "scale=1; ${MEM_LIMIT}/1024/1024/1024" | bc)
        echo "Memory limit    : ${MEM_LIMIT_GB} GB"
    fi
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    # cgroup v1 fallback
    MEM_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    MEM_LIMIT_GB=$(echo "scale=1; ${MEM_LIMIT}/1024/1024/1024" | bc)
    echo "Memory limit    : ${MEM_LIMIT_GB} GB (cgroup v1)"
fi

# memlock ulimit (perlu untuk mmap 23GB)
MEMLOCK=$(ulimit -l)
if [ "${MEMLOCK}" = "unlimited" ]; then
    echo -e "${PASS} memlock       : unlimited (OK untuk lattice mmap)"
else
    echo -e "${FAIL} memlock       : ${MEMLOCK} KB — Anda mungkin perlu --ulimit memlock=-1:-1"
fi
echo

# ============================================================
# 4. Compiler & build tools
# ============================================================
echo "--- Compiler ---"
GCC_VER=$(gcc --version | head -n1)
echo "GCC             : ${GCC_VER}"

CLANG_VER=$(clang --version | head -n1)
echo "Clang           : ${CLANG_VER}"

# Cek apakah GCC ngerti AVX2
if echo "int main(){return 0;}" | gcc -mavx2 -mfma -x c - -o /tmp/avx_test 2>/dev/null; then
    echo -e "${PASS} GCC -mavx2 -mfma compile OK"
    rm -f /tmp/avx_test
else
    echo -e "${FAIL} GCC tidak bisa target AVX2"
fi

# Cek march=native
if echo "int main(){return 0;}" | gcc -march=native -x c - -o /tmp/native_test 2>/dev/null; then
    NATIVE_FLAGS=$(gcc -march=native -E -v - </dev/null 2>&1 | grep cc1 | grep -oP -- '-march=\S+' | head -n1)
    echo -e "${PASS} -march=native works (akan target: ${NATIVE_FLAGS:-unknown})"
    rm -f /tmp/native_test
fi
echo

# ============================================================
# 5. Profiling tools
# ============================================================
echo "--- Profiling ---"
if command -v perf &> /dev/null; then
    PERF_VER=$(perf --version 2>/dev/null || echo "installed")
    echo -e "${PASS} perf available: ${PERF_VER}"

    # Cek apakah perf bisa jalan (perf_event_paranoid)
    if perf stat -e cycles true 2>/dev/null; then
        echo -e "${PASS} perf events readable"
    else
        echo -e "${WARN} perf events di-restrict. Di HOST jalankan:"
        echo "        sudo sysctl -w kernel.perf_event_paranoid=1"
        echo "        (atau =0 untuk lebih permissive)"
    fi
else
    echo -e "${FAIL} perf tidak terinstall"
fi

if command -v valgrind &> /dev/null; then
    echo -e "${PASS} valgrind: $(valgrind --version)"
fi

if command -v hyperfine &> /dev/null; then
    echo -e "${PASS} hyperfine: $(hyperfine --version)"
fi
echo

# ============================================================
# 6. Quick AVX2 micro-test — verify FMA throughput
# ============================================================
echo "--- Quick AVX2 Sanity Test ---"
cat > /tmp/avx_sanity.c << 'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <time.h>

int main() {
    __m256 a = _mm256_set1_ps(1.0001f);
    __m256 b = _mm256_set1_ps(0.9999f);
    __m256 c = _mm256_setzero_ps();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // 1 milyar FMA, unroll 4
    for (long i = 0; i < 250000000L; ++i) {
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
        c = _mm256_fmadd_ps(a, b, c);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // 1 FMA = 16 FLOP (8 mul + 8 add); 1e9 FMA total
    double gflops = (1e9 * 16) / elapsed / 1e9;

    // Print c untuk prevent optimization
    float result[8];
    _mm256_storeu_ps(result, c);

    printf("Elapsed: %.3f sec\n", elapsed);
    printf("Single-core FMA throughput: %.2f GFLOPS\n", gflops);
    printf("(result[0]=%.2f untuk prevent dead code)\n", result[0]);

    if (gflops > 15.0) {
        printf("RESULT: PASS — sesuai estimasi SML spec\n");
        return 0;
    } else if (gflops > 10.0) {
        printf("RESULT: MARGINAL — di bawah target tapi masih usable\n");
        return 0;
    } else {
        printf("RESULT: FAIL — perlu investigasi (CPU throttle? container limit?)\n");
        return 1;
    }
}
EOF

gcc -O3 -march=native -mavx2 -mfma /tmp/avx_sanity.c -o /tmp/avx_sanity 2>/dev/null
if [ -x /tmp/avx_sanity ]; then
    /tmp/avx_sanity
    rm -f /tmp/avx_sanity /tmp/avx_sanity.c
else
    echo -e "${FAIL} Compile error pada AVX sanity test"
fi
echo

echo "================================================"
echo "  Verifikasi selesai."
echo "  Kalau semua ${PASS}, Anda siap mulai Fase 0 dari spec."
echo "================================================"
