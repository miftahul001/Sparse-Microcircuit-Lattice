# SML: Sparse Microcircuit Lattice
### Spesifikasi Arsitektur Neural Topology untuk CPU 6-Core / 32GB RAM

---

**Versi**: 0.1 (Draft Awal)
**Status**: Design Document — Belum diimplementasikan
**Target Host**: HP EliteDesk 800 G4 (Intel i5-8500T, 32GB DDR4, 512GB NVMe, no GPU)
**Ekosistem**: C++ (dengan AVX2 intrinsics, pthread, mmap)
**Penulis Diskusi**: Miftahul Munir + Claude
**Tanggal**: 30 Juni 2026
**Lisensi**: TBD (placeholder untuk keputusan akhir)

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Batasan Hardware sebagai Boundary Condition](#2-batasan-hardware-sebagai-boundary-condition)
3. [Pilihan Ekosistem & Justifikasi](#3-pilihan-ekosistem--justifikasi)
4. [Topologi: Sparse Microcircuit Lattice](#4-topologi-sparse-microcircuit-lattice)
5. [State Transition: Content-Addressed Activation Cascade](#5-state-transition-content-addressed-activation-cascade)
6. [Mekanisme Emergensi](#6-mekanisme-emergensi)
7. [Formulasi Matematis](#7-formulasi-matematis)
8. [Code Flow Reference](#8-code-flow-reference)
9. [Estimasi Performa](#9-estimasi-performa)
10. [Roadmap Implementasi Bertahap](#10-roadmap-implementasi-bertahap)
11. [Open Questions & Risk Register](#11-open-questions--risk-register)
12. [Hubungan dengan ARC-PS](#12-hubungan-dengan-arc-ps)
13. [Glosarium](#13-glosarium)
14. [Referensi & Bacaan Lanjut](#14-referensi--bacaan-lanjut)

---

## 1. Ringkasan Eksekutif

SML (Sparse Microcircuit Lattice) adalah arsitektur neural alternatif yang menolak asumsi inti dari Transformer dan MoE: bahwa komputasi neural memerlukan paralelisme matriks tak terbatas pada GPU. SML dirancang untuk **bertahan hidup** — bukan mendominasi benchmark — pada CPU 6-core dengan 32GB RAM, dengan mempertahankan kapasitas representasi tinggi melalui geometri small-world graph dari ~120.000 mikrosirkit independen.

**Tiga klaim inti**:

1. **Memory bandwidth amortization**: Setiap mikrosirkit menjalankan ~50.000 operasi MAC dari L2 cache, sehingga rasio (compute : memory read) tinggi — menghindari kelaparan bandwidth DDR4.
2. **Content-addressed routing tanpa router yang dipelajari**: Lokality-Sensitive Hashing (LSH) menghasilkan jalur traversal yang berbeda per-input tanpa parameter gating tambahan.
3. **Sub-linear scaling per input**: Input mudah konvergen dalam 2-3 iterasi (~200µs); input kompleks bisa 20-50 iterasi (~5ms). Compute proporsional kesulitan masalah — properti yang tidak dimiliki Transformer.

**Yang tidak diklaim**:

- SML akan mengalahkan GPT-4 pada MMLU. Tidak akan.
- Pelatihan SML lewat backprop konvensional adalah trivial. Bukan.
- Arsitektur ini siap implementasi. Belum — masih ada open questions kritis di [Bagian 11](#11-open-questions--risk-register).

---

## 2. Batasan Hardware sebagai Boundary Condition

Setiap keputusan arsitektur turun dari pembatas ini:

| Komponen | Spesifikasi | Implikasi Arsitektur |
|----------|-------------|----------------------|
| CPU | Intel i5-8500T, 6 core / 6 thread, base 2.10 GHz | Maksimal 6 worker paralel; tidak ada SMT untuk over-subscribe |
| ISA | x86_64 dengan AVX2 (no AVX-512) | Vektor 256-bit; 8 FP32 atau 16 INT16 per instruksi |
| L1d | 32 KB per core | Activation working set < 32 KB |
| L2 | 256 KB per core | **Target ukuran mikrosirkit: ~192 KB** (sisa untuk activation+stack) |
| L3 | 9 MB shared | Buffer komunikasi inter-core; ~50 mikrosirkit hot |
| RAM | 32 GB DDR4, ~30 GB/s peak bandwidth | Total weight footprint ≤ 23 GB (sisa untuk OS+activations+overhead) |
| NVMe | 512 GB | `mmap` weight file; OS handles paging untuk cold microcircuits |

**Konsekuensi non-negotiable**:

- Setiap operasi inferensi yang membutuhkan >1 KB membaca per MAC akan **memory-bound**, bukan compute-bound. Ini batas keras termodinamika.
- Dengan AVX2 FMA throughput ~24 GFLOPS efektif × 6 core = ~144 GFLOPS teoretis, tetapi praktis ~60-80 GFLOPS karena memory stalls. Desain harus memaksimalkan FLOP per byte dibaca.

---

## 3. Pilihan Ekosistem & Justifikasi

**Pilihan: C++ (C++17 minimum, C++20 preferred), dependensi minimal.**

### Alasan menolak Node.js

Node.js bukan pilihan buruk untuk orkestrasi atau plugin modularity, tetapi pertanyaan kelangsungan hidup SML bukan tentang throughput task asinkron, melainkan tentang **bandwidth memori per joule**. Hambatan konkret:

| Faktor | Node.js | C++ |
|--------|---------|-----|
| AVX2 intrinsics | Lewat N-API binding (overhead) | Native (`<immintrin.h>`) |
| Memory alignment 64-byte | Tidak dijamin (Buffer) | `posix_memalign`, `alignas(64)` |
| GC pauses | Ada (V8) | Tidak ada |
| Zero-copy mmap | Sulit (Buffer abstraction) | Langsung (`mmap` + pointer arithmetic) |
| Thread affinity | Terbatas (Worker Threads) | `pthread_setaffinity_np` |
| Latency determinisme | µs-ms variance | ns-µs variance |

### Dependensi yang diperbolehkan

- **Standar**: `pthread`, `mmap`, `<immintrin.h>`, `<cstdint>`, `<atomic>`
- **Opsional**: `xsimd` atau `highway` untuk portability SIMD (jika nanti perlu cross-arch); awalnya AVX2 intrinsics langsung saja
- **Hindari**: Eigen, Boost, OpenMP (terlalu banyak abstraksi yang tidak dikontrol manual)

### Build tooling

- **Compiler**: GCC 11+ atau Clang 14+
- **Flags wajib**: `-O3 -march=native -mavx2 -mfma -funroll-loops`
- **Build system**: CMake (sederhana) atau plain Makefile (preferensi)
- **Test**: GoogleTest atau Catch2 (pilihan terbuka)

---

## 4. Topologi: Sparse Microcircuit Lattice

### 4.1 Anatomi Mikrosirkit

Setiap mikrosirkit `μ_k` adalah struktur self-contained berukuran **192 KB**, dirancang muat di L2 cache (256 KB) dengan ruang sisa untuk activation dan stack:

```cpp
struct alignas(64) Microcircuit {
    int8_t   W[256][384];           // 96 KB  — bobot INT4 dikemas 2/byte
    fp16_t   scales[256];           // 0.5 KB — skala dequantisasi per-baris
    fp16_t   bias[256];             // 0.5 KB
    uint32_t neighbor_lsh[16];      // 64 B   — pointer ke tetangga (12 lokal + 4 shortcut)
    uint32_t self_hash;             // 4 B    — content-addressed identity
    fp16_t   resonance_vec[256];    // 0.5 KB — signature untuk routing
    uint8_t  _padding[];            // padding ke 192 KB
};
// Footprint efektif: 100K parameter per mikrosirkit (post-INT4)
```

### 4.2 Geometri Makro

| Properti | Nilai | Justifikasi |
|----------|-------|-------------|
| Jumlah mikrosirkit | ~120.000 | 23 GB / 192 KB ≈ 120K |
| Total parameter (FP32 equiv) | ~12 miliar | 100K × 120K |
| Footprint RAM aktual | ~23 GB | INT4 quantization |
| Konektivitas per node | 16 (12 lokal + 4 shortcut) | Small-world: clustering tinggi + diameter logaritmik |
| Diameter graf | ~log(120K) ≈ 17 | Reachability semua mikrosirkit dalam <20 hop |

**Topologi small-world** (Watts-Strogatz inspired):

- **12 tetangga lokal**: Mikrosirkit dengan `resonance_vec` terdekat (Hamming distance kecil di LSH space) — mendorong specialization klaster.
- **4 tetangga shortcut**: Acak terdistribusi — memungkinkan analogical jumps lintas domain.

### 4.3 Memory Layout di RAM

```
┌────────────────────────────────────────────────────────────┐
│ Region 0: Lattice (mmap dari file, read-only)              │
│ Offset 0x00000000: μ_0     (192 KB)                        │
│ Offset 0x00030000: μ_1     (192 KB)                        │
│ ...                                                         │
│ Offset 0x5A000000: μ_120000 (192 KB)                       │
│ Total: ~23 GB                                              │
├────────────────────────────────────────────────────────────┤
│ Region 1: LSH Index (in-memory hash table)                 │
│ SimHash 32-bit → list of microcircuit IDs                  │
│ Total: ~50 MB                                              │
├────────────────────────────────────────────────────────────┤
│ Region 2: Activation Workspace (per-thread)                │
│ 6 × (input buffer + output buffer + scratch) ≈ 6 × 8 KB    │
│ Total: ~50 KB                                              │
├────────────────────────────────────────────────────────────┤
│ Region 3: OS + heap + stack                                │
│ Total: ~7-8 GB available                                   │
└────────────────────────────────────────────────────────────┘
```

**Strategi cache locality**:

1. Worker thread mengambil pointer mikrosirkit → CPU prefetch otomatis ke L2.
2. Inner loop sepenuhnya bekerja dari L2 (96 KB bobot + 1 KB scales/bias muat ringan).
3. Hanya output (1 KB) ditulis ke memory shared via L3.

---

## 5. State Transition: Content-Addressed Activation Cascade

### 5.1 Satu Langkah Cascade

Diberikan state aktivasi `a ∈ R^256`:

```
Step 1: Hash konten (~200 ns)
    sig := SimHash_AVX2(a)        // 32-bit signature

Step 2: Resolusi tetangga (~500 ns)
    candidates := LSH_lookup(sig)  // ~30 mikrosirkit kandidat
    top6      := select_top_k(candidates, a, k=6)

Step 3: Dispatch paralel ke 6 core (~30 µs)
    for c in 0..5 parallel:
        y[c] := microcircuit_forward(μ[top6[c]], a)

Step 4: Sinkronisasi (futex wait)
    wait_all_workers()

Step 5: Fusi energy-weighted (~2 µs)
    weights[c] := ||y[c]||² / Σ_j ||y[j]||²
    a_new := Σ_c weights[c] · y[c]

Step 6: Cek konvergensi
    if ||a_new - a|| < ε:
        return decode(a_new)
    else:
        a := a_new; goto Step 1
```

### 5.2 Sifat Penting

- **Tidak ada router yang dipelajari**. Routing emergent dari LSH atas content.
- **Tidak ada attention matrix**. Tidak ada O(N²) operation.
- **Dynamic depth**. Sistem mengalokasikan compute proporsional dengan kesulitan input — secara otomatis, tanpa "early exit" yang di-supervise.
- **Fusion tanpa parameter**. Bobot fusi murni dari magnitudo energi output — mikrosirkit yang lebih confident berkontribusi lebih.

### 5.3 Mengapa ini tidak collapse

Risiko utama desain ini: cascade bisa stuck di trivial fixed point (vektor nol atau vektor seragam). Mitigasi:

1. **GELU non-linearity** memastikan transformasi tidak collapse ke linear map.
2. **Energy-weighted fusion** menekan kontribusi mikrosirkit yang outputnya saturate (sangat besar atau sangat kecil).
3. **Initial activation noise injection**: Setiap input ditambahi noise kecil (σ=0.01) untuk melewati basin atraktor trivial.
4. **Damping factor opsional**: `a_new := α·a_new + (1-α)·a` dengan α≈0.7 untuk mencegah oscillation di awal training.

---

## 6. Mekanisme Emergensi

Pertanyaan kunci: dari mana datangnya kemampuan penalaran kompleks tanpa lapisan tinggi eksplisit?

### 6.1 Tiga Prinsip Emergensi

**(a) Interferensi jalur sebagai operasi komposisi**

Ketika dua rangkaian inferensi yang dimulai dari sub-konsep berbeda mengaktifkan mikrosirkit yang sama di langkah ke-N, mereka konstruktif berinterferensi. Mikrosirkit bersama itu menjadi *simpul komposisi* — secara fungsional setara dengan operasi `join` dalam program. Tidak perlu mendesain operator komposisi; ia muncul dari konvergensi dua representasi terkait ke wilayah LSH yang sama.

**(b) Atraktor titik tetap sebagai abstraksi**

Mengikuti analisis Hopfield, titik tetap dari dinamika `a → T(a)` adalah memori asosiatif yang stabil. Namun di SML, titik tetap bukan vektor tunggal — mereka adalah *atraktor berdimensi tinggi* yang dapat dihuni oleh banyak input semantik-ekuivalen. Atraktor ini **adalah** representasi abstrak. Konsep seperti "kausalitas" atau "negasi" tidak dikodekan di neuron tertentu, melainkan adalah bentuk geometri basin atraktor yang dibagi oleh banyak input yang melibatkan konsep tersebut.

**(c) Small-world routing menghasilkan analogi**

4 shortcut tetangga per mikrosirkit (long-range) berarti dua wilayah konseptual yang jauh hanya berjarak `~log(N) ≈ 17` langkah. Karena cascade umumnya konvergen dalam ~10 langkah, *shortcut yang sebenarnya digunakan* selektif. Sistem belajar (lewat distribusi konten mikrosirkit) shortcut mana yang bermakna. Ini adalah substrat geometris untuk **penalaran analogi**: pemetaan struktur dari domain A ke domain B menjadi traversal dari satu klaster ke klaster lain lewat shortcut bersama.

### 6.2 Penekanan Filosofis

Emergensi di sini bukan magic. Ia adalah konsekuensi terhitung dari:

```
(graf small-world) × (dinamika titik tetap) × (kuantisasi diskretisasi konseptual)
```

Tidak ada satu pun dari tiga komponen ini yang merupakan teknik baru. Yang baru adalah **komposisi mereka sebagai satu-satunya mekanisme komputasi**, tanpa lapisan padat yang membuat mereka tidak relevan.

---

## 7. Formulasi Matematis

### 7.1 Notasi

| Simbol | Definisi |
|--------|----------|
| `M = {μ_1, ..., μ_N}` | Himpunan mikrosirkit, N ≈ 120.000 |
| `a ∈ R^d` | State aktivasi, d = 256 |
| `L: R^d → {0,1}^32` | Fungsi SimHash |
| `N_k(s)` | Operator: k mikrosirkit dengan resonance_vec terdekat dari signature s |
| `f_μ(a)` | Forward function satu mikrosirkit: `GELU(dequant(W_μ)·a + b_μ)` |
| `T: R^d → R^d` | Operator transisi satu langkah |

### 7.2 Operator Transisi

$$
T(a) = \sum_{\mu \in N_6(L(a))} \frac{\|f_\mu(a)\|^2}{\sum_{\mu' \in N_6(L(a))} \|f_{\mu'}(a)\|^2} \cdot f_\mu(a)
$$

### 7.3 Inferensi sebagai Pencarian Titik Tetap

$$
a^* = \lim_{t \to \infty} T^t(a_0), \quad \text{stop ketika } \|T(a) - a\| < \varepsilon
$$

Dalam praktik, `MAX_STEPS = 50` untuk garansi terminasi.

### 7.4 Sifat yang Belum Dibuktikan

⚠️ **Belum dijamin secara matematis**:

- Eksistensi titik tetap untuk semua `a_0` (perlu kontraktivitas pada `T`)
- Konvergensi monotonik
- Uniqueness titik tetap (kemungkinan ada multiple basin)

Mitigasi praktis: tambahkan damping term, batasi MAX_STEPS, monitor `||T(a) - a||` selama inferensi.

---

## 8. Code Flow Reference

### 8.1 Setup (sekali saat boot)

```cpp
#include <sys/mman.h>
#include <pthread.h>
#include <immintrin.h>

// Memory-map lattice file
int fd = open("lattice.bin", O_RDONLY);
void* lattice = mmap(NULL,
                     23ULL << 30,           // 23 GB
                     PROT_READ,
                     MAP_PRIVATE | MAP_POPULATE,
                     fd, 0);
const Microcircuit* M = static_cast<const Microcircuit*>(lattice);

// Build LSH index
LSHIndex idx;
idx.build(M, NUM_MICROCIRCUITS);

// Spawn 6 worker threads, pin ke core fisik
WorkerPool pool(6);
for (int c = 0; c < 6; ++c) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(c, &cpuset);
    pthread_setaffinity_np(pool.threads[c].native_handle(),
                           sizeof(cpu_set_t), &cpuset);
}
```

### 8.2 Inferensi Loop

```cpp
void inference(const float* input, float* output) {
    alignas(32) float a[256];
    load_input(a, input);

    for (int step = 0; step < MAX_STEPS; ++step) {
        // 1. Hash konten
        uint32_t sig = simhash_avx2(a);

        // 2. Resolusi tetangga
        uint32_t neighbor_ids[6];
        idx.resolve_top_k(sig, a, neighbor_ids, 6);

        // 3. Dispatch ke 6 core
        alignas(32) float y[6][256];
        for (int c = 0; c < 6; ++c) {
            pool.submit(c, [&, c]() {
                microcircuit_forward(&M[neighbor_ids[c]], a, y[c]);
            });
        }
        pool.wait_all();

        // 4. Fusi energy-weighted
        alignas(32) float a_new[256];
        energy_weighted_fuse(y, a_new);

        // 5. Cek konvergensi
        float delta = l2_distance_avx2(a, a_new);
        if (delta < EPSILON) break;

        memcpy(a, a_new, sizeof(a));
    }

    project_to_output(a, output);
}
```

### 8.3 Inner Loop Mikrosirkit (Hot Path)

```cpp
void microcircuit_forward(const Microcircuit* M,
                          const float* a,
                          float* y) {
    for (int row = 0; row < 256; ++row) {
        __m256 acc = _mm256_setzero_ps();
        const int8_t* W_row = &M->W[row][0];
        float scale = half_to_float(M->scales[row]);

        for (int col = 0; col < 384; col += 16) {
            // Unpack 16 INT4 → 16 INT8 → 16 FP32 (8+8 lanes)
            __m128i packed = _mm_load_si128(
                reinterpret_cast<const __m128i*>(&W_row[col / 2]));
            __m256i unpacked = unpack_int4_to_int8_avx2(packed);
            __m256 w_lo = int8_to_fp32_lo(unpacked);
            __m256 w_hi = int8_to_fp32_hi(unpacked);

            __m256 a_lo = _mm256_load_ps(&a[col]);
            __m256 a_hi = _mm256_load_ps(&a[col + 8]);

            acc = _mm256_fmadd_ps(w_lo, a_lo, acc);
            acc = _mm256_fmadd_ps(w_hi, a_hi, acc);
        }

        float sum = horizontal_sum_avx2(acc) * scale
                  + half_to_float(M->bias[row]);
        y[row] = gelu_poly(sum);  // Polynomial approx, no exp()
    }
}
```

### 8.4 Helper: SimHash AVX2

```cpp
// 32 hyperplane proyeksi tetap, hasilkan 32-bit signature
uint32_t simhash_avx2(const float* a) {
    static const float HYPERPLANES[32][256];  // pre-loaded
    uint32_t sig = 0;
    for (int h = 0; h < 32; ++h) {
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < 256; i += 8) {
            __m256 hp = _mm256_load_ps(&HYPERPLANES[h][i]);
            __m256 ai = _mm256_load_ps(&a[i]);
            acc = _mm256_fmadd_ps(hp, ai, acc);
        }
        float dot = horizontal_sum_avx2(acc);
        sig |= (dot > 0 ? 1u : 0u) << h;
    }
    return sig;
}
```

---

## 9. Estimasi Performa

### 9.1 Per Langkah Cascade

| Operasi | Latency Estimasi | Catatan |
|---------|------------------|---------|
| SimHash | ~200 ns | 32×256 FMA, fully AVX2 |
| LSH lookup + top-k | ~500 ns | Hash table + small sort |
| Mikrosirkit forward (per core) | ~12 µs | ~50K MAC dari L2 |
| Dispatch + sync overhead | ~5 µs | Futex wakeup |
| Energy-weighted fusion | ~2 µs | 6×256 normalisasi |
| **Total per langkah** | **~20 µs** | Parallel di 6 core |

### 9.2 Per Inferensi

| Skenario | Konvergensi | Total Latency |
|----------|-------------|----------------|
| Input mudah | 2-3 langkah | ~50 µs |
| Input medium | 5-10 langkah | ~150 µs |
| Input kompleks | 20-50 langkah | ~1-5 ms |

### 9.3 Throughput Teoretis

- **Best case** (input mudah, batch 1): ~20.000 inferensi/detik
- **Average case**: ~5.000 inferensi/detik
- **Worst case** (input kompleks): ~200 inferensi/detik

⚠️ **Catatan**: Ini estimasi teoretis berdasarkan model bandwidth + compute. Benchmark aktual bisa berbeda 2-5x. Validasi awal harus pada inner loop dulu sebelum percaya angka end-to-end.

---

## 10. Roadmap Implementasi Bertahap

Pendekatan: **bukti konsep dulu, optimasi belakangan**. Setiap fase harus menghasilkan output yang bisa diukur sebelum lanjut.

### Fase 0: Foundation (Estimasi 1-2 minggu)

**Tujuan**: Setup environment, validasi asumsi AVX2 bisa hit throughput yang diasumsikan.

- [ ] Setup proyek C++ dengan CMake/Makefile
- [ ] Tulis microbenchmark untuk AVX2 FMA throughput murni
- [ ] Tulis microbenchmark untuk L2 cache bandwidth
- [ ] Validasi: bisa hit ≥20 GFLOPS efektif per core?
- [ ] Validasi: latency baca 192KB dari L2 ≤ 10µs?

**Deliverable**: Report "Hardware Capability Validation" — go/no-go untuk lanjut.

### Fase 1: Single Microcircuit Forward (Estimasi 2-3 minggu)

**Tujuan**: Implementasi 1 mikrosirkit yang bisa di-forward, dengan kuantisasi.

- [ ] Struktur data `Microcircuit` dengan alignment
- [ ] INT4 packing/unpacking dengan AVX2
- [ ] Dequantisasi inline (scale × INT4 → FP32)
- [ ] GELU polynomial approximation
- [ ] Forward pass dari random weights, hasil sanity check (no NaN, reasonable magnitude)
- [ ] Benchmark: latency per forward pass ≤ 15 µs?

**Deliverable**: Library function `microcircuit_forward()` yang lulus benchmark.

### Fase 2: Multi-Core Dispatch (Estimasi 2 minggu)

**Tujuan**: 6 mikrosirkit paralel di 6 core, ukur efisiensi paralel.

- [ ] WorkerPool dengan pthread + thread affinity
- [ ] Submit/wait mechanism (futex atau semaphore)
- [ ] Dispatch 6 mikrosirkit ke 6 core
- [ ] Ukur: speedup ≥ 5x (idealnya 5.5x dari 6)?
- [ ] Ukur: dispatch overhead < 10 µs?

**Deliverable**: Parallel forward dengan benchmark.

### Fase 3: LSH Routing (Estimasi 2-3 minggu)

**Tujuan**: SimHash + LSH index + top-k resolution.

- [ ] Implementasi SimHash AVX2
- [ ] Hash table LSH (bucket → microcircuit IDs)
- [ ] Top-k resolution dari kandidat
- [ ] Validasi: hash distribution merata
- [ ] Validasi: top-k results stabil untuk input serupa

**Deliverable**: Routing function dengan unit tests.

### Fase 4: Cascade Loop (Estimasi 2 minggu)

**Tujuan**: Full inference loop dengan convergence check.

- [ ] Energy-weighted fusion
- [ ] Convergence check
- [ ] Damping factor (tuneable)
- [ ] Logging untuk inspeksi: steps-to-converge, energy distribution

**Deliverable**: End-to-end `inference()` function pada random lattice.

### Fase 5: Lattice Storage & mmap (Estimasi 1-2 minggu)

**Tujuan**: Load lattice 23 GB dari disk via mmap.

- [ ] Format file lattice (header + microcircuit blocks)
- [ ] mmap dengan `MAP_POPULATE`
- [ ] Validasi: cold start latency
- [ ] Validasi: working set page faults minimal setelah warmup

**Deliverable**: Loader + serializer.

### Fase 6: Toy Training (Estimasi 4-6 minggu) — PENELITIAN

**Tujuan**: Train SML kecil (~1000 microcircuits) pada toy task. **Fase paling tidak pasti.**

Strategi kandidat:
1. **Equilibrium Propagation** (Scellier & Bengio 2017) — backprop yang adaptable ke fixed-point dynamics
2. **Implicit Differentiation** (Bai et al, Deep Equilibrium Models)
3. **Dictionary learning initialization** + fine-tune lewat policy gradient
4. **Hebbian/local learning rule** sebagai baseline

- [ ] Pilih strategi (decision point major)
- [ ] Implementasi training loop
- [ ] Toy task: XOR-like, atau MNIST-tiny
- [ ] Validasi: bisa converge ke akurasi yang masuk akal?

**Deliverable**: Trained toy lattice + training script.

### Fase 7: Skalakan & ARC Benchmark (Open-Ended)

**Tujuan**: Scale up dan integrasi dengan ARC-PS (jika menarik).

- [ ] Scale ke 10K, 100K, 120K microcircuits
- [ ] Benchmark pada ARC-AGI tasks
- [ ] Eksperimen: SML kecil sebagai **Neural Guide** pengganti MLP 7K parameter di ARC-PS

**Deliverable**: Paper draft atau blog post tentang hasil.

---

## 11. Open Questions & Risk Register

### 11.1 Risiko Teknis

| Risk ID | Deskripsi | Likelihood | Impact | Mitigasi |
|---------|-----------|------------|--------|----------|
| R-01 | AVX2 throughput aktual jauh di bawah teoretis | Medium | High | Fase 0 validation; siap downscale ekspektasi |
| R-02 | Memory bandwidth bottleneck tetap muncul meski cache locality | Medium | Critical | Microbenchmark L2/L3 di Fase 0 |
| R-03 | Cascade tidak konvergen (oscillation) | High | High | Damping factor, MAX_STEPS, monitor norm |
| R-04 | Titik tetap collapse ke trivial (vektor nol/seragam) | Medium | High | Noise injection, GELU, energy weighting |
| R-05 | LSH routing tidak diskriminatif (semua input ke bucket sama) | Medium | High | Mungkin perlu learned hash function |
| R-06 | Backprop tidak applicable; tidak ada training method yang work | High | Critical | Penelitian Fase 6 sebelum scale up |
| R-07 | INT4 quantization terlalu agresif, kehilangan akurasi | Low-Medium | Medium | Fallback ke INT8 (footprint 2x) |

### 11.2 Open Questions (Decision Points)

1. **Pelatihan**: Backprop konvensional bermasalah karena cascade depth variabel. Equilibrium Propagation? Implicit differentiation? Local learning rules? **Decision needed before Fase 6.**

2. **LSH learned vs fixed**: Mulai dengan random hyperplanes, atau langsung learn? **Mulai fixed untuk Fase 3, evaluate di Fase 6.**

3. **Kuantisasi**: INT4 ambisius. INT8 lebih aman tapi 2x memory. **Mulai INT8 di Fase 1, eksperimen INT4 di Fase 5.**

4. **Dimensi state `d`**: 256 adalah guess. Bisa 128 (lebih ramping) atau 512 (lebih ekspresif). **Tuneable parameter; jangan hardcode terlalu dini.**

5. **Jumlah neighbor**: 16 (12+4) adalah guess dari Watts-Strogatz literature. Mungkin 8 cukup. **Eksperimen di Fase 3-4.**

6. **Halt threshold ε**: Berapa? Mungkin learnable. **Mulai dengan ε=0.01, tune empiris.**

### 11.3 Filosofis/Strategis

- Apakah arsitektur ini benar-benar emergent capable, atau hanya glorified nearest-neighbor lookup?
- Apakah lebih masuk akal fokus ke SML, atau dorong ARC-PS dulu sampai matang, baru pertimbangkan SML sebagai Neural Guide?
- Berapa banyak waktu kita siap investasi sebelum ada signal positif/negatif?

---

## 12. Hubungan dengan ARC-PS

SML dan ARC-PS (ARC Program Synthesis) berbagi DNA filosofis:

| Aspek | ARC-PS | SML |
|-------|--------|-----|
| Penolakan | "Satu jaringan padat besar" sebagai jawaban | Sama |
| Bertaruh pada | Struktur (sintesis program simbolik) | Struktur (geometri graf) |
| Verifikasi | Time-as-verifier, eksekusi diskret | Konvergensi titik tetap kontinu |
| Domain natural | Tugas ARC (visual reasoning eksplisit) | Representasi terdistribusi umum |

### Potensi Integrasi

**Hipotesis menarik**: Gunakan SML kecil (mungkin ~5.000 mikrosirkit, ~1 GB footprint) sebagai **Neural Guide** dalam ARC-PS, menggantikan MLP 7.000 parameter saat ini.

Keuntungan potensial:
- Lebih ekspresif daripada MLP shallow
- Sub-linear inferensi: cepat untuk input mudah, lebih dalam untuk input kompleks
- Content-addressed: mungkin lebih cocok untuk routing hipotesis ARC

Kerugian:
- Kompleksitas implementasi jauh lebih tinggi
- Belum proven; ARC-PS sudah punya momentum

**Rekomendasi**: Selesaikan ARC-PS dengan MLP guide dulu. Jika ARC-PS bekerja, eksperimen dengan SML guide sebagai Phase 2.

---

## 13. Glosarium

| Istilah | Definisi |
|---------|----------|
| **Microcircuit (μ_k)** | Unit komputasi atomik SML, 192 KB, ~100K parameter |
| **Lattice** | Himpunan semua mikrosirkit (~120K), tersimpan di RAM via mmap |
| **Resonance vector** | Signature mikrosirkit (FP16, 256-dim) untuk content routing |
| **SimHash** | Locality-sensitive hash; vektor mirip → hash mirip (Hamming distance kecil) |
| **LSH bucket** | Kumpulan mikrosirkit dengan SimHash signature sama/mirip |
| **Cascade** | Iterasi `a → T(a)` sampai konvergen |
| **Fixed point** | Vektor `a*` dimana `T(a*) = a*` (dalam toleransi ε) |
| **Energy-weighted fusion** | Penggabungan output mikrosirkit dengan bobot dari magnitudo |
| **Small-world graph** | Graf dengan clustering tinggi + diameter logaritmik |
| **Shortcut edge** | Edge antar mikrosirkit yang LSH-jauh; untuk analogical jumps |
| **Equilibrium Propagation** | Metode training untuk fixed-point dynamics (alternatif backprop) |
| **AVX2** | Advanced Vector Extensions 2; SIMD 256-bit di x86 |
| **FMA** | Fused Multiply-Add: `a × b + c` dalam 1 instruksi |

---

## 14. Referensi & Bacaan Lanjut

### Inspirasi Teoretis

- Watts, D. & Strogatz, S. (1998). *Collective dynamics of small-world networks*. Nature.
- Hopfield, J. (1982). *Neural networks and physical systems with emergent collective computational abilities*. PNAS.
- Scellier, B. & Bengio, Y. (2017). *Equilibrium Propagation: Bridging the gap between energy-based models and backpropagation*. Frontiers in Computational Neuroscience.
- Bai, S., Kolter, J. Z., & Koltun, V. (2019). *Deep Equilibrium Models*. NeurIPS.
- Charikar, M. (2002). *Similarity estimation techniques from rounding algorithms*. STOC (SimHash).

### Praktis (CPU Inference)

- Intel® Intrinsics Guide (untuk AVX2 reference)
- Agner Fog's Optimization Manuals (instruction tables, microarchitecture)
- llama.cpp source code (referensi CPU inference quantization)
- ggml library (referensi memory layout untuk CPU)

### Latar Belakang (untuk konteks)

- Fedus, W. et al. (2021). *Switch Transformer* (untuk memahami apa yang SML tolak)
- Hinton, G. (2022). *The Forward-Forward Algorithm* (alternative ke backprop)
- Penjelajahan ulang Capsule Networks, HTM (Hierarchical Temporal Memory)

---

## Catatan Akhir

Dokumen ini adalah **spesifikasi desain awal**, bukan janji. Banyak asumsi belum divalidasi. Fase 0 (Hardware Capability Validation) adalah filter pertama: jika hardware tidak bisa hit angka yang diasumsikan, sebagian besar arsitektur perlu redesign.

Pendekatan yang direkomendasikan: **bangun bertahap, ukur di setiap fase, siap pivot**. Jangan implementasi 120K mikrosirkit dulu, baru sadar AVX2 throughput tidak cukup. Bangun 1 mikrosirkit dulu, ukur, kemudian skalakan.

Filosofi yang menyatukan SML dengan proyek Anda yang lain (Backtest Pro, App Manager, ARC-PS): **minimal bootstrap, time-as-verifier, biarkan struktur muncul dari batasan fisik**. Arsitektur ini lahir bukan dari ambisi mengalahkan benchmark, tapi dari pertanyaan: *"Apa yang bisa hidup di silikon yang saya punya?"*

Selama masih hidup berarti masih berproses.

---

**Versi berikutnya** (`v0.2`) akan ditulis setelah Fase 0 selesai, dengan validasi hardware capability dan revisi estimasi performa.
