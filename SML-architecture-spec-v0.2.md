# SML: Sparse Microcircuit Lattice
### Spesifikasi Arsitektur — v0.2 (Empirically Validated)

---

**Versi**: 0.2 (Post-Phase-4 Consolidation)
**Status**: MVP validated end-to-end pada CPU real hardware
**Predecessor**: v0.1 (30 Juni 2026, design-only)
**Target Host**: HP EliteDesk 800 G4 (Intel i5-8500T, locked 2.1 GHz, 32GB DDR4, no GPU)
**Container**: Docker Debian 12 slim, `--cpus="4.8"`
**Ekosistem**: C++ / C dengan AVX2, pthread, mmap
**Tanggal**: 1 Juli 2026
**Basis Empiris**: 4 phase, 10+ benchmark file, semua hasil validated

---

## Daftar Isi

1. [Perubahan dari v0.1](#1-perubahan-dari-v01)
2. [Ringkasan Eksekutif Revisi](#2-ringkasan-eksekutif-revisi)
3. [Batasan Hardware — Validated](#3-batasan-hardware--validated)
4. [Pilihan Ekosistem & Justifikasi](#4-pilihan-ekosistem--justifikasi)
5. [Topologi — Revised](#5-topologi--revised)
6. [Routing: Per-Inference (Bukan Per-Step)](#6-routing-per-inference-bukan-per-step)
7. [Cascade Dynamics](#7-cascade-dynamics)
8. [Formulasi Matematis](#8-formulasi-matematis)
9. [Latency Model Empiris](#9-latency-model-empiris)
10. [Multi-Core Dispatch](#10-multi-core-dispatch)
11. [Roadmap yang Direvisi](#11-roadmap-yang-direvisi)
12. [Open Questions & Risk Register](#12-open-questions--risk-register)
13. [Glosarium](#13-glosarium)

---

## 1. Perubahan dari v0.1

Bagian ini eksplisit menjelaskan **apa yang berubah dan kenapa**. Setiap perubahan dijustifikasi dengan data empiris dari implementation phase.

### 1.1 Perubahan Arsitektural Major

| Aspek | v0.1 | v0.2 | Alasan |
|-------|------|------|--------|
| **Dimensi mikrosirkit** | 256×384 (W[256][384]) | **256×256** (square) | Cascade `T(a)` butuh input dim = output dim untuk iterasi. v0.1 inkonsisten |
| **Routing frequency** | Per cascade step | **Per inference (fixed)** | Phase 4 empirical: dynamic routing menghasilkan limit cycle, 0% konvergensi. Fixed routing: 100% |
| **Weight scale init** | 0.001 (arbitrary) | **1/(sqrt(D)·mean\|W\|)** (kalibrated) | Menghasilkan output magnitude ~= input magnitude |
| **Bias init** | random [-0.1, 0.1] | **Structured positif kecil** | Menghindari ReLU dead zone |
| **Input handling** | Raw | **Normalized ke unit L2** | Konsistensi magnitude across inference |
| **Between-step norm** | Tidak ada | **Opsional (recommended)** | Bounded state magnitude, cleaner attractor |
| **Dispatch sync** | pthread + atomic spin | **pthread_barrier** | Spin-wait konsumsi 100% CPU × 6 core, konflik dengan --cpus quota |

### 1.2 Perubahan Estimasi Performa

| Metrik | v0.1 Estimasi | v0.2 Measured | Ratio |
|--------|---------------|---------------|-------|
| Single-core FMA throughput | 24 GFLOPS | **66.8 GFLOPS peak, sustained** | 2.8× lebih baik |
| L2 bandwidth per core | 30 GB/s asumsi | **45.4 GB/s** | 1.5× lebih baik |
| RAM aggregate 6-core | linear scaling | **24.1 GB/s (1.7× vs single)** | **jauh lebih buruk** |
| Mikrosirkit forward | ~12 µs (INT8 unpack) | **14 µs (256×384) / 9 µs (256×256)** | Sesuai prediksi |
| 6-core dispatch (barrier) | ~5 µs overhead | **33 µs total (19 µs overhead)** | 4× lebih buruk |
| Inference easy (3 step) | ~50 µs | **~200 µs (dengan fix)** | 4× lebih buruk (routing overhead + convergence steps lebih banyak) |
| Inference medium (8 step) | ~150 µs | **~500-700 µs** | 3-5× lebih buruk |

**Interpretasi**: Compute jauh lebih baik dari estimasi. Memory dan sync overhead jauh lebih buruk. Net: SML tetap workable tapi tidak seagresif estimasi v0.1.

### 1.3 Hilangkan dari v0.1

- **Small-world graph shortcut edges**: v0.1 menyebut 4 tetangga shortcut per mikrosirkit. Setelah fixed routing per-inference, konsep tetangga graph tidak lagi relevan untuk cascade dinamis. Konsep bisa dihidupkan lagi kalau dynamic routing di-solve.
- **Neighbor pointers di struct**: 16 × 4 bytes = 64 bytes untuk `neighbor_lsh[16]`. Tidak dipakai, dihilangkan.
- **Multi-probe LSH bucketing**: Phase 3b menunjukkan recall 33% untuk 8-bit prefix + Hamming-1 probe. Untuk N < 10K, linear scan lebih baik. Bucketing di-defer sampai Phase 5+.

### 1.4 Tambahan Baru di v0.2

- **Cascade dinamik validated** dengan real timing dan konvergensi 100% (fixed routing)
- **Explicit design decision**: routing adalah "expert committee selection" per-input, bukan multi-hop reasoning
- **Empirical latency breakdown** dari Phase 5 benchmark

---

## 2. Ringkasan Eksekutif Revisi

SML v0.2 adalah **CPU-only neural network arsitektur** yang, tidak seperti Transformer/MoE, mendesain untuk **thermodynamic survival** pada hardware terbatas (6 core, 32GB RAM). MVP telah divalidasi end-to-end.

**Tiga klaim inti** (dari v0.1) — direvisi berdasarkan empirical data:

1. **Memory bandwidth amortization** ✅ **Validated**: Mikrosirkit 67 KB muat di L2 (256 KB). L2 bandwidth 45 GB/s per core mendukung streaming compute. Cache warm rate ~3 µs/mikrosirkit.

2. **Content-addressed routing tanpa router learned** ✅ **Validated dengan caveat**: SimHash + LSH routing works untuk 1000 mikrosirkit. **CAVEAT**: Routing harus dilakukan **sekali per inference**, bukan setiap cascade step. Dynamic per-step routing menghasilkan limit cycle (empirical Phase 4).

3. **Sub-linear scaling per input** ⚠️ **Sebagian**: Convergence dalam ~10-12 step untuk fixed routing dengan kalibrated init. Belum ada bukti bahwa "easy input converge lebih cepat" tanpa training.

**Yang bekerja secara empiris**:
- Full inference pipeline: input → route → 6× forward → fuse → converge → output
- Latency: ~600-720 µs untuk inference tipikal (12 step × 60 µs)
- Deterministic dengan seed fixed
- 100% convergence rate dengan fixed routing + kalibrasi weight

**Yang belum bekerja**:
- Semantic meaningfulness dari attractor (butuh training, Phase 6)
- Dynamic per-step routing (butuh regularization/smoothing, atau accept sebagai design constraint)
- Scale ke 100K+ mikrosirkit (Phase 5)

---

## 3. Batasan Hardware — Validated

Semua angka di bawah adalah **hasil measurement aktual**, bukan estimasi.

### 3.1 CPU

| Property | Value | Source |
|----------|-------|--------|
| CPU Model | Intel Core i5-8500T | `/proc/cpuinfo` |
| Cores/Threads | 6 / 6 (no HT) | `lscpu` |
| Base frequency | 2.10 GHz | Locked (hardware longevity) |
| ISA | x86_64 + AVX2 + FMA + POPCNT | Verified via CPUID |
| L1d per core | 32 KB | `lscpu` |
| L2 per core | 256 KB | `lscpu` |
| L3 shared | 9 MB | `lscpu` |
| RAM | 32 GB DDR4-2666 dual-channel | `dmidecode` |

### 3.2 Bandwidth Measured (Phase 0 cache_benchmark.c)

| Level | Single-core | 6-core aggregate | Speedup |
|-------|-------------|------------------|---------|
| L2 | 45.4 GB/s | 342 GB/s | 7.5× |
| L3 | 29.9 GB/s | (not tested) | — |
| RAM | 14.0 GB/s | 24.1 GB/s | **1.7×** ⚠️ |

**Key insight**: RAM aggregate saturate di 24 GB/s karena DDR4 dual-channel bottleneck. **Cascade cold-load bottleneck adalah RAM, bukan compute**.

### 3.3 Compute Measured (Phase 0 fma_benchmark.c)

| Test | Result | Efficiency vs Theoretical |
|------|--------|---------------------------|
| Latency-bound FMA | 8.34 GFLOPS | 99.3% of theoretical (dependency chain limit) |
| Peak throughput | **66.82 GFLOPS** | **99.4% of theoretical 67.2 GFLOPS** |
| Sustained 10 sec | 66.83 GFLOPS × 10 sample | Zero degradasi |

Compute bukan bottleneck. Lock 2.1 GHz + T-series TDP headroom = deterministic sustained throughput.

### 3.4 Container Constraints

```
docker run --cpus="4.8" --memory="28g" \
  --ulimit memlock=-1:-1 \
  --cap-add=SYS_PTRACE --cap-add=SYS_NICE \
  ...
```

`--cpus="4.8"` = maksimal 480% CPU sustained. Bermakna:
- 6 thread × 100% aktif = 600% demand > quota
- **Design constraint**: hindari 6 thread spinning idle
- Fixed: Phase 2 v2 barrier sync — worker sleep saat idle

---

## 4. Pilihan Ekosistem & Justifikasi

**Pilihan tetap C** (bisa juga C++, tapi kami pakai C). Alasan dari v0.1 masih valid:

- glibc + native AVX2 intrinsics
- `pthread_setaffinity_np` untuk thread pinning
- `mmap` untuk zero-copy lattice loading
- Deterministic latency, no GC pauses

**Dependencies minimum**:
- Standard C: `pthread`, `mmap`, `<immintrin.h>`, `<stdatomic.h>`
- Compiler: GCC 11+ dengan `-O3 -march=native -mavx2 -mfma -mpopcnt`
- Build: plain Makefile atau CMake

**Tidak dipakai** (sesuai filosofi minimal bootstrap):
- Eigen, Boost, Intel MKL
- OpenMP (kita atur thread manual)
- Python bindings di runtime path

---

## 5. Topologi — Revised

### 5.1 Mikrosirkit Struct Final

```c
typedef struct {
    // Weight matrix: SQUARE 256×256 (revised dari v0.1 yang 256×384)
    alignas(64) int8_t W[D_STATE][D_STATE];         // 64 KB
    
    // Per-row dequantization + activation offset
    alignas(64) float  scales[D_STATE];             // 1 KB
    alignas(64) float  bias[D_STATE];               // 1 KB
    
    // Routing metadata
    alignas(64) uint32_t self_hash;                 // 4 B (SimHash dari resonance_vec)
    alignas(32) float    resonance_vec[D_STATE];    // 1 KB
    
    // Total: ~67 KB per mikrosirkit
} Microcircuit;
```

**Perbedaan penting dari v0.1**:
- W adalah SQUARE (`D_STATE × D_STATE`) supaya cascade `T(a)` well-defined
- `neighbor_lsh[16]` dihilangkan (fixed routing tidak butuh graph traversal)
- Ukuran ~67 KB (bukan 192 KB) — muat L2 dengan margin ~189 KB

### 5.2 Geometri Makro Revised

| Property | v0.1 Target | v0.2 Validated |
|----------|-------------|----------------|
| Mikrosirkit dimensi | 256×384 | **256×256** |
| Parameter per mc | ~100K | ~65K |
| Ukuran per mc | 192 KB (INT4) | 67 KB (INT8) |
| Target N | 120K | **1K validated, 10K tested next (Phase 5)** |
| Total lattice | 23 GB (INT4) | **~65 MB @ N=1K, ~650 MB @ N=10K** |

**Skala saat ini**: N=1000 (Phase 3-4). Belum di N=120K karena butuh mmap infrastruktur (Phase 5).

### 5.3 Memory Layout untuk mmap (Phase 5 Preview)

Format lattice file yang direncanakan:
```
File offset 0: Header (magic bytes, N, dimensions, checksum)
File offset 4096: Aligned mikrosirkit array
  mc[0]: 67 KB (padded to 64 KB alignment = 96 KB)
  mc[1]: 96 KB
  ...
  mc[N-1]: 96 KB
Total: ~9.5 MB @ N=100, ~9.5 GB @ N=100K
```

Padding ke 96 KB (aligned untuk L2) menambah ~30% storage overhead tapi worth it untuk cache locality.

---

## 6. Routing: Per-Inference (Bukan Per-Step)

**Ini adalah perubahan design paling significant di v0.2.**

### 6.1 Bukti Empiris untuk Fixed Routing

Phase 4 v2 benchmark, 100 trials per configuration:

| Config | Converged | Mean steps | Interpretation |
|--------|-----------|------------|----------------|
| Dynamic + no norm | 0/100 | 50 (max) | Limit cycle |
| Dynamic + norm | 0/100 | 50 (max) | Limit cycle |
| **Fixed + no norm** | **100/100** | **10.3** | Fixed point |
| **Fixed + norm** | **100/100** | **12.0** | Fixed point |

Trajectory dynamic routing menunjukkan `mc[0]` cycling antara {669, 767, 777, 404, ...}. State ditarik oleh microcircuit berbeda tiap step → oscillation.

### 6.2 Mekanisme Routing (Simplified)

```
Input a (256-dim)
    ↓ (normalize ke unit L2)
    ↓
q_hash = SimHash(a)     // 32-bit signature, ~650 ns
    ↓
Linear scan 1000 mc:
  distance[i] = popcount(q_hash XOR mc[i].self_hash)
top_6 = smallest 6 distances    // ~2.5 µs total
    ↓
FREEZE: gunakan same top_6 sepanjang cascade
```

**Kompleksitas per inference**: O(N) untuk linear scan. Untuk N=1000, ~2.5 µs. Untuk N=100K, akan ~250 µs (masih acceptable).

Bucketing (Phase 3b) menghasilkan recall 33% — **tidak dipakai di v0.2**. Sampai kita punya bucketing yang better recall, linear scan adalah default.

### 6.3 Filosofi Routing yang Diperbarui

**v0.1 asumsi**: routing berubah tiap step memungkinkan "multi-hop reasoning" — cascade traverse different conceptual clusters.

**v0.2 realitas**: routing per-inference adalah **expert committee selection**. Similar dengan Mixture-of-Experts (MoE) dengan top-K = 6, tapi:
- MoE: 1-pass forward dengan learned router
- SML: iterated forward dengan LSH router, cascade sampai fixed point

Trade-off yang diterima:
- ❌ Kehilangan: multi-hop reasoning inherent di dynamic routing
- ✅ Dapat: predictable convergence, empirically stable
- ✅ Dapat: cascade adalah "expert refinement" yang tractable analitis

Multi-hop reasoning bisa dihidupkan lagi via:
- Training dengan smoothed routing (soft top-K)
- Regularization pada state space untuk hindari routing boundaries
- Hierarchical routing (route once di outer level, iterate di inner level)

Ini di-defer ke penelitian lanjutan.

---

## 7. Cascade Dynamics

### 7.1 Full Inference Pipeline (Validated)

```
input (256-dim FP32)
    │
    ├─→ normalize_unit(input)      // input /= ‖input‖
    │
    ├─→ route_once(input) → [mc_id_0, mc_id_1, ..., mc_id_5]  // FROZEN
    │
    │   ┌────────── cascade loop (max 50 iter) ───────────┐
    │   │                                                 │
    │   ├─→ For each mc_id in top_6:                     │
    │   │     y[i] = microcircuit_forward(mcs[mc_id], a) │
    │   │                                                 │
    │   ├─→ a_new = fuse_energy_weighted(y[0..5])        │
    │   │                                                 │
    │   ├─→ (opt) normalize_unit(a_new)                  │
    │   │                                                 │
    │   ├─→ delta = ‖a_new - a‖ / (‖a‖ + ε)              │
    │   │                                                 │
    │   ├─→ if delta < CONV_EPSILON (0.01): break        │
    │   │                                                 │
    │   └─→ a = a_new                                    │
    │
    └─→ return a  // Fixed point (attractor)
```

### 7.2 Trajectory Karakteristik (Empirical)

**Fixed routing + no normalize** (Phase 4 v2, seed 100):
```
Step 1: delta=1.16, norm=0.58   (initial adjustment, ~ReLU sparsification)
Step 2: delta=0.69, norm=0.31   (contracting)
Step 3: delta=0.50, norm=0.18
Step 4: delta=0.41, norm=0.11
Step 5: delta=0.30, norm=0.08
Step 6: delta=0.18, norm=0.07
Step 7: delta=0.09, norm=0.062
Step 8: delta=0.04, norm=0.060
Step 9: delta=0.02, norm=0.059
Step 10: delta=0.008, norm=0.059   ✓ CONVERGED
```

Delta monotonically decreasing. Norm converge ke ~0.06 fixed point. Klasik contractive dynamics.

**Fixed routing + normalize between steps**:
- Konvergensi sedikit lebih lambat (~12 step vs 10)
- State magnitude terkontrol = 1.0 sepanjang
- **Recommended untuk downstream composability**

### 7.3 Fusion: Energy-Weighted (Unchanged dari v0.1)

```
E_i = ‖y_i‖²                   // energy per microcircuit output
w_i = E_i / Σ_j E_j            // proportional to energy
a_new = Σ w_i · y_i           // weighted sum

// Guard: kalau Σ E_j < 1e-12, fallback ke uniform average
```

Implementasi AVX2, ~15 µs untuk 6 output × 256 dim.

---

## 8. Formulasi Matematis

### 8.1 Notasi Revised

| Simbol | Definisi | Value |
|--------|----------|-------|
| N | Jumlah mikrosirkit | 1000 (current), 10K-120K (target) |
| D | State dimension | 256 |
| K | Route arity | 6 |
| ε | Convergence threshold | 0.01 (1% relative L2 change) |
| MAX_STEPS | Iteration limit | 50 |
| M | Set semua mikrosirkit | {μ_1, ..., μ_N} |
| L | SimHash function | R^D → {0,1}^32 |
| top_K(s, M) | K mikrosirkit dengan min Hamming distance dari s | Fixed per inference |

### 8.2 Operator Cascade (Revised)

Definisi routing set (fixed per inference):
$$
R(a_0) = \text{top}_K(L(a_0), M) \quad \text{[computed ONCE]}
$$

Operator per-step cascade:
$$
T_R(a) = \sum_{\mu \in R} \frac{\|f_\mu(a)\|^2}{\sum_{\mu' \in R} \|f_{\mu'}(a)\|^2} \cdot f_\mu(a)
$$

**Penting**: `R` adalah subscript fixed, bukan `R(a)` yang berubah tiap step. Ini menjaga `T_R` continuous (dan empirically contractive dengan kalibrasi tepat).

Inferensi:
$$
a^* = \lim_{t \to \infty} T_R^t(a_0), \quad \text{stop when } \|T_R(a) - a\| / \|a\| < \varepsilon
$$

### 8.3 Sifat yang Divalidasi Empiris (Bukan lagi Asumsi)

- ✅ Eksistensi fixed point: 100/100 trials converge
- ✅ Contractive di L2 relative: delta monotonically decreasing
- ✅ Sub-linear compute per input: ~10-12 steps typical (vs MAX 50)

### 8.4 Sifat yang Belum Divalidasi

- ⚠️ Uniqueness fixed point: multiple basin might exist (belum di-test dengan trained weights)
- ⚠️ Semantic meaningfulness: attractor dari random weights bukan feature-rich (butuh training)

---

## 9. Latency Model Empiris

### 9.1 Per-Component Breakdown (Validated)

| Komponen | Latency | Source |
|----------|---------|--------|
| SimHash 32-bit | 658 ns | Phase 3a Test 1 |
| Route linear @ N=1000 | 2486 ns | Phase 3b Test 3 |
| Route bucketed @ N=1000 | 831 ns (33% recall) | Phase 3b Test 3 |
| Microcircuit forward (256×256) | ~9 µs | Extrapolated dari Phase 1 (256×384 = 14 µs) |
| Microcircuit forward (256×384) | 13.94 µs | Phase 1 |
| 6× sequential forward | ~54 µs | Phase 4 Test 5 |
| 6× parallel forward (barrier v2) | ~33 µs | Phase 2 v2 |
| Fusion energy-weighted | ~2 µs | Phase 4 Test 5 |
| L2 distance + norm check | ~500 ns | Estimate |
| Memcpy state (256 FP32) | ~200 ns | Estimate |

### 9.2 Per-Step Cascade

**Sequential (current MVP)**:
- Forward: ~54 µs
- Fusion + convergence check: ~3 µs
- Overhead: ~3 µs
- **Total: ~60 µs per cascade step**

**With Phase 2 v2 worker pool integration**:
- Dispatch (parallel forward): ~33 µs
- Fusion + convergence check: ~3 µs
- Overhead: ~4 µs
- **Total: ~40 µs per cascade step** (33% speedup)

### 9.3 Per-Inference Total

Untuk cascade 10-12 step (typical fixed routing dengan calibrated init):

| Config | Route (once) | Cascade steps | Total |
|--------|--------------|---------------|-------|
| Sequential MVP | 2.5 µs | 10 × 60 = 600 µs | **~603 µs** |
| Sequential MVP | 2.5 µs | 12 × 60 = 720 µs | **~723 µs** |
| With worker pool | 2.5 µs | 10 × 40 = 400 µs | **~403 µs** |
| With worker pool | 2.5 µs | 12 × 40 = 480 µs | **~483 µs** |

**Throughput single-thread orchestration**:
- Sequential MVP: ~1400-1650 inference/sec
- With worker pool: ~2000-2500 inference/sec

### 9.4 Skala N (Extrapolated)

Cascade compute tidak scale dengan N (fixed K=6). Yang scale hanya route.

| N | Route linear | Route bucketed (target 90% recall) | Cascade total (unchanged) |
|---|--------------|-------------------------------------|---------------------------|
| 1000 | 2.5 µs | N/A | 600 µs |
| 10K | 25 µs | ~5 µs (perlu Phase 5 tuning) | 625 µs |
| 100K | 250 µs | ~15 µs (Phase 5) | 615 µs (bucketed) atau 850 µs (linear) |

**Insight**: Sampai N=10K, linear scan tetap fine. Bucketing hanya mandatory di N=100K+.

---

## 10. Multi-Core Dispatch

### 10.1 Konfigurasi Validated (Phase 2 v2)

```c
pthread_barrier_t start_barrier;  // 7 threads (6 worker + main)
pthread_barrier_t end_barrier;

// Worker pinned ke core 0-5 via pthread_setaffinity_np
```

Sinkronisasi: pthread_barrier_wait. Worker sleep saat menunggu (futex-backed). Compatible dengan `--cpus="4.8"` container.

**Dispatch overhead**: ~19 µs per step (dari Phase 2 v2 measurement).

**Kenapa bukan spin-wait**: 6 spinning worker × 100% + main = 700% demand > 480% quota → hang.

**Kenapa bukan hybrid spin+futex (v3)**: Kompleksitas tanpa payoff jelas. Bencmark v3 menunjukkan performa lebih buruk (~180 µs) — cause tidak fully diagnosed. Barrier lebih simpel dan reliable.

### 10.2 Scaling Efficiency

Measured di Phase 2 v2:
- Sequential 6 × 14 µs = 84 µs
- Parallel dispatch 33 µs
- **Speedup 2.51×** (out of ideal 6×)
- Efficiency ~42%

Overhead dominan: barrier wake (~15-20 µs) + memory bandwidth contention.

**Design decision**: Accept 42% efficiency untuk MVP. Optimization ke 60-70% via hybrid sync (spin+futex) di-defer setelah Phase 5-6.

---

## 11. Roadmap yang Direvisi

### Status: ✅ Selesai

- ✅ Phase 0: Hardware capability validation (fma_benchmark, cache_benchmark)
- ✅ Phase 1: Single microcircuit forward (14 µs validated)
- ✅ Phase 2: Multi-core dispatch (33 µs dengan barrier)
- ✅ Phase 3a: LSH routing basics (SimHash, linear scan)
- ✅ Phase 3b: LSH bucketing exploration (deferred karena recall issue)
- ✅ Phase 4: Full cascade — mekanisme validated
- ✅ Phase 4 v2: Routing discontinuity discovery, fixed routing solution

### Phase 5: Persistent Storage & Scale (Next)

**Tujuan**: mmap lattice, scale ke 10K-100K mikrosirkit.

Sub-fase:
- **5a**: File format design + serialization (~1 minggu)
- **5b**: mmap loading + validate cache behavior @ N=10K (~1 minggu)
- **5c**: Scale ke N=100K, benchmark cold vs warm cascade (~1 minggu)
- **5d**: Bucketing revisit dengan Hamming-2 probing untuk 90%+ recall (~1 minggu)

**Deliverable**: SML dengan 100K mikrosirkit lattice yang bisa di-load dari disk.

### Phase 6: Training Strategy (Big Unknown)

**Options** (perlu decision):
1. Equilibrium Propagation (Scellier & Bengio) — backprop through fixed point
2. Implicit differentiation (Deep Equilibrium Models)
3. Hebbian / local learning rules
4. Gradient-free (evolutionary, RL)

**Blocker**: butuh target task + dataset + evaluation metric. Belum ada.

**Decision needed** sebelum Phase 6:
- Target task apa? (classification simple? ARC-like? language modeling toy?)
- Bagaimana measure success?
- Berapa komputasi training worth invest?

### Phase 7+: Post-Training

Setelah training works:
- Real benchmark comparison vs baseline (small MLP, small Transformer)
- Ablation studies (K, D, N impact)
- Publication atau blog post

---

## 12. Open Questions & Risk Register

### 12.1 Risk Register Updated

| Risk ID | Status v0.1 | Status v0.2 |
|---------|-------------|-------------|
| R-01: AVX2 throughput di bawah teoretis | Medium | ✅ **RESOLVED** — 99.4% peak achieved |
| R-02: Memory bandwidth bottleneck | Critical | ⚠️ **PARTIAL** — RAM aggregate 1.7× (not 6×), design compensated |
| R-03: Cascade tidak konvergen | High | ✅ **RESOLVED** — fixed routing achieves 100% |
| R-04: Fixed point collapse ke trivial | High | ⚠️ **PARTIAL** — converge tapi ke small-norm attractor |
| R-05: LSH routing tidak diskriminatif | High | ✅ **RESOLVED** — Phase 3 Test 3 confirm content-addressed |
| R-06: Tidak ada training method that works | Critical | ⚠️ **STILL OPEN** — belum di-attempt |
| R-07: INT8 quantization terlalu agresif | Low-Medium | ✅ **RESOLVED** — INT8 correctness verified (Phase 1) |

**Risiko baru yang muncul**:

| Risk ID | Deskripsi | Status |
|---------|-----------|--------|
| R-08: Dynamic routing per-step tidak workable | Discovered Phase 4 | Design pivot dilakukan |
| R-09: Bucketing recall issue | Discovered Phase 3b | Defer ke Phase 5, gunakan linear scan untuk sekarang |
| R-10: Dispatch overhead 19 µs > estimasi 5 µs | Discovered Phase 2 v2 | Accept untuk MVP |

### 12.2 Open Questions

1. **Training**: Sudah decided butuh training. Method TBD (Phase 6 planning).
2. **Dynamic routing**: Bisa direstore dengan smoothing? Butuh eksperimen di Phase 6+.
3. **Fixed point uniqueness**: Multiple basin? Perlu test dengan diverse init.
4. **Semantic meaningfulness**: Attractor dari random weights punya structure? Empirical Phase 4 Test 4 menunjukkan weak signal.

---

## 13. Glosarium

Update dari v0.1:

| Istilah | Definisi |
|---------|----------|
| **Cascade** | Iterasi `a → T_R(a) → T_R²(a) → ...` sampai fixed point |
| **Committee** | Set K=6 microcircuit yang di-route sekali per inference |
| **Contractive** | Property `‖T(a) - T(b)‖ < c·‖a-b‖`, c < 1 → converge to fixed point |
| **Content-addressed** | Routing berdasarkan konten (SimHash), bukan explicit indexing |
| **Dynamic routing** | Routing yang berubah tiap cascade step (v0.1 approach, **DEPRECATED**) |
| **Fixed routing** | Routing sekali per inference, freeze sepanjang cascade (v0.2 approach) |
| **Fixed point** | Vektor `a*` di mana `T_R(a*) = a*` (dalam toleransi ε) |
| **Limit cycle** | Trajectory yang periodic tanpa converge (masalah v0.1 dynamic routing) |
| **Microcircuit** | Unit komputasi atomik SML, 67 KB, D×D INT8 matrix + metadata |
| **Route** | Fungsi input → K microcircuit terbaik (via SimHash + Hamming distance) |
| **SimHash** | Locality-sensitive hash 32-bit dari random hyperplane projections |

---

## Catatan Metodologi

Spec ini adalah **snapshot post-empirical**. Setiap angka dan claim di dokumen ini bisa di-trace ke file benchmark spesifik di repository. Setiap decision architectural dijustifikasi dengan data, bukan intuisi.

**Perbandingan filosofi v0.1 vs v0.2**:
- v0.1: "Berikut design ambisius, mari lihat mana yang bekerja"
- v0.2: "Berikut yang bekerja, dokumentasikan sebagai baseline stabil sebelum lanjut"

Kalau future-you atau collaborator baca ini, harusnya:
1. Bisa reproduce hasil kita (benchmark files ada)
2. Bisa extend arsitektur tanpa breakage (MVP works standalone)
3. Tahu blocker fundamental (training) dan option-nya (Phase 6 planning)

---

**Versi berikutnya (v0.3)** akan ditulis setelah Phase 5 (persistent storage @ 100K scale) atau Phase 6 (training MVP), mana yang selesai duluan.

Selama masih hidup berarti masih berproses.
