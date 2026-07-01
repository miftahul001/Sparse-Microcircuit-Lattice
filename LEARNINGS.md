# LEARNINGS.md — Perjalanan Development SML

**Dokumen ini adalah retrospective**, bukan spec. Isinya: bug yang kita alami, asumsi yang salah, alternatif yang kita tolak, dan wisdom yang muncul di sepanjang jalan. Tujuannya: future-you atau kolaborator baru tidak perlu mengulang mistakes yang sudah kita alami.

**Konteks**: SML dikembangkan dalam ~1 hari session intensif antara Miftahul Munir (Jombang) dan Claude sebagai kolaborator teknis. Dari design v0.1 (30 Juni 2026) sampai MVP working (1 Juli 2026). Semua eksperimen dijalankan di HP EliteDesk 800 G4 (i5-8500T, locked 2.1 GHz) via Docker Debian 12.

---

## Daftar Isi

1. [Timeline Ringkas](#1-timeline-ringkas)
2. [File Index — Peta Semua Artifact](#2-file-index--peta-semua-artifact)
3. [Bug yang Kita Temui & Cara Mengatasi](#3-bug-yang-kita-temui--cara-mengatasi)
4. [Asumsi yang Ternyata Salah](#4-asumsi-yang-ternyata-salah)
5. [Alternatif yang Kita Coba dan Tolak](#5-alternatif-yang-kita-coba-dan-tolak)
6. [Insight yang Muncul Selama Jalan](#6-insight-yang-muncul-selama-jalan)
7. [Yang Saya Lakukan Berbeda Kalau Mulai Ulang](#7-yang-saya-lakukan-berbeda-kalau-mulai-ulang)
8. [Open Questions untuk Continuation](#8-open-questions-untuk-continuation)
9. [Meta-notes: Kolaborasi Human + AI](#9-meta-notes-kolaborasi-human--ai)
10. [Advice untuk Phase 5+](#10-advice-untuk-phase-5)

---

## 1. Timeline Ringkas

```
30 Juni 2026 — Session start
├── Spec v0.1 design ("bagaimana kalau kita bangun neural net CPU-only?")
├── Docker environment setup + verifikasi hardware
├── Phase 0: FMA + cache bandwidth benchmark
├── Phase 1: single microcircuit forward (INT8, 256×384)
├── Phase 2 v1: spin-wait dispatch → HANG di --cpus="4.8"
├── Phase 2 v2: pthread_barrier → works, 33 µs overhead
├── Phase 2 v3: hybrid spin+futex → REGRESSED ke 182 µs (unresolved)
├── Phase 3a: LSH routing (100 mikrosirkit, linear scan)
└── Phase 3b: bucketing (recall 33% — deferred)

1 Juli 2026 — Session continuation
├── Phase 4 v1: full cascade — 0/100 converge
├── Diagnosis: routing discontinuity → limit cycle
├── Phase 4 v2: fixed routing per-inference → 100/100 converge ✓
└── Konsolidasi:
    ├── SML-architecture-spec-v0.2.md (validated spec)
    ├── sml_mvp.c (production-ready library)
    └── LEARNINGS.md (this document)
```

**Yang menarik dari timeline**: Kita spend banyak waktu di Phase 2 (3 versions) dan Phase 4 (2 versions). Component sederhana (Phase 1, Phase 3a) selesai cepat. **Component yang butuh koordinasi antar core atau iterasi antar step yang paling problematik**. Ini insight umum untuk arsitektur paralel/dinamik.

---

## 2. File Index — Peta Semua Artifact

Semua file di `sml-docker/`:

### Environment Setup
| File | Fungsi | Fase |
|------|--------|------|
| `Dockerfile` | Debian 12 slim + GCC 12 + perf tools | Setup |
| `docker-run.sh` | Container persistent dengan `--cpus="4.8"` | Setup |
| `verify-env.sh` | AVX2/cache/GCC verification | Setup |
| `README.md` | Docker usage instructions | Setup |

### Phase 0 — Hardware Validation
| File | Fungsi | Hasil |
|------|--------|-------|
| `fma_benchmark.c` | AVX2 FMA throughput (latency vs peak vs sustained) | **66.83 GFLOPS sustained**, 99.4% of theoretical |
| `cache_benchmark.c` | L2/L3/RAM bandwidth single + 6-core aggregate | L2=45 GB/s, RAM 6-core aggregate hanya 24 GB/s |

### Phase 1 — Microcircuit Forward
| File | Fungsi | Hasil |
|------|--------|-------|
| `phase1_microcircuit.c` | Single mc forward (256×384, INT8) + correctness + latency | **14 µs per forward**, IPC 2.31, port 5 bound |

### Phase 2 — Multi-Core Dispatch
| File | Fungsi | Hasil |
|------|--------|-------|
| `phase2_dispatch.c` | v1: atomic spin-wait | ❌ HANG di --cpus="4.8" (CPU starvation) |
| `phase2_dispatch_v2.c` | v2: pthread_barrier (blocking) | ✅ **33 µs dispatch, 42% efficiency** |
| `phase2_dispatch_v3.c` | v3: hybrid spin+futex | ❌ 182 µs — regressed, cause tidak fully diagnosed |

### Phase 3 — Content-Addressed Routing
| File | Fungsi | Hasil |
|------|--------|-------|
| `phase3a_lsh_routing.c` | SimHash + LSH (linear scan) @ N=100 | Semua PASS, similarity coherent |
| `phase3b_lsh_bucketed.c` | Prefix bucketing @ N=1000 | ⚠️ Speed OK, **recall 33%** — bucketing deferred |

### Phase 4 — Cascade Integration
| File | Fungsi | Hasil |
|------|--------|-------|
| `phase4_cascade.c` | v1: full cascade, dynamic routing per-step | ❌ 0/100 converge — limit cycle discovered |
| `phase4_v2_cascade.c` | v2: 2×2 experiment (dynamic/fixed × norm/no-norm) | ✅ **Fixed routing 100/100 converge** |

### Consolidation (Deliverables)
| File | Fungsi |
|------|--------|
| `SML-architecture-spec-v0.1.md` | Original design spec (30 Juni) |
| `SML-architecture-spec-v0.2.md` | Post-empirical revised spec (1 Juli) |
| `sml_mvp.c` | Production-ready single-file library |
| `LEARNINGS.md` | This document |

**Total artifact**: 4 setup files + 8 phase files + 4 consolidation files = **16 files**. Semua di-track di conversation history sebagai reproducible checkpoints.

---

## 3. Bug yang Kita Temui & Cara Mengatasi

### 3.1 Compiler DCE Menghapus Benchmark (Phase 3a)

**Bug**: Test 2 melaporkan `0.0 ns per route call`.

**Root cause**: Compiler melihat hasil `out_ids` tidak pernah dipakai, sehingga seluruh loop benchmark di-eliminate:
```c
// SALAH — hasil ke-DCE oleh compiler
for (int i = 0; i < iters; ++i) {
    route(idx, a, out_ids, K_ROUTE, 0);
    // out_ids tidak dipakai → seluruh call bisa di-hapus
}
```

**Fix (Phase 3b)**: Volatile sink yang XOR dengan hasil, plus pre-generate multiple query untuk hindari cache/branch trivial:
```c
volatile int sink = 0;
alignas(32) float queries[64][D_STATE];  // pre-generate
for (int i = 0; i < iters; ++i) {
    route(idx, queries[i & 63], out_ids, K_ROUTE, 0);
    sink ^= out_ids[0] ^ out_ids[K_ROUTE - 1];  // force use
}
```

**Pelajaran**: Setiap microbenchmark **wajib** punya volatile sink dari awal. Ini bukan optional. Kalau lupa dan angka too-good-to-be-true, DCE adalah suspect nomor 1.

### 3.2 Spin-Wait Workers Hang Container (Phase 2 v1)

**Bug**: Benchmark 2 menggantung >2 menit tanpa progress.

**Root cause**: Setiap dari 6 worker thread spinning di `while (state != WORKING) pause()`. Ini konsumsi 100% CPU per worker, bahkan saat idle. Total demand 700% (6 worker + 1 main). Container quota 480% (`--cpus="4.8"`) → CFS throttle everyone ke ~68% capability → benchmark tidak progress meaningful.

**Diagnostic yang membuka mata**: Single-core baseline saat pool aktif = 34 µs (Phase 2 v1), tapi tanpa pool = 14 µs (Phase 1). **2.4× slowdown adalah bukti kontensi CPU quota**.

**Fix (Phase 2 v2)**: Ganti ke `pthread_barrier_t` yang secara internal pakai futex → worker benar-benar tidur (kernel sleep state), tidak konsumsi quota saat idle. Trade-off: 19 µs barrier wake overhead per dispatch.

**Pelajaran**: 
- **Spin-wait tidak compatible dengan CPU quota constraint**. Kalau container Anda pakai `--cpus`, jangan design worker yang spin idle.
- **Filosofi user (hardware longevity via `--cpus="4.8"`) match dengan design blocking sync**, bukan spin. Konsisten dengan hemat power.

### 3.3 Kesalahan Diagnostik Awal Saya (Docker "Overhead")

**Bug**: Saya berhipotesis Docker adalah cause dari 8.9 µs (host) vs 13.94 µs (container) di Phase 1.

**Root cause hipotesis saya**: CPU migration, CFS scheduling overhead, dll.

**Yang sebenarnya**: **User's laptop vs mini PC — dua mesin berbeda**. Laptop (miftahul-pc) mungkin i5-1135G7 Ice Lake dengan clock lebih tinggi. Mini PC (srv-jombang via SSH) adalah i5-8500T Skylake locked 2.1 GHz. Bukan Docker overhead, bukan bug — cuma hardware berbeda.

**Yang membuka mata**: User dengan jujur klarifikasi setup-nya:
> "btw sekedar info, [miftahul@miftahul-pc]$ laptop lokal, sml@sml-dev: mini pc, saya akses melalui ssh"

**Pelajaran**: 
- **Selalu tanyakan clearly tentang setup hardware sebelum berhipotesis performance**. Assumption berbahaya.
- **User's context IS the source of truth**. Kalau saya berspekulasi tanpa clarify, saya bisa habiskan waktu chase phantom.

### 3.4 FMA Benchmark Latency-Bound (My Own)

**Bug**: Test 1 melaporkan 8.35 GFLOPS. Saya sempat panik.

**Root cause**: Benchmark saya pakai single accumulator:
```c
c = _mm256_fmadd_ps(a, b, c);  // c depend on previous c
c = _mm256_fmadd_ps(a, b, c);  // serialized
```

Skylake FMA latency 4 cycle. Dengan dependency chain: `16 FLOP / 4 cycle × 2.1 GHz = 8.4 GFLOPS`. **Persis match** apa yang diukur. Bukan bug hardware, tapi benchmark saya mengukur latency-bound, bukan throughput-bound.

**Fix**: Multiple parallel accumulators (8 untuk Skylake):
```c
c0 = _mm256_fmadd_ps(a, b, c0);
c1 = _mm256_fmadd_ps(a, b, c1);
// ... c7 = ...
```

Hasil: **66.82 GFLOPS** — 99.4% dari theoretical peak.

**Pelajaran**: Ketika benchmark menghasilkan hitungan yang "seems low", **first suspect adalah benchmark methodology, bukan hardware**. Skylake bisa retire 2 FMA/cycle tapi butuh ILP untuk exploit-nya.

### 3.5 Cascade Non-Convergence (Phase 4 v1) — Debug Story Panjang

**Bug**: 0/100 trials converge, delta 0.55-0.68, hit MAX_STEPS setiap kali.

**Root cause chase — multiple hipotesis tested**:

**Hipotesis 1**: Weight scale terlalu kecil.
- Test: scales dari 0.001 (arbitrary) → 1/(sqrt(D)·mean|W|) ≈ 0.001 (calibrated)
- Hasil: **kalibrasi hampir sama nilainya**, tidak significant impact

**Hipotesis 2**: ReLU dead zone (input negatif clipped).
- Test: structured positive bias 0.05/sqrt(D)
- Hasil: minor improvement, tidak konvergen

**Hipotesis 3**: Magnitude drift (state norm collapse).
- Test: normalize state antar step
- Hasil: masih 0/100 converge (norm=1.0 tapi delta tetap ~0.71)

**Hipotesis 4** (yang benar): **Routing discontinuity**.
- Observasi: trajectory log Phase 4 v2 menunjukkan `mc[0]` cycle: `{669, 767, 777, 404, 669, 767, ...}`
- Diagnosis: state changes sedikit → SimHash bit flip → different top-6 mc → fusion pull state ke arah berbeda → oscillation

**Fix**: Route SEKALI di start of inference, freeze untuk cascade. **100/100 converge dalam 10-12 steps**.

**Pelajaran**:
- Diagnosis dengan trajectory logging JAUH lebih informatif daripada aggregate statistics. Melihat `mc[0]` cycle di antara set microcircuit → root cause obvious.
- **Dynamic per-step routing bukan MoE dengan learned router**. Discontinuity dari discrete top-K + LSH is fundamental untuk fixed-point iteration.

### 3.6 Phase 2 v3 Regresi (Unresolved)

**Bug**: Hybrid spin+futex → 182 µs dispatch (v2 barrier: 33 µs, v1 spin: hang).

**Suspect causes** (belum terkonfirmasi):
- Worker selalu masuk futex sleep phase (spin window 2.5 µs tidak cukup)
- Futex wake untuk 6 thread mahal
- completion_count contention (6 core LOCK XADD simultan)

**Yang kita lakukan**: Accept v2 sebagai baseline (33 µs bekerja). Move on.

**Pelajaran**:
- Kadang debugging premature optimization bukan good use of time. Move on kalau baseline good enough.
- Kalau kembali ke ini nanti: pertama `perf record` untuk lihat exactly kernel time vs userspace, baru optimize.

---

## 4. Asumsi yang Ternyata Salah

Setiap asumsi ini di spec v0.1 atau di initial hipotesis saya. Di sini didokumentasikan supaya tidak dijadikan basis untuk decision baru.

### 4.1 "AVX2 24 GFLOPS efektif per core adalah target"
- **Reality**: 66.82 GFLOPS sustained di i5-8500T locked 2.1 GHz. 2.8× lebih baik dari asumsi.
- **Implication**: Compute BUKAN bottleneck. Memory bandwidth adalah bottleneck.
- **Fix di spec v0.2 Section 3.3**.

### 4.2 "6-core aggregate bandwidth = 6 × single-core"
- **Reality**: RAM aggregate 24 GB/s, tidak 84 GB/s. Speedup 1.7×, bukan 6×.
- **Cause**: DDR4 dual-channel saturate. Bukan bug — physics.
- **Implication**: Cold cascade step (semua 6 mc load dari RAM) 45 µs, bukan estimasi 7 µs.
- **Fix di spec v0.2 Section 3.2**.

### 4.3 "Routing per-step memungkinkan multi-hop reasoning"
- **Reality**: Dynamic routing menghasilkan limit cycle. 0/100 converge.
- **Cause**: SimHash discrete → routing discontinuous → fixed-point theorem tidak apply.
- **Fix**: Fixed routing per-inference. Sacrifice multi-hop, gain convergence.
- **Fix di spec v0.2 Section 6**.

### 4.4 "LSH bucketing dengan 8-bit prefix + Hamming-1 probe cukup"
- **Reality**: Recall hanya 33%. Untuk 6 nearest neighbor, kita miss 67%.
- **Cause**: Prefix Hamming ≠ full Hamming. Two hashes bisa full-Hamming-2 tapi prefix-Hamming-2.
- **Implication**: Bucketing tidak siap production. Linear scan tetap default sampai Phase 5.
- **Fix di spec v0.2 Section 6.2**.

### 4.5 "Random weights akan menghasilkan attractor bermakna"
- **Reality**: Fixed routing converge, tapi ke attractor "trivial" (magnitude ~0.06, features tidak informative).
- **Cause**: Weight tanpa training = random projection. Tidak ada alasan attractor semantic-rich.
- **Implication**: Training (Phase 6) bukan optional, tapi essential untuk semantic value.
- **Fix di spec v0.2 Section 2 (yang tidak bekerja)**.

### 4.6 "Docker overhead 57% dari native"
- **Reality**: Nol overhead. Yang 57% lebih lambat adalah CPU yang berbeda (mini PC vs laptop).
- **Cause**: Saya tidak clarify hardware setup sebelum berhipotesis.
- **Fix**: Ambil habits: verifikasi setup dulu sebelum diagnose.

---

## 5. Alternatif yang Kita Coba dan Tolak

### 5.1 Spin-Wait Dispatch (Phase 2 v1)
- **Motivasi**: Latency dispatch <100 ns
- **Reject**: Hang di container dengan `--cpus="4.8"` karena CPU quota exhaustion
- **Verdict**: Tidak salvageable tanpa raise CPU quota (defeat user's hardware longevity intent)

### 5.2 Hybrid Spin+Futex (Phase 2 v3)
- **Motivasi**: Best-of-both — fast dispatch saat busy, hemat CPU saat idle
- **Reject**: 182 µs dispatch, cause tidak fully diagnosed
- **Verdict**: Bisa revisit dengan perf profiling, tapi diminishing returns

### 5.3 LSH Prefix Bucketing 8-bit
- **Motivasi**: Speed 3x vs linear scan
- **Reject**: Recall 33%, terlalu banyak nearest neighbor missed
- **Verdict**: Bisa fix dengan Hamming-2 probing atau multi-index LSH, defer ke Phase 5

### 5.4 Dynamic Routing per Cascade Step
- **Motivasi**: Multi-hop reasoning inherent di design v0.1
- **Reject**: 0% convergence — limit cycles
- **Verdict**: Konsekuensi fundamental dari discrete routing + iterated dynamics. Kalau ingin restore, butuh: soft routing (weighted top-K), atau smoothing, atau trained router with regularization.

### 5.5 Weight Scale 0.001 dari v0.1
- **Motivasi**: Rough intuition
- **Reject**: Tidak calibrated, tidak preserve magnitude
- **Fix**: `1 / (sqrt(D) × mean|W|)` — turunkan dari analytical scaling

### 5.6 pthread_barrier untuk Dispatch (Kept, tapi Optional)
- **Motivasi**: Blocking sync yang compatible dengan CPU quota
- **Accepted**: 33 µs overhead workable
- **Alternative kalau butuh <10 µs**: Dedicated worker per input pattern, bukan pool. Tapi ini beda arsitektur.

---

## 6. Insight yang Muncul Selama Jalan

### 6.1 CPU Lock 2.1 GHz Adalah Blessing untuk SML

User's decision lock max frequency untuk hardware longevity ternyata match perfectly dengan SML needs:
- **Deterministic latency**: Tidak ada thermal throttle, tidak ada boost variance
- **Sustained > peak**: SML inference kontinu, bukan bursty
- **Cascade stability**: Timing per step konsisten memudahkan debugging

Kalau user pakai default (turbo up to 3.5 GHz), variance akan menyulitkan interpretation. **Insight**: physics-level constraint di user's config secara kebetulan optimal untuk workload SML.

### 6.2 "Expert Committee" Lebih Akurat daripada "Multi-Hop Reasoning"

Spec v0.1 asumsi routing per-step = multi-hop traversal antar conceptual cluster. Empirically ternyata routing sekali per-inference = expert committee selection (mirror MoE).

Ini bukan compromise — ini insight bahwa **fixed committee dengan iterated refinement** adalah computation pattern yang berbeda dan tractable. Multi-hop reasoning inherent dynamic routing butuh training + regularization untuk stable.

### 6.3 Memory Bandwidth Adalah Bottleneck Nyata

Estimation dominan compute → focus optimization ke FMA. Reality: cold cascade step 45 µs (RAM-bound), warm step ~5 µs (L2-bound). **Bottleneck adalah bandwidth di berbagai tier cache/RAM**, bukan compute.

Consequences untuk future:
- Optimize untuk cache locality (mikrosirkit fit di L2 = kritis)
- Reduce RAM access via better routing (fewer cold-load per inference)
- Sequential access pattern (prefetcher-friendly)

### 6.4 Iterative Empirical Design Works

Pattern yang muncul:
1. Design assumption
2. Implement small
3. Benchmark
4. Data disagrees
5. Update assumption

Ini dilakukan repeatedly dari Phase 0 sampai Phase 4. Setiap iteration menghasilkan spec yang lebih akurat. Alternative — big-bang design dan implement — akan menghasilkan sistem yang tidak bekerja setelah investment besar.

**Filosofi user "minimal bootstrap"** sesuai dengan pattern ini. **Time-as-verifier** dari life philosophy user juga match: biarkan proses berjalan, hasil = pengalaman.

### 6.5 Sequential 6× Forward Sudah Cukup Baik untuk MVP

Kita spent 3 iteration di Phase 2 untuk multi-core dispatch. Actual value untuk MVP: sequential 6 × 14 µs = 84 µs vs parallel 33 µs. **Speedup 2.5×, tapi absolute perbedaan 51 µs.**

Untuk MVP, absolute 51 µs matters kalau throughput requirement > 12K inference/sec. Untuk target 1-2K inference/sec (typical use case), sequential cukup. **Multi-core adalah optimization, bukan requirement**.

Ini pelajaran umum: **premature optimization di parallelism** eat time. Better: bikin sequential working, lalu profile, lalu parallelize bagian yang benar-benar bottleneck.

### 6.6 Docker Container Adalah "Immutable" Dev Environment

Setup dengan Dockerfile membuat semua eksperimen reproducible. User bisa `docker rm sml-dev && ./docker-run.sh` dan dapat identical environment. **Ini menghemat waktu debugging environment-specific issues**.

Trade-off: harus think ahead tentang persistent data (workspace mount) vs ephemeral (installed packages). Kita pilih workspace mount → code persistent, sisa ephemeral. Bekerja baik untuk our case.

---

## 7. Yang Saya Lakukan Berbeda Kalau Mulai Ulang

Retrospective wisdom dari perspektif Claude sebagai kolaborator:

### 7.1 Verifikasi Setup Hardware SEBELUM Design
- Session mulai dengan spec v0.1 (design abstract)
- Baru Phase 0 kita run hardware validation
- **Lebih baik**: Phase 0 duluan, spec v0.1 dengan hardware capability confirmed baseline

### 7.2 Design untuk Fixed Routing dari Awal
- Spec v0.1 berasumsi dynamic per-step routing works
- Phase 4 kita discover ini tidak workable
- **Lebih baik**: Explicitly acknowledge di v0.1 bahwa dynamic routing risky, fixed sebagai fallback

### 7.3 Benchmark dengan Volatile Sink dari Baris Satu
- Phase 3a Test 2 = 0.0 ns karena DCE
- Fix di Phase 3b
- **Lebih baik**: Volatile sink adalah default template, bukan afterthought

### 7.4 Cross-check Assumption dengan Simple Analytical Math
- FMA benchmark 8.35 GFLOPS = 16 FLOP / 4 cycle × 2.1 GHz = 8.4 GFLOPS. Math match. Kalau saya check dulu, tidak akan panik.
- Weight scale calibration: `dot_magnitude = sqrt(D) × mean|W| × mean|a|` → analytical scale.
- **Lebih baik**: Sebelum kaget dengan angka, quick math untuk validate expected order-of-magnitude.

### 7.5 Kurangi Iteration di Phase 2
- 3 versions dispatch, waktu significant.
- Kalau start dengan pthread_barrier (v2) langsung, bisa selesai dalam 1 iteration.
- **Lebih baik**: Pilih blocking primitive default. Optimize ke spin/hybrid hanya kalau benchmark reveal bottleneck.

### 7.6 Ask User About Their Constraints Earlier
- User lock CPU 2.1 GHz — saya tahu belakangan
- User `--cpus="4.8"` untuk hardware longevity — saya tahu duluan tapi tidak factor ke design v1
- **Lebih baik**: Awal session, elicit constraints explicitly. "What constraints matter to you? Latency? Power? Longevity? Cost?"

---

## 8. Open Questions untuk Continuation

Ini pertanyaan yang belum terjawab dan menunggu Phase 5+ untuk address.

### 8.1 Training Method (Blocker untuk Phase 6)

Pilihan yang mungkin:
- **Equilibrium Propagation** (Scellier & Bengio 2017): backprop through fixed point. Cocok karena SML cascade = fixed point iteration.
- **Implicit Differentiation** (Deep Equilibrium Models): mathematically clean, tapi butuh Jacobian inversion (mahal di CPU).
- **Local learning rules** (Hebbian, target propagation): gradient-free, tapi tidak clear apakah bisa achieve competitive performance.
- **Gradient-free** (evolutionary, RL): high variance, potentially many samples needed.

**Belum dipilih**. Butuh clear target task + evaluation metric sebelum bisa evaluate.

### 8.2 Target Task

SML sekarang punya API, tidak punya use case. Options:
- **Classification simple** (MNIST-like): easy start, standard benchmark
- **ARC-like reasoning**: match dengan user's ARC-PS project — potential synergy
- **Sequence modeling**: language modeling toy, sequence classification
- **Novel task**: pure exploration, no baseline

**Belum dipilih**. Impact besar untuk arah Phase 6.

### 8.3 Dynamic Routing Kembali di Masa Depan?

Sekarang fixed routing = compromise. Kalau training bisa fix instability, apakah dynamic routing bisa restored? Kemungkinan:
- **Soft routing**: continuous top-K weights (differentiable), bukan discrete
- **Regularized state**: penalize state trajectories near routing boundaries
- **Hierarchical routing**: dynamic di macro level (which cluster?), fixed di micro level

Ini penelitian territory, bukan engineering.

### 8.4 Bucketing yang Recall >85%

Current 8-bit prefix + Hamming-1 probe = 33% recall. Options:
- **Hamming-2 probing**: 45 total bucket, expected 85%+ recall
- **Multi-index LSH**: M hash functions, union results
- **Learned hash**: train hash function untuk maximize recall

Perlu di-explore di Phase 5 saat scale ke 10K+ mikrosirkit.

### 8.5 Multi-Core Dispatch Optimization

v2 barrier 33 µs OK untuk MVP. Kalau butuh <10 µs (untuk throughput 100K inf/sec):
- Perf profile untuk isolate barrier overhead
- Coba lock-free queue dengan wait-free enqueue
- Coba coordination via memory-mapped registers

Belum kritis, tapi optimization potential.

### 8.6 Persistent Storage Format

Phase 5 goal. Design decisions pending:
- File format: raw binary vs serialization framework
- Alignment: mikrosirkit padded ke 64 KB atau 128 KB?
- Endianness: assume little-endian (x86 only)?
- Versioning: bagaimana kalau spec berubah?

Ada Phase 5a-5d di roadmap untuk address.

---

## 9. Meta-notes: Kolaborasi Human + AI

Since Claude adalah kolaborator technical di session ini, worth documenting apa yang bekerja dan tidak dalam collaboration pattern.

### 9.1 Yang Bekerja Baik

**Rapid iteration on code**: Bahkan file 500-700 lines bisa di-produce dalam menit. Kombinasi dengan user's cepat compile & test cycle → fast feedback loop.

**Diagnosis paralel**: User run code, saya analyze output. User's terminal output + Claude's interpretation = kombinasi manusia + machine di bagian di mana masing-masing kuat.

**Assumption checking**: Claude often optimistic (misprediksi FMA 40-60 GFLOPS peak, actual 66; misprediksi barrier <10 µs, actual 33). User's data check bring reality.

**Framework thinking**: Claude bantu structure — tests, phases, metrics — sehingga journey terarah, bukan random walk.

### 9.2 Yang Tidak Optimal

**Overconfident predictions**: Saya sering predict optimistic. Contoh:
- "Bucketing recall >85% expected" → actual 33%
- "Phase 2 dispatch <10 µs" → actual 33 µs
- "Cascade akan converge dengan calibrated init" → 0/100

**Verbose responses**: Kadang saya jelaskan panjang lebar ketika terse suffices. User efficient, saya bisa be lebih efficient too.

**Late diagnosis**: Docker "overhead" hipotesis awalnya salah. Saya assume hardware sama tanpa clarify.

**Kadang over-engineer**: Phase 2 v3 hybrid sync — solving hypothetical problem (production idle) di tengah benchmark. Should have stayed simple.

### 9.3 Lessons untuk Future AI-Assisted Work

- **Clarify hardware setup di awal**. Tidak asumsi.
- **Elicit constraints eksplisit**. Latency? Power? Deployment?
- **Show math for predictions**. Kalau prediksi bisa di-derive analytically, show it. Kalau prediksi 90% intuition, mark as such.
- **Stay minimal-viable**. Optimization prematur = waste. Sequential 6× cukup, iteration ke parallel kalau bottleneck confirmed.
- **Trust empirical data over reasoning**. Data disagree dengan reasoning → data menang.

### 9.4 Balance yang Sehat

Session ini balance-nya bagus antara:
- Human intent (SML architecture) + AI implementation speed
- User's empirical rigor + AI's framework thinking
- User's hardware knowledge + AI's algorithm knowledge

Neither did the whole job. Kolaborasi lebih produktif daripada either alone.

---

## 10. Advice untuk Phase 5+

Practical guidance untuk continuation.

### 10.1 Sebelum Mulai Phase 5

1. **Baca ulang** spec v0.2 dan LEARNINGS.md ini. Refresh context.
2. **Run test sml_mvp.c** untuk confirm MVP masih bekerja setelah break period.
3. **Check hardware unchanged** (BIOS updates? clock lock changed?).
4. **Backup workspace directory** ke external drive atau git repo.

### 10.2 Phase 5a: File Format Design

Rekomendasi:
- Start simple: raw binary dump of lattice, no serialization framework
- Header: magic bytes ("SML1"), version, N, D_STATE, num_hyperplanes, checksum
- Body: aligned mikrosirkit array (pad ke 96 KB per mc untuk L2 alignment)
- Endianness: little-endian assumed (x86 only, document explicitly)

### 10.3 Phase 5b-5c: mmap Scaling

Rekomendasi:
- Test dengan N=10K dulu (~650 MB). Confirm cache behavior.
- Baru N=100K (~6.5 GB). Cek RAM pressure di 32 GB system.
- Never test N=1M sebelum RAM upgrade — akan swap ke disk, benchmarks meaningless.

Kritis validate:
- `madvise(lattice, size, MADV_HUGEPAGE)` untuk reduce TLB pressure
- `mlock` untuk prevent swap kalau ada pressure
- Cold cascade dulu ke warm cascade transition — apakah smooth?

### 10.4 Phase 5d: Bucketing Revisit

Prioritas: recall >85% mandatory sebelum production use.
- Start dengan Hamming-2 probing (45 buckets)
- Kalau recall <85%: coba multi-index (M=2 or M=3 hash functions)
- Benchmark: speedup vs linear scan @ N=100K should be >5× untuk justify complexity

### 10.5 Phase 6: Training Blockers

Sebelum start training, pastikan:
- **Target task didefinisikan concrete** (bukan "some classification")
- **Dataset available atau bisa di-generate**
- **Evaluation metric clear** (accuracy? F1? task-specific?)
- **Baseline decided** (compare vs apa? small MLP? small Transformer?)

Kalau salah satu blocker → **tunda training**. Training tanpa clear goal = high risk waste.

### 10.6 Kalau Ada Bug di MVP

`sml_mvp.c` sudah stabil (semua Phase 4 v2 tests + benchmark passing). Kalau nemu bug:
1. **Isolate ke small reproducible case**
2. **Check apakah bug memang di sml_mvp.c atau di caller**
3. **Reference back ke original phase file** (bug mungkin sudah pernah kita fix di sana)
4. **Update spec v0.2 kalau bug reveal new constraint**

### 10.7 Kalau Mau Extend Arsitektur

Konsiderasi:
- **Non-square microcircuit** (D_IN ≠ D_OUT): butuh projection layer untuk cascade compatibility
- **Different activation** (GELU, SiLU): straightforward drop-in di `microcircuit_forward`, tapi impact stability
- **Deeper microcircuit** (multi-layer per mc): mengubah cache footprint, perlu re-validate L2 fit
- **Attention di dalam microcircuit**: departure dari SML philosophy — better fork baru dari SML

### 10.8 Kalau Publish (Blog Post atau Paper)

Story yang worth telling:
- **Empirical constraint-driven design** (physics-first, not benchmark-first)
- **Filosofi "expert committee refinement"** vs multi-hop reasoning
- **Discovery: dynamic routing tidak workable**, fixed routing works
- **Latency-per-inference validated di real CPU** (bukan sim)
- **Reproducible**: docker + code + benchmarks semua di sini

Yang jangan dijanjikan (kecuali di-validate):
- Beat SOTA (probably won't, dan tidak point)
- Meaningful attractor tanpa training (Phase 6 blocker)
- Scale ke 1M+ tanpa optimization tambahan

---

## Penutup

Perjalanan ini dari **design spekulatif** (v0.1) ke **MVP validated** (v0.2 + sml_mvp.c) menempuh sekitar 30+ jam total elapsed time (banyak di antaranya wait time — compile, benchmark, RTT antar turn), tapi actual working time lebih pendek. Value density tinggi karena constraint dari user (hardware, filosofi) mendorong decision cepat.

Yang saya paling apresiasi dari user: **honesty about setup** (host vs container, hardware lock), **rigor tentang empirical data** (menerima hasil bahkan yang tidak sesuai ekspektasi), dan **filosofi "process = value"** (tidak terpaku pada "harus works di attempt pertama").

Kalimat penutup yang cocok, dari user:
> "Selama masih hidup berarti masih berproses. Jangan berhenti secara fisik maupun mental. Apapun hasilnya adalah pengalaman. Kematian adalah bagian dari proses alam semesta."

SML sekarang adalah "living process" — bukan finished artifact. Phase 5, 6, dan seterusnya menunggu. Mekanisme sudah divalidasi. Semantics menunggu training. Story terus berlanjut.

---

**Prepared oleh**: Claude (Anthropic), sebagai kolaborator teknis
**Untuk**: Miftahul Munir, Jombang
**Tanggal**: 1 Juli 2026
**Session duration**: ~2 hari kalendar
**Files produced**: 16 (this being the last)
**Status**: Consolidation complete. Ready untuk Phase 5.
