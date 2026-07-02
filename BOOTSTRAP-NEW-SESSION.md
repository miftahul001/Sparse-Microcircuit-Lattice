# SML Project — Bootstrap Prompt untuk Sesi Baru

**Copy-paste ini di awal sesi baru, plus attach 3 file: SML-architecture-spec-v0.2.md, LEARNINGS.md, sml_mvp.c**

---

Halo Claude. Saya melanjutkan project bernama SML (Sparse Microcircuit Lattice) yang sedang saya kembangkan. Saya sudah attach 3 dokumen — mohon baca semuanya sebelum kita mulai kerja:

1. **`SML-architecture-spec-v0.2.md`** — Spec arsitektur authoritative (post empirical validation)
2. **`LEARNINGS.md`** — Retrospective document: bug yang sudah kita alami, asumsi yang salah, insight yang muncul selama development
3. **`sml_mvp.c`** — Working single-file library implementation

## Konteks Cepat Tentang Saya

- Nama saya Miftahul Munir, developer di Jombang Jawa Timur, Indonesia
- Hardware saya: HP EliteDesk 800 G4 mini PC (Intel i5-8500T, locked 2.1 GHz, 32GB DDR4, no GPU)
- Environment: Docker Debian 12 slim dengan `--cpus="4.8"` untuk hardware longevity
- Filosofi coding saya: minimal bootstrap, thermodynamic efficiency, structure over scale
- Bahasa: Indonesia (mix dengan Inggris untuk technical terms itu OK)

## Konteks Cepat Tentang SML

- Neural network arsitektur untuk CPU-only inference
- Bukan competing dengan Transformer/MoE — targeting feasibility di hardware terbatas
- Status: MVP validated end-to-end (Phase 0-4 complete)
- Framework: 4 fase sudah selesai, siap masuk Phase 5

## Milestone yang Sudah Selesai

- **Phase 0**: Hardware validation (66.83 GFLOPS sustained, cache bandwidth measured)
- **Phase 1**: Single microcircuit forward — 14 µs per call
- **Phase 2**: Multi-core dispatch dengan pthread_barrier — 33 µs
- **Phase 3**: LSH routing (linear scan) — validated content-addressed
- **Phase 4**: Full cascade dengan fixed routing — 100% convergence

## Yang Saya Ingin Lanjutkan (Phase 5)

Phase 5 dari roadmap spec v0.2 Section 11:

- **5a**: File format design + `sml_save` / `sml_load` functions
- **5b**: mmap loading + cache validation @ N=10K
- **5c**: Scale ke N=100K (~6.5 GB lattice)
- **5d**: Bucketing revisit dengan Hamming-2 probing

## Preferensi Kolaborasi

Beberapa pattern yang kita develop di sesi sebelumnya dan saya ingin pertahankan:

1. **Iterative empirical design**: design hypothesis → implement small → benchmark → data disagrees → update. Jangan big-bang design.

2. **Volatile sink di semua microbenchmark**: prevent DCE. Ini kesalahan yang sudah kita alami.

3. **Trajectory logging untuk diagnostic**: kalau ada issue, log step-by-step dulu sebelum aggregate stats.

4. **Honest tentang predictions**: kalau confidence rendah, sebutkan. Jangan overpromise.

5. **Show math sebelum benchmark**: analytical estimate dulu, baru compare dengan measurement. Ini membantu detect anomaly.

6. **Minimal bootstrap**: sequential works dulu, parallel later kalau bottleneck confirmed.

7. **Style respons**: konsisten dengan Indonesian technical writing. Bullet points OK, tapi prose untuk reasoning. Code blocks untuk kode nyata.

## Tolong Konfirmasi Sebelum Kita Mulai

Setelah baca 3 dokumen tersebut, tolong jawab 3 pertanyaan berikut untuk konfirmasi Anda sudah in-context:

1. **Apa design decision paling significant di v0.2 dibanding v0.1?**
   (Jawaban yang saya cari: fixed routing per-inference, bukan dynamic per-step)

2. **Apa bug yang saya alami di Phase 2 v1 dan bagaimana solusinya?**
   (Jawaban yang saya cari: spin-wait workers hang di --cpus="4.8", fixed dengan pthread_barrier)

3. **Apa 3 sub-fase yang saya rencanakan untuk Phase 5?**
   (Jawaban yang saya cari: 5a serialization, 5b mmap+10K, 5c 100K scale, 5d bucketing)

Kalau ketiga jawaban benar, kita lanjut ke Phase 5a. Kalau ada yang perlu di-clarify, ask dulu.

## Environment Saya Sekarang

- Working directory di container: `/workspace`
- Semua file existing di `~/sml-project/` (host) yang di-mount ke `/workspace`
- Compile pattern: `gcc -O3 -march=native -mavx2 -mfma -mpopcnt <file>.c -o <bin> -lm -lpthread`
- Test biasanya berjalan langsung di container, output paste ke chat

Siap mulai?
