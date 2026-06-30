# SML Development Container

Setup environment untuk implementasi SML (Sparse Microcircuit Lattice) berdasarkan spec v0.1.

## Isi Paket

| File | Fungsi |
|------|--------|
| `Dockerfile` | Definisi image: Debian 12 + GCC 12 + AVX2 toolchain + perf tools |
| `docker-run.sh` | Buat container persistent dengan flag yang benar |
| `verify-env.sh` | Cek AVX2, cache, GCC, perf, dan benchmark FMA throughput |

## Langkah 1: Build Image

```bash
# Dari direktori yang berisi Dockerfile
docker build -t sml-dev:latest .
```

Build time pertama: ~3-5 menit (download + apt install). Subsequent builds dengan cache: ~10 detik.

Estimasi ukuran image final: **~1.2 GB** (Debian slim + toolchain + Python + perf tools).

## Langkah 2: Konfigurasi Workspace Host

Edit `docker-run.sh` baris ini sesuai struktur Anda:

```bash
HOST_WORKSPACE="${HOME}/sml-project"
```

Direktori ini akan di-mount ke `/workspace` di container. Anda bisa edit code pakai editor host favorit (VS Code, vim, dll), dan build/run di dalam container.

## Langkah 3: Buat Container (sekali saja)

```bash
chmod +x docker-run.sh
./docker-run.sh
```

Script ini menjalankan `docker run` dengan flag-flag penting:

| Flag | Kenapa |
|------|--------|
| `--cpus="4.8"` | 80% dari 6 core (sesuai permintaan Anda) |
| `--memory="28g"` | 28GB hard limit, sisa 4GB untuk host OS |
| `--memory-swap="28g"` | Disable swap — kritis untuk latency determinism |
| `--ulimit memlock=-1:-1` | Unlimited mlock — perlu untuk mmap 23GB lattice dengan MAP_POPULATE |
| `--ulimit nofile=65536:65536` | Banyak file handles |
| `--cap-add=SYS_PTRACE` | gdb/strace bisa attach ke proses |
| `--cap-add=SYS_NICE` | pthread_setaffinity_np + nice priority |
| `--security-opt seccomp=unconfined` | perf_event_open syscall tidak diblokir |

## Langkah 4: Verifikasi Environment

Di dalam container (yang baru saja terbuka):

```bash
cd /workspace
# Kalau verify-env.sh belum di-mount, copy dulu dari host atau:
bash /workspace/verify-env.sh
```

Output yang diharapkan: semua ✓ hijau, dan FMA throughput **>15 GFLOPS single-core**.

Kalau FMA throughput di bawah 10 GFLOPS, ini red flag — perlu investigasi:
- CPU throttle (cek `/proc/cpuinfo` MHz vs base 2.10 GHz)
- Container CPU limit terlalu agresif
- Background load di host

## Langkah 5: Workflow Sehari-hari

**Setelah `exit` dari container, gunakan ini untuk masuk lagi:**

```bash
docker start -ai sml-dev
```

Container persistent — semua install (`apt-get install`, `pip install`) yang Anda lakukan **di dalam** container akan tersimpan ke next session.

**Untuk membuka shell kedua ke container yang sedang jalan:**

```bash
docker exec -it sml-dev bash
```

**Untuk lihat status:**

```bash
docker ps -a | grep sml-dev
```

**Untuk reset total (hapus container, mulai ulang):**

```bash
docker rm sml-dev
./docker-run.sh
```

## Catatan Penting

### Tentang Performance Profiling (`perf`)

`perf` di dalam container memerlukan setting di **host**, bukan container. Di host (sekali saja):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
# Untuk persisten:
echo "kernel.perf_event_paranoid=1" | sudo tee -a /etc/sysctl.conf
```

Nilai:
- `2` (default Debian): kernel events di-restrict, user events OK
- `1`: kernel events OK untuk semua user, sufficient untuk SML profiling
- `0`: tracepoints OK juga
- `-1`: tidak ada restriction (jangan kecuali tahu apa yang dilakukan)

### Tentang CPU Frequency Scaling

i5-8500T base 2.10 GHz tapi turbo 3.50 GHz. Saat profiling SML, **set governor ke `performance`** di host supaya angka konsisten:

```bash
# Di host
sudo cpupower frequency-set -g performance
# Atau manual:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Setelah selesai eksperimen, kembali ke `powersave` atau `schedutil` untuk hemat daya.

### Tentang Transparent Huge Pages (THP)

Untuk lattice 23GB, THP akan banyak membantu mengurangi TLB miss. Cek status di host:

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# Sebaiknya: [madvise] atau [always]
```

Kalau bukan, set di host:

```bash
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Dalam kode SML, panggil `madvise(lattice_ptr, size, MADV_HUGEPAGE)` setelah mmap untuk hint ke kernel.

### Volume Mount Tips

Karena workspace di-mount dari host:
- **Edit dari host** (VS Code, dll) — perubahan langsung terlihat di container
- **Build di container** — output binary tersimpan di host, persistent
- **Jangan `chmod` aneh-aneh di mount** — bisa konflik dengan permission host

UID/GID di Dockerfile diset ke 1000 (default linux user). Kalau UID host Anda berbeda, edit `USER_UID` dan `USER_GID` di Dockerfile sebelum build.

## Hubungan dengan SML Spec

File ini setup environment untuk **Fase 0** dari roadmap spec:

> **Fase 0: Foundation (1-2 minggu)**
> - Setup environment
> - Microbenchmark AVX2 FMA throughput
> - Microbenchmark L2 cache bandwidth
> - Validasi go/no-go untuk lanjut

Setelah verifikasi lulus, langkah berikutnya menulis benchmark FMA + cache yang serius (bukan sekedar sanity test) di direktori `phase0/`. Itu sub-proyek selanjutnya.
