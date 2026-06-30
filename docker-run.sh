#!/usr/bin/env bash
# ============================================================
# SML Container Run Script
# Creates a PERSISTENT container (no --rm) yang bisa
# di-start/stop berkali-kali dengan `docker start -ai sml-dev`
# ============================================================

set -euo pipefail

# ============================================================
# Konfigurasi — sesuaikan sesuai kebutuhan
# ============================================================
CONTAINER_NAME="sml-dev"
IMAGE_NAME="sml-dev:latest"

# CPU: 80% dari 6 core = 4.8
CPU_LIMIT="4.8"

# Memory: 28GB dari 32GB (sisa 4GB untuk host OS)
# Lattice SML = 23GB + workspace = ~28GB cukup
MEMORY_LIMIT="28g"
MEMORY_SWAP="28g"  # disable swap (penting untuk latency determinism)

# Host workspace mount — sesuaikan dengan path Anda
HOST_WORKSPACE="${HOME}/sml-project"

# ============================================================
# Buat workspace di host kalau belum ada
# ============================================================
if [ ! -d "${HOST_WORKSPACE}" ]; then
    echo "Creating workspace directory: ${HOST_WORKSPACE}"
    mkdir -p "${HOST_WORKSPACE}"
fi

# ============================================================
# Cek apakah container sudah ada
# ============================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' sudah ada."
    echo "Gunakan: docker start -ai ${CONTAINER_NAME}"
    echo "Atau hapus dulu: docker rm ${CONTAINER_NAME}"
    exit 1
fi

# ============================================================
# Buat dan jalankan container
# ============================================================
# Penjelasan flag penting:
#
#   --cpus="4.8"           80% dari 6 core fisik
#   --memory="28g"          Hard limit memory
#   --memory-swap="28g"     Disable swap (= memory limit) — kritis untuk latency
#   --ulimit memlock=-1:-1  Unlimited mlock (perlu untuk mmap 23GB + MAP_POPULATE)
#   --ulimit nofile=...     Banyak file handles (untuk worker threads)
#   --cap-add=SYS_PTRACE    gdb/strace bisa attach ke proses
#   --cap-add=SYS_NICE      pthread_setaffinity_np + nice/realtime priority
#   --security-opt seccomp=unconfined  perf_event_open syscall tidak diblokir
#   -v /workspace mount     Edit code di host, build/run di container
#
docker run -it \
    --name "${CONTAINER_NAME}" \
    --hostname sml-dev \
    --cpus="${CPU_LIMIT}" \
    --memory="${MEMORY_LIMIT}" \
    --memory-swap="${MEMORY_SWAP}" \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=65536:65536 \
    --cap-add=SYS_PTRACE \
    --cap-add=SYS_NICE \
    --security-opt seccomp=unconfined \
    -v "${HOST_WORKSPACE}:/workspace" \
    -v "${HOME}/.gitconfig:/home/sml/.gitconfig:ro" \
    -w /workspace \
    "${IMAGE_NAME}"

# Setelah keluar dari `docker run -it`, container tetap ada (tidak --rm).
# Untuk masuk lagi: docker start -ai ${CONTAINER_NAME}
