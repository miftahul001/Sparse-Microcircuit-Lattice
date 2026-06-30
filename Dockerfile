# ============================================================
# SML Development Environment
# Target: C++ AVX2 inference development on i5-8500T
# Base: Debian 12 (bookworm) slim — glibc, GCC 12.2, ~75MB base
# ============================================================
FROM debian:12-slim

# Non-interactive frontend untuk apt
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set timezone (optional, sesuaikan jika perlu)
ENV TZ=Asia/Jakarta

# ============================================================
# Layer 1: Core build toolchain
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Compiler & build system
    build-essential \
    gcc-12 \
    g++-12 \
    clang-14 \
    clang-format-14 \
    clang-tidy-14 \
    cmake \
    make \
    ninja-build \
    pkg-config \
    # Version control
    git \
    git-lfs \
    # Basic utilities
    ca-certificates \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 2: Performance analysis & debugging tools
# CRITICAL untuk SML: kita perlu measure cache, memory bandwidth, FLOPS
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Debugging
    gdb \
    lldb-14 \
    valgrind \
    strace \
    ltrace \
    # Profiling (perf is the most important tool here)
    linux-perf \
    google-perftools \
    libgoogle-perftools-dev \
    # Static analysis
    cppcheck \
    # Cache analysis (kcachegrind requires X, skip; use callgrind output)
    # Hardware info
    hwloc \
    numactl \
    libnuma-dev \
    # Memory/CPU monitoring
    htop \
    iotop \
    sysstat \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 3: Testing & benchmarking
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtest-dev \
    libbenchmark-dev \
    libbenchmark-tools \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 4: Python untuk utility scripts (BUKAN ML training)
# Untuk: generate test lattice, parse benchmark output, simple plots
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 5: Developer comfort (vim, tmux, etc — boleh skip kalau mount editor host)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    tmux \
    less \
    tree \
    file \
    bc \
    jq \
    ripgrep \
    fd-find \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Layer 6: Install hyperfine untuk benchmarking CLI
# (Debian 12 punya version yang OK, tapi kita install via release untuk newer)
# ============================================================
RUN curl -sSL https://github.com/sharkdp/hyperfine/releases/download/v1.18.0/hyperfine_1.18.0_amd64.deb \
        -o /tmp/hyperfine.deb && \
    apt-get install -y /tmp/hyperfine.deb && \
    rm /tmp/hyperfine.deb && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Set GCC 12 dan Clang 14 sebagai default
# ============================================================
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# ============================================================
# Setup user non-root (best practice; uid 1000 cocok dengan host umum)
# ============================================================
ARG USERNAME=sml
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} && \
    # Beri akses ke perf events tanpa sudo (akan tetap perlu host-level config)
    apt-get update && apt-get install -y --no-install-recommends sudo && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Workspace setup
# ============================================================
WORKDIR /workspace
RUN chown -R ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}

# ============================================================
# Verifikasi default — akan jalan setiap docker start
# ============================================================
CMD ["/bin/bash"]
