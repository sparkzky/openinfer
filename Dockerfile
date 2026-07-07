# Dev image for openinfer — gives a contributor a working `cargo build --release`
# for the default `qwen3` feature with zero extra setup.
#
# This is a DEV image, NOT a runtime/serving image:
#   - no model weights, no per-SM prebuilt binaries
#   - no NVIDIA GPU driver (the host driver is used via `--gpus`)
#
# Rust channel is intentionally bare `nightly` to match rust-toolchain.toml and
# CI (.github/workflows/ci.yml). The rust-toolchain.toml pinned in the mounted
# repo drives the final toolchain on every `cargo` invocation.

FROM nvidia/cuda:13.3.0-devel-ubuntu24.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ── System build essentials ──────────────────────────────────────────────────
# The devel base already ships build-essential (gcc/g++/make), git, and the CUDA
# 13.3 toolkit (nvcc, cuBLAS) at /usr/local/cuda-13.3. Added below is what the
# build additionally needs:
#   protobuf-compiler — prost/tonic codegen (openinfer-sim, server frontend)
#   cmake, pkg-config — native-dependency discovery
#   libclang-dev      — bindgen FFI for the feature-gated openinfer-comm crates
#   curl, ca-certificates — rustup / uv install
#   python3, python3-venv — uv and the opt-in Triton layer
# apt package versions are transitively pinned by the base image
# (nvidia/cuda:13.3.0-devel-ubuntu24.04); pinning individual apt versions is
# brittle across base-image refreshes, so DL3008 is intentionally not applied.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      protobuf-compiler \
      cmake \
      pkg-config \
      libclang-dev \
      curl \
      ca-certificates \
      python3 \
      python3-venv \
 && rm -rf /var/lib/apt/lists/*

# ── uv (pinned to 0.11.26) — the project's Python tool ──────────────────────
# Used by the opt-in Triton layer below and the onboarding toolchain check.
RUN curl -fsSL https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-unknown-linux-gnu.tar.gz \
 | tar -xz -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv

# ── Non-root dev user ────────────────────────────────────────────────────────
RUN useradd --create-home --uid 1000 --shell /bin/bash openinfer \
 && mkdir -p /workspaces/openinfer \
 && chown -R openinfer:openinfer /workspaces

# ── Rust toolchain (matches rust-toolchain.toml: channel = "nightly") ────────
# Installed AS the dev user so the cargo registry/git cache is writable at build
# time. Components rustfmt + clippy match rust-toolchain.toml.
USER openinfer
ENV HOME=/home/openinfer
RUN curl --proto '=https' --tlsv1.2 -LsSf https://sh.rustup.rs \
 | sh -s -- -y --default-toolchain nightly --profile minimal \
 && "$HOME/.cargo/bin/rustup" component add rustfmt clippy

# ── CUDA environment (base image symlinks /usr/local/cuda -> cuda-13.3) ──────
# OPENINFER_CUDA_SM is deliberately NOT pinned: with `--gpus` the build.rs
# auto-detects the host GPU's compute capability via nvidia-smi. For a GPU-less
# build, pass `-e OPENINFER_CUDA_SM=<cap>` (e.g. 86, 120).
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/home/openinfer/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

WORKDIR /workspaces/openinfer

# Pin the toolchain file into the image so a bare `cargo` resolves nightly even
# before the repo is mounted. The mounted repo's own rust-toolchain.toml takes
# precedence (closer to the working directory) at runtime.
COPY --chown=openinfer:openinfer rust-toolchain.toml ./

# ────────────────────────────────────────────────────────────────────────────
# OPT-IN — Triton venv for the `qwen35-4b` build-time AOT kernel path.
# NOT needed for the default qwen3 build. Uncomment the two lines below to bake a
# Triton venv into the image and point OPENINFER_TRITON_PYTHON at it:
#
#   USER root
#   RUN uv venv /opt/triton-venv --python 3.12 \
#    && uv pip install --python /opt/triton-venv/bin/python triton
#   USER openinfer
#   ENV OPENINFER_TRITON_PYTHON=/opt/triton-venv/bin/python
#
# Or create the venv at runtime in your mounted checkout (the documented flow,
# see docs/playbooks/developer-onboarding.md and README.md):
#   uv venv && uv pip install triton     # build.rs auto-detects .venv/bin/python
# ────────────────────────────────────────────────────────────────────────────

CMD ["bash"]
