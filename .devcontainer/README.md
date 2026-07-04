# openinfer dev container

One-click dev environment: Rust (nightly) + CUDA 13.3 toolkit + build essentials.
No model weights, no GPU driver — the host driver is passed through with `--gpus`.

## Open it

- **VS Code**: install the *Dev Containers* extension, open this repo, run
  **Dev Containers: Reopen in Container**.
- **GitHub Codespaces**: the `devcontainer.json` here is picked up automatically.

Requirements on the host: Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
so `--gpus=all` reaches the container.

## The default build (zero extra steps)

The container ships everything the default `qwen3` build needs. With `--gpus`,
`build.rs` auto-detects the host GPU's compute capability:

```bash
cargo build --release          # pure Rust + CUDA, no Python anywhere
cargo test --release --workspace --lib
```

> No GPU inside the build? `build.rs` needs an SM target, so pass one explicitly:
> `OPENINFER_CUDA_SM=120 cargo build --release` (e.g. `86`, `90a`, `120`).

## Opt-in: Triton for `qwen35-4b`

The default build needs no Python. For the `qwen35-4b` feature (build-time Triton
AOT kernels), create a venv in your mounted checkout — `build.rs` auto-detects
`.venv/bin/python`:

```bash
uv venv && uv pip install triton
cargo build --release --features qwen35-4b
```

(You can instead bake the venv into the image — see the commented layer at the
bottom of `../Dockerfile`.)

## Environment

`CUDA_HOME` (`/usr/local/cuda`), `PATH`, and `LD_LIBRARY_PATH` are set in the
image. `OPENINFER_CUDA_SM` is intentionally left unset so the real GPU is
auto-detected. For the full env-var reference see
[docs/playbooks/developer-onboarding.md](../docs/playbooks/developer-onboarding.md)
and the main [README](../README.md).
