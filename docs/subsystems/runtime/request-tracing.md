# Request Tracing

> **TL;DR:** Request tracing first slice implemented behind `--request-tracing`: the vLLM bridge carries an opt-in trace through `GenerateRequest`, Qwen3 and `openinfer-sim` record scheduler/forward events, and terminal summaries reuse the benchmark-compatible `openinfer_http_trace` JSON log contract.
>
> **Last touched:** 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - confirmed request tracing belongs under shared subsystem docs and must be indexed.
  - `docs/roadmap/execution.md` - request tracing is a `Next` cross-model infrastructure task: spans should cover frontend, scheduler, and forward steps, then bridge measured-vs-simulated gaps.
  - `docs/roadmap/direction.md` - tracing is the online leg of the ledger -> simulator -> tracing loop, used for attribution rather than exact prediction.
  - `docs/subsystems/frontend/cpu-profiling-baseline.md` - frontend overhead is currently measured indirectly; it explicitly asks for timestamps at bridge submit, first token receive, and output emission.
  - `docs/subsystems/scheduler/scheduler.md` - Qwen3's scheduler is a single GPU-owning thread with FCFS prefill priority, unified prefill+decode steps, and existing `TokenEvent::Scheduled` timestamps.
  - `docs/subsystems/kernels/kernel-op-reports.md` - Qwen3 already has feature-gated kernel-call tracing for offline model reports; request tracing should not turn that heavy path on in normal serving.
  - `openinfer-engine/src/engine.rs` - `GenerateRequest` already carries `request_id` and `queued_at_unix_s`; `TokenEvent::Scheduled` already carries queue/schedule timestamps.
  - `openinfer-vllm-frontend/src/bridge.rs` - `LocalEngineBridge::start_request`, `run_request_stream`, `send_token_output`, and `output_loop` are the frontend insertion points.
  - `openinfer-qwen3-4b/src/scheduler.rs`, `openinfer-qwen3-4b/src/scheduler/plan.rs`, `openinfer-qwen3-4b/src/scheduler/effects.rs` - Qwen3 has plan/effects boundaries where scheduler-step spans can be added without changing model math.
  - `openinfer-qwen35-4b/src/scheduler.rs` - Qwen3.5 uses older direct event emission, so full parity there should be follow-up after the first contract is proven.
  - `openinfer-sim/src/lib.rs` - simulated engine is the cheapest way to verify frontend trace propagation without CUDA.
  - `openinfer-server/src/trace_reporter.rs` and workspace `Cargo.toml` - fastrace, OpenTelemetry, and a Chrome trace file reporter already exist, but are not currently wired into serving.
  - `scripts/bench_http_serving.py` and `tests/test_bench_http_serving.py` - server-log attribution already parses `openinfer_http_trace`; changing the marker would break the existing benchmark workflow unless the script is updated.
  - `openinfer-deepseek-v4/src/direct/scheduler.rs` - DeepSeek V4 already emits coarse `openinfer_http_trace` lines, so the first shared contract should preserve that marker and converge field names rather than invent a parallel log stream.
- **Relevant history**:
  - `docs/subsystems/frontend/cpu-profiling-baseline.md` shows the immediate value: decompose the observed ~145ms frontend TTFT overhead rather than infer it from perf samples.
  - `docs/subsystems/kernels/kernel-op-reports.md` shows an existing separation that should be preserved: kernel DAG tracing is feature-gated/offline; request tracing is runtime/serving.
- **Plan**:
  1. Review this spec and settle the first implementation boundary: Qwen3 + simulated engine first, with other model schedulers exporting only existing coarse events until follow-up.
  2. After approval, write a task-by-task implementation plan covering `openinfer-engine`, `openinfer-vllm-frontend`, `openinfer-server`, `openinfer-sim`, `scripts/bench_http_serving.py` if needed, and Qwen3 scheduler files.
  3. Implement tracing behind an explicit opt-in so normal serving keeps near-zero overhead.
  4. Verify with unit tests for span accounting and simulated-engine frontend tests before any GPU checks.
- **Risks / open questions**:
  - OTLP support is already dependency-declared, but wiring a production exporter may widen the first PR. The recommended first pass emits structured log summaries only, while keeping the span model OTLP-compatible.
  - A runtime trace cannot collect per-kernel DAG details without enabling the existing `kernel-call-trace` feature, which is intentionally not normal serving. Forward spans should name phase and batch shape, not every kernel.
  - The first implementation should avoid touching all model schedulers at once. Qwen3 gives the cleanest boundary, `openinfer-sim` gives fast frontend validation, and DeepSeek V4's existing trace output can be left as-is until a convergence follow-up.

## Spec

### Success Criteria

The first request-tracing slice is complete when:

1. Serving can be started with tracing disabled and pays only a cheap disabled-branch cost.
2. Serving can be started with request tracing enabled and produces one request trace per completion with stable request identity.
3. Each trace can attribute TTFT into at least: frontend bridge submit, engine queue wait, scheduler prefill/unified forward, first token emission, bridge output send, and terminal completion.
4. Qwen3 scheduler steps record phase (`prefill`, `decode`, `unified`), pending/decode batch sizes, prompt/computed/cached token counts where available, and step outcome.
5. `openinfer-sim` exercises the same frontend trace propagation without CUDA.
6. The output format is machine-readable by the existing `scripts/bench_http_serving.py --server-log` parser and contains enough stable fields for future simulator comparison.

### Recommended Approach

Use a small `openinfer-engine::request_trace` module with explicit, allocation-light helpers. The request/event contract already lives in `openinfer-engine`; putting trace context in `openinfer-core` would create the wrong dependency direction because `openinfer-core` depends on `openinfer-engine`.

- A `RequestTraceConfig` read once from CLI/env.
- A `RequestTraceContext` carried in `GenerateRequest` as an optional field, so scheduler threads can append events without depending on frontend modules.
- Tiny event/span helpers that are no-ops when tracing is disabled.
- Span names and fields that are stable strings, so future OTLP and simulator joins do not depend on display text.

This keeps model schedulers independent from frontend details and avoids making `tracing` or OpenTelemetry calls ad hoc at every call site.

### Alternatives Considered

1. **Use only `tracing` spans.** This fits Rust conventions and OpenTelemetry export, but cross-thread/request correlation needs careful context propagation through channels. It also risks mixing user-request spans with normal logs unless the subscriber setup is redesigned first.
2. **Use only fastrace.** The repo already has `FileReporter`, and Chrome trace output is useful for local diagnostics. The downside is that OTLP export is not the natural path, and it does not directly feed the current benchmark log parser.
3. **Use explicit request trace events with optional adapters.** This is the recommended first pass. It is the least invasive, easy to test, can emit structured `openinfer_http_trace` summaries immediately, and can later adapt to fastrace/OTLP without rewriting scheduler code.

### Trace Model

Record coarse request lifecycle events:

- `frontend.request_received`: bridge receives an `EngineCoreRequest`.
- `frontend.submit_begin` / `frontend.submit_end`: bridge converts and submits `GenerateRequest`.
- `scheduler.admitted`: request enters an executable prefill/unified step; this aligns with `TokenEvent::Scheduled`.
- `scheduler.step`: one scheduler iteration with phase, pending batch size, decode batch size, and result.
- `model.forward`: executor forward span around `execute_prefill`, `execute_decode`, or `execute_unified`.
- `scheduler.first_token`: first generated token sent to the request channel.
- `frontend.first_token_received`: bridge receives first `TokenEvent::Token`.
- `frontend.output_sent`: bridge sends `EngineCoreOutputs` to vLLM output IPC.
- `request.finished`, `request.error`, or `request.rejected`: terminal outcome.

The first pass should store monotonic durations for local calculations and Unix seconds for compatibility with existing `TokenEvent`/benchmark correlation.

### Output Contract

When enabled, emit one structured summary log line per terminal request using the existing marker consumed by benchmark tooling:

```text
openinfer_http_trace {"request_id":"...","prompt_tokens":128,"completion_tokens":64,"finish_reason":"length","frontend_to_queue_ms":...,"admission_queue_ms":...,"scheduler_prefill_ms":...,"scheduler_decode_ms":...,"first_token_emit_unix_s":...,"active_set_size":...,"decode_batch_size_max":...}
```

This deliberately keeps `scripts/bench_http_serving.py --server-log` compatible. If a later OTLP/Chrome-trace adapter wants a more generic name, add it as an adapter-level concern rather than changing the benchmark log marker.

Optional local diagnostics may later write Chrome trace JSON through the existing `FileReporter`, but that is not part of the first implementation gate.

### Scope Boundaries

In scope for the first implementation:

- Shared trace event/span API in `openinfer-engine`.
- CLI or env flag to enable request tracing in `openinfer-server`.
- Frontend bridge instrumentation in `openinfer-vllm-frontend`.
- Qwen3 scheduler plan/effects instrumentation.
- Simulated engine instrumentation sufficient for frontend tests.
- Preserve or update `scripts/bench_http_serving.py --server-log` compatibility.
- Unit/integration tests that do not require CUDA.

Out of scope for the first implementation:

- Per-kernel runtime tracing in normal serving.
- Full Qwen3.5/Kimi/DeepSeek scheduler parity.
- Distributed/multi-process trace propagation beyond preserving request IDs and trace headers for future work.
- Making OTLP mandatory for local development.

## Execution Log

- 2026-06-17: Added `openinfer-engine::request_trace` with disabled/enabled trace handles, stable summary fields, JSON emission, and unit coverage for disabled behavior and lifecycle duration math.
- 2026-06-17: Added `GenerateRequest.trace` and updated direct request constructors in Qwen3, Qwen3.5 tests, Kimi, DeepSeek V4, and `openinfer-sim` to pass disabled traces unless explicitly testing tracing.
- 2026-06-17: Threaded `RequestTraceConfig` through `openinfer-vllm-frontend` while preserving existing public `serve*` functions as disabled-by-default wrappers. Added `serve_with_trace_config`, `serve_model_with_trace_config`, and `serve_model_with_lora_routes_and_trace_config` for opt-in callers.
- 2026-06-17: Added `--request-tracing` to `openinfer-server`; when enabled, the bridge emits one `openinfer_http_trace {json}` summary per terminal request.
- 2026-06-17: Instrumented the bridge stream path for request received, submit, scheduler admitted, first token received, output sent, and terminal error/rejected/finished events.
- 2026-06-17: Instrumented Qwen3 scheduler state so trace handles survive pending/active transitions; plan execution records prefill/decode/unified forward durations plus scheduler phase/batch metadata; effects record first-token and terminal outcomes.
- 2026-06-17: Instrumented `openinfer-sim` with scheduler admission, simulated prefill duration, first token, and terminal trace completion for CUDA-free trace semantics.
- 2026-06-17: Extended benchmark parser coverage to preserve `scheduler_prefill_ms` and `scheduler_decode_ms`; `scripts/bench_http_serving.py` did not require changes.

Commands run:

```bash
cargo fmt --all --check
cargo test -p openinfer-engine request_trace --lib
python3 -m unittest tests/test_bench_http_serving.py
git diff --check
```

All commands above passed.

Blocked local verification:

```bash
cargo check -p openinfer-engine -p openinfer-vllm-frontend -p openinfer-sim -p openinfer-server
cargo check -p openinfer-qwen3-4b --lib
OPENINFER_CUDA_SM=80 cargo check -p openinfer-engine -p openinfer-vllm-frontend -p openinfer-sim -p openinfer-server
OPENINFER_CUDA_SM=80 cargo check -p openinfer-qwen3-4b --lib
```

These checks did not reach the request-tracing Rust code on this macOS host. Without `OPENINFER_CUDA_SM`, `openinfer-kernels` fails GPU SM detection. With `OPENINFER_CUDA_SM=80`, the build proceeds to CUDA compilation and fails running `nvcc`; `pegaflow-proto` also fails because `protoc` is not installed; Qwen3 additionally pulls `io-uring`, which does not compile on macOS due missing Linux libc symbols.

## Debrief

This slice intentionally favors the existing benchmark log path over a broader tracing backend. That keeps the first implementation small and immediately useful for TTFT attribution while leaving Chrome trace and OTLP as adapters over the same request event model.

The main design adjustment from the original wording is that the shared trace API lives in `openinfer-engine`, not `openinfer-core`, because `openinfer-core` depends on `openinfer-engine`. `openinfer-core::request_trace` re-exports the API so model crates that already import through core do not need a second dependency edge.

Remaining follow-ups:

- Run frontend/sim/Qwen3 scheduler tests on a Linux CUDA host with `protoc` and `nvcc` available.
- Add parity instrumentation for Qwen3.5, Kimi, and DeepSeek scheduler paths after the Qwen3 contract is validated under real serving.
- Add optional Chrome trace or OTLP export as an adapter; do not change the `openinfer_http_trace` marker consumed by benchmark tooling.
