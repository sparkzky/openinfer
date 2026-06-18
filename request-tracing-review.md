# Request Tracing Review Notes

> Created: 2026-06-17
>
> Scope: review findings for the request tracing implementation on branch `feat/request-tracing`.

## 2026-06-17 Frontend / Server Review

Reviewer: `Locke` (`019ed3f3-14e6-7b32-aa7b-b7ff340e8174`)

### Findings

- **Important** — `openinfer-vllm-frontend/src/bridge.rs:218-231, 261-264, 321-331`

  `run_request_stream` only calls `trace.finish(...)` and `emit_trace_summary(...)` on `Finished` / `Error` / `Rejected`. If `self.handle.submit(...)` fails, if `token_rx.recv()` returns `None`, or if `send_token_output(...)` fails because the output side is gone, the function returns early with no terminal summary at all.

  **Impact:** the `--request-tracing` contract is not "one summary per request" on shutdown / disconnect / submission-failure paths, so traces go missing exactly on the paths audited. Bench log consumers will see gaps rather than a completed record.

  **Suggestion:** funnel every non-success exit through a single terminalization helper that records a terminal reason, calls `trace.finish(...)` once, and emits the summary before returning.

## 2026-06-17 Qwen3 / Sim Review

Reviewer: `Socrates` (`019ed3f3-3982-75d2-aff9-2fbcbb1e529a`)

### Findings

- **Important** — `openinfer-qwen3-4b/src/scheduler/plan.rs:60-73,94-116`

  `scheduler.admitted` is recorded on every prefill/unified execution pass, not just on the request's first admission. Chunked-prefill requests return to `prefilling` with `prefill_pos > 0`, but `record_pending_admitted()` has no guard, so later chunks re-emit the same admission marker.

  **Impact:** raw traces overcount admissions and make the admission timestamp semantically mean "this chunk entered the GPU" instead of "this request was admitted once." The summary hides it via `first_event_at`, but any raw-event consumer or future simulator join will see duplicate admissions.

  **Suggestion:** only record `scheduler.admitted` when `prefill_pos == 0`, or move it to the first-chunk resolve path so it fires once per request.

- **Important** — `openinfer-qwen3-4b/src/scheduler.rs:987-1007`

  `fail_touched_requests()` emits `TokenEvent::Error` and drops executor state, but it never calls `trace.finish(...)` on the affected `PendingRequest` / `ActiveRequestState` traces.

  **Impact:** request-trace summaries never exist for execution failures unless some outer caller happens to finish the trace later. That makes the scheduler-side trace lifecycle incomplete and breaks direct-caller / engine-side consumers of `RequestTrace::summary()` for error completions.

  **Suggestion:** finalize every affected trace with an error terminal before or alongside sending `TokenEvent::Error`. The frontend bridge can still mirror the terminal event, but the scheduler should not depend on it.

- **Minor** — `openinfer-qwen3-4b/src/scheduler.rs:908-946`

  Admission rejects are collapsed into `finish_reason: "error"` for both KV/context rejects and missing-LoRA rejects.

  **Impact:** trace consumers cannot distinguish admission rejection from runtime execution failure, which is a semantic mismatch with the request-tracing spec's `rejected` terminal outcome.

  **Suggestion:** preserve a distinct rejected terminal reason in the trace contract, or document clearly that rejection is intentionally folded into error for benchmark compatibility.

- **Minor** — `openinfer-sim/src/lib.rs:80-113,115-162`

  The simulated engine returns early on closed `Scheduled`, `PromptTokens`, or `Token` sends without writing any terminal trace, so the frontend-free path can drop summaries silently on disconnect/cancellation.

  **Impact:** the sim no longer exercises the same "always produce a terminal trace" contract that the real scheduler is supposed to satisfy, which weakens the frontend-free direct-caller validation path.

  **Suggestion:** on send failure, emit a terminal error/rejected trace before returning, rather than just exiting the task.

### Reviewer Note

Static review only; CUDA/protoc/io-uring-dependent checks were not run on this host.

## 2026-06-17 Shared Contract / API Review

Reviewer: `Einstein` (`019ed3f2-f0c8-7792-a296-b19b2c9acdac`)

### Findings

- **Important** — `openinfer-engine/src/request_trace.rs:123-141`, `openinfer-vllm-frontend/src/bridge.rs:208-222`

  The disabled path is not actually "cheap": `record()` and `finish()` eagerly call `unix_now_s()` before checking whether tracing is enabled, so every no-op trace call still pays a wall-clock read. The bridge also clones `request_id` just to hand an owned string into `RequestTrace::from_config`, even when tracing is off.

  **Impact:** the default-disabled mode still adds per-request overhead on the hot path, which conflicts with the spec's "cheap disabled-branch cost" requirement.

  **Suggestion:** guard before capturing timestamps, and avoid cloning the request id unless tracing is enabled.

- **Important** — `openinfer-engine/src/request_trace.rs:37-42, 143-185`

  The trace state only stores Unix timestamps, and every duration is derived by subtracting wall-clock values and clamping negatives to zero. That misses the spec's monotonic-duration requirement, and `terminal_unix_s` is recorded but never serialized, so the summary has no explicit terminal-completion slice.

  **Impact:** clock adjustments or timestamp skew can silently corrupt latency attribution, and the JSON contract cannot represent request completion separately from last output flush.

  **Suggestion:** store a monotonic `Instant`-based elapsed timeline alongside Unix time, and add a serialized terminal/completion field or total request duration.

- **Minor** — `openinfer-engine/src/request_trace.rs:190-193`, `openinfer-vllm-frontend/src/bridge.rs:421-424`

  `summary_json()` turns serialization failure into `None`, and the bridge then drops that `None` silently. That makes "tracing disabled", "trace incomplete", and "trace failed to serialize" indistinguishable at the log boundary.

  **Impact:** an enabled request can lose its benchmark trace without any diagnostic, which is a bad failure mode for a contract meant to feed `--server-log` attribution.

  **Suggestion:** emit a warning when an enabled trace fails to serialize, or make the JSON emission path return a result so failures are visible.

### Reviewer Note

No deadlock or data-race problem found in the mutex usage itself; the lock is short-lived and not re-entrant.
