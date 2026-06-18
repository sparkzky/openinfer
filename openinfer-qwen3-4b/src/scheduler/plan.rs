use anyhow::Result;
use openinfer_core::request_trace::RequestTraceFields;
use rand::rngs::StdRng;
use std::time::Instant;

use crate::executor::{
    DecodePlan, DecodeResult, DecodeStepItem, ModelExecutor, PrefillPlan, PrefillResult,
    PrefillStepItem, UnifiedPlan, UnifiedResult,
};

use super::{ActiveRequestState, PendingRequest};

pub(super) enum ExecutionPlan {
    Prefill { pending: Vec<PendingRequest> },
    Decode,
    Unified { pending: Vec<PendingRequest> },
}

pub(super) enum ExecutionArtifacts {
    Prefill {
        pending: Vec<PendingRequest>,
        result: PrefillResult,
        /// Stamped before the forward pass ran — downstream metrics split
        /// queue time (queued→scheduled) from prefill time (scheduled→first
        /// token), so stamping after execution would fold prefill into queue.
        scheduled_at_unix_s: f64,
    },
    Decode {
        result: DecodeResult,
    },
    Unified {
        pending: Vec<PendingRequest>,
        result: UnifiedResult,
        scheduled_at_unix_s: f64,
    },
}

pub(super) fn build_next_plan(
    have_active: bool,
    pending: Vec<PendingRequest>,
) -> Option<ExecutionPlan> {
    if !pending.is_empty() && have_active {
        Some(ExecutionPlan::Unified { pending })
    } else if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

pub(super) fn execute_plan(
    executor: &mut impl ModelExecutor,
    active: &mut [ActiveRequestState],
    plan: ExecutionPlan,
    rng: &mut StdRng,
) -> Result<ExecutionArtifacts> {
    match plan {
        ExecutionPlan::Prefill { pending } => {
            let scheduled_at_unix_s = openinfer_core::engine::unix_now_s();
            record_pending_admitted(&pending, scheduled_at_unix_s);
            let indices: Vec<usize> = (0..pending.len()).collect();
            let requests = build_prefill_items(&pending, &indices, rng);
            let any_echo = pending.iter().any(|req| req.echo);
            let started = Instant::now();
            let mut result = executor.execute_prefill(PrefillPlan {
                requests: &requests,
                echo: any_echo,
            })?;
            let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
            record_pending_forward(&pending, "model.forward.prefill", duration_ms);
            record_pending_step(&pending, "scheduler.step", "prefill", pending.len(), 0);
            sort_prefill_results(&mut result.requests);
            Ok(ExecutionArtifacts::Prefill {
                pending,
                result,
                scheduled_at_unix_s,
            })
        }
        ExecutionPlan::Decode => {
            let indices: Vec<usize> = (0..active.len()).collect();
            let requests = build_decode_items(active, &indices, rng);
            let started = Instant::now();
            let mut result = executor.execute_decode(DecodePlan {
                requests: &requests,
            })?;
            let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
            record_active_forward(active, "model.forward.decode", duration_ms);
            record_active_step(active, "scheduler.step", "decode", 0, active.len());
            sort_decode_results(&mut result.requests);
            Ok(ExecutionArtifacts::Decode { result })
        }
        ExecutionPlan::Unified { pending } => {
            let scheduled_at_unix_s = openinfer_core::engine::unix_now_s();
            record_pending_admitted(&pending, scheduled_at_unix_s);
            let pending_indices: Vec<usize> = (0..pending.len()).collect();
            let active_indices: Vec<usize> = (0..active.len()).collect();
            let prefill_requests = build_prefill_items(&pending, &pending_indices, rng);
            let decode_requests = build_decode_items(active, &active_indices, rng);
            let started = Instant::now();
            let mut result = executor.execute_unified(UnifiedPlan {
                prefill_requests: &prefill_requests,
                decode_requests: &decode_requests,
            })?;
            let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
            record_pending_forward(&pending, "model.forward.unified", duration_ms);
            record_active_forward(active, "model.forward.unified", duration_ms);
            record_pending_step(
                &pending,
                "scheduler.step",
                "unified",
                pending.len(),
                active.len(),
            );
            record_active_step(
                active,
                "scheduler.step",
                "unified",
                pending.len(),
                active.len(),
            );
            sort_prefill_results(&mut result.prefill_requests);
            sort_decode_results(&mut result.decode_requests);
            Ok(ExecutionArtifacts::Unified {
                pending,
                result,
                scheduled_at_unix_s,
            })
        }
    }
}

fn record_pending_admitted(pending: &[PendingRequest], scheduled_at_unix_s: f64) {
    for req in pending {
        req.trace.record_at(
            "scheduler.admitted",
            scheduled_at_unix_s,
            RequestTraceFields {
                prompt_tokens: Some(req.prompt_tokens.len()),
                cached_tokens: Some(req.cached_tokens),
                ..Default::default()
            },
        );
    }
}

fn record_pending_forward(pending: &[PendingRequest], name: &'static str, duration_ms: f64) {
    for req in pending {
        req.trace.record(
            name,
            RequestTraceFields {
                duration_ms: Some(duration_ms),
                ..Default::default()
            },
        );
    }
}

fn record_active_forward(active: &[ActiveRequestState], name: &'static str, duration_ms: f64) {
    for req in active {
        req.trace.record(
            name,
            RequestTraceFields {
                duration_ms: Some(duration_ms),
                ..Default::default()
            },
        );
    }
}

fn record_pending_step(
    pending: &[PendingRequest],
    name: &'static str,
    phase: &'static str,
    pending_batch_size: usize,
    decode_batch_size: usize,
) {
    for req in pending {
        req.trace.record(
            name,
            RequestTraceFields {
                phase: Some(phase),
                active_set_size: Some(pending_batch_size + decode_batch_size),
                decode_batch_size: Some(decode_batch_size),
                ..Default::default()
            },
        );
    }
}

fn record_active_step(
    active: &[ActiveRequestState],
    name: &'static str,
    phase: &'static str,
    pending_batch_size: usize,
    decode_batch_size: usize,
) {
    for req in active {
        req.trace.record(
            name,
            RequestTraceFields {
                phase: Some(phase),
                active_set_size: Some(pending_batch_size + decode_batch_size),
                decode_batch_size: Some(decode_batch_size),
                ..Default::default()
            },
        );
    }
}

fn build_prefill_items(
    pending: &[PendingRequest],
    indices: &[usize],
    rng: &mut StdRng,
) -> Vec<PrefillStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &pending[index];
            PrefillStepItem {
                request_id: r.request_id,
                prompt_tokens: r.prompt_tokens.clone(),
                max_output_tokens: r.max_tokens,
                params: r.params,
                logprobs: r.logprobs,
                echo: r.echo,
                lora_adapter: r.lora_adapter.clone(),
                random_val: rand::RngExt::random(rng),
                cached_tokens: r.cached_tokens,
                chunk_budget: r.step_chunk,
                chunk_start: 0,
                chunk_tokens: 0,
            }
        })
        .collect()
}

fn build_decode_items(
    active: &[ActiveRequestState],
    indices: &[usize],
    rng: &mut StdRng,
) -> Vec<DecodeStepItem> {
    indices
        .iter()
        .map(|&index| {
            let r = &active[index];
            DecodeStepItem {
                request_id: r.request_id,
                token_id: r.last_token,
                params: r.params,
                logprobs: r.logprobs,
                lora_adapter: r.lora_adapter.clone(),
                random_val: rand::RngExt::random(rng),
            }
        })
        .collect()
}

fn sort_prefill_results(results: &mut [crate::executor::PrefillRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

fn sort_decode_results(results: &mut [crate::executor::DecodeRequestResult]) {
    results.sort_by_key(|result| result.request_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::RequestId;
    use openinfer_core::sampler::SamplingParams;

    fn pending() -> PendingRequest {
        let (token_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        PendingRequest {
            request_id: RequestId::new(0),
            lora_adapter: None,
            trace: openinfer_core::request_trace::RequestTrace::disabled(),
            prompt_tokens: vec![1, 2, 3],
            params: SamplingParams::default(),
            max_tokens: 8,
            token_tx,
            logprobs: 0,
            echo: false,
            queued_at_unix_s: None,
            prefetch_offered: false,
            prefill_pos: 0,
            step_chunk: 3,
            cached_tokens: 0,
        }
    }

    // The plan selector is the whole batch-formation policy: what the scheduler
    // does each tick is fully determined by (have_active, has_pending). Pin the
    // 2×2 truth table so a policy regression can't slip through silently.
    #[test]
    fn plan_selection_follows_active_and_pending_state() {
        assert!(
            build_next_plan(false, vec![]).is_none(),
            "idle scheduler (no active, no pending) produces no plan"
        );
        assert!(
            matches!(build_next_plan(true, vec![]), Some(ExecutionPlan::Decode)),
            "active-only ticks decode the running batch"
        );
        assert!(
            matches!(
                build_next_plan(false, vec![pending()]),
                Some(ExecutionPlan::Prefill { pending }) if pending.len() == 1
            ),
            "pending-only prefills the new arrivals"
        );
        assert!(
            matches!(
                build_next_plan(true, vec![pending()]),
                Some(ExecutionPlan::Unified { pending }) if pending.len() == 1
            ),
            "active + pending fuses prefill and decode into one unified step"
        );
    }
}
