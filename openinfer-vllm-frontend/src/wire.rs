use anyhow::{Result, bail};
use vllm_engine_core_client::protocol::logprobs::{
    PositionLogprobs, TokenLogprob as WireTokenLogprob,
};
use vllm_engine_core_client::protocol::{EngineCoreFinishReason, EngineCoreSamplingParams};

use openinfer_engine::engine::{FinishReason, TokenLogprob};
use openinfer_engine::sampler::SamplingParams;

pub(crate) const LORA_ADAPTER_XARG: &str = "openinfer_lora_adapter";

pub(crate) fn to_wire_position_logprobs(
    token_id: u32,
    logprob: Option<TokenLogprob>,
) -> Option<PositionLogprobs> {
    let lp = logprob?;
    let mut entries = Vec::with_capacity(1 + lp.top_logprobs.len());
    // openinfer-core does not currently expose the sampled token's vocab rank.
    // rank: 1 is correct for greedy sampling, where the sampled token is top-1,
    // and is a lossy placeholder for non-greedy sampling.
    // See discussion on PR #96.
    entries.push(WireTokenLogprob {
        token_id,
        logprob: lp.logprob,
        rank: 1,
    });
    for (index, (alt_id, alt_logprob)) in lp.top_logprobs.into_iter().enumerate() {
        if alt_id == token_id {
            continue;
        }
        entries.push(WireTokenLogprob {
            token_id: alt_id,
            logprob: alt_logprob,
            rank: (index + 1) as u32,
        });
    }
    Some(PositionLogprobs { entries })
}

/// Convert the wire-layer sampling params into the engine contract.
///
/// Returns `Err` (surfaced as a rejected request at the bridge) when the
/// client set a sampling knob the engine does not yet implement, so the
/// caller gets an explicit signal instead of the silent fallback that used
/// to drop every field below `top_p`. Currently rejected: frequency,
/// presence, and repetition penalties (slice 2, #490).
pub(crate) fn convert_sampling(params: &EngineCoreSamplingParams) -> Result<SamplingParams> {
    // The vLLM frontend lowers a client `ignore_eos=true` to `_eos_token_id:
    // None`, but `_all_stop_token_ids` always carries the model EOS set (it
    // exists for min_tokens masking, not stop detection). Deriving ignore_eos
    // from all_stop_token_ids would therefore void every ignore_eos request on
    // models with a real EOS. Only `_eos_token_id` and the client's explicit
    // `stop_token_ids` express a stop intent.
    let ignore_eos = params.eos_token_id.is_none() && params.stop_token_ids.is_empty();

    // Penalties need a per-request output-token-count pipeline that lands in
    // slice 2 (#490); reject explicitly instead of silently ignoring.
    if params.frequency_penalty != 0.0 {
        bail!("frequency_penalty is not yet supported");
    }
    if params.presence_penalty != 0.0 {
        bail!("presence_penalty is not yet supported");
    }
    if params.repetition_penalty != 1.0 {
        bail!("repetition_penalty is not yet supported");
    }

    // min_p range: [0, 1]. 0 disables the filter (fast path). Clamp tiny
    // negatives to 0 rather than rejecting — the wire default is already 0
    // and a stray -0.0 / subnormal should not be a client-visible error.
    let min_p = if params.min_p < 0.0 { 0.0 } else { params.min_p };
    if min_p > 1.0 {
        bail!("min_p must be in [0, 1], got {min_p}");
    }

    // Per-request seed: the wire type carries i64; the GPU sampler wants u64.
    // A negative seed is rejected (philox keys are unsigned).
    let seed = match params.seed {
        None => None,
        Some(s) if s < 0 => bail!("seed must be non-negative, got {s}"),
        Some(s) => Some(s as u64),
    };

    if params.temperature <= 0.0 {
        return Ok(SamplingParams {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos,
            // Greedy ignores min_p and seed by definition (argmax is
            // deterministic), but we still carry them so a request that later
            // switches to sampling stays faithful to its params.
            min_p,
            seed,
        });
    }

    Ok(SamplingParams {
        temperature: params.temperature,
        top_k: if params.top_k == 0 {
            -1
        } else {
            i32::try_from(params.top_k).unwrap_or(i32::MAX)
        },
        top_p: params.top_p,
        ignore_eos,
        min_p,
        seed,
    })
}

pub(crate) fn requested_logprobs(params: &EngineCoreSamplingParams) -> usize {
    params
        .logprobs
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

pub(crate) fn lora_adapter_from_sampling_params(
    params: &EngineCoreSamplingParams,
) -> Result<Option<String>> {
    let Some(extra_args) = params.extra_args.as_ref() else {
        return Ok(None);
    };
    let Some(value) = extra_args.get(LORA_ADAPTER_XARG) else {
        return Ok(None);
    };
    match value.as_str() {
        Some(name) if !name.is_empty() => Ok(Some(name.to_string())),
        Some(_) => bail!("{LORA_ADAPTER_XARG} must not be empty"),
        None => bail!("{LORA_ADAPTER_XARG} must be a string"),
    }
}

pub(crate) fn convert_finish_reason(reason: FinishReason) -> EngineCoreFinishReason {
    match reason {
        FinishReason::Length => EngineCoreFinishReason::Length,
        FinishReason::Stop => EngineCoreFinishReason::Stop,
        FinishReason::Error => EngineCoreFinishReason::Error,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use vllm_engine_core_client::protocol::logprobs::{Logprobs, MaybeWireLogprobs};

    use super::*;

    fn to_wire_logprobs(token_id: u32, logprob: Option<TokenLogprob>) -> Option<MaybeWireLogprobs> {
        let position = to_wire_position_logprobs(token_id, logprob)?;
        Some(MaybeWireLogprobs::Direct(Logprobs {
            positions: vec![position],
        }))
    }

    #[test]
    fn convert_sampling_honors_ignore_eos_lowering() {
        // ignore_eos=true lowering: _eos_token_id=None while
        // _all_stop_token_ids still carries the model EOS set.
        let mut params = EngineCoreSamplingParams::for_test();
        params.all_stop_token_ids = BTreeSet::from([163_586]);
        assert!(convert_sampling(&params).unwrap().ignore_eos);

        // Normal request: _eos_token_id present.
        params.eos_token_id = Some(163_586);
        assert!(!convert_sampling(&params).unwrap().ignore_eos);

        // Explicit client stop tokens keep EOS detection on even when the
        // frontend dropped _eos_token_id.
        params.eos_token_id = None;
        params.stop_token_ids = vec![42];
        assert!(!convert_sampling(&params).unwrap().ignore_eos);
    }

    #[test]
    fn convert_sampling_threads_min_p() {
        let mut params = EngineCoreSamplingParams::for_test();
        // Default min_p is 0 — disabled, fast path.
        assert_eq!(convert_sampling(&params).unwrap().min_p, 0.0);

        // A set min_p flows through verbatim.
        params.min_p = 0.05;
        assert_eq!(convert_sampling(&params).unwrap().min_p, 0.05);

        // Negative min_p clamps to 0 rather than erroring.
        params.min_p = -1.0;
        assert_eq!(convert_sampling(&params).unwrap().min_p, 0.0);

        // min_p > 1 is rejected.
        params.min_p = 1.5;
        assert!(convert_sampling(&params).is_err());
    }

    #[test]
    fn convert_sampling_threads_seed() {
        let mut params = EngineCoreSamplingParams::for_test();
        // Default seed is None — defers to engine-wide seed.
        assert_eq!(convert_sampling(&params).unwrap().seed, None);

        // A set seed flows through as u64.
        params.seed = Some(42);
        assert_eq!(convert_sampling(&params).unwrap().seed, Some(42));

        // Negative seed is rejected.
        params.seed = Some(-1);
        assert!(convert_sampling(&params).is_err());
    }

    #[test]
    fn convert_sampling_rejects_penalties() {
        // frequency_penalty
        let mut params = EngineCoreSamplingParams::for_test();
        params.frequency_penalty = 0.5;
        assert!(convert_sampling(&params).is_err());

        // presence_penalty
        let mut params = EngineCoreSamplingParams::for_test();
        params.presence_penalty = 0.5;
        assert!(convert_sampling(&params).is_err());

        // repetition_penalty (default is 1.0 = no-op)
        let mut params = EngineCoreSamplingParams::for_test();
        params.repetition_penalty = 1.5;
        assert!(convert_sampling(&params).is_err());
    }

    #[test]
    fn lora_adapter_from_sampling_params_reads_proxy_xarg() {
        let mut params = EngineCoreSamplingParams::for_test();
        params.extra_args = Some(HashMap::from([(
            LORA_ADAPTER_XARG.to_string(),
            serde_json::Value::String("adapter-a".to_string()),
        )]));

        assert_eq!(
            lora_adapter_from_sampling_params(&params)
                .expect("extract adapter")
                .as_deref(),
            Some("adapter-a")
        );
    }

    fn assert_logprob_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= f32::EPSILON,
            "logprob mismatch: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn to_wire_logprobs_emits_sampled_then_alternatives() {
        let lp = TokenLogprob {
            logprob: -0.5,
            top_logprobs: vec![(7, -0.5), (42, -1.5)],
        };
        let wire = to_wire_logprobs(7, Some(lp)).expect("logprob payload");
        let direct = match wire {
            MaybeWireLogprobs::Direct(d) => d,
            MaybeWireLogprobs::Wire(_) => panic!("expected Direct logprobs"),
        };
        assert_eq!(direct.positions.len(), 1);
        let entries = &direct.positions[0].entries;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].token_id, 7);
        assert_logprob_eq(entries[0].logprob, -0.5);
        assert_eq!(entries[0].rank, 1);
        assert_eq!(entries[1].token_id, 42);
        assert_logprob_eq(entries[1].logprob, -1.5);
        assert_eq!(entries[1].rank, 2);
    }

    #[test]
    fn to_wire_logprobs_keeps_distinct_top_k_alternatives() {
        let lp = TokenLogprob {
            logprob: -0.5,
            top_logprobs: vec![(8, -1.0), (9, -1.5)],
        };
        let wire = to_wire_logprobs(7, Some(lp)).expect("logprob payload");
        let direct = match wire {
            MaybeWireLogprobs::Direct(d) => d,
            MaybeWireLogprobs::Wire(_) => panic!("expected Direct logprobs"),
        };
        assert_eq!(direct.positions.len(), 1);
        let entries = &direct.positions[0].entries;
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].token_id, 7);
        assert_logprob_eq(entries[0].logprob, -0.5);
        assert_eq!(entries[0].rank, 1);
        assert_eq!(entries[1].token_id, 8);
        assert_logprob_eq(entries[1].logprob, -1.0);
        assert_eq!(entries[1].rank, 1);
        assert_eq!(entries[2].token_id, 9);
        assert_logprob_eq(entries[2].logprob, -1.5);
        assert_eq!(entries[2].rank, 2);
    }
}
