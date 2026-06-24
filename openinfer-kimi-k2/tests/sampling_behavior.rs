//! Behavioral coverage for the honored sampling params, ported from
//! openinfer-qwen3-4b (#222).
//!
//! The frontend guard pins "unsupported params are rejected"; these tests pin
//! the other half — `temperature` / `top_k` / `top_p` actually steer the
//! sampler. The silent-greedy-fallback failure class (params dropped on the way
//! to the sampler) is caught by the diversity assertion; the collapse identities
//! (`top_k=1`, tiny `top_p`) catch the inverse (sampling ignoring the masks).
//!
//! Per-request seeds are rejected at the frontend and the legacy per-row sampler
//! is not run-to-run deterministic, so seed determinism is intentionally not
//! asserted here.
//!
//! Requires 8 CUDA GPUs and Kimi-K2.6 weights (Kimi-K2 has no single-GPU path);
//! skips cleanly when the model is absent (point `OPENINFER_TEST_MODEL_PATH` at
//! the weights to run it).

use std::path::Path;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, EpBackend, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::parallel::ParallelConfig;
use openinfer_core::sampler::SamplingParams;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Kimi-K2.6");
const GENERATED_TOKENS: usize = 32;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping kimi-k2 sampling_behavior: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Start the Kimi-K2 engine on the mandatory TP1/DP8/EP8 topology (matches
/// `vllm_golden_gate`). Eager (no CUDA graph) so the sampler path is the
/// default one.
fn start_engine(path: &str) -> EngineHandle {
    openinfer_kimi_k2::start_engine(
        Path::new(path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: (0..8).collect(),
            parallel_config: Some(ParallelConfig::new(1, 8)),
            ep_backend: EpBackend::DeepEp,
            seed: 42,
        },
    )
    .expect("failed to start engine")
}

/// Submit one request and collect the generated token ids until `Finished`.
fn generate(handle: &EngineHandle, prompt_tokens: Vec<u32>, params: SamplingParams) -> Vec<u32> {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params,
            max_tokens: GENERATED_TOKENS,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut tokens = Vec::new();
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, .. }) => tokens.push(id),
            Some(TokenEvent::Scheduled { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => return tokens,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

#[test]
fn sampling_params_steer_the_sampler() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = start_engine(&model_path);
    let tokenizer = common::load_tokenizer(&model_path);

    // Branchy continuation so unrestricted sampling has real entropy to show.
    let prompt = "Here is a short story about a dragon. Once upon a time";
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");

    // Warm the prefix cache first so every measured run takes the same
    // (cache-hit) prefill path; a cold-vs-warm comparison could flip a
    // near-tie token through bf16 wobble and fail spuriously.
    let _ = generate(&handle, prompt_tokens.clone(), SamplingParams::default());

    // Greedy is repeatable token-for-token.
    let greedy = generate(&handle, prompt_tokens.clone(), SamplingParams::default());
    assert_eq!(
        greedy.len(),
        GENERATED_TOKENS,
        "EOS inside a 32-token story tail is unexpected"
    );
    let greedy_again = generate(&handle, prompt_tokens.clone(), SamplingParams::default());
    assert_eq!(greedy, greedy_again, "greedy decode must be deterministic");

    // top_k = 1 leaves a single survivor: identical to greedy at any
    // temperature.
    let top_k_one = generate(
        &handle,
        prompt_tokens.clone(),
        SamplingParams {
            temperature: 0.8,
            top_k: 1,
            ..SamplingParams::default()
        },
    );
    assert_eq!(top_k_one, greedy, "top_k=1 must collapse to greedy");

    // The nucleus is the smallest descending-probability prefix whose mass
    // reaches top_p. The largest probability is always >= 1/vocab, so with
    // top_p below that bound the nucleus is exactly the argmax for ANY
    // distribution — the identity holds even on high-entropy steps where the
    // top token carries little mass. (Kimi-K2 vocab is 163_840 → 1/vocab ≈
    // 6.1e-6, well above the 1e-6 threshold used here.)
    let top_p_tiny = generate(
        &handle,
        prompt_tokens.clone(),
        SamplingParams {
            temperature: 1.0,
            top_p: 1e-6,
            ..SamplingParams::default()
        },
    );
    assert_eq!(top_p_tiny, greedy, "top_p=1e-6 must collapse to greedy");

    // Unrestricted high-temperature sampling must actually sample: four runs
    // of 32 tokens collapsing to one sequence means the params were dropped
    // on the way to the sampler (silent greedy fallback).
    let hot = SamplingParams {
        temperature: 1.5,
        top_k: -1,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    let runs: Vec<Vec<u32>> = (0..4)
        .map(|_| generate(&handle, prompt_tokens.clone(), hot))
        .collect();
    assert!(
        runs.iter().any(|run| *run != runs[0]),
        "4 high-temperature runs were token-identical — sampling params are not reaching the sampler"
    );
}
