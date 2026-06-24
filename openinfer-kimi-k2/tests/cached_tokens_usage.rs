//! Prefix-cache observability IT for Kimi-K2, ported from openinfer-qwen3-4b (#222).
//!
//! The frontend reports `usage.prompt_tokens_details.cached_tokens` from
//! `TokenEvent::Scheduled`. This test pins the engine half of that contract:
//! a cold prompt reports zero cached tokens, a warm repeat of the same prompt
//! reports a nonzero full-block count, and the count never claims the whole
//! prompt (the last token is always recomputed).
//!
//! NOTE: this test depends on #351 — Kimi-K2 currently reports `cached_tokens=0`
//! from the `Scheduled` event (the prefix match is not propagated). It is marked
//! `#[ignore]` so it cannot break CI before #351 lands; delete the `#[ignore]`
//! line when merging #351 and confirm it passes on the serving hardware.
//!
//! Requires 8 CUDA GPUs and Kimi-K2.6 weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, EpBackend, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::parallel::ParallelConfig;
use openinfer_core::sampler::SamplingParams;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Kimi-K2.6");
/// Kimi-K2's KV page size (`KIMI_KV_PAGE_SIZE` in `src/runner/worker.rs`).
const KV_BLOCK_SIZE: usize = 16;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping kimi-k2 cached_tokens_usage: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Start the Kimi-K2 engine on the mandatory TP1/DP8/EP8 topology (matches
/// `vllm_golden_gate`).
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

/// Submit `prompt_tokens`, drain the stream to `Finished`, and return the
/// `cached_tokens` carried by the `Scheduled` event.
fn run_and_capture_cached(handle: &EngineHandle, prompt_tokens: Vec<u32>) -> usize {
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 4,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut cached = None;
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Scheduled { cached_tokens, .. }) => {
                assert!(
                    cached.replace(cached_tokens).is_none(),
                    "Scheduled must be emitted exactly once per request"
                );
            }
            Some(TokenEvent::Token { .. } | TokenEvent::PromptTokens { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
    cached.expect("Scheduled event must precede Finished")
}

#[test]
#[ignore = "depends on #351: Kimi-K2 must report cached_tokens from Scheduled; remove this line after #351 lands"]
fn warm_repeat_reports_cached_tokens() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = start_engine(&model_path);
    let tokenizer = common::load_tokenizer(&model_path);

    let prompt = "The kv cache stores attention keys and values for every \
        generated token so the model never recomputes earlier positions. "
        .repeat(8);
    let prompt_tokens = tokenizer.encode(&prompt, false).expect("encode failed");
    let prompt_len = prompt_tokens.len();
    assert!(
        prompt_len > 2 * KV_BLOCK_SIZE,
        "prompt must span multiple KV blocks for a meaningful hit"
    );

    let cold = run_and_capture_cached(&handle, prompt_tokens.clone());
    assert_eq!(cold, 0, "cold run must report zero cached tokens");

    let warm = run_and_capture_cached(&handle, prompt_tokens);
    assert!(warm > 0, "warm repeat must report a prefix-cache hit");
    assert!(
        warm < prompt_len,
        "at least the last prompt token is always recomputed (warm={warm}, prompt={prompt_len})"
    );
    assert_eq!(
        warm % KV_BLOCK_SIZE,
        0,
        "hits are matched in full blocks (warm={warm})"
    );
    assert_eq!(
        warm,
        (prompt_len - 1) / KV_BLOCK_SIZE * KV_BLOCK_SIZE,
        "warm hit must cover every cacheable full block (prompt={prompt_len})"
    );
}
