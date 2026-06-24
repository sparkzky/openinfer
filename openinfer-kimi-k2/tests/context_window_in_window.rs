//! In-window context regression IT for Kimi-K2, ported from openinfer-qwen3-4b (#222).
//!
//! Companion to `context_window.rs` (which proves *oversized* prompts are
//! rejected). This proves the *positive* side of the same admission contract: a
//! prompt that is inside the position-encoding window (`max_position_embeddings`
//! = 262_144 for Kimi-K2) must be served end-to-end.
//!
//! qwen3's version targeted a since-fixed hardcoded 4096-entry RoPE table. The
//! Kimi-K2 analog boundary is the yarn `original_max_position_embeddings` (4096,
//! `KIMI_K2_YARN_ORIGINAL_MAX_POS` in `src/config.rs`): position 4096 is the
//! first index where yarn long-rope scaling kicks in. A 4097-token prompt spans
//! positions 0..=4096 and therefore exercises the yarn-extension RoPE path
//! while staying far inside the 262_144 max context. If the yarn extension table
//! ever silently regresses (missing/wrong-scale entries past 4096) this test
//! fails or crashes instead of passing.
//!
//! Lives in its own test binary for the same reason as `context_window.rs`:
//! `cargo test` serializes test binaries but parallelizes `#[test]`s within one
//! binary, so two engines on one set of GPUs would contend. One engine per file.
//!
//! Requires 8 CUDA GPUs and Kimi-K2.6 weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::engine::{
    EngineLoadOptions, EpBackend, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::parallel::ParallelConfig;
use openinfer_core::sampler::SamplingParams;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Kimi-K2.6");
/// 4097 tokens → positions 0..=4096. Position 4096 is the first index past the
/// yarn `original_max_position_embeddings` (4096); serving this prompt requires
/// the yarn-extension RoPE table to actually exist and be indexed at 4096. Token
/// id 1 is a valid vocab id — a forward pass actually runs here (unlike the
/// rejection test, which never reaches prefill).
const IN_WINDOW_PROMPT_TOKENS: usize = 4097;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping kimi-k2 context_window_in_window: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

#[test]
fn in_window_prompt_past_yarn_original_max_is_served() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = openinfer_kimi_k2::start_engine(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            enable_prefill_profile: false,
            device_ordinals: (0..8).collect(),
            parallel_config: Some(ParallelConfig::new(1, 8)),
            ep_backend: EpBackend::DeepEp,
            seed: 42,
        },
    )
    .expect("failed to start engine");

    let prompt_tokens = vec![1u32; IN_WINDOW_PROMPT_TOKENS];
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    let mut generated = 0usize;
    loop {
        match rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { .. }) => generated += 1,
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => {
                panic!("in-window prompt errored (yarn-extension RoPE not exercised?): {message}")
            }
            Some(TokenEvent::Rejected { message, .. }) => {
                panic!("in-window prompt was wrongly rejected: {message}")
            }
            None => panic!("scheduler channel closed without Finished"),
        }
    }

    assert_eq!(
        generated, 1,
        "expected exactly one generated token for a 4097-token prompt with max_tokens=1"
    );
}
