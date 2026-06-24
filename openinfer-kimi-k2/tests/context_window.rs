//! Context-window admission IT for Kimi-K2, ported from openinfer-qwen3-4b (#222).
//!
//! A prompt longer than the model's position-encoding window must be rejected at
//! admission with a context-length error — and crucially *before* any prefill, so
//! the oversized sequence never reaches the RoPE kernel (whose bounds trap would
//! otherwise take down the CUDA context). Admission rejects on prompt length
//! alone, so this stays cheap despite the oversized prompt: no forward pass runs.
//! After the rejection the engine must keep serving normal requests.
//!
//! Lives in its own test binary (not `scheduler_robustness.rs`) because `cargo
//! test` runs test binaries sequentially but parallelizes `#[test]`s within one
//! binary — two engines on one set of GPUs would contend. One engine-starting
//! test per file keeps them serialized.
//!
//! Requires 8 CUDA GPUs and Kimi-K2.6 weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, EpBackend, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::parallel::ParallelConfig;
use openinfer_core::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Kimi-K2.6");
/// Kimi-K2's `max_position_embeddings` is 262_144 (`openinfer-kimi-k2/src/config.rs`).
/// 300_000 overflows it outright. Token id is irrelevant — the request is
/// rejected before any embedding lookup.
const OVERSIZED_PROMPT_TOKENS: usize = 300_000;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping kimi-k2 context_window: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
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

/// Submit `prompt` and block until the request finishes; returns the decoded text.
fn generate_text(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
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
            Some(TokenEvent::PromptTokens { .. } | TokenEvent::Scheduled { .. }) => {}
            Some(TokenEvent::Finished { .. }) => break,
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("scheduler channel closed without Finished"),
        }
    }
    tokenizer.decode(&tokens, true).expect("decode failed")
}

#[test]
fn oversized_prompt_is_rejected_with_context_length_error() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = start_engine(&model_path);

    let prompt_tokens = vec![1u32; OVERSIZED_PROMPT_TOKENS];
    let (token_tx, mut rx) = TokenSink::standalone();
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 8,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");

    match rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected { message, .. }) => {
            assert!(
                message.contains("context length"),
                "expected a context-length rejection, got: {message}"
            );
        }
        Some(TokenEvent::Error { message, .. }) => {
            panic!("oversized prompt errored instead of clean rejection: {message}")
        }
        _ => panic!("oversized prompt should be rejected at admission"),
    }

    // The engine must keep serving normal requests after the rejection.
    let tokenizer = common::load_tokenizer(&model_path);
    let text = generate_text(&handle, &tokenizer, "Hello", 5);
    assert!(
        !text.is_empty(),
        "scheduler dead after context-length rejection"
    );
}
