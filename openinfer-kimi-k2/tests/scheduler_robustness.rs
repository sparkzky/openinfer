//! Scheduler robustness IT for Kimi-K2, ported from openinfer-qwen3-4b (#222).
//!
//! The qwen3 version additionally asserted on the `--batch-invariant` GEMM-N
//! pin policy (`numeric_policy()`/`pin_counters()`); that machinery belongs to
//! qwen3's low-level `Qwen3Executor` and has no Kimi-K2 equivalent, so this port
//! keeps only the model-agnostic serving contract: a client that hangs up
//! mid-flight (its receiver is dropped before scheduling) must not wedge the
//! engine. The submit still succeeds, the scheduler retires the orphaned request
//! when its sends start failing, and later requests are served. It drives the
//! real engine + `submit` rather than a mocked scheduler, so it exercises the
//! actual send-failure retirement path.
//!
//! Requires 8 CUDA GPUs and Kimi-K2.6 weights; skips cleanly when the model is
//! absent (point `OPENINFER_TEST_MODEL_PATH` at the weights to run it).

use std::path::Path;
use std::time::Duration;

use openinfer_core::engine::{
    EngineHandle, EngineLoadOptions, EpBackend, GenerateRequest, TokenEvent, TokenSink,
};
use openinfer_core::parallel::ParallelConfig;
use openinfer_core::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Kimi-K2.6");

fn model_path_or_skip() -> Option<String> {
    match std::env::var("OPENINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping kimi-k2 scheduler_robustness: {MODEL_PATH}/config.json is missing; set OPENINFER_TEST_MODEL_PATH to run it"
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

/// A client that drops its receiver before the request is scheduled must not
/// wedge the engine: the submit still succeeds, the scheduler retires the
/// orphaned request when its sends start failing, and later requests are served.
#[test]
fn scheduler_survives_consumer_drop() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let handle = start_engine(&model_path);
    let tokenizer = common::load_tokenizer(&model_path);

    // Submit, then drop the receiver immediately — the scheduler should notice
    // the send failures and retire the request rather than spinning on it.
    let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
    let (token_tx, rx) = TokenSink::standalone();
    drop(rx);
    handle
        .submit(GenerateRequest {
            request_id: None,
            queued_at_unix_s: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens: 10,
            lora_adapter: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit failed");
    std::thread::sleep(Duration::from_millis(500));

    // Barrier: drain the dropped orphan before the counted run (else its prefill
    // could leak into the subsequent request).
    let _ = generate_text(&handle, &tokenizer, "Hello", 1);

    let text = generate_text(&handle, &tokenizer, "Hello", 5);
    assert!(!text.is_empty(), "scheduler dead after consumer drop");
}
