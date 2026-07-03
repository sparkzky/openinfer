/// Temperature/top-k/top-p sampling parameters carried end-to-end from the
/// HTTP layer to the GPU sampler.
///
/// `min_p` and `seed` are wired by #490 slice 1; the penalties
/// (frequency/presence/repetition) are rejected at the frontend until slice 2
/// lands a real kernel for them.
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub ignore_eos: bool,
    /// Minimum probability ratio threshold. A token survives only if its
    /// probability >= `min_p * max_prob_in_row`. `0.0` disables the filter
    /// (the fast path). Range: `[0, 1]`.
    pub min_p: f32,
    /// Per-request RNG seed for non-greedy sampling. `None` defers to the
    /// engine-wide seed (current behavior); `Some(s)` makes the request's
    /// sampled sequence reproducible regardless of batch composition.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
            min_p: 0.0,
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Greedy means argmax: temperature below the sampling epsilon (the
    /// temperature -> 0 limit is argmax regardless of top_p, and 1/temperature
    /// overflows long before that; vLLM draws the same line at 1e-5) or
    /// top_k == 1 (a single token survives the mask). Everything else requires
    /// a real sampling pass.
    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-5 || self.top_k == 1
    }
}
