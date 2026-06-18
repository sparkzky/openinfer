use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

pub const TRACE_LOG_MARKER: &str = "openinfer_http_trace";

#[derive(Clone, Debug, Default)]
pub struct RequestTraceConfig {
    pub enabled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RequestTraceFields {
    pub phase: Option<&'static str>,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub duration_ms: Option<f64>,
    pub active_set_size: Option<usize>,
    pub decode_batch_size: Option<usize>,
    pub cached_tokens: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct RequestTraceTerminal {
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct RequestTrace {
    inner: Option<Arc<Mutex<RequestTraceState>>>,
}

#[derive(Clone, Debug)]
struct RequestTraceState {
    request_id: String,
    arrival_unix_s: f64,
    events: Vec<RequestTraceEvent>,
    terminal: Option<RequestTraceTerminal>,
    terminal_unix_s: Option<f64>,
}

#[derive(Clone, Debug)]
struct RequestTraceEvent {
    name: &'static str,
    at_unix_s: f64,
    fields: RequestTraceFields,
}

#[derive(Clone, Debug, Serialize)]
pub struct RequestTraceSummary {
    pub request_id: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queued_at_unix_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduled_at_unix_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_token_emit_unix_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontend_to_queue_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission_queue_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_prefill_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_unified_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_flush_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_set_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_batch_size_max: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scheduler_phases: Vec<String>,
}

impl RequestTrace {
    #[must_use]
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    #[must_use]
    pub fn enabled(request_id: String, arrival_unix_s: f64) -> Self {
        Self {
            inner: Some(Arc::new(Mutex::new(RequestTraceState {
                request_id,
                arrival_unix_s,
                events: Vec::new(),
                terminal: None,
                terminal_unix_s: None,
            }))),
        }
    }

    #[must_use]
    pub fn from_config(
        config: &RequestTraceConfig,
        request_id: String,
        arrival_unix_s: f64,
    ) -> Self {
        if config.enabled {
            Self::enabled(request_id, arrival_unix_s)
        } else {
            Self::disabled()
        }
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    pub fn record(&self, name: &'static str, fields: RequestTraceFields) {
        self.record_at(name, unix_now_s(), fields);
    }

    pub fn record_at(&self, name: &'static str, at_unix_s: f64, fields: RequestTraceFields) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = inner.lock().expect("request trace mutex poisoned");
        state.events.push(RequestTraceEvent {
            name,
            at_unix_s,
            fields,
        });
    }

    pub fn finish(&self, terminal: RequestTraceTerminal) {
        self.finish_at(unix_now_s(), terminal);
    }

    pub fn finish_at(&self, at_unix_s: f64, terminal: RequestTraceTerminal) {
        let Some(inner) = &self.inner else {
            return;
        };
        let mut state = inner.lock().expect("request trace mutex poisoned");
        state.terminal = Some(terminal);
        state.terminal_unix_s = Some(at_unix_s);
    }

    #[must_use]
    pub fn summary(&self) -> Option<RequestTraceSummary> {
        let inner = self.inner.as_ref()?;
        let state = inner.lock().expect("request trace mutex poisoned");
        let terminal = state.terminal.as_ref()?;

        let submit_end = first_event_at(&state.events, "frontend.submit_end");
        let admitted = first_event_at(&state.events, "scheduler.admitted");
        let first_token = first_event_at(&state.events, "scheduler.first_token")
            .or_else(|| first_event_at(&state.events, "frontend.first_token_received"));
        let output_sent = first_event_at(&state.events, "frontend.output_sent");

        Some(RequestTraceSummary {
            request_id: state.request_id.clone(),
            prompt_tokens: terminal.prompt_tokens,
            completion_tokens: terminal.completion_tokens,
            finish_reason: terminal.finish_reason.clone(),
            queued_at_unix_s: Some(state.arrival_unix_s),
            scheduled_at_unix_s: admitted,
            first_token_emit_unix_s: first_token,
            frontend_to_queue_ms: submit_end.map(|end| ms_between(state.arrival_unix_s, end)),
            admission_queue_ms: submit_end
                .zip(admitted)
                .map(|(start, end)| ms_between(start, end)),
            scheduler_prefill_ms: sum_duration(&state.events, "model.forward.prefill"),
            scheduler_decode_ms: sum_duration(&state.events, "model.forward.decode"),
            scheduler_unified_ms: sum_duration(&state.events, "model.forward.unified"),
            stream_flush_ms: first_token
                .zip(output_sent)
                .map(|(start, end)| ms_between(start, end)),
            active_set_size: max_field(&state.events, |fields| fields.active_set_size),
            decode_batch_size_max: max_field(&state.events, |fields| fields.decode_batch_size),
            cached_tokens: max_field(&state.events, |fields| fields.cached_tokens),
            scheduler_phases: scheduler_phases(&state.events),
        })
    }

    #[must_use]
    pub fn summary_json(&self) -> Option<String> {
        self.summary()
            .and_then(|summary| serde_json::to_string(&summary).ok())
    }
}

impl Default for RequestTrace {
    fn default() -> Self {
        Self::disabled()
    }
}

fn first_event_at(events: &[RequestTraceEvent], name: &'static str) -> Option<f64> {
    events
        .iter()
        .find(|event| event.name == name)
        .map(|event| event.at_unix_s)
}

fn sum_duration(events: &[RequestTraceEvent], name: &'static str) -> Option<f64> {
    let mut total = 0.0;
    let mut seen = false;
    for event in events.iter().filter(|event| event.name == name) {
        if let Some(duration_ms) = event.fields.duration_ms {
            total += duration_ms;
            seen = true;
        }
    }
    seen.then_some(round_ms(total))
}

fn max_field(
    events: &[RequestTraceEvent],
    field: impl Fn(&RequestTraceFields) -> Option<usize>,
) -> Option<usize> {
    events.iter().filter_map(|event| field(&event.fields)).max()
}

fn scheduler_phases(events: &[RequestTraceEvent]) -> Vec<String> {
    let mut phases = Vec::new();
    for phase in events
        .iter()
        .filter(|event| event.name == "scheduler.step")
        .filter_map(|event| event.fields.phase)
    {
        if !phases.iter().any(|existing| existing == phase) {
            phases.push(phase.to_string());
        }
    }
    phases
}

fn ms_between(start_unix_s: f64, end_unix_s: f64) -> f64 {
    round_ms((end_unix_s - start_unix_s).max(0.0) * 1000.0)
}

fn round_ms(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before UNIX_EPOCH")
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_trace_does_not_record_summary() {
        let trace = RequestTrace::disabled();
        trace.record("frontend.request_received", RequestTraceFields::default());
        assert!(trace.summary().is_none());
    }

    #[test]
    fn summary_computes_core_phase_durations() {
        let trace = RequestTrace::enabled("req-1".to_string(), 10.0);
        trace.record_at(
            "frontend.request_received",
            10.000,
            RequestTraceFields::default(),
        );
        trace.record_at("frontend.submit_end", 10.010, RequestTraceFields::default());
        trace.record_at(
            "scheduler.admitted",
            10.030,
            RequestTraceFields {
                prompt_tokens: Some(8),
                ..Default::default()
            },
        );
        trace.record_at(
            "model.forward.prefill",
            10.200,
            RequestTraceFields {
                duration_ms: Some(170.0),
                ..Default::default()
            },
        );
        trace.record_at(
            "scheduler.first_token",
            10.210,
            RequestTraceFields::default(),
        );
        trace.record_at(
            "frontend.output_sent",
            10.260,
            RequestTraceFields::default(),
        );
        trace.finish_at(
            10.300,
            RequestTraceTerminal {
                finish_reason: "length".to_string(),
                prompt_tokens: 8,
                completion_tokens: 4,
            },
        );

        let summary = trace.summary().expect("enabled trace has summary");
        assert_eq!(summary.request_id, "req-1");
        assert_eq!(summary.prompt_tokens, 8);
        assert_eq!(summary.completion_tokens, 4);
        assert_eq!(summary.finish_reason, "length");
        assert_eq!(summary.frontend_to_queue_ms, Some(10.0));
        assert_eq!(summary.admission_queue_ms, Some(20.0));
        assert_eq!(summary.scheduler_prefill_ms, Some(170.0));
        assert_eq!(summary.stream_flush_ms, Some(50.0));
    }
}
