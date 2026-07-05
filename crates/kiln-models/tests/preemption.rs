//! Preemption, Cancel, and resume correctness (SPEC §6.1 / Phase 4 part 3).
//!
//! The invariant under test is *correctness*, not just liveness: a request
//! preempted under memory pressure must resume — re-prefill + replay of
//! its generated tokens — onto **bit-identical** greedy output to a run
//! that was never preempted (CLAUDE.md determinism), including for a
//! committed golden fixture. Scenarios force preemption with artificially
//! small block pools under concurrent requests.
//!
//! Skips (with a note) when `KILN_TEST_MODELS` is unset or Metal is
//! unavailable. Single `#[test]` because the kiln-mlx live-object counter
//! is process-global.

#![cfg(feature = "metal")]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use kiln_engine::{
    Engine, EngineConfig, EngineRequest, FinishKind, FinishSummary, PenaltyOptions, Priority,
    SamplingOptions, SeqEvent,
};
use kiln_mlx::{Stream, debug};
use kiln_models::LlamaModel;
use kiln_tokenize::{ChatMessage, ChatTemplate, Tokenizer};
use minijinja::Value;

const MODEL_NAME: &str = "llama-3.2-1b-4bit";
/// Must match PINNED_DATE_STRING in scripts/gen-golden.py.
const PINNED_DATE_STRING: &str = "26 Jul 2024";
/// Safety bound for scripted scenarios (they need a few hundred steps).
const MAX_STEPS: usize = 2000;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var_os("KILN_TEST_MODELS")?;
    let dir = PathBuf::from(root).join(MODEL_NAME);
    dir.join("config.json").is_file().then_some(dir)
}

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    prompt: String,
    chat_template: bool,
    max_tokens: usize,
    expected_token_ids: Vec<u32>,
    #[allow(dead_code)]
    mlx_lm_version: String,
    weights_revision: String,
}

/// Collected outputs plus the cancel handle for one request.
struct Collected {
    tokens: Rc<RefCell<Vec<u32>>>,
    finish: Rc<RefCell<Option<FinishSummary>>>,
    cancel: Arc<AtomicBool>,
}

impl Collected {
    fn summary(&self) -> FinishSummary {
        self.finish.borrow().clone().expect("request finished")
    }

    fn tokens(&self) -> Vec<u32> {
        self.tokens.borrow().clone()
    }
}

fn request(prompt: &[u32], max_tokens: usize, priority: Priority) -> (EngineRequest, Collected) {
    let tokens = Rc::new(RefCell::new(Vec::new()));
    let finish = Rc::new(RefCell::new(None));
    let cancel = Arc::new(AtomicBool::new(false));
    let (t, f) = (Rc::clone(&tokens), Rc::clone(&finish));
    let request = EngineRequest {
        prompt: prompt.to_vec(),
        max_tokens,
        sampling: SamplingOptions::default(), // greedy
        penalties: PenaltyOptions {
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
        penalty_window: 0,
        stop_tokens: std::collections::HashSet::new(),
        priority,
        cancel: Arc::clone(&cancel),
        on_event: Box::new(move |event| {
            match event {
                SeqEvent::Token(token) => t.borrow_mut().push(token),
                SeqEvent::PrefixHit { .. } => {}
                SeqEvent::Finished(summary) => *f.borrow_mut() = Some(summary),
            }
            true
        }),
    };
    (
        request,
        Collected {
            tokens,
            finish,
            cancel,
        },
    )
}

fn drain(engine: &mut Engine<&LlamaModel>) {
    for _ in 0..MAX_STEPS {
        if engine.is_idle() {
            return;
        }
        engine.step().expect("engine step");
    }
    panic!("engine failed to drain within {MAX_STEPS} steps (livelock?)");
}

fn new_engine(model: &LlamaModel, config: EngineConfig) -> Engine<&LlamaModel> {
    Engine::new(model, model.kv_dims(), config, Stream::gpu()).expect("engine builds")
}

/// Runs one request alone (no pressure, same config ⇒ same chunk
/// boundaries) and returns its token stream — the resume-parity reference.
fn solo(model: &LlamaModel, config: EngineConfig, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    let mut engine = new_engine(model, config);
    let (request, out) = request(prompt, max_tokens, Priority::Interactive);
    engine.submit(request);
    drain(&mut engine);
    let summary = out.summary();
    assert_eq!(
        summary.reason,
        FinishKind::Length,
        "solo reference run must complete: {summary:?}"
    );
    assert_eq!(summary.preemptions, 0, "solo run cannot be preempted");
    out.tokens()
}

#[test]
fn preemption_resumes_bit_exact_and_cancel_is_bounded() {
    if !kiln_mlx::memory::metal_is_available() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let Some(model_dir) = model_dir() else {
        eprintln!("skipping: KILN_TEST_MODELS not set or {MODEL_NAME} missing");
        return;
    };

    let baseline = debug::live_objects();
    {
        let stream = Stream::gpu();
        let model = LlamaModel::load(&model_dir, &stream).expect("model loads");
        let tokenizer = Tokenizer::from_model_dir(&model_dir).expect("tokenizer loads");

        let text = "Paged attention splits the key-value cache into fixed-size blocks. ".repeat(24);
        let ids = tokenizer.encode(&text, true).expect("encodes");
        assert!(ids.len() >= 283, "need enough tokens to slice prompts");
        // Senior request: 100-token prompt growing to 160 slots (5 blocks).
        let senior = &ids[..100];
        // Junior request: 33-token prompt growing to 81 slots (3 blocks).
        let junior = &ids[100..133];

        // Pool: 6 blocks x 32 = 192 slots. senior(160) + junior(96 worst
        // case) exceed it mid-decode, but each fits alone — preemption,
        // never an error.
        // Cache off: this suite pins exact preemption counts and pool
        // arithmetic; donated blocks (evictable, SPEC §6.3) would shift
        // when OutOfBlocks fires. Cache-on preemption interplay is covered
        // by kiln-models/tests/prefix_cache.rs.
        let squeeze = EngineConfig {
            num_blocks: 6,
            prefix_cache: false,
            ..EngineConfig::default()
        };
        let solo_senior = solo(&model, squeeze.clone(), senior, 60);
        let solo_junior = solo(&model, squeeze.clone(), junior, 48);

        // 1) Same priority: the most recently admitted request is the
        //    victim; it resumes and both streams stay bit-exact.
        {
            let mut engine = new_engine(&model, squeeze.clone());
            let (a, a_out) = request(senior, 60, Priority::Interactive);
            let (b, b_out) = request(junior, 48, Priority::Interactive);
            engine.submit(a);
            engine.submit(b);
            drain(&mut engine);
            let (a_summary, b_summary) = (a_out.summary(), b_out.summary());
            assert_eq!(a_summary.reason, FinishKind::Length);
            assert_eq!(b_summary.reason, FinishKind::Length);
            assert!(
                b_summary.preemptions >= 1,
                "junior was never preempted: {b_summary:?}"
            );
            assert_eq!(a_summary.preemptions, 0, "senior must not be preempted");
            assert!(engine.preemptions() >= 1, "engine preemption counter");
            assert_eq!(a_out.tokens(), solo_senior, "senior stream disturbed");
            assert_eq!(
                b_out.tokens(),
                solo_junior,
                "preempted junior did not resume onto its solo stream"
            );
        }
        eprintln!("same-priority preemption: victim resumed bit-exact");

        // 2) Priority order: a BATCH request self-preempts under pressure
        //    even though it is senior to the INTERACTIVE one (SPEC §6.1:
        //    lowest priority first, then most recently admitted).
        {
            let mut engine = new_engine(&model, squeeze.clone());
            let (a, a_out) = request(senior, 60, Priority::Batch);
            let (b, b_out) = request(junior, 48, Priority::Interactive);
            engine.submit(a);
            engine.submit(b);
            drain(&mut engine);
            let (a_summary, b_summary) = (a_out.summary(), b_out.summary());
            assert_eq!(a_summary.reason, FinishKind::Length);
            assert_eq!(b_summary.reason, FinishKind::Length);
            assert!(
                a_summary.preemptions >= 1,
                "BATCH request was never preempted: {a_summary:?}"
            );
            assert_eq!(
                b_summary.preemptions, 0,
                "INTERACTIVE request must survive a BATCH competitor"
            );
            // Priority changes scheduling, never tokens.
            assert_eq!(a_out.tokens(), solo_senior, "batch stream diverged");
            assert_eq!(b_out.tokens(), solo_junior, "interactive stream diverged");
        }
        eprintln!("priority preemption: BATCH yielded to INTERACTIVE, both bit-exact");

        // 3a) Cancel mid-stream stops within the proto's 2-step budget.
        {
            let mut engine = new_engine(&model, squeeze.clone());
            let (a, a_out) = request(junior, 60, Priority::Interactive);
            engine.submit(a);
            while a_out.tokens.borrow().len() < 3 {
                engine.step().expect("engine step");
            }
            let at_cancel = a_out.tokens.borrow().len();
            a_out.cancel.store(true, Ordering::Release);
            let mut steps = 0;
            while a_out.finish.borrow().is_none() {
                assert!(steps < 2, "cancel not honored within 2 engine steps");
                engine.step().expect("engine step");
                steps += 1;
            }
            let summary = a_out.summary();
            assert_eq!(summary.reason, FinishKind::Cancelled);
            assert_eq!(
                a_out.tokens.borrow().len(),
                at_cancel,
                "tokens emitted after the cancel flag was set"
            );
            assert!(engine.is_idle(), "cancelled request still active");
        }
        eprintln!("cancel honored within {} step(s) of the flag", 1);

        // 3b) Cancel lands while the request sits preempted in WAITING.
        {
            let mut engine = new_engine(&model, squeeze.clone());
            let (a, a_out) = request(senior, 60, Priority::Interactive);
            let (b, b_out) = request(junior, 48, Priority::Interactive);
            engine.submit(a);
            engine.submit(b);
            let mut steps = 0;
            while engine.preemptions() == 0 {
                assert!(steps < MAX_STEPS, "scenario never preempted");
                assert!(!engine.is_idle(), "drained without preempting");
                engine.step().expect("engine step");
                steps += 1;
            }
            assert!(
                b_out.finish.borrow().is_none(),
                "junior finished instead of being preempted"
            );
            let at_cancel = b_out.tokens.borrow().len();
            assert!(
                at_cancel > 0,
                "junior should have streamed before preemption"
            );
            b_out.cancel.store(true, Ordering::Release);
            let mut cancel_steps = 0;
            while b_out.finish.borrow().is_none() {
                assert!(
                    cancel_steps < 2,
                    "waiting-queue cancel not honored within 2 steps"
                );
                engine.step().expect("engine step");
                cancel_steps += 1;
            }
            let b_summary = b_out.summary();
            assert_eq!(b_summary.reason, FinishKind::Cancelled);
            assert_eq!(b_summary.preemptions, 1);
            assert_eq!(b_summary.completion_tokens as usize, at_cancel);
            drain(&mut engine);
            let a_summary = a_out.summary();
            assert_eq!(a_summary.reason, FinishKind::Length);
            assert_eq!(
                a_out.tokens(),
                solo_senior,
                "survivor disturbed by preempt-then-cancel"
            );
        }
        eprintln!("cancel-while-preempted ok, survivor bit-exact");

        // 4) Golden fixture under forced preemption: the fixture request
        //    is preempted mid-decode, resumes, and must still reproduce
        //    the committed mlx-lm reference token-for-token.
        {
            let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/golden")
                .join(MODEL_NAME)
                .join("chat-code.json");
            let fixture: Fixture = serde_json::from_str(
                &std::fs::read_to_string(&fixture_path).expect("fixture readable"),
            )
            .expect("fixture parses");
            let local_revision = std::fs::read_to_string(model_dir.join(".kiln-revision"))
                .map(|text| text.trim().to_owned())
                .unwrap_or_default();
            assert_eq!(
                fixture.weights_revision, local_revision,
                "fixture was generated for a different weights revision"
            );
            assert!(fixture.chat_template, "chat-code is a chat fixture");
            let template = ChatTemplate::from_model_dir(&model_dir).expect("template loads");
            let rendered = template
                .render_with(
                    &[ChatMessage {
                        role: "user".into(),
                        content: fixture.prompt.clone(),
                    }],
                    true,
                    &[("date_string", Value::from(PINNED_DATE_STRING))],
                )
                .expect("template renders");
            // BOS contract: the rendered template already contains BOS.
            let fixture_ids = tokenizer.encode(&rendered, false).expect("encodes");

            // Pool: 8 blocks (256 slots). The pressure request grows to
            // 160 slots (5 blocks); the fixture (47 prompt + 128
            // generated = 175 slots, 6 blocks) collides mid-decode, is
            // preempted as the most recent, and resumes after the
            // pressure request finishes.
            let pressured = EngineConfig {
                num_blocks: 8,
                prefix_cache: false,
                ..EngineConfig::default()
            };
            let mut engine = new_engine(&model, pressured.clone());
            let (p, p_out) = request(senior, 60, Priority::Interactive);
            let (g, g_out) = request(&fixture_ids, fixture.max_tokens, Priority::Interactive);
            engine.submit(p);
            engine.submit(g);
            drain(&mut engine);
            let (p_summary, g_summary) = (p_out.summary(), g_out.summary());
            assert_eq!(p_summary.reason, FinishKind::Length);
            assert_eq!(g_summary.reason, FinishKind::Length);
            assert!(
                g_summary.preemptions >= 1,
                "fixture request was never preempted (resize the scenario): {g_summary:?}"
            );
            assert_eq!(
                g_out.tokens(),
                fixture.expected_token_ids,
                "golden parity broken by preemption (prompt tokens: {})",
                fixture_ids.len()
            );
        }
        eprintln!("golden chat-code under preemption: exact match after resume");

        // 5) Sustained pressure: a BATCH request outcompeted by a stream
        //    of younger INTERACTIVE arrivals is preempted on every cycle
        //    but never starved — arrivals are finite, seniority is stable,
        //    and it resumes onto its solo stream (drain()'s MAX_STEPS
        //    guard is the liveness bound).
        {
            let mut engine = new_engine(&model, squeeze.clone());
            let (b, b_out) = request(senior, 60, Priority::Batch);
            let (i1, i1_out) = request(junior, 48, Priority::Interactive);
            engine.submit(b);
            engine.submit(i1);
            let mut steps = 0;
            while engine.preemptions() == 0 {
                assert!(steps < MAX_STEPS, "scenario never preempted");
                engine.step().expect("engine step");
                steps += 1;
            }
            assert!(
                b_out.finish.borrow().is_none(),
                "BATCH request finished instead of yielding"
            );
            // Second younger INTERACTIVE arrival while the BATCH request
            // sits preempted: it forces a second cycle after the resume.
            let (i2, i2_out) = request(junior, 48, Priority::Interactive);
            engine.submit(i2);
            drain(&mut engine);
            let b_summary = b_out.summary();
            assert_eq!(b_summary.reason, FinishKind::Length);
            assert!(
                b_summary.preemptions >= 2,
                "sustained pressure should preempt the BATCH request on \
                 every cycle: {b_summary:?}"
            );
            assert_eq!(
                b_out.tokens(),
                solo_senior,
                "twice-preempted request did not resume onto its solo stream"
            );
            for (name, out) in [("i1", &i1_out), ("i2", &i2_out)] {
                let summary = out.summary();
                assert_eq!(summary.reason, FinishKind::Length, "{name}");
                assert_eq!(summary.preemptions, 0, "{name} must not be preempted");
                assert_eq!(out.tokens(), solo_junior, "{name} stream diverged");
            }
        }
        eprintln!("double preemption under sustained pressure: resumed bit-exact both times");

        // 6) Arrival seniority is stable across preemption: after J is
        //    preempted and resumes, a younger equal-priority K must be
        //    the next victim — a (buggy) arrival renewed on re-admission
        //    would mark J newest and victimize it again.
        {
            let kprompt = &ids[133..198]; // 65 tokens; grows to 113 slots (4 blocks)
            let solo_k = solo(&model, squeeze.clone(), kprompt, 48);
            let mut engine = new_engine(&model, squeeze.clone());
            let (a, a_out) = request(senior, 60, Priority::Interactive);
            let (j, j_out) = request(junior, 48, Priority::Interactive);
            engine.submit(a);
            engine.submit(j);
            // Cycle 1: A's growth preempts J (most recent). Run A out.
            let mut steps = 0;
            while a_out.finish.borrow().is_none() {
                assert!(steps < MAX_STEPS, "senior never finished");
                engine.step().expect("engine step");
                steps += 1;
            }
            assert!(engine.preemptions() >= 1, "cycle 1 never preempted J");
            assert!(j_out.finish.borrow().is_none(), "J finished prematurely");
            // J resumes now; K arrives younger. When the pool fills again
            // the victim must be K, not the once-preempted J.
            let (k, k_out) = request(kprompt, 48, Priority::Interactive);
            engine.submit(k);
            drain(&mut engine);
            let (j_summary, k_summary) = (j_out.summary(), k_out.summary());
            assert_eq!(j_summary.reason, FinishKind::Length);
            assert_eq!(k_summary.reason, FinishKind::Length);
            assert_eq!(
                j_summary.preemptions, 1,
                "J's seniority must survive its resume — the younger K is \
                 the next victim: {j_summary:?}"
            );
            assert!(
                k_summary.preemptions >= 1,
                "pressure between J and K never landed (resize the scenario): {k_summary:?}"
            );
            assert_eq!(a_out.tokens(), solo_senior, "senior stream diverged");
            assert_eq!(j_out.tokens(), solo_junior, "J stream diverged");
            assert_eq!(k_out.tokens(), solo_k, "K stream diverged");
        }
        eprintln!("arrival seniority stable across resume: younger K preempted, J untouched");

        // 7) Preemption mid-prefill: with a small prefill chunk the junior
        //    is still PREFILLING (zero tokens streamed) when the pool
        //    fills; it must rewind to WAITING and later re-prefill from
        //    scratch onto its solo stream.
        {
            // Pool 9 = 288 slots; chunk 8 stretches the junior's 149-token
            // prefill across 19 iterations. The 120-token senior needs its
            // 5th block 10 decode steps in — inside that window — so the
            // junior's own chunk 17 (slot 129, its 5th block) finds the
            // pool full at 128/149 prompt tokens processed.
            let chunked = EngineConfig {
                num_blocks: 9,
                prefill_chunk: 8,
                prefix_cache: false,
                ..EngineConfig::default()
            };
            let a7prompt = &ids[..120]; // 120 tokens -> 180 slots total
            let jprompt = &ids[133..283]; // 150 tokens -> 174 slots total
            let solo_chunk_senior = solo(&model, chunked.clone(), a7prompt, 60);
            let solo_chunk_j = solo(&model, chunked.clone(), jprompt, 24);
            let mut engine = new_engine(&model, chunked.clone());
            let (a, a_out) = request(a7prompt, 60, Priority::Interactive);
            let (j, j_out) = request(jprompt, 24, Priority::Interactive);
            engine.submit(a);
            engine.submit(j);
            let mut steps = 0;
            while engine.preemptions() == 0 {
                assert!(steps < MAX_STEPS, "scenario never preempted");
                assert!(!engine.is_idle(), "drained without preempting");
                engine.step().expect("engine step");
                steps += 1;
            }
            assert!(
                j_out.tokens.borrow().is_empty(),
                "junior should be preempted mid-prefill, before any token"
            );
            drain(&mut engine);
            let (a_summary, j_summary) = (a_out.summary(), j_out.summary());
            assert_eq!(a_summary.reason, FinishKind::Length);
            assert_eq!(j_summary.reason, FinishKind::Length);
            assert_eq!(a_summary.preemptions, 0, "senior must not be preempted");
            assert!(j_summary.preemptions >= 1, "junior was never preempted");
            assert_eq!(
                a_out.tokens(),
                solo_chunk_senior,
                "senior stream diverged under mid-prefill preemption"
            );
            assert_eq!(
                j_out.tokens(),
                solo_chunk_j,
                "mid-prefill preempted request did not re-prefill onto its solo stream"
            );
        }
        eprintln!("mid-prefill preemption: junior re-prefilled from scratch, bit-exact");

        // 8) Phase 4 closeout: preemption under the batch-16 load shape,
        //    not a 2-3 request micro-pool. A 16-deep interactive burst
        //    (8 × A: 64-token prompts, 48 new tokens; 8 × B: 96-token
        //    prompts, 32 new tokens, interleaved) runs against a pool that
        //    admits all 16 — the batch genuinely reaches width 16 — but
        //    cannot hold their full growth (peak demand 64 blocks vs 54),
        //    so the youngest requests are preempted mid-decode as seniors
        //    grow, sit out, resume, and every one of the 16 streams must
        //    still equal its solo run bit-exact.
        {
            let a_prompt = &ids[..64];
            let b_prompt = &ids[64..160];
            let wide = EngineConfig {
                num_blocks: 54,
                prefix_cache: false,
                ..EngineConfig::default()
            };
            let solo_a = solo(&model, wide.clone(), a_prompt, 48);
            let solo_b = solo(&model, wide.clone(), b_prompt, 32);
            let mut engine = new_engine(&model, wide.clone());
            let outs: Vec<Collected> = (0..16)
                .map(|k| {
                    let (req, out) = if k % 2 == 0 {
                        request(a_prompt, 48, Priority::Interactive)
                    } else {
                        request(b_prompt, 32, Priority::Interactive)
                    };
                    engine.submit(req);
                    out
                })
                .collect();
            let mut max_width = 0;
            let mut steps = 0;
            while !engine.is_idle() {
                assert!(steps < MAX_STEPS, "batch-16 pressure scenario livelocked");
                engine.step().expect("engine step");
                max_width = max_width.max(engine.num_running());
                steps += 1;
            }
            assert_eq!(
                max_width, 16,
                "the load never reached the batch-16 shape (resize the scenario)"
            );
            assert!(
                engine.preemptions() >= 1,
                "pool pressure never landed at width 16 (resize the scenario)"
            );
            for (k, out) in outs.iter().enumerate() {
                let summary = out.summary();
                assert_eq!(
                    summary.reason,
                    FinishKind::Length,
                    "request {k} must finish under batch-16 pressure: {summary:?}"
                );
                let want = if k % 2 == 0 { &solo_a } else { &solo_b };
                assert_eq!(
                    &out.tokens(),
                    want,
                    "request {k} diverged from its solo stream under batch-16 pressure \
                     (preemptions: {})",
                    summary.preemptions
                );
            }
            // Victim order still holds at this width: the two most senior
            // requests must never be preempted.
            assert_eq!(outs[0].summary().preemptions, 0, "senior A preempted");
            assert_eq!(outs[1].summary().preemptions, 0, "senior B preempted");
            let preempted = outs
                .iter()
                .filter(|out| out.summary().preemptions > 0)
                .count();
            eprintln!(
                "batch-16 pressure: width {max_width}, {} preemption(s) across \
                 {preempted} request(s), all 16 streams bit-exact",
                engine.preemptions()
            );
        }
    }
    assert_eq!(
        debug::live_objects(),
        baseline,
        "preemption run leaked mlx handles"
    );
}
