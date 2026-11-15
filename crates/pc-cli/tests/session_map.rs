//! Replay of a session on a map model (SPEC sections 8t and 8u). A 1.0.0 session can hold map
//! runs with the rows the old engine computed. Replay goes through `model_equilibrium` and
//! `run_start`. Both refuse an equilibrium or an IVP run on a map, so those records fail and
//! name the refusal, while an ITER record replays.

mod common;

use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use pc_cli::session::{Block, Engine, EqResult, Evaluator, Item, Kind, ModelRec, RunRec, Session};
use serde_json::Value;

/// The logistic map, as in `crates/pc-ode/tests/corpus/logistic_map.ode`.
const LOGISTIC_MAP: &str = include_str!("../../pc-ode/tests/corpus/logistic_map.ode");
/// The same right-hand side with `@ meth=discrete`.
const LOGISTIC_DISCRETE: &str = "par r=2\ninit x=0.2\nx'=r*x*(1-x)\n@ meth=discrete\ndone\n";
const EQ_OPTS: &str = r#"{"max_newton":50,"tol_x":1e-10,"tol_f":1e-10}"#;
const IVP_PROBLEM: &str =
    r#"{"type":"IVP","tspan":[0.0,5.0],"options":{"rtol":1e-8,"atol":1e-10,"max_steps":1000}}"#;
const ITER_PROBLEM: &str = r#"{"type":"ITER","tspan":[0,5]}"#;
/// Part of the refusal text of SPEC 8t, kept in SPEC 8u.
const REFUSAL: &str = "the model is a map";

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("session_map");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A session with an equilibrium record and a trajectory record on `source`. The stored
/// values are placeholders, since replay is refused before it compares anything.
fn map_session(source: &str) -> Session {
    let eq = RunRec {
        kind: Kind::Equilibrium,
        problem: EQ_OPTS.to_owned(),
        status: String::new(),
        ty: String::new(),
        free: Vec::new(),
        grid: Vec::new(),
        n: 1,
        u0: Block::row(vec![0.5]),
        p: Block::row(vec![2.0]),
        rows: Vec::new(),
        specials: Vec::new(),
        result: Some(EqResult {
            u: Block::row(vec![0.0]),
            eig_re: Block::row(vec![2.0]),
            eig_im: Block::row(vec![0.0]),
        }),
    };
    let ivp = RunRec {
        kind: Kind::Simulate,
        problem: IVP_PROBLEM.to_owned(),
        status: r#"{"state":"done"}"#.to_owned(),
        ty: "IVP".to_owned(),
        free: Vec::new(),
        grid: Vec::new(),
        n: 1,
        u0: Block::row(vec![0.2]),
        p: Block::row(vec![2.0]),
        rows: vec![Block::new(2, 2, vec![0.0, 0.2, 5.0, 0.5])],
        specials: vec![Block::new(0, 2, Vec::new())],
        result: None,
    };
    session(
        source,
        vec![eq, ivp],
        vec![Item { run: 1, branch: 0 }, Item { run: 2, branch: 1 }],
    )
}

/// A session with one ITER record of the logistic map at r = 2.5 from 0.1 over [0 5]. The
/// rows come from a loop that keeps the association of the model, (r*x)*(1-x), so a replay
/// matches them bit for bit.
fn iter_session(source: &str) -> Session {
    let r = 2.5;
    let mut x = 0.1_f64;
    let mut rows = vec![0.0, x];
    for k in 1..=5 {
        x = (r * x) * (1.0 - x);
        rows.extend([f64::from(k), x]);
    }
    let iter = RunRec {
        kind: Kind::Simulate,
        problem: ITER_PROBLEM.to_owned(),
        status: r#"{"state":"done","kind":"ITER","n":1,"stride":2,"branches":[{"stop":"End","points":6,"specials":[],"warnings":[]}]}"#.to_owned(),
        ty: "ITER".to_owned(),
        free: Vec::new(),
        grid: Vec::new(),
        n: 1,
        u0: Block::row(vec![0.1]),
        p: Block::row(vec![r]),
        rows: vec![Block::new(6, 2, rows)],
        specials: vec![Block::new(0, 2, Vec::new())],
        result: None,
    };
    session(source, vec![iter], vec![Item { run: 1, branch: 1 }])
}

fn session(source: &str, runs: Vec<RunRec>, items: Vec<Item>) -> Session {
    let engine = Engine::load().unwrap();
    let m = engine.model(source, Evaluator::Jit).unwrap();
    let (name, version) = engine.version();
    Session {
        created: "2026-11-14T00:00:00Z".to_owned(),
        engine_name: name,
        engine_version: version,
        platform: "rust-test".to_owned(),
        matlab: "none".to_owned(),
        model: ModelRec {
            name: "logistic.ode".to_owned(),
            source: source.to_owned(),
            sha256: m.sha256().to_owned(),
            evaluator: "jit".to_owned(),
        },
        runs,
        items,
    }
}

#[test]
fn replay_of_a_map_session_fails_every_run_with_the_refusal() {
    for (tag, source) in [("map", LOGISTIC_MAP), ("discrete", LOGISTIC_DISCRETE)] {
        let file = scratch().join(format!("{tag}.pcsession.json"));
        std::fs::write(&file, map_session(source).to_text()).unwrap();
        let o = run(&scratch(), &["session", "replay", file.to_str().unwrap()]);
        let (out, err) = (stdout(&o), stderr(&o));
        assert_eq!(code(&o), 1, "{tag}: stdout {out}\nstderr {err}");
        let report: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(report["pass"], false, "{tag}: {report}");
        let runs = report["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2, "{tag}: {report}");
        for r in runs {
            assert_eq!(r["pass"], false, "{tag}: {r}");
            let msg = r["message"].as_str().unwrap();
            assert!(msg.starts_with("replay failed: "), "{tag}: {msg}");
            assert!(msg.contains(REFUSAL), "{tag}: {msg}");
        }
    }
}

#[test]
fn replay_of_an_iter_record_on_a_map_passes() {
    for (tag, source) in [("map", LOGISTIC_MAP), ("discrete", LOGISTIC_DISCRETE)] {
        let file = scratch().join(format!("{tag}_iter.pcsession.json"));
        std::fs::write(&file, iter_session(source).to_text()).unwrap();
        let o = run(&scratch(), &["session", "replay", file.to_str().unwrap()]);
        let (out, err) = (stdout(&o), stderr(&o));
        assert_eq!(code(&o), 0, "{tag}: stdout {out}\nstderr {err}");
        let report: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(report["pass"], true, "{tag}: {report}");
        let r = &report["runs"][0];
        assert_eq!(r["type"], "ITER", "{tag}: {r}");
        assert_eq!(r["identical"], true, "{tag}: {r}");
    }
}
