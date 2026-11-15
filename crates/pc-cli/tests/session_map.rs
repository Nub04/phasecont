//! Replay of a session on a map model (SPEC section 8t). A 1.0.0 session can hold map runs
//! with the rows the old engine computed. Replay goes through `model_equilibrium` and
//! `run_start`, which both refuse maps, so every map run fails and names the refusal.

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
/// Part of the refusal text of SPEC 8t.
const REFUSAL: &str = "the model is a map";

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("session_map");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A session with an equilibrium record and a trajectory record on `source`. The stored
/// values are placeholders, since replay is refused before it compares anything.
fn map_session(source: &str) -> Session {
    let engine = Engine::load().unwrap();
    let m = engine.model(source, Evaluator::Jit).unwrap();
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
        runs: vec![eq, ivp],
        items: vec![Item { run: 1, branch: 0 }, Item { run: 2, branch: 1 }],
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
