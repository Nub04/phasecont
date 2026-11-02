//! The acceptance test of SPEC §14 for sessions, in Rust: a full Morris-Lecar diagram goes
//! through the same engine functions that MATLAB calls (`model_equilibrium`, `run_start`,
//! `run_poll`, `run_copy`), is written as a version 1 session file, and `pc-cli session
//! replay` must reproduce it. The diagram holds a Newton solve, the branch of equilibria in
//! I with its Hopf point, the branch of cycles from that point with its fold of cycles, the
//! curve of folds and the curve of Hopf points in (I, phi), and one trajectory.
//!
//! The cycles use a coarse mesh (NTST 40, NCOL 4), so the engine warns that the mesh is too
//! coarse. The warnings are part of the recorded status and the replay must repeat them.

mod common;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use common::{code, run, stderr, stdout};
use pc_cli::session::{
    Block, Engine, EqResult, Evaluator, Item, Kind, ModelHandle, ModelRec, RunRec, Session,
};
use serde_json::Value;

const NTST: usize = 40;
const NCOL: usize = 4;
const CREATED: &str = "2026-11-01T00:00:00Z";
const EQ_OPTS: &str = r#"{"max_newton":50,"tol_x":1e-10,"tol_f":1e-10}"#;
const MODEL_NAME: &str = "morris_lecar.ode";

fn ml_source() -> String {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-engine/tests/models/morris_lecar.ode");
    std::fs::read_to_string(path).expect("cannot read the Morris-Lecar model")
}

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("session_ml");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/+fixtures/ml_small.pcsession.json")
}

/// Morris-Lecar parameters (I, phi, C, gL, gK, gCa, vL, vK, vCa, v1, v2, v3, v4).
fn ml_params(i: f64, phi: f64) -> Vec<f64> {
    vec![
        i, phi, 20.0, 2.0, 8.0, 4.4, -60.0, -84.0, 120.0, -1.2, 18.0, 2.0, 30.0,
    ]
}

/// The number of values of an orbit on the collocation grid, for two states.
const ORBIT: usize = 2 * (NTST * NCOL + 1);

// ---------------------------------------------------------------------------------------
// Building runs.

fn equilibrium_run(m: &ModelHandle, guess: &[f64], p: &[f64]) -> RunRec {
    let result: EqResult = m.equilibrium(EQ_OPTS, guess, p).unwrap();
    RunRec {
        kind: Kind::Equilibrium,
        problem: EQ_OPTS.to_owned(),
        status: String::new(),
        ty: String::new(),
        free: Vec::new(),
        grid: Vec::new(),
        n: guess.len(),
        u0: Block::row(guess.to_vec()),
        p: Block::row(p.to_vec()),
        rows: Vec::new(),
        specials: Vec::new(),
        result: Some(result),
    }
}

/// A run through `run_start`.
fn started(
    m: &ModelHandle,
    (kind, ty, free, grid): (Kind, &str, &[&str], Vec<usize>),
    problem: String,
    u0: &[f64],
    p: &[f64],
) -> RunRec {
    let ex = m
        .run(&problem, u0, p)
        .unwrap_or_else(|e| panic!("{ty} run: {e}"));
    RunRec {
        kind,
        problem,
        status: ex.status,
        ty: ty.to_owned(),
        free: free.iter().map(|s| (*s).to_owned()).collect(),
        grid,
        n: 2,
        u0: Block::row(u0.to_vec()),
        p: Block::row(p.to_vec()),
        rows: ex.rows,
        specials: ex.specials,
        result: None,
    }
}

/// The special row labeled `label` of the first branch of `rec`, with its position.
fn special_row<'a>(rec: &'a RunRec, label: &str) -> &'a [f64] {
    let status: Value = serde_json::from_str(&rec.status).unwrap();
    let list = status["branches"][0]["specials"].as_array().unwrap();
    let at = list
        .iter()
        .position(|s| s["label"] == label)
        .unwrap_or_else(|| panic!("no {label} point in {}", rec.status));
    let b = &rec.specials[0];
    &b.data()[at * b.cols()..(at + 1) * b.cols()]
}

fn mesh_json(mesh: &[f64]) -> String {
    let items: Vec<String> = mesh.iter().map(|v| format!("{v:?}")).collect();
    items.join(",")
}

fn ep_problem(bounds: (f64, f64), max_steps: usize) -> String {
    format!(
        r#"{{"type":"EP","free":"I","start_is_special":false,"options":{{"direction":1,"bounds":[{:?},{:?}],"max_steps":{max_steps},"hmax":2.0,"h0":0.5,"detect":{{"lp":true,"bp":true,"h":true}}}}}}"#,
        bounds.0, bounds.1
    )
}

fn lc_problem() -> String {
    format!(
        r#"{{"type":"LC","free":"I","ntst":{NTST},"ncol":{NCOL},"start":"hopf","options":{{"bounds":[80.0,120.0],"max_steps":300,"max_newton":15,"hmax":0.5}}}}"#
    )
}

fn lpc_problem(mesh: &[f64]) -> String {
    format!(
        r#"{{"type":"LPC","free":["I","phi"],"ntst":{NTST},"ncol":{NCOL},"mesh":[{}],"options":{{"bounds":[0.02,0.06],"max_steps":100,"max_newton":15,"hmax":0.5}}}}"#,
        mesh_json(mesh)
    )
}

const H_PROBLEM: &str = r#"{"type":"H","free":["I","phi"],"options":{"bounds":[0.02,0.06],"max_steps":200,"max_newton":15,"hmax":0.5}}"#;
const IVP_PROBLEM: &str =
    r#"{"type":"IVP","tspan":[0.0,300.0],"options":{"rtol":1e-8,"atol":1e-10,"max_steps":100000}}"#;

fn session_of(eng: &Engine, m: &ModelHandle, source: String, runs: Vec<RunRec>) -> Session {
    let items = runs
        .iter()
        .enumerate()
        .flat_map(|(i, r)| {
            if r.kind == Kind::Equilibrium {
                vec![Item {
                    run: i + 1,
                    branch: 0,
                }]
            } else {
                (1..=r.rows.len())
                    .map(|b| Item {
                        run: i + 1,
                        branch: b,
                    })
                    .collect()
            }
        })
        .collect();
    let (name, version) = eng.version();
    Session {
        created: CREATED.to_owned(),
        engine_name: name,
        engine_version: version,
        platform: "rust-test".to_owned(),
        matlab: "none".to_owned(),
        model: ModelRec {
            name: MODEL_NAME.to_owned(),
            source,
            sha256: m.sha256().to_owned(),
            evaluator: "jit".to_owned(),
        },
        runs,
        items,
    }
}

/// The full diagram: six runs.
fn full_diagram(eng: &Engine) -> (Session, Vec<(&'static str, f64)>) {
    let source = ml_source();
    let m = eng.model(&source, Evaluator::Jit).unwrap();
    let mut times = Vec::new();
    let mut clock = |what: &'static str, t: Instant| times.push((what, t.elapsed().as_secs_f64()));

    let t = Instant::now();
    let p80 = ml_params(80.0, 0.04);
    let eq = equilibrium_run(&m, &[-20.0, 0.1], &p80);
    clock("equilibrium", t);

    // Branch of equilibria in I from the rest state at I = 0, up to the Hopf point.
    let t = Instant::now();
    let p0 = ml_params(0.0, 0.04);
    let rest = equilibrium_run(&m, &[-60.0, 0.0], &p0);
    let rest_u = rest.result.as_ref().unwrap().u.data().to_vec();
    let ep = started(
        &m,
        (Kind::Continuation, "EP", &["I"], Vec::new()),
        ep_problem((0.0, 150.0), 400),
        &rest_u,
        &p0,
    );
    clock("EP", t);
    let h = special_row(&ep, "H");
    let (i_hopf, h_u) = (h[0], vec![h[1], h[2]]);
    let p_hopf = ml_params(i_hopf, 0.04);

    // Cycles from the Hopf point, with the fold of cycles below it.
    let t = Instant::now();
    let lc = started(
        &m,
        (Kind::Continuation, "LC", &["I"], vec![NTST, NCOL]),
        lc_problem(),
        &h_u,
        &p_hopf,
    );
    clock("LC", t);
    let fold = special_row(&lc, "LPC");
    let mut start = fold[1..2 + ORBIT].to_vec();
    assert_eq!(start.len(), ORBIT + 1);
    let mesh = fold[2 + ORBIT..3 + ORBIT + NTST].to_vec();
    let p_fold = ml_params(fold[0], 0.04);

    let t = Instant::now();
    let lpc = started(
        &m,
        (Kind::Continuation, "LPC", &["I", "phi"], vec![NTST, NCOL]),
        lpc_problem(&mesh),
        &start,
        &p_fold,
    );
    clock("LPC curve", t);
    start.clear();

    let t = Instant::now();
    let hopf_curve = started(
        &m,
        (Kind::Continuation, "H", &["I", "phi"], Vec::new()),
        H_PROBLEM.to_owned(),
        &h_u,
        &p_hopf,
    );
    clock("H curve", t);

    let t = Instant::now();
    let ivp = started(
        &m,
        (Kind::Simulate, "IVP", &[], Vec::new()),
        IVP_PROBLEM.to_owned(),
        &[-20.0, 0.1],
        &ml_params(100.0, 0.04),
    );
    clock("IVP", t);

    let runs = vec![eq, ep, lc, lpc, hopf_curve, ivp];
    (session_of(eng, &m, source, runs), times)
}

/// The small session of the MATLAB fixture: a Newton solve, a short branch of equilibria
/// with its Hopf point, and a short trajectory with no special points.
fn small_diagram(eng: &Engine) -> Session {
    let source = ml_source();
    let m = eng.model(&source, Evaluator::Jit).unwrap();
    let p60 = ml_params(60.0, 0.04);
    let eq = equilibrium_run(&m, &[-30.0, 0.1], &p60);
    let u = eq.result.as_ref().unwrap().u.data().to_vec();
    let ep = started(
        &m,
        (Kind::Continuation, "EP", &["I"], Vec::new()),
        ep_problem((60.0, 100.0), 100),
        &u,
        &p60,
    );
    let ivp = started(
        &m,
        (Kind::Simulate, "IVP", &[], Vec::new()),
        r#"{"type":"IVP","tspan":[0.0,12.0],"options":{"rtol":1e-6,"atol":1e-9,"max_steps":1000}}"#
            .to_owned(),
        &[-20.0, 0.1],
        &ml_params(100.0, 0.04),
    );
    session_of(eng, &m, source, vec![eq, ep, ivp])
}

/// The text of the full diagram, built once for all tests.
fn diagram_text() -> &'static str {
    static TEXT: OnceLock<String> = OnceLock::new();
    TEXT.get_or_init(|| {
        let eng = Engine::load().unwrap();
        let (s, times) = full_diagram(&eng);
        for (what, secs) in times {
            println!("built {what}: {secs:.2} s");
        }
        s.to_text()
    })
}

fn write(name: &str, text: &str) -> PathBuf {
    let path = scratch().join(name);
    std::fs::write(&path, text).unwrap();
    path
}

/// Runs `pc-cli session replay` on a file and returns the exit code, the report and stderr.
fn replay(file: &Path, extra: &[&str]) -> (i32, Value, String) {
    let mut args = vec!["session", "replay", file.to_str().unwrap()];
    args.extend_from_slice(extra);
    let o = run(&scratch(), &args);
    let out = stdout(&o);
    let report = serde_json::from_str(&out).unwrap_or(Value::Null);
    (code(&o), report, stderr(&o))
}

// ---------------------------------------------------------------------------------------
// The tests.

/// The same cycle problem run twice in one process gives the same bits, status text
/// included. Replay reports `identical` only because of this.
#[test]
fn the_cycle_run_is_bit_deterministic() {
    let eng = Engine::load().unwrap();
    let m = eng.model(&ml_source(), Evaluator::Jit).unwrap();
    let p0 = ml_params(0.0, 0.04);
    let rest = equilibrium_run(&m, &[-60.0, 0.0], &p0);
    let rest_u = rest.result.as_ref().unwrap().u.data().to_vec();
    let ep = started(
        &m,
        (Kind::Continuation, "EP", &["I"], Vec::new()),
        ep_problem((0.0, 150.0), 400),
        &rest_u,
        &p0,
    );
    let h = special_row(&ep, "H");
    let (h_u, p) = (vec![h[1], h[2]], ml_params(h[0], 0.04));
    let t = Instant::now();
    let a = m.run(&lc_problem(), &h_u, &p).unwrap();
    let first = t.elapsed();
    let b = m.run(&lc_problem(), &h_u, &p).unwrap();
    println!("LC run {first:?}, then {:?}", t.elapsed() - first);
    assert_eq!(a.status, b.status);
    assert_eq!(a.rows.len(), b.rows.len());
    for (x, y) in a.rows.iter().zip(&b.rows) {
        assert!(x.same_bits(y), "the rows of two identical runs differ");
    }
    for (x, y) in a.specials.iter().zip(&b.specials) {
        assert!(
            x.same_bits(y),
            "the special rows of two identical runs differ"
        );
    }
}

#[test]
fn the_diagram_has_the_expected_runs() {
    let s = Session::parse(diagram_text()).unwrap();
    let kinds: Vec<_> = s.runs.iter().map(|r| (r.kind, r.ty.as_str())).collect();
    assert_eq!(
        kinds,
        [
            (Kind::Equilibrium, ""),
            (Kind::Continuation, "EP"),
            (Kind::Continuation, "LC"),
            (Kind::Continuation, "LPC"),
            (Kind::Continuation, "H"),
            (Kind::Simulate, "IVP"),
        ]
    );
    // One item per run, and the equilibrium has branch 0.
    assert_eq!(s.items.len(), 6);
    assert_eq!(s.items[0], Item { run: 1, branch: 0 });
    assert_eq!(s.runs[2].grid, [NTST, NCOL]);
    assert_eq!(s.runs[2].rows[0].cols(), 2 + ORBIT + (NTST + 1) + 4 + 3 + 2);
    // The Hopf point is subcritical, so the branch of cycles leaves it toward lower I and
    // the fold of cycles lies below the Hopf point.
    let ep_rows = &s.runs[1].rows[0];
    assert_eq!(ep_rows.cols(), 12);
    let hopf = special_i(&s.runs[1], "H");
    let fold = special_i(&s.runs[2], "LPC");
    assert!((hopf - 93.8576).abs() < 1e-3, "Hopf at I = {hopf}");
    assert!(fold < hopf - 4.0 && fold > 80.0, "fold at I = {fold}");
    // Both curves start at phi = 0.04 and end on the bound phi = 0.06.
    for run in [3, 4] {
        let r = &s.runs[run].rows[0];
        assert_eq!(r.data()[0], 0.04);
        let last = r.data()[(r.rows() - 1) * r.cols()];
        assert!((last - 0.06).abs() < 1e-9, "run {run} ends at phi = {last}");
    }
    let ivp = &s.runs[5].rows[0];
    assert_eq!(ivp.cols(), 3);
    assert_eq!(ivp.data()[(ivp.rows() - 1) * 3], 300.0);
}

fn special_i(rec: &RunRec, label: &str) -> f64 {
    special_row(rec, label)[0]
}

#[test]
fn the_diagram_survives_a_text_round_trip() {
    let s = Session::parse(diagram_text()).unwrap();
    assert_eq!(s.to_text(), diagram_text());
    let again = Session::parse(&s.to_text()).unwrap();
    for (a, b) in s.runs.iter().zip(&again.runs) {
        assert_eq!(a.problem, b.problem);
        assert_eq!(a.status, b.status);
        assert!(a.u0.same_bits(&b.u0) && a.p.same_bits(&b.p));
        for (x, y) in a
            .rows
            .iter()
            .zip(&b.rows)
            .chain(a.specials.iter().zip(&b.specials))
        {
            assert!(x.same_bits(y));
        }
    }
}

/// SPEC §14: the replay of the diagram passes on every run, and every run is bit-identical.
#[test]
fn replay_reproduces_the_full_diagram() {
    let file = write("diagram.pcsession.json", diagram_text());
    let t = Instant::now();
    let (c, report, err) = replay(&file, &[]);
    println!("replay took {:?}", t.elapsed());
    assert_eq!(c, 0, "stderr: {err}\nreport: {report}");
    assert_eq!(report["pass"], true);
    assert_eq!(report["tol"], 1e-8);
    assert_eq!(report["evaluator"], "jit");
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 6);
    for (i, r) in runs.iter().enumerate() {
        assert_eq!(r["index"], i + 1);
        assert_eq!(r["pass"], true, "run {}: {r}", i + 1);
        assert_eq!(r["identical"], true, "run {}: {r}", i + 1);
        assert_eq!(r["max_rel_diff"], 0.0, "run {}: {r}", i + 1);
        assert_eq!(r["labels_match"], true);
        assert_eq!(r["stop_match"], true);
        assert_eq!(r["rows_match"], true);
    }
    let types: Vec<_> = runs.iter().map(|r| r["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["", "EP", "LC", "LPC", "H", "IVP"]);
}

/// Changes one number of the EP branch (run 2) by 1e-3.
fn with_changed_row() -> Session {
    let mut s = Session::parse(diagram_text()).unwrap();
    let b = &s.runs[1].rows[0];
    let mut data = b.data().to_vec();
    data[10 * b.cols() + 1] += 1e-3;
    s.runs[1].rows[0] = Block::new(b.rows(), b.cols(), data);
    s
}

#[test]
fn a_changed_row_fails_and_names_its_run() {
    let file = write("changed_row.pcsession.json", &with_changed_row().to_text());
    let (c, report, err) = replay(&file, &[]);
    assert_eq!(c, 1, "stderr: {err}");
    assert_eq!(report["pass"], false);
    let runs = report["runs"].as_array().unwrap();
    for (i, r) in runs.iter().enumerate() {
        assert_eq!(r["pass"], i != 1, "run {}: {r}", i + 1);
    }
    let bad = &runs[1];
    assert_eq!(bad["index"], 2);
    assert_eq!(bad["identical"], false);
    let diff = bad["max_rel_diff"].as_f64().unwrap();
    assert!(diff > 1e-6 && diff < 1e-3, "max_rel_diff {diff}");
    assert!(
        bad["message"].as_str().unwrap().contains("differ"),
        "{}",
        bad["message"]
    );
}

#[test]
fn a_loose_tolerance_accepts_the_change_but_not_as_identical() {
    let file = write("loose.pcsession.json", &with_changed_row().to_text());
    let (c, report, err) = replay(&file, &["--tol", "1e-2"]);
    assert_eq!(c, 0, "stderr: {err}\nreport: {report}");
    assert_eq!(report["tol"], 1e-2);
    let run = &report["runs"][1];
    assert_eq!(run["pass"], true);
    assert_eq!(run["identical"], false);
}

#[test]
fn a_changed_source_is_a_hash_mismatch() {
    let mut s = Session::parse(diagram_text()).unwrap();
    s.model.source.push_str("# edited\n");
    let file = write("source.pcsession.json", &s.to_text());
    let (c, report, err) = replay(&file, &[]);
    assert_eq!(c, 2, "report: {report}");
    assert!(err.contains("hash mismatch"), "stderr: {err}");
    assert!(report.is_null(), "a hash mismatch prints no report");
}

#[test]
fn an_unknown_version_is_a_bad_file() {
    let mut v: Value = serde_json::from_str(diagram_text()).unwrap();
    v["version"] = Value::from(2);
    let file = write("version.pcsession.json", &v.to_string());
    let (c, _, err) = replay(&file, &[]);
    assert_eq!(c, 2);
    assert!(
        err.contains("unsupported session version 2"),
        "stderr: {err}"
    );
}

#[test]
fn a_missing_file_is_a_bad_file() {
    let (c, _, err) = replay(&scratch().join("no-such-file.json"), &[]);
    assert_eq!(c, 2);
    assert!(err.contains("cannot read"), "stderr: {err}");
}

/// The committed fixture the MATLAB tests decode. `PC_UPDATE_FIXTURE=1` rewrites it. The
/// fixture must always replay, so a change of the engine that moves the numbers shows up
/// here and not in MATLAB.
#[test]
fn the_fixture_replays() {
    let path = fixture_path();
    if std::env::var_os("PC_UPDATE_FIXTURE").is_some() {
        let eng = Engine::load().unwrap();
        std::fs::write(&path, small_diagram(&eng).to_text()).unwrap();
    }
    let s =
        Session::read(&path).unwrap_or_else(|e| panic!("{e}; regenerate with PC_UPDATE_FIXTURE=1"));
    // Runs with different fields, an empty list of special points, and non-square blocks.
    assert_eq!(s.runs.len(), 3);
    assert_eq!(s.runs[0].kind, Kind::Equilibrium);
    assert!(s.runs[0].rows.is_empty() && s.runs[0].result.is_some());
    assert_eq!(s.runs[1].specials[0].rows(), 1);
    assert_eq!(s.runs[2].specials[0].rows(), 0);
    for r in &s.runs[1..] {
        let b = &r.rows[0];
        assert!(b.rows() > b.cols(), "{}x{}", b.rows(), b.cols());
    }
    let (c, report, err) = replay(&path, &[]);
    assert_eq!(c, 0, "stderr: {err}\nreport: {report}");
    assert_eq!(report["pass"], true);
}

/// The file the MATLAB writer made from the fixture, named by `PC_REENCODED_SESSION`, decodes
/// to the same bits as the fixture. SessionFormatTest.m writes it. This test is ignored in
/// a plain `cargo test`; run it with `--ignored` after the MATLAB suite.
#[test]
#[ignore = "needs the file that SessionFormatTest.m writes; set PC_REENCODED_SESSION"]
fn the_matlab_writer_matches_the_fixture() {
    let path = std::env::var("PC_REENCODED_SESSION")
        .expect("set PC_REENCODED_SESSION to the file SessionFormatTest.m wrote");
    let theirs = Session::read(Path::new(&path)).unwrap();
    let ours = Session::read(&fixture_path()).unwrap();
    assert_eq!(theirs.runs.len(), ours.runs.len());
    for (i, (a, b)) in ours.runs.iter().zip(&theirs.runs).enumerate() {
        let tag = format!("run {}", i + 1);
        assert!(a.u0.same_bits(&b.u0), "{tag}: u0");
        assert!(a.p.same_bits(&b.p), "{tag}: p");
        assert_eq!(a.rows.len(), b.rows.len(), "{tag}");
        for (x, y) in a
            .rows
            .iter()
            .zip(&b.rows)
            .chain(a.specials.iter().zip(&b.specials))
        {
            assert_eq!((x.rows(), x.cols()), (y.rows(), y.cols()), "{tag}: shape");
            assert!(x.same_bits(y), "{tag}: bits");
        }
        match (&a.result, &b.result) {
            (Some(x), Some(y)) => {
                assert!(
                    x.u.same_bits(&y.u) && x.eig_re.same_bits(&y.eig_re),
                    "{tag}"
                );
                assert!(x.eig_im.same_bits(&y.eig_im), "{tag}");
            }
            (None, None) => {}
            _ => panic!("{tag}: one run has a result and the other has none"),
        }
    }
}
