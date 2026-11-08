//! BPC labels along a branch of delay cycles (SPEC section 8q).
//!
//! The oracle ZBLK is the oscillator `x = sin t`, `T = 2 pi`, at `tau = pi/2`, plus the
//! decoupled block `z' = (mu - m0) z - z^3`, whose variational equation `zeta' = (mu - m0) zeta`
//! has a singular discretization exactly at `mu = m0`. Run
//! `cargo test --test dde_cycle_bpc -- --ignored --nocapture` to print the measured values
//! behind the bounds.

use pc_engine::{
    DdeCycleBranch, DdeCyclePoint, DdeCycleSetup, DdeCycleSpecial, DdeCycleStart,
    DdeFloquetOptions, DdeMeshAdapt, DdeModel, EngineError, Evaluator, Options, StopReason,
    build_dde, dde_continue_cycle, dde_continue_cycle_adapt, dde_continue_cycle_bpc,
    dde_continue_cycle_floquet,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

const NTST: usize = 20;
const NCOL: usize = 4;
const M0: f64 = 0.3;

// Bounds, 4 to 50 times the values printed by `print_tables` (SPEC section 8q). The ones
// at rounding level (ALPHA_TOL, SEQ_BPC_TOL) are 15 to 50 times larger so that another
// platform's libm does not trip them.
const ALPHA_TOL: f64 = 1e-13;
const ADAPT_ALPHA_TOL: f64 = 3e-10;
const ADAPT_PERIOD_TOL: f64 = 5e-8;
const SEQ_BPC_TOL: f64 = 1e-13;
const SEQ_LPC_TOL: f64 = 2e-11;
const JIT_ALPHA_TOL: f64 = 1e-9;

fn dde(src: &str, which: Evaluator) -> DdeModel {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    match build_dde(&l.expanded, &l.info, which) {
        Ok(d) => d,
        Err(e) => panic!("build_dde failed: {e}"),
    }
}

/// ZBLK, with the `z` equation given as the right-hand side of `z'`.
fn zblk_with(zrhs: &str) -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.1, tau={tau:?}, m0={M0:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         z'={zrhs}\n\
         done\n"
    )
}

fn zblk() -> String {
    zblk_with("(mu-m0)*z-z^3")
}

/// ZBLK with a second identical block `w`, a double branch point (test 8).
fn double_block() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.1, tau={tau:?}, m0={M0:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         z'=(mu-m0)*z-z^3\n\
         w'=(mu-m0)*w-w^3\n\
         done\n"
    )
}

/// The cubic model of Section 8o, test 3, with the `z` block of test 5.
fn cubic_z() -> String {
    let tau = PI / 2.0;
    format!(
        "par a=0.25, tau={tau:?}, m0=0.2\n\
         x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)\n\
         z'=(a-m0)*z-z^3\n\
         done\n"
    )
}

fn cubic() -> String {
    let tau = PI / 2.0;
    format!(
        "par a=0.25, tau={tau:?}\n\
         x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)\n\
         done\n"
    )
}

fn osc() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.3, tau={tau:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         done\n"
    )
}

fn ns_model() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.3, tau={tau:?}, s=0.3\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         y'=-s*delay(y,1)\n\
         done\n"
    )
}

fn setup(params: &[f64], free: usize, ntst: usize) -> DdeCycleSetup {
    DdeCycleSetup {
        params: params.to_vec(),
        free,
        ntst,
        ncol: NCOL,
    }
}

/// The profile with `n` values per base point on the uniform mesh: component 0 is
/// `amp sin(2 pi s)`, the others are 0.
fn start_sine(ntst: usize, n: usize, amp: f64) -> DdeCycleStart {
    let nq = ntst * NCOL;
    let mut profile = vec![0.0; n * (nq + 1)];
    for q in 0..=nq {
        let s = q as f64 / nq as f64;
        profile[q * n] = amp * (2.0 * PI * s).sin();
    }
    DdeCycleStart {
        profile,
        period: 2.0 * PI,
    }
}

fn run_opts(bounds: (f64, f64), direction: f64, hmax: f64) -> Options {
    Options {
        bounds,
        direction,
        hmax,
        max_steps: 400,
        max_newton: 10,
        ..Options::default()
    }
}

fn fl(nhist: usize, count: usize) -> DdeFloquetOptions {
    DdeFloquetOptions { nhist, count }
}

fn bpc(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    f: Option<&DdeFloquetOptions>,
    a: Option<&DdeMeshAdapt>,
) -> Result<DdeCycleBranch, EngineError> {
    let cancel = AtomicBool::new(false);
    dde_continue_cycle_bpc(m, st, start, o, f, a, &cancel)
}

fn bpc_ok(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    f: Option<&DdeFloquetOptions>,
    a: Option<&DdeMeshAdapt>,
    what: &str,
) -> DdeCycleBranch {
    bpc(m, st, start, o, f, a).unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn specials<'a>(b: &'a DdeCycleBranch, label: &str) -> Vec<&'a DdeCycleSpecial> {
    b.specials.iter().filter(|s| s.label == label).collect()
}

fn labels(b: &DdeCycleBranch) -> Vec<String> {
    b.specials.iter().map(|s| s.label.clone()).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pair_bits(v: &[(f64, f64)]) -> Vec<(u64, u64)> {
    v.iter().map(|z| (z.0.to_bits(), z.1.to_bits())).collect()
}

fn same_point(p: &DdeCyclePoint, q: &DdeCyclePoint) -> bool {
    p.alpha.to_bits() == q.alpha.to_bits()
        && p.period.to_bits() == q.period.to_bits()
        && bits(&p.profile) == bits(&q.profile)
        && bits(&p.mesh) == bits(&q.mesh)
        && pair_bits(&p.multipliers) == pair_bits(&q.multipliers)
}

fn same_special(a: &DdeCycleSpecial, b: &DdeCycleSpecial) -> bool {
    a.label == b.label && a.index == b.index && same_point(&a.point, &b.point)
}

/// One `BPC` at `m0` to `tol`, and nothing else.
fn only_bpc(b: &DdeCycleBranch, m0: f64, tol: f64, what: &str) {
    assert_eq!(b.stop, StopReason::Bounds, "{what}: stop {:?}", b.stop);
    assert_eq!(labels(b), vec!["BPC".to_string()], "{what}: specials");
    let a = b.specials[0].point.alpha;
    assert!(
        (a - m0).abs() <= tol,
        "{what}: BPC at {a}, error {:e}",
        a - m0
    );
}

/// A ZBLK run from `mu0` (the `mu` of the start).
struct Zblk {
    m: DdeModel,
    st: DdeCycleSetup,
    start: DdeCycleStart,
}

fn zblk_case(src: &str, mu0: f64, ntst: usize, which: Evaluator) -> Zblk {
    Zblk {
        m: dde(src, which),
        st: setup(&[mu0, PI / 2.0, M0], 0, ntst),
        start: start_sine(ntst, 2, 1.0),
    }
}

// Test 1: the exact branch point, both directions, and no BPC without the opt-in.
#[test]
fn the_exact_branch_point() {
    for (mu0, dir) in [(0.1, 1.0), (0.6, -1.0)] {
        let mut c = zblk_case(&zblk(), mu0, NTST, Evaluator::Jit);
        let o = run_opts((0.1, 0.6), dir, 0.1);
        let b = bpc_ok(&mut c.m, &c.st, &c.start, &o, None, None, "zblk");
        only_bpc(&b, M0, ALPHA_TOL, &format!("direction {dir}"));
        let cancel = AtomicBool::new(false);
        let f = dde_continue_cycle_floquet(&mut c.m, &c.st, &c.start, &o, None, &cancel).unwrap();
        assert!(specials(&f, "BPC").is_empty(), "floquet run labels BPC");
        let a = dde_continue_cycle_adapt(
            &mut c.m,
            &c.st,
            &c.start,
            &o,
            None,
            &DdeMeshAdapt { every: 3 },
            &cancel,
        )
        .unwrap();
        assert!(specials(&a, "BPC").is_empty(), "adapt run labels BPC");
        let p = dde_continue_cycle(&mut c.m, &c.st, &c.start, &o, &cancel).unwrap();
        assert!(specials(&p, "BPC").is_empty(), "plain run labels BPC");
    }
}

// Test 2: step size and an off-grid start.
#[test]
fn step_size_and_an_off_grid_start() {
    for hmax in [0.083, 0.31] {
        let mut c = zblk_case(&zblk(), 0.137, NTST, Evaluator::Jit);
        let o = run_opts((0.1, 0.6), 1.0, hmax);
        let b = bpc_ok(&mut c.m, &c.st, &c.start, &o, None, None, "off grid");
        only_bpc(&b, M0, ALPHA_TOL, &format!("hmax {hmax}"));
    }
}

// Test 3: Floquet on, and a fine mesh.
#[test]
fn floquet_and_a_fine_mesh() {
    let mut c = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let b = bpc_ok(
        &mut c.m,
        &c.st,
        &c.start,
        &o,
        Some(&fl(20, 4)),
        None,
        "floquet",
    );
    only_bpc(&b, M0, ALPHA_TOL, "floquet");
    let mut c = zblk_case(&zblk(), 0.1, 100, Evaluator::Jit);
    let b = bpc_ok(&mut c.m, &c.st, &c.start, &o, None, None, "n = 100");
    only_bpc(&b, M0, ALPHA_TOL, "N = 100");
}

// Test 4: a nonlinear coefficient in the block.
#[test]
fn a_nonlinear_coefficient() {
    let src = zblk_with("(exp(8*mu)-exp(8*m0))*z-z^3");
    let mut c = zblk_case(&src, 0.137, NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.31);
    let b = bpc_ok(&mut c.m, &c.st, &c.start, &o, None, None, "nonlinear");
    only_bpc(&b, M0, ALPHA_TOL, "nonlinear block");
}

// Test 5: branch point, fold, branch point.
#[test]
fn branch_point_fold_branch_point() {
    let mut m = dde(&cubic_z(), Evaluator::Jit);
    let st = setup(&[0.25, PI / 2.0, 0.2], 0, NTST);
    let start = start_sine(NTST, 2, 1.5_f64.sqrt());
    let o = run_opts((-0.5, 0.4), -1.0, 0.1);
    let b = bpc_ok(
        &mut m,
        &st,
        &start,
        &o,
        Some(&fl(20, 4)),
        None,
        "bpc lpc bpc",
    );
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    assert_eq!(
        labels(&b),
        vec!["BPC".to_string(), "LPC".to_string(), "BPC".to_string()],
        "specials"
    );
    for s in specials(&b, "BPC") {
        assert!(
            (s.point.alpha - 0.2).abs() <= SEQ_BPC_TOL,
            "BPC at {}",
            s.point.alpha
        );
    }
    let lpc = specials(&b, "LPC");
    assert!(
        lpc[0].point.alpha.abs() <= SEQ_LPC_TOL,
        "LPC at {}",
        lpc[0].point.alpha
    );
}

// Test 6: no false label.
#[test]
fn no_false_label() {
    let f = fl(20, 4);
    // ZBLK with m0 = 1.0: no branch point in the range.
    let mut m = dde(&zblk(), Evaluator::Jit);
    let st = setup(&[0.1, PI / 2.0, 1.0], 0, NTST);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let b = bpc_ok(
        &mut m,
        &st,
        &start_sine(NTST, 2, 1.0),
        &o,
        Some(&f),
        None,
        "m0 = 1",
    );
    assert_eq!(b.stop, StopReason::Bounds, "m0 = 1: stop");
    assert!(specials(&b, "BPC").is_empty(), "m0 = 1: {:?}", labels(&b));

    // The oscillator alone on the mu-line.
    let mut m = dde(&osc(), Evaluator::Jit);
    let st = setup(&[0.3, PI / 2.0], 0, NTST);
    let o = run_opts((0.1, 0.9), 1.0, 0.2);
    let b = bpc_ok(
        &mut m,
        &st,
        &start_sine(NTST, 1, 1.0),
        &o,
        Some(&f),
        None,
        "osc",
    );
    assert_eq!(b.stop, StopReason::Bounds, "osc: stop");
    assert!(specials(&b, "BPC").is_empty(), "osc: {:?}", labels(&b));

    // The cubic model without z: the fold is an LPC and no BPC.
    let mut m = dde(&cubic(), Evaluator::Jit);
    let st = setup(&[0.25, PI / 2.0], 0, NTST);
    let o = run_opts((-0.5, 0.4), -1.0, 0.1);
    let b = bpc_ok(
        &mut m,
        &st,
        &start_sine(NTST, 1, 1.5_f64.sqrt()),
        &o,
        Some(&f),
        None,
        "cubic",
    );
    assert_eq!(labels(&b), vec!["LPC".to_string()], "cubic: specials");

    // The NS model: the NS stays labeled, and there is no BPC.
    let mut m = dde(&ns_model(), Evaluator::Jit);
    let st = setup(&[0.3, PI / 2.0, 0.3], 2, NTST);
    let o = run_opts((0.1, 2.0), 1.0, 0.2);
    let b = bpc_ok(
        &mut m,
        &st,
        &start_sine(NTST, 2, 1.0),
        &o,
        Some(&fl(16, 6)),
        None,
        "ns",
    );
    assert_eq!(labels(&b), vec!["NS".to_string()], "ns: specials");
}

// Test 7: with mesh adaptation.
#[test]
fn mesh_adaptation() {
    let mut c = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let b = bpc_ok(
        &mut c.m,
        &c.st,
        &c.start,
        &o,
        None,
        Some(&DdeMeshAdapt { every: 3 }),
        "adapt",
    );
    only_bpc(&b, M0, ADAPT_ALPHA_TOL, "adapt");
    let p = &b.specials[0].point;
    assert!(
        (p.period - 2.0 * PI).abs() <= ADAPT_PERIOD_TOL,
        "T - 2 pi = {:e}",
        p.period - 2.0 * PI
    );
    for (i, q) in b.points.iter().enumerate() {
        assert_eq!(q.mesh.len(), NTST + 1, "point {i}: mesh length");
        assert_eq!(q.mesh[0], 0.0, "point {i}: first");
        assert_eq!(q.mesh[NTST], 1.0, "point {i}: last");
        assert!(q.mesh.windows(2).all(|w| w[0] < w[1]), "point {i}: order");
    }
}

// Test 8: a double branch point is missed (limitation).
#[test]
fn a_double_branch_point_is_missed() {
    let tau = PI / 2.0;
    let mut m = dde(&double_block(), Evaluator::Jit);
    let st = setup(&[0.137, tau, M0], 0, NTST);
    let o = run_opts((0.1, 0.6), 1.0, 0.083);
    let b = bpc_ok(
        &mut m,
        &st,
        &start_sine(NTST, 3, 1.0),
        &o,
        None,
        None,
        "double",
    );
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    assert!(
        specials(&b, "BPC").is_empty(),
        "a double branch point was labeled: {:?}",
        labels(&b)
    );
}

fn same_run(a: &DdeCycleBranch, b: &DdeCycleBranch, what: &str) {
    assert_eq!(a.points.len(), b.points.len(), "{what}: points");
    assert_eq!(a.stop, b.stop, "{what}: stop");
    for (i, (p, q)) in a.points.iter().zip(&b.points).enumerate() {
        assert!(same_point(p, q), "{what}: point {i} differs");
    }
    assert_eq!(a.specials.len(), b.specials.len(), "{what}: specials");
    for (i, (p, q)) in a.specials.iter().zip(&b.specials).enumerate() {
        assert!(same_special(p, q), "{what}: special {i} differs");
    }
}

// Test 9: where there is no BPC the run is the run without the test, bit for bit.
#[test]
fn the_points_do_not_change() {
    let cancel = AtomicBool::new(false);
    let tau = PI / 2.0;
    // The mu-line of Section 8n, both directions.
    for (dir, mu0) in [(1.0, 0.3), (-1.0, 0.3)] {
        let mut m = dde(&osc(), Evaluator::Jit);
        let st = setup(&[mu0, tau], 0, NTST);
        let start = start_sine(NTST, 1, 1.0);
        let o = run_opts((0.1, 0.9), dir, 0.2);
        let a = dde_continue_cycle_floquet(&mut m, &st, &start, &o, None, &cancel).unwrap();
        let b = bpc_ok(&mut m, &st, &start, &o, None, None, "mu-line");
        same_run(&a, &b, &format!("mu-line {dir}"));
    }
    // The runs of Section 8o, tests 1 to 3.
    let f = fl(16, 6);
    let mut m = dde(&ns_model(), Evaluator::Jit);
    let st = setup(&[0.3, tau, 0.3], 2, NTST);
    let start = start_sine(NTST, 2, 1.0);
    let o = run_opts((0.1, 2.0), 1.0, 0.2);
    let a = dde_continue_cycle_floquet(&mut m, &st, &start, &o, Some(&f), &cancel).unwrap();
    let b = bpc_ok(&mut m, &st, &start, &o, Some(&f), None, "ns");
    same_run(&a, &b, "section 8o test 1");

    let pd_src = format!(
        "par mu=0.3, tau={tau:?}, s=0.4, g=0.3\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         y'=s*(-delay(x,tau)*y+x*z)-g*y\n\
         z'=s*(x*y+delay(x,tau)*z)-g*z\n\
         done\n"
    );
    let mut m = dde(&pd_src, Evaluator::Jit);
    let st = setup(&[0.3, tau, 0.4, 0.3], 2, NTST);
    let start = start_sine(NTST, 3, 1.0);
    let o = run_opts((0.3, 0.8), 1.0, 0.1);
    let a = dde_continue_cycle_floquet(&mut m, &st, &start, &o, Some(&f), &cancel).unwrap();
    let b = bpc_ok(&mut m, &st, &start, &o, Some(&f), None, "pd");
    same_run(&a, &b, "section 8o test 2");

    let mut m = dde(&cubic(), Evaluator::Jit);
    let st = setup(&[0.25, tau], 0, NTST);
    let start = start_sine(NTST, 1, 1.5_f64.sqrt());
    let o = run_opts((-0.5, 0.4), -1.0, 0.1);
    let a = dde_continue_cycle_floquet(&mut m, &st, &start, &o, None, &cancel).unwrap();
    let b = bpc_ok(&mut m, &st, &start, &o, None, None, "cubic");
    same_run(&a, &b, "section 8o test 3");

    // With adapt, the mu-line of Section 8p, test 3.
    let mut m = dde(&osc(), Evaluator::Jit);
    let st = setup(&[0.3, tau], 0, NTST);
    let start = start_sine(NTST, 1, 1.0);
    let o = run_opts((0.1, 0.9), 1.0, 0.2);
    let ad = DdeMeshAdapt { every: 1 };
    let a = dde_continue_cycle_adapt(&mut m, &st, &start, &o, None, &ad, &cancel).unwrap();
    let b = bpc_ok(&mut m, &st, &start, &o, None, Some(&ad), "adapt");
    same_run(&a, &b, "section 8p test 3");
}

// Test 11: errors, in the order of Sections 8o and 8p.
#[test]
fn errors_are_reported_before_the_run() {
    let mut c = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let msg_of = |r: Result<DdeCycleBranch, EngineError>, what: &str| match r {
        Err(EngineError::BadProblem(m)) => m,
        other => panic!("{what}: expected BadProblem, got {other:?}"),
    };
    let m = msg_of(
        bpc(
            &mut c.m,
            &c.st,
            &c.start,
            &o,
            None,
            Some(&DdeMeshAdapt { every: 0 }),
        ),
        "every 0",
    );
    assert!(m.starts_with("adapt: "), "every 0: {m}");
    let one = setup(&[0.1, PI / 2.0, M0], 0, 1);
    let start1 = start_sine(1, 2, 1.0);
    let m = msg_of(
        bpc(
            &mut c.m,
            &one,
            &start1,
            &o,
            None,
            Some(&DdeMeshAdapt { every: 1 }),
        ),
        "ntst 1",
    );
    assert!(m.starts_with("adapt: "), "ntst 1: {m}");
    let m = msg_of(
        bpc(&mut c.m, &c.st, &c.start, &o, Some(&fl(0, 1)), None),
        "floquet",
    );
    assert!(m.starts_with("floquet: "), "floquet: {m}");
    let bad_step = Options {
        hmax: -1.0,
        ..o.clone()
    };
    let m = msg_of(
        bpc(
            &mut c.m,
            &c.st,
            &c.start,
            &bad_step,
            Some(&fl(0, 1)),
            Some(&DdeMeshAdapt { every: 0 }),
        ),
        "step control",
    );
    assert!(m.contains("step control"), "step control: {m}");
    let set = AtomicBool::new(true);
    let r = dde_continue_cycle_bpc(&mut c.m, &c.st, &c.start, &o, None, None, &set);
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
    set.store(false, Ordering::Relaxed);
}

// Test 12: the two evaluators.
#[test]
fn jit_and_fallback_agree() {
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let mut j = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    let mut f = zblk_case(&zblk(), 0.1, NTST, Evaluator::Fallback);
    let a = bpc_ok(&mut j.m, &j.st, &j.start, &o, None, None, "jit");
    let b = bpc_ok(&mut f.m, &f.st, &f.start, &o, None, None, "fallback");
    assert_eq!(a.points.len(), b.points.len(), "points");
    assert_eq!(a.specials.len(), b.specials.len(), "specials");
    only_bpc(&b, M0, ALPHA_TOL, "fallback");
    let (x, y) = (a.specials[0].point.alpha, b.specials[0].point.alpha);
    assert!((x - y).abs() <= JIT_ALPHA_TOL, "jit {x}, fallback {y}");
}

/// Prints the values behind the bounds (SPEC section 8q, "Measured values").
#[test]
#[ignore]
fn print_tables() {
    let row = |what: &str, b: &DdeCycleBranch| {
        for s in &b.specials {
            println!(
                "{what}: {} at point {} alpha = {:.15}, alpha - m0 = {:.3e}, T - 2 pi = {:.3e}",
                s.label,
                s.index,
                s.point.alpha,
                s.point.alpha - M0,
                s.point.period - 2.0 * PI
            );
        }
    };
    for (mu0, dir) in [(0.1, 1.0), (0.6, -1.0)] {
        let mut c = zblk_case(&zblk(), mu0, NTST, Evaluator::Jit);
        let o = run_opts((0.1, 0.6), dir, 0.1);
        row(
            &format!("test 1 dir {dir}"),
            &bpc(&mut c.m, &c.st, &c.start, &o, None, None).unwrap(),
        );
    }
    for hmax in [0.083, 0.31] {
        let mut c = zblk_case(&zblk(), 0.137, NTST, Evaluator::Jit);
        let o = run_opts((0.1, 0.6), 1.0, hmax);
        row(
            &format!("test 2 hmax {hmax}"),
            &bpc(&mut c.m, &c.st, &c.start, &o, None, None).unwrap(),
        );
    }
    let mut c = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    row(
        "test 3 floquet",
        &bpc(&mut c.m, &c.st, &c.start, &o, Some(&fl(20, 4)), None).unwrap(),
    );
    let mut c = zblk_case(&zblk(), 0.1, 100, Evaluator::Jit);
    row(
        "test 3 N = 100",
        &bpc(&mut c.m, &c.st, &c.start, &o, None, None).unwrap(),
    );
    let src = zblk_with("(exp(8*mu)-exp(8*m0))*z-z^3");
    let mut c = zblk_case(&src, 0.137, NTST, Evaluator::Jit);
    let o31 = run_opts((0.1, 0.6), 1.0, 0.31);
    row(
        "test 4",
        &bpc(&mut c.m, &c.st, &c.start, &o31, None, None).unwrap(),
    );
    let mut m = dde(&cubic_z(), Evaluator::Jit);
    let st = setup(&[0.25, PI / 2.0, 0.2], 0, NTST);
    let o5 = run_opts((-0.5, 0.4), -1.0, 0.1);
    let b = bpc(
        &mut m,
        &st,
        &start_sine(NTST, 2, 1.5_f64.sqrt()),
        &o5,
        Some(&fl(20, 4)),
        None,
    )
    .unwrap();
    for s in &b.specials {
        println!(
            "test 5: {} at point {} a = {:.15}, a - 0.2 = {:.3e}",
            s.label,
            s.index,
            s.point.alpha,
            s.point.alpha - 0.2
        );
    }
    let mut c = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    row(
        "test 7 adapt",
        &bpc(
            &mut c.m,
            &c.st,
            &c.start,
            &o,
            None,
            Some(&DdeMeshAdapt { every: 3 }),
        )
        .unwrap(),
    );
}
