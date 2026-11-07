//! LPC, PD and NS labels along a branch of delay cycles (SPEC section 8o).
//!
//! Every model has the exact cycle `x = sin t`, `T = 2 pi`, at `tau = pi/2`, so the
//! oracles are closed forms: an extra linear block gives a Neimark-Sacker point at
//! `s = pi/2` (test 1) and a period doubling at `s = sqrt(0.34)` (test 2), and the cubic
//! model has a fold of cycles at `a = 0` (test 3). Run
//! `cargo test --test dde_cycle_labels -- --ignored --nocapture` to print the measured
//! values behind the bounds.

use pc_engine::{
    DdeCycleBranch, DdeCycleSetup, DdeCycleSpecial, DdeCycleStart, DdeFloquetOptions, DdeModel,
    EngineError, Evaluator, Options, StopReason, build_dde, dde_continue_cycle,
    dde_continue_cycle_floquet, dde_floquet_multipliers,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

const NTST: usize = 20;
const NCOL: usize = 4;

/// The NS parameter, `pi/2`, and the argument of the multiplier there, `4 pi - pi^2`.
const NS_S: f64 = PI / 2.0;
const NS_THETA: f64 = 4.0 * PI - PI * PI;
/// The PD parameter `sqrt(1/4 + g^2)` with `g = 0.3`.
const G: f64 = 0.3;

// Bounds, about 2 to 3 times the values printed by `print_tables` (SPEC section 8o).
// Three of them are wider than that because the measured value is at rounding level
// (NS_ALPHA_TOL, PD_MULT_TOL) or zero (JIT_ALPHA_TOL), and rounding differs by platform.
const NS_ALPHA_TOL: f64 = 1e-12;
const NS_THETA_TOL: f64 = 4e-9;
const PD_ALPHA_TOL: f64 = 1e-11;
const PD_MULT_TOL: f64 = 1e-12;
const PD_END_TOL: f64 = 4e-10;
const LPC_ALPHA_TOL: f64 = 2e-11;
const LPC_RHO_TOL: f64 = 1e-10;
const LPC_PERIOD_TOL: f64 = 8e-11;
const LPC_CURVE_TOL: f64 = 2e-9;
const JIT_ALPHA_TOL: f64 = 1e-10;

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

/// Test 1: the oscillator plus `y' = -s y(t - 1)`; parameters `(mu, tau, s)`.
fn ns_source() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.3, tau={tau:?}, s=0.3\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         y'=-s*delay(y,1)\n\
         done\n"
    )
}

/// Test 2: the oscillator plus the rotating block; parameters `(mu, tau, s, g)`.
fn pd_source() -> String {
    let tau = PI / 2.0;
    format!(
        "par mu=0.3, tau={tau:?}, s=0.4, g={G:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         y'=s*(-delay(x,tau)*y+x*z)-g*y\n\
         z'=s*(x*y+delay(x,tau)*z)-g*z\n\
         done\n"
    )
}

/// Test 3: the cubic model; parameters `(a, tau)`.
fn lpc_source() -> String {
    let tau = PI / 2.0;
    format!(
        "par a=0.25, tau={tau:?}\n\
         x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)\n\
         done\n"
    )
}

fn setup(params: &[f64], free: usize) -> DdeCycleSetup {
    DdeCycleSetup {
        params: params.to_vec(),
        free,
        ntst: NTST,
        ncol: NCOL,
    }
}

/// The profile with `n` values per base point; `f(c, s)` is component `c` at `s` in [0, 1].
fn profile(n: usize, f: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    let nq = NTST * NCOL;
    let mut out = Vec::with_capacity(n * (nq + 1));
    for q in 0..=nq {
        let s = q as f64 / nq as f64;
        for c in 0..n {
            out.push(f(c, s));
        }
    }
    out
}

fn start_of(profile: Vec<f64>) -> DdeCycleStart {
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

fn tight_opts(bounds: (f64, f64), direction: f64, hmax: f64) -> Options {
    Options {
        tol_x: 1e-11,
        tol_f: 1e-10,
        ..run_opts(bounds, direction, hmax)
    }
}

fn run(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    fl: Option<&DdeFloquetOptions>,
) -> Result<DdeCycleBranch, EngineError> {
    let cancel = AtomicBool::new(false);
    dde_continue_cycle_floquet(m, st, start, o, fl, &cancel)
}

fn run_ok(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    fl: Option<&DdeFloquetOptions>,
    what: &str,
) -> DdeCycleBranch {
    run(m, st, start, o, fl).unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn fl(nhist: usize, count: usize) -> DdeFloquetOptions {
    DdeFloquetOptions { nhist, count }
}

/// Test 1 setup: start `s = 0.3` upward.
struct Ns {
    m: DdeModel,
    st: DdeCycleSetup,
    start: DdeCycleStart,
    o: Options,
}

fn ns_case(which: Evaluator, tight: bool) -> Ns {
    let o = if tight {
        tight_opts((0.1, 2.0), 1.0, 0.2)
    } else {
        run_opts((0.1, 2.0), 1.0, 0.2)
    };
    Ns {
        m: dde(&ns_source(), which),
        st: setup(&[0.3, PI / 2.0, 0.3], 2),
        start: start_of(profile(
            2,
            |c, s| {
                if c == 0 { (2.0 * PI * s).sin() } else { 0.0 }
            },
        )),
        o,
    }
}

/// Test 2 setup: start `s = 0.4` upward to `hi`.
struct Pd {
    m: DdeModel,
    st: DdeCycleSetup,
    start: DdeCycleStart,
    o: Options,
}

fn pd_case(which: Evaluator, hi: f64, tight: bool) -> Pd {
    let o = if tight {
        tight_opts((0.3, hi), 1.0, 0.1)
    } else {
        run_opts((0.3, hi), 1.0, 0.1)
    };
    Pd {
        m: dde(&pd_source(), which),
        st: setup(&[0.3, PI / 2.0, 0.4, G], 2),
        start: start_of(profile(
            3,
            |c, s| {
                if c == 0 { (2.0 * PI * s).sin() } else { 0.0 }
            },
        )),
        o,
    }
}

/// Test 3 setup: start `a = 0.25` on the upper branch, downward through the fold.
struct Lpc {
    m: DdeModel,
    st: DdeCycleSetup,
    start: DdeCycleStart,
    o: Options,
}

fn lpc_case(which: Evaluator) -> Lpc {
    let r = 1.5_f64.sqrt();
    Lpc {
        m: dde(&lpc_source(), which),
        st: setup(&[0.25, PI / 2.0], 0),
        start: start_of(profile(1, |_, s| r * (2.0 * PI * s).sin())),
        o: run_opts((-0.5, 0.4), -1.0, 0.1),
    }
}

fn specials<'a>(b: &'a DdeCycleBranch, label: &str) -> Vec<&'a DdeCycleSpecial> {
    b.specials.iter().filter(|s| s.label == label).collect()
}

fn labels(b: &DdeCycleBranch) -> Vec<String> {
    b.specials.iter().map(|s| s.label.clone()).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter()
        .zip(b)
        .fold(0.0, |s: f64, (x, y)| s.max((x - y).abs()))
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

fn modulus(z: (f64, f64)) -> f64 {
    z.0.hypot(z.1)
}

/// The smallest distance from `target` to a multiplier of the point.
fn nearest(mults: &[(f64, f64)], target: (f64, f64)) -> f64 {
    mults
        .iter()
        .map(|&z| dist(z, target))
        .fold(f64::INFINITY, f64::min)
}

/// `rho^2` of test 3: the mean over the mesh nodes of `u(t)^2 + u(t - T/4)^2`, which is
/// `r^2` for the orbit `r sin t` at `tau = pi/2`.
fn rho_squared(profile: &[f64]) -> f64 {
    let nodes = NTST;
    let shift = NTST / 4;
    let at = |j: usize| profile[(j % nodes) * NCOL];
    let sum: f64 = (0..nodes)
        .map(|j| {
            let k = (j + nodes - shift) % nodes;
            at(j) * at(j) + at(k) * at(k)
        })
        .sum();
    sum / nodes as f64
}

/// Whether the two branches agree bit for bit in their points.
fn same_points(a: &DdeCycleBranch, b: &DdeCycleBranch) -> bool {
    a.points.len() == b.points.len()
        && a.points.iter().zip(&b.points).all(|(p, q)| {
            p.alpha == q.alpha
                && p.period == q.period
                && p.profile == q.profile
                && p.h == q.h
                && p.newton_iters == q.newton_iters
        })
}

fn same_special(a: &DdeCycleSpecial, b: &DdeCycleSpecial) -> bool {
    a.label == b.label
        && a.index == b.index
        && a.point.alpha == b.point.alpha
        && a.point.period == b.point.period
        && a.point.profile == b.point.profile
}

// Test 1: the NS point of the extra block, closed form.
#[test]
fn ns_point_at_pi_over_two() {
    let mut c = ns_case(Evaluator::Jit, false);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "ns run");
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    let ns = specials(&b, "NS");
    assert_eq!(ns.len(), 1, "specials: {:?}", labels(&b));
    assert!(
        (ns[0].point.alpha - NS_S).abs() < NS_ALPHA_TOL,
        "NS at {} instead of {NS_S}",
        ns[0].point.alpha
    );
    let m = &ns[0].point.multipliers;
    let target = (NS_THETA.cos(), NS_THETA.sin());
    let target_c = (target.0, -target.1);
    assert!(
        nearest(m, target) < NS_THETA_TOL,
        "no e^(+i theta) in {m:?}"
    );
    assert!(
        nearest(m, target_c) < NS_THETA_TOL,
        "no e^(-i theta) in {m:?}"
    );
    // The double real root at s = 1/e is no label.
    let e = (-1.0_f64).exp();
    assert!(
        b.specials.iter().all(|s| (s.point.alpha - e).abs() > 0.05),
        "a special near 1/e: {:?}",
        labels(&b)
    );
}

// Test 2: the PD point of the rotating block, closed form.
#[test]
fn pd_point_and_the_end_of_the_run() {
    let mut c = pd_case(Evaluator::Jit, 0.8, false);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "pd run");
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    assert_eq!(labels(&b), vec!["PD".to_string()], "specials");
    let pd = &b.specials[0];
    let s_pd = (0.25 + G * G).sqrt();
    assert!(
        (pd.point.alpha - s_pd).abs() < PD_ALPHA_TOL,
        "PD at {} instead of {s_pd}",
        pd.point.alpha
    );
    assert!(
        nearest(&pd.point.multipliers, (-1.0, 0.0)) < PD_MULT_TOL,
        "no multiplier at -1 in {:?}",
        pd.point.multipliers
    );

    let mut c = pd_case(Evaluator::Jit, 0.7, false);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "pd end");
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    let last = b.points.last().unwrap();
    assert!((last.alpha - 0.7).abs() < 1e-9, "ends at {}", last.alpha);
    let lead = last.multipliers[0];
    let want = -(2.0 * PI * (-G + (0.7_f64 * 0.7 - 0.25).sqrt())).exp();
    assert!(
        dist(lead, (want, 0.0)) < PD_END_TOL,
        "leading multiplier {lead:?}, expected {want}"
    );
    // At s = 1/2 the pair of real multipliers meets and turns complex: no label.
    assert_eq!(labels(&b), vec!["PD".to_string()], "specials");
}

// Test 3: the fold of cycles of the cubic model.
#[test]
fn lpc_fold_of_cycles() {
    let mut c = lpc_case(Evaluator::Jit);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "lpc run");
    assert_eq!(b.stop, StopReason::Bounds, "stop");
    assert_eq!(labels(&b), vec!["LPC".to_string()], "specials");
    let lpc = &b.specials[0];
    assert!(
        lpc.point.alpha.abs() < LPC_ALPHA_TOL,
        "LPC at a = {}",
        lpc.point.alpha
    );
    assert!(
        (rho_squared(&lpc.point.profile) - 1.0).abs() < LPC_RHO_TOL,
        "rho^2 = {}",
        rho_squared(&lpc.point.profile)
    );
    assert!(
        (lpc.point.period - 2.0 * PI).abs() < LPC_PERIOD_TOL,
        "T = {}",
        lpc.point.period
    );
    // The curve (rho^2 - 1)^2 = a, on the upper branch before the fold, below it after.
    for (i, p) in b.points.iter().enumerate() {
        let rho = rho_squared(&p.profile);
        assert!(
            ((rho - 1.0).powi(2) - p.alpha).abs() < LPC_CURVE_TOL,
            "point {i}: a = {}, rho^2 = {rho}",
            p.alpha
        );
        if p.alpha > 1e-3 {
            let upper = i < lpc.index;
            assert_eq!(
                rho > 1.0,
                upper,
                "point {i}: a = {}, rho^2 = {rho}",
                p.alpha
            );
        }
    }
    assert!(lpc.index > 0 && lpc.index < b.points.len());
    let last = b.points.last().unwrap();
    assert!((last.alpha - 0.4).abs() < 1e-9, "ends at {}", last.alpha);

    // With the multipliers on, the LPC special is the same one.
    let mut c = lpc_case(Evaluator::Jit);
    let f = run_ok(
        &mut c.m,
        &c.st,
        &c.start,
        &c.o,
        Some(&fl(12, 4)),
        "lpc floquet",
    );
    let lf = specials(&f, "LPC");
    assert_eq!(lf.len(), 1, "specials: {:?}", labels(&f));
    assert!(same_special(lf[0], lpc), "the LPC special moved");
}

// Test 4: the labels do not change the path.
#[test]
fn points_do_not_depend_on_the_multipliers() {
    let mut c = ns_case(Evaluator::Jit, false);
    let off = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "ns off");
    let on = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "ns on");
    assert!(same_points(&off, &on), "test 1 points differ");

    let mut c = pd_case(Evaluator::Jit, 0.8, false);
    let off = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "pd off");
    let on = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "pd on");
    assert!(same_points(&off, &on), "test 2 points differ");

    let mut c = lpc_case(Evaluator::Jit);
    let off = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "lpc off");
    let on = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(12, 4)), "lpc on");
    assert!(same_points(&off, &on), "test 3 points differ");
    let a = specials(&off, "LPC");
    let b = specials(&on, "LPC");
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert!(same_special(a[0], b[0]), "the LPC specials differ");
}

// Test 5: the multipliers of a point are those of `floquet_multipliers`.
#[test]
fn point_multipliers_are_floquet_multipliers() {
    let mut c = pd_case(Evaluator::Jit, 0.8, false);
    let opts = fl(16, 6);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&opts), "pd run");
    assert!(b.points.len() > 3);
    let cancel = AtomicBool::new(false);
    for (i, p) in b.points.iter().enumerate() {
        assert_eq!(p.multipliers.len(), 6, "point {i}");
        assert!(
            p.multipliers
                .windows(2)
                .all(|w| modulus(w[0]) >= modulus(w[1])),
            "point {i} not sorted by modulus: {:?}",
            p.multipliers
        );
        let mut params = c.st.params.clone();
        params[c.st.free] = p.alpha;
        let st = DdeCycleSetup {
            params,
            ..c.st.clone()
        };
        let cycle = DdeCycleStart {
            profile: p.profile.clone(),
            period: p.period,
        };
        let f = dde_floquet_multipliers(&mut c.m, &st, &cycle, &opts, &cancel).unwrap();
        assert_eq!(p.multipliers, f.multipliers, "point {i}");
        assert_eq!(p.trivial_error, f.trivial_error, "point {i}");
    }
    let mut c = pd_case(Evaluator::Jit, 0.8, false);
    let off = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "pd off");
    assert!(
        off.points
            .iter()
            .all(|p| p.multipliers.is_empty() && p.trivial_error.is_nan()),
        "points without floquet carry multipliers"
    );
    assert!(off.specials.iter().all(|s| s.point.multipliers.is_empty()));
}

// Test 6: errors.
#[test]
fn errors_are_reported_before_the_run() {
    let mut c = ns_case(Evaluator::Jit, false);
    let n_d = 2 * 17;
    for (what, f) in [
        ("nhist 0", fl(0, 1)),
        ("count 0", fl(16, 0)),
        ("count above d", fl(16, n_d + 1)),
        ("stage above the limit", fl(130, 1)),
    ] {
        match run(&mut c.m, &c.st, &c.start, &c.o, Some(&f)) {
            Err(EngineError::BadProblem(msg)) => {
                assert!(msg.starts_with("floquet: "), "{what}: message {msg}");
            }
            other => panic!("{what}: expected BadProblem, got {other:?}"),
        }
    }
    // `build_dde` refuses a model without a delay call, so the test empties the list of a
    // built model.
    let mut nodelay = ns_case(Evaluator::Jit, false);
    nodelay.m.delays.clear();
    match run(
        &mut nodelay.m,
        &nodelay.st,
        &nodelay.start,
        &nodelay.o,
        Some(&fl(4, 1)),
    ) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.starts_with("floquet: "), "no delay: message {msg}");
        }
        other => panic!("no delay: expected BadProblem, got {other:?}"),
    }
    // A set cancel flag.
    let cancel = AtomicBool::new(true);
    let r = dde_continue_cycle_floquet(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), &cancel);
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
    cancel.store(false, Ordering::Relaxed);
    // The step-control error comes before the floquet error.
    let bad_step = Options {
        hmax: -1.0,
        ..c.o.clone()
    };
    match run(&mut c.m, &c.st, &c.start, &bad_step, Some(&fl(0, 0))) {
        Err(EngineError::BadProblem(msg)) => {
            assert!(msg.contains("step control"), "message: {msg}");
        }
        other => panic!("step control: expected BadProblem, got {other:?}"),
    }
    // `continue_cycle` is the `None` wrapper.
    let cancel = AtomicBool::new(false);
    let a = dde_continue_cycle(&mut c.m, &c.st, &c.start, &c.o, &cancel).unwrap();
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "none");
    // NaN is not equal to itself, so the comparison goes field by field.
    assert!(same_points(&a, &b), "points");
    assert_eq!(a.specials.len(), b.specials.len());
    assert!(
        a.specials
            .iter()
            .zip(&b.specials)
            .all(|(p, q)| same_special(p, q))
    );
    assert_eq!(a.stop, b.stop);
    assert_eq!(a.warnings, b.warnings);
}

// Test 7: the interpreter and the compiled evaluator find the same points.
#[test]
fn jit_and_fallback_agree() {
    let mut a = ns_case(Evaluator::Jit, true);
    let mut b = ns_case(Evaluator::Fallback, true);
    let f = fl(16, 6);
    let ra = run_ok(&mut a.m, &a.st, &a.start, &a.o, Some(&f), "ns jit");
    let rb = run_ok(&mut b.m, &b.st, &b.start, &b.o, Some(&f), "ns fallback");
    compare_specials(&ra, &rb, "test 1");

    let mut a = pd_case(Evaluator::Jit, 0.8, true);
    let mut b = pd_case(Evaluator::Fallback, 0.8, true);
    let ra = run_ok(&mut a.m, &a.st, &a.start, &a.o, Some(&f), "pd jit");
    let rb = run_ok(&mut b.m, &b.st, &b.start, &b.o, Some(&f), "pd fallback");
    compare_specials(&ra, &rb, "test 2");
}

fn compare_specials(a: &DdeCycleBranch, b: &DdeCycleBranch, what: &str) {
    assert_eq!(a.specials.len(), b.specials.len(), "{what}: count");
    for (x, y) in a.specials.iter().zip(&b.specials) {
        assert_eq!(x.label, y.label, "{what}: label");
        assert_eq!(x.index, y.index, "{what}: index");
        assert!(
            (x.point.alpha - y.point.alpha).abs() < JIT_ALPHA_TOL,
            "{what}: alpha {} against {}",
            x.point.alpha,
            y.point.alpha
        );
        assert!(
            max_diff(&x.point.profile, &y.point.profile) < 1e-6,
            "{what}: profile"
        );
    }
}

/// Prints the values behind the bounds of section 8o.
#[test]
#[ignore = "prints the measured tables of SPEC section 8o"]
fn print_tables() {
    let mut c = ns_case(Evaluator::Jit, false);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "ns");
    for s in &b.specials {
        let m = &s.point.multipliers;
        println!(
            "NS run: {} at s = {:.12} (error {:.3e}), index {}, points {}, steps stop {:?}",
            s.label,
            s.point.alpha,
            s.point.alpha - NS_S,
            s.index,
            b.points.len(),
            b.stop
        );
        println!(
            "  nearest e^(+i theta) {:.3e}, e^(-i theta) {:.3e}",
            nearest(m, (NS_THETA.cos(), NS_THETA.sin())),
            nearest(m, (NS_THETA.cos(), -NS_THETA.sin()))
        );
    }
    let mut c = pd_case(Evaluator::Jit, 0.8, false);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "pd");
    let s_pd = (0.25 + G * G).sqrt();
    for s in &b.specials {
        println!(
            "PD run: {} at s = {:.12} (error {:.3e}), nearest -1 {:.3e}, points {}",
            s.label,
            s.point.alpha,
            s.point.alpha - s_pd,
            nearest(&s.point.multipliers, (-1.0, 0.0)),
            b.points.len()
        );
    }
    let mut c = pd_case(Evaluator::Jit, 0.7, false);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, Some(&fl(16, 6)), "pd end");
    let last = b.points.last().unwrap();
    let want = -(2.0 * PI * (-G + (0.49_f64 - 0.25).sqrt())).exp();
    println!(
        "PD end: s = {}, leading {:?}, expected {want}, error {:.3e}",
        last.alpha,
        last.multipliers[0],
        dist(last.multipliers[0], (want, 0.0))
    );
    let mut c = lpc_case(Evaluator::Jit);
    let b = run_ok(&mut c.m, &c.st, &c.start, &c.o, None, "lpc");
    let mut worst: f64 = 0.0;
    for p in &b.points {
        let rho = rho_squared(&p.profile);
        worst = worst.max(((rho - 1.0).powi(2) - p.alpha).abs());
    }
    for s in &b.specials {
        println!(
            "LPC run: {} at a = {:.3e}, rho^2 - 1 = {:.3e}, T - 2 pi = {:.3e}, index {}, \
             points {}, worst curve error {worst:.3e}",
            s.label,
            s.point.alpha,
            rho_squared(&s.point.profile) - 1.0,
            s.point.period - 2.0 * PI,
            s.index,
            b.points.len()
        );
    }
    let mut a = ns_case(Evaluator::Jit, true);
    let mut f = ns_case(Evaluator::Fallback, true);
    let ra = run_ok(&mut a.m, &a.st, &a.start, &a.o, Some(&fl(16, 6)), "ns jit");
    let rb = run_ok(&mut f.m, &f.st, &f.start, &f.o, Some(&fl(16, 6)), "ns fb");
    for (x, y) in ra.specials.iter().zip(&rb.specials) {
        println!(
            "Jit vs Fallback test 1: {} alpha diff {:.3e}",
            x.label,
            (x.point.alpha - y.point.alpha).abs()
        );
    }
    let mut a = pd_case(Evaluator::Jit, 0.8, true);
    let mut f = pd_case(Evaluator::Fallback, 0.8, true);
    let ra = run_ok(&mut a.m, &a.st, &a.start, &a.o, Some(&fl(16, 6)), "pd jit");
    let rb = run_ok(&mut f.m, &f.st, &f.start, &f.o, Some(&fl(16, 6)), "pd fb");
    for (x, y) in ra.specials.iter().zip(&rb.specials) {
        println!(
            "Jit vs Fallback test 2: {} alpha diff {:.3e}",
            x.label,
            (x.point.alpha - y.point.alpha).abs()
        );
    }
}
