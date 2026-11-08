//! Branch switching at a BPC of a delay cycle (SPEC section 8r).
//!
//! The oracle is ZBLK of section 8q: the oscillator `x = sin t`, `T = 2 pi`, at `tau = pi/2`,
//! plus the decoupled block `z' = (mu - m0) z - z^3`. The other branch `z^2 = mu - m0` lies on
//! the discretized problem exactly at every mesh point. Run
//! `cargo test --test dde_cycle_switch -- --ignored --nocapture` to print the measured values
//! behind the bounds.

use pc_engine::{
    DdeCycleBranch, DdeCyclePoint, DdeCycleSetup, DdeCycleStart, DdeCycleSwitch, DdeFloquetOptions,
    DdeMeshAdapt, DdeModel, EngineError, Evaluator, Options, StopReason, build_dde,
    dde_continue_cycle_bpc, dde_continue_cycle_switch, dde_cycle_jacobian, dde_switch_cycle_bpc,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

const NTST: usize = 20;
const NCOL: usize = 4;
const M0: f64 = 0.3;

// Bounds of SPEC section 8r. They were set before the code existed; the measured values and
// the revisions are in the section after the run of `print_tables`.
const ALPHA_TOL: f64 = 1e-12;
const NORM_TOL: f64 = 1e-12;
const SMALL_ENTRY: f64 = 1e-6;
const Z_ENTRY_TOL: f64 = 1e-6;
const JT_TOL: f64 = 1e-8;
const Z_FLAT_TOL: f64 = 1e-9;
const FAMILY_TOL: f64 = 1e-9;
const X_TOL: f64 = 1e-7;
const T_TOL: f64 = 1e-10;
const END_TOL: f64 = 1e-12;
const TANGENT_AGREE: f64 = 1e-6;
const NONLINEAR_TOL: f64 = 1e-8;
const FOLD_TOL: f64 = 1e-8;

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

/// The cubic model of section 8o, test 3, with the `z` block of section 8q, test 5.
fn cubic_z() -> String {
    let tau = PI / 2.0;
    format!(
        "par a=0.25, tau={tau:?}, m0=0.2\n\
         x'=-delay(x,tau)+x*(a-(x^2+delay(x,tau)^2-1)^2)\n\
         z'=(a-m0)*z-z^3\n\
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

fn bpc_ok(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    start: &DdeCycleStart,
    o: &Options,
    f: Option<&DdeFloquetOptions>,
    a: Option<&DdeMeshAdapt>,
    what: &str,
) -> DdeCycleBranch {
    let cancel = AtomicBool::new(false);
    dde_continue_cycle_bpc(m, st, start, o, f, a, &cancel).unwrap_or_else(|e| panic!("{what}: {e}"))
}

fn labels(b: &DdeCycleBranch) -> Vec<String> {
    b.specials.iter().map(|s| s.label.clone()).collect()
}

/// A ZBLK-like run from `mu0` (the first parameter of the start).
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

impl Zblk {
    /// The first run, with the branch-point test on, from the start of the case.
    fn first(&mut self, o: &Options) -> DdeCycleBranch {
        bpc_ok(
            &mut self.m,
            &self.st,
            &self.start,
            o,
            None,
            None,
            "first run",
        )
    }

    fn switch(&mut self, b: &DdeCycleBranch, k: usize) -> Result<DdeCycleSwitch, EngineError> {
        dde_switch_cycle_bpc(&mut self.m, &self.st, b, k)
    }

    fn switched(
        &mut self,
        sw: &DdeCycleSwitch,
        o: &Options,
        f: Option<&DdeFloquetOptions>,
        bpc: bool,
    ) -> Result<DdeCycleBranch, EngineError> {
        let cancel = AtomicBool::new(false);
        dde_continue_cycle_switch(&mut self.m, &self.st, sw, o, f, bpc, &cancel)
    }
}

/// The standard first run of section 8r (ZBLK from `mu = 0.1`, bounds (0.1, 0.6), direction
/// +1, `hmax` 0.1) and the switch at its only special.
fn standard(ntst: usize, which: Evaluator) -> (Zblk, DdeCycleSwitch) {
    let mut c = zblk_case(&zblk(), 0.1, ntst, which);
    let b = c.first(&run_opts((0.1, 0.6), 1.0, 0.1));
    assert_eq!(labels(&b), vec!["BPC".to_string()], "first run specials");
    let sw = c.switch(&b, 0).unwrap_or_else(|e| panic!("switch: {e}"));
    (c, sw)
}

fn zvals(p: &DdeCyclePoint) -> Vec<f64> {
    p.profile.chunks(2).map(|c| c[1]).collect()
}

fn refused(r: Result<DdeCycleSwitch, EngineError>, msg: &str, what: &str) {
    match r {
        Err(EngineError::BadProblem(m)) => assert_eq!(m, msg, "{what}"),
        other => panic!("{what}: expected BadProblem({msg:?}), got {other:?}"),
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The size of the unknowns `(profile, T, alpha)` for `n = 2` on a mesh of `ntst` intervals.
fn dim2(ntst: usize) -> usize {
    2 * (ntst * NCOL + 1) + 2
}

/// Test 1, part 1: the tangent of the pitchfork, `1/sqrt(Nm + 1)` in every `z` entry.
fn check_pitchfork_tangent(sw: &DdeCycleSwitch, ntst: usize, what: &str) {
    let nq1 = ntst * NCOL + 1;
    let dim = dim2(ntst);
    let t = &sw.tangent;
    assert_eq!(t.len(), dim, "{what}: tangent length");
    assert!(
        (norm2(t) - 1.0).abs() <= NORM_TOL,
        "{what}: norm {}",
        norm2(t)
    );
    assert!(
        t[dim - 2].abs() <= SMALL_ENTRY,
        "{what}: T entry {}",
        t[dim - 2]
    );
    assert!(
        t[dim - 1].abs() <= SMALL_ENTRY,
        "{what}: alpha entry {}",
        t[dim - 1]
    );
    let want = 1.0 / (nq1 as f64).sqrt();
    for q in 0..nq1 {
        assert!(
            t[2 * q].abs() <= SMALL_ENTRY,
            "{what}: x entry {q} is {}",
            t[2 * q]
        );
        assert!(
            (t[2 * q + 1] - want).abs() <= Z_ENTRY_TOL,
            "{what}: z entry {q} is {}, want {want}",
            t[2 * q + 1]
        );
    }
    assert!(
        (sw.alpha - M0).abs() <= ALPHA_TOL,
        "{what}: alpha {}",
        sw.alpha
    );
}

/// `J t` for the Jacobian at the start of `sw`, with the phase reference at the start profile,
/// as `(||J t||_inf, ||J||_inf)` (the second is the largest absolute row sum).
fn jacobian_residual(c: &mut Zblk, sw: &DdeCycleSwitch) -> (f64, f64) {
    let dim = sw.tangent.len();
    let mut x = sw.start.profile.clone();
    x.push(sw.start.period);
    x.push(sw.alpha);
    let j = dde_cycle_jacobian(&mut c.m, &c.st, &sw.start.profile, &x).expect("jacobian");
    let rows = dim - 1;
    let mut jt = 0.0f64;
    let mut jn = 0.0f64;
    for i in 0..rows {
        let mut s = 0.0;
        let mut r = 0.0;
        for k in 0..dim {
            s += j[i + k * rows] * sw.tangent[k];
            r += j[i + k * rows].abs();
        }
        jt = jt.max(s.abs());
        jn = jn.max(r);
    }
    (jt, jn)
}

/// The points of a switched run: the `z` values of a point agree, `z` has the sign of `sign`
/// after the first point, and `resid(z, alpha)` is small.
fn check_family(
    b: &DdeCycleBranch,
    sign: f64,
    resid: &dyn Fn(f64, f64) -> f64,
    tol: f64,
    what: &str,
) {
    for (i, p) in b.points.iter().enumerate() {
        let z = zvals(p);
        let lo = z.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            hi - lo <= Z_FLAT_TOL,
            "{what}: point {i}: z spread {}",
            hi - lo
        );
        if i > 0 {
            assert!(sign * z[0] > 0.0, "{what}: point {i}: z = {}", z[0]);
        }
        let r = resid(z[0], p.alpha);
        assert!(r.abs() <= tol, "{what}: point {i}: residual {r:e}");
    }
}

/// The `x` values and the period of every point of a ZBLK run.
fn check_x(b: &DdeCycleBranch, ntst: usize, what: &str) {
    let nq = ntst * NCOL;
    for (i, p) in b.points.iter().enumerate() {
        assert!(
            (p.period - 2.0 * PI).abs() <= T_TOL,
            "{what}: point {i}: T - 2 pi = {:e}",
            p.period - 2.0 * PI
        );
        for q in 0..=nq {
            let want = (2.0 * PI * q as f64 / nq as f64).sin();
            let got = p.profile[2 * q];
            assert!(
                (got - want).abs() <= X_TOL,
                "{what}: point {i}: x at {q} is {got}, want {want}"
            );
        }
    }
}

fn pitchfork_resid(z: f64, a: f64) -> f64 {
    z * z - (a - M0)
}

/// Test 2 for one direction: the run along the switched branch.
fn check_switched_run(c: &mut Zblk, sw: &DdeCycleSwitch, ntst: usize, dir: f64, what: &str) {
    let o = run_opts((0.1, 0.6), dir, 0.1);
    let b = c
        .switched(sw, &o, None, false)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(b.stop, StopReason::Bounds, "{what}: stop {:?}", b.stop);
    let first = &b.points[0];
    assert!(
        (first.alpha - M0).abs() <= ALPHA_TOL,
        "{what}: first alpha {}",
        first.alpha
    );
    check_family(&b, dir, &pitchfork_resid, FAMILY_TOL, what);
    for w in b.points.windows(2) {
        assert!(
            w[1].alpha >= w[0].alpha,
            "{what}: alpha falls from {} to {}",
            w[0].alpha,
            w[1].alpha
        );
    }
    let last = b.points.last().expect("points");
    assert!(
        (last.alpha - 0.6).abs() <= END_TOL,
        "{what}: last alpha {}",
        last.alpha
    );
    check_x(&b, ntst, what);
}

// Test 1: the pitchfork.
#[test]
fn the_pitchfork_tangent() {
    let (mut c, sw) = standard(NTST, Evaluator::Jit);
    check_pitchfork_tangent(&sw, NTST, "pitchfork");
    assert!(
        (sw.start.period - 2.0 * PI).abs() <= T_TOL,
        "T - 2 pi = {:e}",
        sw.start.period - 2.0 * PI
    );
    let (jt, jn) = jacobian_residual(&mut c, &sw);
    assert!(jt <= JT_TOL * jn, "||J t|| = {jt:e}, ||J|| = {jn:e}");
}

// Test 2: the switched run, both directions.
#[test]
fn the_switched_run() {
    let (mut c, sw) = standard(NTST, Evaluator::Jit);
    check_switched_run(&mut c, &sw, NTST, 1.0, "direction +1");
    check_switched_run(&mut c, &sw, NTST, -1.0, "direction -1");
}

// Test 3: the transcritical form.
#[test]
fn the_transcritical_form() {
    let src = zblk_with("(mu-m0)*z-z^2");
    let mut c = zblk_case(&src, 0.1, NTST, Evaluator::Jit);
    let b = c.first(&run_opts((0.1, 0.6), 1.0, 0.1));
    assert_eq!(labels(&b), vec!["BPC".to_string()], "first run specials");
    let sw = c.switch(&b, 0).unwrap_or_else(|e| panic!("switch: {e}"));
    let dim = dim2(NTST);
    let t = &sw.tangent;
    assert_eq!(t.len(), dim, "tangent length");
    assert!((norm2(t) - 1.0).abs() <= NORM_TOL, "norm {}", norm2(t));
    let ta = t[dim - 1];
    assert!(ta > 0.0, "t_alpha = {ta}");
    let nq1 = NTST * NCOL + 1;
    for q in 0..nq1 {
        assert!(
            (t[2 * q + 1] - ta).abs() <= TANGENT_AGREE,
            "z entry {q} is {}, t_alpha is {ta}",
            t[2 * q + 1]
        );
    }
    let resid = |z: f64, a: f64| z - (a - M0);
    let up = c
        .switched(&sw, &run_opts((0.1, 0.6), 1.0, 0.1), None, false)
        .expect("direction +1");
    assert_eq!(up.stop, StopReason::Bounds, "stop +1");
    check_family(&up, 1.0, &resid, FAMILY_TOL, "transcritical +1");
    assert!(
        up.points.windows(2).all(|w| w[1].alpha >= w[0].alpha),
        "alpha does not rise"
    );
    let down = c
        .switched(&sw, &run_opts((0.1, 0.6), -1.0, 0.1), None, false)
        .expect("direction -1");
    assert_eq!(down.stop, StopReason::Bounds, "stop -1");
    check_family(&down, -1.0, &resid, FAMILY_TOL, "transcritical -1");
    assert!(
        down.points.windows(2).all(|w| w[1].alpha <= w[0].alpha),
        "alpha does not fall"
    );
    let last = down.points.last().expect("points");
    assert!(
        (last.alpha - 0.1).abs() <= END_TOL,
        "last alpha {}",
        last.alpha
    );
}

// Test 4: the direction and the steps of the first run do not matter.
#[test]
fn the_first_run_does_not_matter() {
    let (_, reference) = standard(NTST, Evaluator::Jit);
    for (mu0, dir, hmax) in [(0.6, -1.0, 0.1), (0.137, 1.0, 0.31), (0.137, 1.0, 0.083)] {
        let what = format!("mu0 {mu0}, direction {dir}, hmax {hmax}");
        let mut c = zblk_case(&zblk(), mu0, NTST, Evaluator::Jit);
        let b = c.first(&run_opts((0.1, 0.6), dir, hmax));
        let sw = c.switch(&b, 0).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            (sw.alpha - reference.alpha).abs() <= ALPHA_TOL,
            "{what}: alpha {} against {}",
            sw.alpha,
            reference.alpha
        );
        let d = max_abs_diff(&sw.tangent, &reference.tangent);
        assert!(d <= TANGENT_AGREE, "{what}: tangents differ by {d:e}");
    }
}

// Test 5: a nonlinear coefficient in the block.
#[test]
fn a_nonlinear_coefficient() {
    let src = zblk_with("(exp(8*mu)-exp(8*m0))*z-z^3");
    let mut c = zblk_case(&src, 0.137, NTST, Evaluator::Jit);
    let b = c.first(&run_opts((0.1, 0.6), 1.0, 0.31));
    assert_eq!(labels(&b), vec!["BPC".to_string()], "first run specials");
    let sw = c.switch(&b, 0).unwrap_or_else(|e| panic!("switch: {e}"));
    let r = c
        .switched(&sw, &run_opts((0.1, 0.4), 1.0, 0.1), None, false)
        .expect("switched run");
    assert_eq!(r.stop, StopReason::Bounds, "stop {:?}", r.stop);
    let resid = |z: f64, a: f64| z * z - ((8.0 * a).exp() - (8.0 * M0).exp());
    check_family(&r, 1.0, &resid, NONLINEAR_TOL, "nonlinear");
    let last = r.points.last().expect("points");
    assert!(
        (last.alpha - 0.4).abs() <= END_TOL,
        "last alpha {}",
        last.alpha
    );
}

// Test 6: the size and the fallback evaluator.
#[test]
fn the_size_and_the_fallback_evaluator() {
    let (mut big, sw) = standard(100, Evaluator::Jit);
    check_pitchfork_tangent(&sw, 100, "n = 100");
    let (jt, jn) = jacobian_residual(&mut big, &sw);
    assert!(
        jt <= JT_TOL * jn,
        "n = 100: ||J t|| = {jt:e}, ||J|| = {jn:e}"
    );
    check_switched_run(&mut big, &sw, 100, 1.0, "n = 100, +1");
    check_switched_run(&mut big, &sw, 100, -1.0, "n = 100, -1");

    let (mut jit, jit_sw) = standard(NTST, Evaluator::Jit);
    let (mut fb, fb_sw) = standard(NTST, Evaluator::Fallback);
    let d = max_abs_diff(&jit_sw.tangent, &fb_sw.tangent);
    assert!(d <= TANGENT_AGREE, "fallback tangent differs by {d:e}");
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let a = jit.switched(&jit_sw, &o, None, false).expect("jit run");
    let b = fb.switched(&fb_sw, &o, None, false).expect("fallback run");
    assert!(
        a.points.len().abs_diff(b.points.len()) <= 2,
        "{} points against {}",
        a.points.len(),
        b.points.len()
    );
}

// Test 7: no second label at the start.
#[test]
fn no_second_label_at_the_start() {
    let (mut c, sw) = standard(NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let b = c
        .switched(&sw, &o, Some(&fl(20, 4)), true)
        .expect("switched run");
    assert_eq!(b.stop, StopReason::Bounds, "stop {:?}", b.stop);
    assert!(b.specials.is_empty(), "specials: {:?}", labels(&b));
    assert!(b.warnings.is_empty(), "warnings: {:?}", b.warnings);
    for (i, p) in b.points.iter().enumerate() {
        assert_eq!(p.multipliers.len(), 4, "point {i}: multipliers");
    }
}

// Test 8: the model with a fold, switching at both branch points.
#[test]
fn switching_around_a_fold() {
    let mut m = dde(&cubic_z(), Evaluator::Jit);
    let st = setup(&[0.25, PI / 2.0, 0.2], 0, NTST);
    let start = start_sine(NTST, 2, 1.5_f64.sqrt());
    let o = run_opts((-0.5, 0.4), -1.0, 0.1);
    let b = bpc_ok(&mut m, &st, &start, &o, Some(&fl(20, 4)), None, "first run");
    assert_eq!(
        labels(&b),
        vec!["BPC".to_string(), "LPC".to_string(), "BPC".to_string()],
        "specials"
    );
    let resid = |z: f64, a: f64| z * z - (a - 0.2);
    for k in [0, 2] {
        let sw = dde_switch_cycle_bpc(&mut m, &st, &b, k)
            .unwrap_or_else(|e| panic!("switch at special {k}: {e}"));
        let cancel = AtomicBool::new(false);
        let r = dde_continue_cycle_switch(
            &mut m,
            &st,
            &sw,
            &run_opts((-0.5, 0.4), 1.0, 0.1),
            None,
            false,
            &cancel,
        )
        .unwrap_or_else(|e| panic!("run from special {k}: {e}"));
        assert_eq!(r.stop, StopReason::Bounds, "special {k}: stop {:?}", r.stop);
        check_family(&r, 1.0, &resid, FOLD_TOL, &format!("special {k}"));
    }
    refused(
        dde_switch_cycle_bpc(&mut m, &st, &b, 1),
        "special point 1 is a LPC, not a BPC",
        "special 1",
    );
}

// Test 9: the refusals.
#[test]
fn the_refusals() {
    let mut c = zblk_case(&zblk(), 0.1, NTST, Evaluator::Jit);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let b = c.first(&o);
    assert_eq!(labels(&b), vec!["BPC".to_string()], "first run specials");
    let sw = c.switch(&b, 0).expect("switch");

    refused(c.switch(&b, 7), "no special point 7", "out of range");

    let mut first = b.clone();
    first.specials[0].index = 0;
    refused(
        c.switch(&first, 0),
        "the BPC is the first point of the branch; switching needs the point before it",
        "index 0",
    );
    let mut past = b.clone();
    past.specials[0].index = past.points.len() + 1;
    refused(
        c.switch(&past, 0),
        "the BPC is the first point of the branch; switching needs the point before it",
        "index past the points",
    );

    let adapted = bpc_ok(
        &mut c.m,
        &c.st,
        &c.start,
        &o,
        None,
        Some(&DdeMeshAdapt { every: 3 }),
        "adapt run",
    );
    refused(
        c.switch(&adapted, 0),
        "the branch has an adapted mesh",
        "adapted mesh",
    );

    let mut short = b.clone();
    short.specials[0].point.profile.pop();
    refused(
        c.switch(&short, 0),
        "the BPC does not fit the model",
        "profile of the wrong length",
    );
    let mut nan = b.clone();
    nan.specials[0].point.profile[3] = f64::NAN;
    refused(
        c.switch(&nan, 0),
        "the BPC does not fit the model",
        "profile with a NaN",
    );

    let mut same = b.clone();
    let before = same.specials[0].index - 1;
    same.specials[0].point = same.points[before].clone();
    refused(
        c.switch(&same, 0),
        "the BPC coincides with the point before it",
        "the point before it",
    );

    // An ordinary point of the mu-line: the Jacobian has rank dim - 1.
    let mut ordinary = b.clone();
    ordinary.specials[0].index = 1;
    ordinary.specials[0].point = b.points[1].clone();
    refused(
        c.switch(&ordinary, 0),
        "BPC is not simple",
        "an ordinary point",
    );

    // The BPC of this branch for a model whose branch point has a double root: the form is 0.
    let mut flat = dde(&zblk_with("(mu-m0)^2*z-z^3"), Evaluator::Jit);
    refused(
        dde_switch_cycle_bpc(&mut flat, &c.st, &b, 0),
        "BPC is not simple",
        "a vanishing form",
    );

    // continue_cycle_switch.
    let mut bad = sw.clone();
    bad.tangent.pop();
    let e = c.switched(&bad, &o, None, false).unwrap_err();
    assert_eq!(
        e,
        EngineError::BadProblem("the switch does not fit the setup".into()),
        "short tangent"
    );
    let mut nan_t = sw.clone();
    nan_t.tangent[0] = f64::NAN;
    let e = c.switched(&nan_t, &o, None, false).unwrap_err();
    assert_eq!(
        e,
        EngineError::BadProblem("the switch does not fit the setup".into()),
        "tangent with a NaN"
    );
    let mut steps = o.clone();
    steps.hmin = 0.0;
    let e = c.switched(&bad, &steps, None, false).unwrap_err();
    assert_eq!(
        e,
        EngineError::BadProblem("bad step control options".into()),
        "the step controls come first"
    );
    let cancel = AtomicBool::new(true);
    let e = dde_continue_cycle_switch(&mut c.m, &c.st, &sw, &o, None, false, &cancel).unwrap_err();
    assert_eq!(e, EngineError::Cancelled, "a cancel flag set at the call");
    assert!(cancel.load(Ordering::Relaxed));
}

/// Prints the values behind the bounds (SPEC section 8r, "Measured values").
#[test]
#[ignore = "prints the measured values of SPEC section 8r"]
fn print_tables() {
    let (mut c, sw) = standard(NTST, Evaluator::Jit);
    let nq1 = NTST * NCOL + 1;
    let dim = dim2(NTST);
    let t = &sw.tangent;
    let want = 1.0 / (nq1 as f64).sqrt();
    let xmax = (0..nq1).fold(0.0f64, |m, q| m.max(t[2 * q].abs()));
    let zerr = (0..nq1).fold(0.0f64, |m, q| m.max((t[2 * q + 1] - want).abs()));
    let (jt, jn) = jacobian_residual(&mut c, &sw);
    println!("test 1: N = {NTST}, m = {NCOL}, dim = {dim}");
    println!("  alpha - m0        {:e}", sw.alpha - M0);
    println!("  T - 2 pi          {:e}", sw.start.period - 2.0 * PI);
    println!("  | ||t|| - 1 |     {:e}", (norm2(t) - 1.0).abs());
    println!("  max |x entry|     {xmax:e}");
    println!("  T entry           {:e}", t[dim - 2]);
    println!("  alpha entry       {:e}", t[dim - 1]);
    println!("  max z entry err   {zerr:e}");
    println!("  ||J t|| / ||J||   {:e}  ({jt:e} / {jn:e})", jt / jn);
    for dir in [1.0, -1.0] {
        let o = run_opts((0.1, 0.6), dir, 0.1);
        let b = c.switched(&sw, &o, None, false).expect("switched run");
        let mut zspread = 0.0f64;
        let mut fam = 0.0f64;
        let mut xerr = 0.0f64;
        let mut terr = 0.0f64;
        for p in &b.points {
            let z = zvals(p);
            let lo = z.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            zspread = zspread.max(hi - lo);
            fam = fam.max(pitchfork_resid(z[0], p.alpha).abs());
            terr = terr.max((p.period - 2.0 * PI).abs());
            for q in 0..=NTST * NCOL {
                let s = (2.0 * PI * q as f64 / (NTST * NCOL) as f64).sin();
                xerr = xerr.max((p.profile[2 * q] - s).abs());
            }
        }
        let last = b.points.last().expect("points");
        println!(
            "test 2, direction {dir}: {} points, stop {:?}",
            b.points.len(),
            b.stop
        );
        println!("  first alpha - m0  {:e}", b.points[0].alpha - M0);
        println!("  z spread          {zspread:e}");
        println!("  z^2 - (a - m0)    {fam:e}");
        println!("  x error           {xerr:e}");
        println!("  T error           {terr:e}");
        println!("  last alpha - 0.6  {:e}", last.alpha - 0.6);
    }

    let src = zblk_with("(mu-m0)*z-z^2");
    let mut tc = zblk_case(&src, 0.1, NTST, Evaluator::Jit);
    let b = tc.first(&run_opts((0.1, 0.6), 1.0, 0.1));
    let tsw = tc.switch(&b, 0).expect("transcritical switch");
    let ta = tsw.tangent[dim - 1];
    let zdiff = (0..nq1).fold(0.0f64, |m, q| m.max((tsw.tangent[2 * q + 1] - ta).abs()));
    println!(
        "test 3: alpha - m0 {:e}, t_alpha {ta:e}, max |t_z - t_alpha| {zdiff:e}",
        tsw.alpha - M0
    );
    for dir in [1.0, -1.0] {
        let r = tc
            .switched(&tsw, &run_opts((0.1, 0.6), dir, 0.1), None, false)
            .expect("transcritical run");
        let fam = r
            .points
            .iter()
            .fold(0.0f64, |m, p| m.max((zvals(p)[0] - (p.alpha - M0)).abs()));
        println!(
            "  direction {dir}: {} points, stop {:?}, max |z - (a - m0)| {fam:e}, last alpha {}",
            r.points.len(),
            r.stop,
            r.points.last().expect("points").alpha
        );
    }

    for (mu0, dir, hmax) in [(0.6, -1.0, 0.1), (0.137, 1.0, 0.31), (0.137, 1.0, 0.083)] {
        let mut c4 = zblk_case(&zblk(), mu0, NTST, Evaluator::Jit);
        let b = c4.first(&run_opts((0.1, 0.6), dir, hmax));
        let s4 = c4.switch(&b, 0).expect("switch");
        println!(
            "test 4, mu0 {mu0}, direction {dir}, hmax {hmax}: alpha diff {:e}, tangent diff {:e}",
            s4.alpha - sw.alpha,
            max_abs_diff(&s4.tangent, &sw.tangent)
        );
    }

    let src = zblk_with("(exp(8*mu)-exp(8*m0))*z-z^3");
    let mut c5 = zblk_case(&src, 0.137, NTST, Evaluator::Jit);
    let b = c5.first(&run_opts((0.1, 0.6), 1.0, 0.31));
    let s5 = c5.switch(&b, 0).expect("switch");
    let r = c5
        .switched(&s5, &run_opts((0.1, 0.4), 1.0, 0.1), None, false)
        .expect("nonlinear run");
    let fam = r.points.iter().fold(0.0f64, |m, p| {
        let z = zvals(p)[0];
        m.max((z * z - ((8.0 * p.alpha).exp() - (8.0 * M0).exp())).abs())
    });
    println!(
        "test 5: alpha - m0 {:e}, {} points, max |z^2 - (e^8a - e^8m0)| {fam:e}",
        s5.alpha - M0,
        r.points.len()
    );

    let (mut big, bsw) = standard(100, Evaluator::Jit);
    let (bjt, bjn) = jacobian_residual(&mut big, &bsw);
    println!(
        "test 6, N = 100: alpha - m0 {:e}, ||J t|| / ||J|| {:e}",
        bsw.alpha - M0,
        bjt / bjn
    );
    let (mut fb, fsw) = standard(NTST, Evaluator::Fallback);
    let o = run_opts((0.1, 0.6), 1.0, 0.1);
    let a = c.switched(&sw, &o, None, false).expect("jit run");
    let f = fb.switched(&fsw, &o, None, false).expect("fallback run");
    println!(
        "  fallback: tangent diff {:e}, {} points against {}",
        max_abs_diff(&sw.tangent, &fsw.tangent),
        f.points.len(),
        a.points.len()
    );

    let mut m = dde(&cubic_z(), Evaluator::Jit);
    let st = setup(&[0.25, PI / 2.0, 0.2], 0, NTST);
    let start = start_sine(NTST, 2, 1.5_f64.sqrt());
    let bf = bpc_ok(
        &mut m,
        &st,
        &start,
        &run_opts((-0.5, 0.4), -1.0, 0.1),
        Some(&fl(20, 4)),
        None,
        "fold run",
    );
    for k in [0, 2] {
        let s8 = dde_switch_cycle_bpc(&mut m, &st, &bf, k).expect("switch");
        let cancel = AtomicBool::new(false);
        let r = dde_continue_cycle_switch(
            &mut m,
            &st,
            &s8,
            &run_opts((-0.5, 0.4), 1.0, 0.1),
            None,
            false,
            &cancel,
        )
        .expect("run");
        let fam = r.points.iter().fold(0.0f64, |mx, p| {
            let z = zvals(p)[0];
            mx.max((z * z - (p.alpha - 0.2)).abs())
        });
        println!(
            "test 8, special {k}: alpha - 0.2 {:e}, {} points, max |z^2 - (a - 0.2)| {fam:e}",
            s8.alpha - 0.2,
            r.points.len()
        );
    }
}
