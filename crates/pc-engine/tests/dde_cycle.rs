//! Periodic orbits of a delay equation by collocation (SPEC section 8l).
//!
//! The model `x' = -x(t - tau) + mu x (1 - x^2 - x(t - tau)^2)` has the exact orbit
//! `x = sin t`, `T = 2 pi`, for every `mu` and for `tau = pi/2 + 2 pi K`
//! (`references/dde-cycle-oracle.json`). `TWO` has no known orbit; it checks the Jacobian
//! with a parameter delay and a literal delay.

use pc_engine::{
    CollocationBasis, DdeCycle, DdeCycleSetup, DdeCycleStart, DdeHopfCycle, DdeModel, EngineError,
    Evaluator, Options, build_dde, dde_correct_cycle, dde_cycle_jacobian, dde_cycle_residual,
    dde_hopf_cycle,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

const TAU_H: f64 = 1.327_237_281_892_678_3;
const OMEGA_H: f64 = 0.953_939_201_416_945_7;

fn osc_source(tau: f64) -> String {
    format!(
        "par mu=0.3, tau={tau:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         done\n"
    )
}

const TWO: &str = "par a=0.9, tau1=1.1\n\
    x'=-a*x+0.5*delay(y,tau1)-0.3*x*y\n\
    y'=-y+0.8*delay(x,0.7)+0.2*delay(y,tau1)*delay(x,0.7)\n\
    done\n";

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

fn both() -> [Evaluator; 2] {
    [Evaluator::Jit, Evaluator::Fallback]
}

fn tau_k(k: usize) -> f64 {
    PI / 2.0 + 2.0 * PI * k as f64
}

fn opts() -> Options {
    Options {
        max_newton: 30,
        ..Options::default()
    }
}

/// The tolerances of test 6, which compare two evaluators to 1e-10.
fn tight() -> Options {
    Options {
        max_newton: 30,
        tol_x: 1e-11,
        tol_f: 1e-10,
        ..Options::default()
    }
}

fn setup(params: &[f64], free: usize, ntst: usize, ncol: usize) -> DdeCycleSetup {
    DdeCycleSetup {
        params: params.to_vec(),
        free,
        ntst,
        ncol,
    }
}

/// Base point `q` of the uniform mesh is `q / (N m)`.
fn base(ntst: usize, ncol: usize) -> Vec<f64> {
    let nq = ntst * ncol;
    (0..=nq).map(|q| q as f64 / nq as f64).collect()
}

/// `f(c, s)` for component `c` at every base point, point after point.
fn profile(n: usize, ntst: usize, ncol: usize, f: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    let mut out = Vec::new();
    for s in base(ntst, ncol) {
        for c in 0..n {
            out.push(f(c, s));
        }
    }
    out
}

/// The unknown vector: the profile, then `T`, then `alpha`.
fn unknowns(prof: &[f64], period: f64, alpha: f64) -> Vec<f64> {
    let mut x = prof.to_vec();
    x.push(period);
    x.push(alpha);
    x
}

fn sine(ntst: usize, ncol: usize) -> Vec<f64> {
    profile(1, ntst, ncol, |_, s| (2.0 * PI * s).sin())
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |a: f64, x| a.max(x.abs()))
}

fn bad<T: std::fmt::Debug>(r: Result<T, EngineError>, what: &str) {
    match r {
        Err(EngineError::BadProblem(_)) => {}
        other => panic!("{what}: expected BadProblem, got {other:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_correct(
    m: &mut DdeModel,
    params: &[f64],
    free: usize,
    ntst: usize,
    ncol: usize,
    start: &[f64],
    period: f64,
    o: &Options,
) -> Result<DdeCycle, EngineError> {
    let cancel = AtomicBool::new(false);
    dde_correct_cycle(
        m,
        &setup(params, free, ntst, ncol),
        &DdeCycleStart {
            profile: start.to_vec(),
            period,
        },
        o,
        &cancel,
    )
}

fn run_hopf(
    m: &mut DdeModel,
    params: &[f64],
    free: usize,
    ntst: usize,
    ncol: usize,
    eps: f64,
    o: &Options,
) -> Result<DdeCycle, EngineError> {
    let cancel = AtomicBool::new(false);
    dde_hopf_cycle(
        m,
        &setup(params, free, ntst, ncol),
        &DdeHopfCycle {
            x: vec![0.0],
            omega: OMEGA_H,
            eps,
        },
        o,
        &cancel,
    )
}

// ---------------------------------------------------------------------------------------
// Test 1: the Jacobian against central differences.

struct JacCase {
    name: &'static str,
    src: String,
    params: Vec<f64>,
    free: usize,
    ntst: usize,
    ncol: usize,
    period: f64,
}

fn jac_cases() -> Vec<JacCase> {
    let mut v = Vec::new();
    for (free, name) in [(1, "tau"), (0, "mu")] {
        v.push(JacCase {
            name: if free == 1 {
                "OSC K0 free tau"
            } else {
                "OSC K0 free mu"
            },
            src: osc_source(tau_k(0)),
            params: vec![0.3, tau_k(0)],
            free,
            ntst: 6,
            ncol: 3,
            period: 6.1,
        });
        let _ = name;
        v.push(JacCase {
            name: if free == 1 {
                "OSC K1 free tau"
            } else {
                "OSC K1 free mu"
            },
            src: osc_source(tau_k(1)),
            params: vec![0.3, tau_k(1)],
            free,
            ntst: 5,
            ncol: 4,
            period: 6.1,
        });
    }
    for (free, name) in [(1, "TWO free tau1"), (0, "TWO free a")] {
        v.push(JacCase {
            name,
            src: TWO.to_string(),
            params: vec![0.9, 1.1],
            free,
            ntst: 5,
            ncol: 3,
            period: 5.3,
        });
    }
    v
}

fn test_profile(c: &JacCase, n: usize) -> Vec<f64> {
    profile(n, c.ntst, c.ncol, |k, s| {
        0.2 * k as f64 + 0.5 * (2.0 * PI * s + 0.3 * k as f64).sin() + 0.1 * (4.0 * PI * s).cos()
    })
}

/// Every shifted point of every Gauss point is at least 1e-3 (in `s`) from a mesh node.
fn assert_off_nodes(m: &DdeModel, c: &JacCase, alpha: f64) {
    let basis = CollocationBasis::new(c.ncol).unwrap();
    let mut p = c.params.clone();
    p[c.free] = alpha;
    let nn = c.ntst as f64;
    for slot in &m.delays {
        let tau = match slot.time {
            pc_ir::DelayTime::Param(j) => p[j],
            pc_ir::DelayTime::Lit(v) => v,
        };
        for j in 0..c.ntst {
            for z in basis.nodes() {
                let s = (j as f64 + z) / nn;
                let sp = (s - tau / c.period).rem_euclid(1.0);
                let dist = (0..=c.ntst)
                    .map(|k| (sp - k as f64 / nn).abs())
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    dist >= 1e-3,
                    "{}: shifted point {sp} is {dist} from a node",
                    c.name
                );
            }
        }
    }
}

fn jacobian_of(c: &JacCase, which: Evaluator) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut m = dde(&c.src, which);
    let n = 1 + usize::from(c.src == TWO);
    let prof = test_profile(c, n);
    let alpha = c.params[c.free];
    assert_off_nodes(&m, c, alpha);
    let x = unknowns(&prof, c.period, alpha);
    let st = setup(&c.params, c.free, c.ntst, c.ncol);
    let jac = dde_cycle_jacobian(&mut m, &st, &prof, &x).unwrap();
    (jac, x, prof)
}

#[test]
fn jacobian_matches_central_differences() {
    for c in jac_cases() {
        let mut m = dde(&c.src, Evaluator::Jit);
        let (jac, x, prof) = jacobian_of(&c, Evaluator::Jit);
        let dim = x.len();
        let rows = dim - 1;
        assert_eq!(jac.len(), rows * dim, "{}", c.name);
        let st = setup(&c.params, c.free, c.ntst, c.ncol);
        for col in 0..dim {
            let h = 1e-6 * (1.0 + x[col].abs());
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[col] += h;
            xm[col] -= h;
            // `alpha` is the last unknown and the model's parameter, so the residual reads
            // it from `x`.
            let rp = dde_cycle_residual(&mut m, &st, &prof, &xp).unwrap();
            let rm = dde_cycle_residual(&mut m, &st, &prof, &xm).unwrap();
            for r in 0..rows {
                let fd = (rp[r] - rm[r]) / (2.0 * h);
                let j = jac[r + rows * col];
                assert!(
                    (j - fd).abs() <= 1e-6 * (1.0 + j.abs()),
                    "{}: row {r} column {col}: J = {j}, FD = {fd}",
                    c.name
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Test 2: the residual of the exact orbit.

/// `(K, N, m, bound)`: the largest absolute residual of the exact orbit.
const RESIDUAL: [(usize, usize, usize, f64); 12] = [
    (0, 10, 3, 3e-3),
    (0, 10, 4, 3e-4),
    (0, 20, 3, 4e-4),
    (0, 20, 4, 2e-5),
    (0, 40, 3, 5e-5),
    (0, 40, 4, 1.2e-6),
    (1, 10, 3, 3e-3),
    (1, 10, 4, 3e-4),
    (1, 20, 3, 4e-4),
    (1, 20, 4, 2e-5),
    (1, 40, 3, 5e-5),
    (1, 40, 4, 1.2e-6),
];

fn exact_residual(k: usize, ntst: usize, ncol: usize) -> f64 {
    let mut m = dde(&osc_source(tau_k(k)), Evaluator::Jit);
    let prof = sine(ntst, ncol);
    let x = unknowns(&prof, 2.0 * PI, tau_k(k));
    let st = setup(&[0.3, tau_k(k)], 1, ntst, ncol);
    max_abs(&dde_cycle_residual(&mut m, &st, &prof, &x).unwrap())
}

#[test]
fn exact_orbit_has_a_small_residual_that_falls_with_n() {
    let mut last: Option<(usize, usize, f64)> = None;
    for (k, n, m, bound) in RESIDUAL {
        let r = exact_residual(k, n, m);
        assert!(
            r <= bound,
            "K={k} N={n} m={m}: residual {r:e} above {bound:e}"
        );
        if let Some((pk, pm, pr)) = last
            && pk == k
            && pm == m
            && pr > 1e-13
            && r > 1e-13
        {
            let factor = 2f64.powi(m as i32 - 1);
            assert!(
                pr / r >= factor,
                "K={k} m={m}: N {} to {n} reduced {pr:e} to {r:e}, factor {} below {factor}",
                n / 2,
                pr / r
            );
        }
        last = Some((k, m, r));
    }
}

// ---------------------------------------------------------------------------------------
// Test 3: Newton from a perturbed start.

/// `(K, N, m, bound on |T - 2 pi|, bound on max |u_q - sin 2 pi s_q|)`.
const NEWTON: [(usize, usize, usize, f64, f64); 12] = [
    (0, 10, 3, 4e-4, 2.5e-4),
    (0, 10, 4, 4e-9, 4e-6),
    (0, 20, 3, 1.2e-7, 5e-6),
    (0, 20, 4, 5e-11, 1.3e-7),
    (0, 40, 3, 2e-9, 3e-7),
    (0, 40, 4, 1e-12, 4e-9),
    (1, 10, 3, 2.5e-4, 5e-4),
    (1, 10, 4, 2e-9, 4e-6),
    (1, 20, 3, 7e-8, 5.5e-6),
    (1, 20, 4, 3e-11, 1.3e-7),
    (1, 40, 3, 1e-9, 3.3e-7),
    (1, 40, 4, 1e-12, 4e-9),
];

fn perturbed(ntst: usize, ncol: usize) -> Vec<f64> {
    profile(1, ntst, ncol, |_, s| {
        1.2 * (2.0 * PI * s).sin() + 0.05 * (4.0 * PI * s).sin()
    })
}

fn newton_run(k: usize, ntst: usize, ncol: usize, which: Evaluator, o: &Options) -> DdeCycle {
    let mut m = dde(&osc_source(tau_k(k)), which);
    run_correct(
        &mut m,
        &[0.3, tau_k(k)],
        1,
        ntst,
        ncol,
        &perturbed(ntst, ncol),
        6.0,
        o,
    )
    .unwrap_or_else(|e| panic!("K={k} N={ntst} m={ncol}: {e}"))
}

#[test]
fn newton_reaches_the_exact_orbit() {
    for (k, n, m, dt, du) in NEWTON {
        let r = newton_run(k, n, m, Evaluator::Jit, &opts());
        let exact = sine(n, m);
        let err = r
            .profile
            .iter()
            .zip(&exact)
            .fold(0.0, |a: f64, (u, e)| a.max((u - e).abs()));
        assert!(
            (r.period - 2.0 * PI).abs() <= dt,
            "K={k} N={n} m={m}: T = {} off by {:e}, bound {dt:e}",
            r.period,
            (r.period - 2.0 * PI).abs()
        );
        assert!(
            err <= du,
            "K={k} N={n} m={m}: profile error {err:e} above {du:e}"
        );
        assert_eq!(r.alpha, tau_k(k), "alpha is fixed");
        assert!(
            r.residual <= 1e-9,
            "K={k} N={n} m={m}: residual {:e}",
            r.residual
        );
        assert_eq!(r.profile.len(), n * m + 1);
    }
}

// ---------------------------------------------------------------------------------------
// Test 4: the Hopf start.

fn hopf_at(eps: f64, which: Evaluator, o: &Options) -> DdeCycle {
    let mut m = dde(&osc_source(TAU_H), which);
    run_hopf(&mut m, &[0.3, TAU_H], 1, 20, 4, eps, o).unwrap_or_else(|e| panic!("eps = {eps}: {e}"))
}

#[test]
fn hopf_start_follows_the_normal_form() {
    let period0 = 2.0 * PI / OMEGA_H;
    let runs: Vec<(f64, DdeCycle)> = [0.04, 0.02, 0.01]
        .iter()
        .map(|&e| (e, hopf_at(e, Evaluator::Jit, &opts())))
        .collect();
    let da: Vec<f64> = runs.iter().map(|(_, r)| r.alpha - TAU_H).collect();
    for (d, (e, _)) in da.iter().zip(&runs) {
        assert!(d.abs() > 1e-12, "eps = {e}: alpha did not move");
    }
    for i in 0..2 {
        let ratio = da[i + 1] / da[i];
        assert!(
            (ratio - 0.25).abs() <= 0.005,
            "d_alpha ratio {ratio} for eps {} to {}",
            runs[i].0,
            runs[i + 1].0
        );
    }
    let dt: Vec<f64> = runs
        .iter()
        .map(|(_, r)| (r.period - period0).abs())
        .collect();
    assert!(dt[2] <= 2e-4, "T off by {} at eps = 0.01", dt[2]);
    let fall = dt[1] / dt[2];
    assert!(
        (fall - 4.0).abs() <= 0.1,
        "T error fell by {fall}, expected about 4"
    );
}

// ---------------------------------------------------------------------------------------
// Test 5: errors.

fn osc_params() -> Vec<f64> {
    vec![0.3, tau_k(0)]
}

#[test]
fn bad_setups_are_bad_problems() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let o = opts();
    let good = |ntst: usize, ncol: usize| sine(ntst, ncol);
    let pr = osc_params();
    bad(run_correct(&mut m, &pr, 1, 0, 3, &[0.0], 6.0, &o), "ntst 0");
    bad(
        run_correct(&mut m, &pr, 1, 10, 1, &good(10, 3), 6.0, &o),
        "ncol 1",
    );
    bad(
        run_correct(&mut m, &pr, 1, 10, 8, &good(10, 3), 6.0, &o),
        "ncol 8",
    );
    bad(
        run_correct(&mut m, &pr, 2, 10, 3, &good(10, 3), 6.0, &o),
        "free = n params",
    );
    bad(
        run_correct(&mut m, &[0.3], 1, 10, 3, &good(10, 3), 6.0, &o),
        "short params",
    );
    bad(
        run_correct(
            &mut m,
            &[0.3, tau_k(0), 1.0],
            1,
            10,
            3,
            &good(10, 3),
            6.0,
            &o,
        ),
        "long params",
    );
    bad(
        run_correct(
            &mut m,
            &[f64::NAN, tau_k(0)],
            1,
            10,
            3,
            &good(10, 3),
            6.0,
            &o,
        ),
        "NaN parameter",
    );
    bad(
        run_correct(&mut m, &[0.3, 0.0], 1, 10, 3, &good(10, 3), 6.0, &o),
        "tau 0",
    );
    bad(
        run_correct(&mut m, &[0.3, -1.0], 1, 10, 3, &good(10, 3), 6.0, &o),
        "tau -1",
    );
    bad(
        run_correct(&mut m, &pr, 1, 10, 3, &good(10, 4), 6.0, &o),
        "profile length",
    );
    let mut nan = good(10, 3);
    nan[7] = f64::NAN;
    bad(
        run_correct(&mut m, &pr, 1, 10, 3, &nan, 6.0, &o),
        "NaN in the profile",
    );
    bad(
        run_correct(&mut m, &pr, 1, 10, 3, &good(10, 3), 0.0, &o),
        "period 0",
    );
    bad(
        run_correct(&mut m, &pr, 1, 10, 3, &good(10, 3), -1.0, &o),
        "period -1",
    );
    bad(
        run_correct(&mut m, &pr, 1, 10, 3, &good(10, 3), f64::NAN, &o),
        "period NaN",
    );
    bad(
        run_correct(&mut m, &pr, 1, 1000, 4, &vec![0.0; 4001], 6.0, &o),
        "dim above the limit",
    );
    bad(
        run_correct(&mut m, &pr, 1, usize::MAX / 2, 4, &[0.0], 6.0, &o),
        "ntst * ncol overflows",
    );
}

#[test]
fn bad_hopf_starts_are_bad_problems() {
    let mut m = dde(&osc_source(TAU_H), Evaluator::Jit);
    let st = setup(&[0.3, TAU_H], 1, 20, 4);
    let o = opts();
    let cancel = AtomicBool::new(false);
    let mut run = |x: Vec<f64>, omega: f64, eps: f64| {
        dde_hopf_cycle(&mut m, &st, &DdeHopfCycle { x, omega, eps }, &o, &cancel)
    };
    bad(run(vec![0.0, 0.0], OMEGA_H, 0.01), "x length");
    bad(run(vec![f64::NAN], OMEGA_H, 0.01), "NaN x");
    for omega in [0.0, -1.0, f64::NAN] {
        bad(run(vec![0.0], omega, 0.01), "omega");
    }
    for eps in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        match run(vec![0.0], OMEGA_H, eps) {
            Err(EngineError::BadProblem(msg)) => {
                assert!(msg.contains("eps"), "message for eps = {eps}: {msg}")
            }
            other => panic!("eps = {eps}: expected BadProblem, got {other:?}"),
        }
    }
    bad(
        run(vec![0.0], 2.0, 0.01),
        "omega that is not a Hopf frequency",
    );
}

#[test]
fn a_set_cancel_flag_stops_the_run() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let cancel = AtomicBool::new(true);
    let r = dde_correct_cycle(
        &mut m,
        &setup(&osc_params(), 1, 10, 3),
        &DdeCycleStart {
            profile: perturbed(10, 3),
            period: 6.0,
        },
        &opts(),
        &cancel,
    );
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
    assert!(cancel.load(Ordering::Relaxed));
}

// ---------------------------------------------------------------------------------------
// Test 6: the two evaluators agree.

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + x.abs()),
            "{what}[{i}]: {x} against {y}"
        );
    }
}

#[test]
fn jit_and_fallback_agree() {
    for c in jac_cases() {
        let (j0, _, _) = jacobian_of(&c, both()[0].clone());
        let (j1, _, _) = jacobian_of(&c, both()[1].clone());
        close(&j0, &j1, 1e-10, &format!("{} Jacobian", c.name));
    }
    for k in [0, 1] {
        let a = newton_run(k, 20, 4, both()[0].clone(), &tight());
        let b = newton_run(k, 20, 4, both()[1].clone(), &tight());
        close(&a.profile, &b.profile, 1e-10, &format!("K={k} profile"));
        close(
            &[a.period, a.alpha],
            &[b.period, b.alpha],
            1e-10,
            &format!("K={k} T, alpha"),
        );
    }
    let a = hopf_at(0.02, both()[0].clone(), &tight());
    let b = hopf_at(0.02, both()[1].clone(), &tight());
    close(&a.profile, &b.profile, 1e-10, "Hopf profile");
    close(
        &[a.period, a.alpha],
        &[b.period, b.alpha],
        1e-10,
        "Hopf T, alpha",
    );
}

// ---------------------------------------------------------------------------------------
// Not a test: prints the numbers for the tables of SPEC section 8l.
//   cargo test -p pc-engine --test dde_cycle print_tables -- --ignored --nocapture

#[test]
#[ignore]
fn print_tables() {
    for (k, n, m, _) in RESIDUAL {
        println!("RESID K={k} N={n} m={m} {:e}", exact_residual(k, n, m));
    }
    for (k, n, m, _, _) in NEWTON {
        let r = newton_run(k, n, m, Evaluator::Jit, &opts());
        let exact = sine(n, m);
        let err = r
            .profile
            .iter()
            .zip(&exact)
            .fold(0.0, |a: f64, (u, e)| a.max((u - e).abs()));
        println!(
            "NEWTON K={k} N={n} m={m} dT={:e} du={err:e} iters={} res={:e}",
            (r.period - 2.0 * PI).abs(),
            r.iters,
            r.residual
        );
    }
    let period0 = 2.0 * PI / OMEGA_H;
    for eps in [0.04, 0.02, 0.01] {
        let r = hopf_at(eps, Evaluator::Jit, &opts());
        println!(
            "HOPF eps={eps} dalpha={:e} dT={:e} iters={} res={:e}",
            r.alpha - TAU_H,
            (r.period - period0).abs(),
            r.iters,
            r.residual
        );
    }
}
