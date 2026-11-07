//! Floquet multipliers of a periodic orbit of a delay equation (SPEC section 8m).
//!
//! `OSC` has the exact orbit `x = sin t`, `T = 2 pi`, for `tau = pi/2 + 2 pi K`. The oracle
//! file `references/dde-cycle-oracle.json` holds its multipliers from a solver with a
//! different method (about nine digits). `FORCED` adds a state that the first equation does
//! not read, so one multiplier is known in closed form.

use pc_engine::{
    DdeCycleSetup, DdeCycleStart, DdeFloquet, DdeFloquetOptions, DdeModel, EngineError, Evaluator,
    Options, build_dde, dde_correct_cycle, dde_floquet_multipliers,
};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};

const ORACLE: &str = include_str!("../../../references/dde-cycle-oracle.json");

fn osc_source(tau: f64) -> String {
    format!(
        "par mu=0.3, tau={tau:?}\n\
         x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
         done\n"
    )
}

const OSCLIT: &str = "par mu=0.3\n\
    x'=-delay(x,1.5707963267948966)+mu*x*(1-x^2-delay(x,1.5707963267948966)^2)\n\
    done\n";

const FORCED: &str = "par mu=0.3, tau=1.5707963267948966\n\
    x'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\n\
    y'=-0.1*y+0.5*delay(x,tau)\n\
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

fn tau_k(k: usize) -> f64 {
    PI / 2.0 + 2.0 * PI * k as f64
}

fn opts() -> Options {
    Options {
        max_newton: 30,
        ..Options::default()
    }
}

/// The corrector tolerances of the test that compares two evaluators to 1e-9.
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

fn base(ntst: usize, ncol: usize) -> Vec<f64> {
    let nq = ntst * ncol;
    (0..=nq).map(|q| q as f64 / nq as f64).collect()
}

fn profile(n: usize, ntst: usize, ncol: usize, f: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    let mut out = Vec::new();
    for s in base(ntst, ncol) {
        for c in 0..n {
            out.push(f(c, s));
        }
    }
    out
}

/// The start of test 3 of section 8l.
fn perturbed(n: usize, ntst: usize, ncol: usize) -> Vec<f64> {
    profile(n, ntst, ncol, |c, s| {
        if c == 0 {
            1.2 * (2.0 * PI * s).sin() + 0.05 * (4.0 * PI * s).sin()
        } else {
            0.0
        }
    })
}

fn bad<T: std::fmt::Debug>(r: Result<T, EngineError>, what: &str) {
    match r {
        Err(EngineError::BadProblem(_)) => {}
        other => panic!("{what}: expected BadProblem, got {other:?}"),
    }
}

/// The orbit of `OSC` at `(mu, tau_k)` with `tau` fixed, corrected from the perturbed start.
fn osc_orbit(
    m: &mut DdeModel,
    mu: f64,
    k: usize,
    ntst: usize,
    ncol: usize,
    o: &Options,
) -> (DdeCycleSetup, DdeCycleStart) {
    let st = setup(&[mu, tau_k(k)], 1, ntst, ncol);
    let cancel = AtomicBool::new(false);
    let c = dde_correct_cycle(
        m,
        &st,
        &DdeCycleStart {
            profile: perturbed(1, ntst, ncol),
            period: 6.0,
        },
        o,
        &cancel,
    )
    .unwrap();
    (
        st,
        DdeCycleStart {
            profile: c.profile,
            period: c.period,
        },
    )
}

fn floquet(
    m: &mut DdeModel,
    st: &DdeCycleSetup,
    cyc: &DdeCycleStart,
    nhist: usize,
    count: usize,
) -> Result<DdeFloquet, EngineError> {
    let cancel = AtomicBool::new(false);
    dde_floquet_multipliers(m, st, cyc, &DdeFloquetOptions { nhist, count }, &cancel)
}

fn modulus(z: (f64, f64)) -> f64 {
    z.0.hypot(z.1)
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// The nontrivial multipliers with a nonnegative imaginary part, largest modulus first: the
/// entry closest to 1 is dropped, and so is the lower member of each conjugate pair.
fn leading(all: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let triv = (0..all.len())
        .min_by(|&i, &j| dist(all[i], (1.0, 0.0)).total_cmp(&dist(all[j], (1.0, 0.0))))
        .unwrap();
    let mut v: Vec<(f64, f64)> = all
        .iter()
        .enumerate()
        .filter(|&(i, z)| i != triv && z.1 >= 0.0)
        .map(|(_, z)| *z)
        .collect();
    v.sort_by(|a, b| modulus(*b).total_cmp(&modulus(*a)));
    v
}

// ---------------------------------------------------------------------------------------
// The oracle rows, read from the JSON file by hand (the crate has no JSON dependency).

struct Row {
    k: usize,
    mu: f64,
    mult: Vec<(f64, f64)>,
}

fn numbers(s: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E') {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.parse::<f64>().unwrap());
            cur.clear();
        }
    }
    out
}

fn oracle_rows() -> Vec<Row> {
    let at = ORACLE.find("\"rows\"").unwrap();
    ORACLE[at..]
        .split("\"K\":")
        .skip(1)
        .map(|chunk| {
            let mu_at = chunk.find("\"mu\":").unwrap();
            let mm_at = chunk.find("\"multipliers\":").unwrap();
            let k = numbers(&chunk[..mu_at])[0] as usize;
            let mu = numbers(&chunk[mu_at + 5..mm_at])[0];
            let flat = numbers(&chunk[mm_at + 14..]);
            assert_eq!(flat.len() % 2, 0);
            let mult = flat.chunks(2).map(|p| (p[0], p[1])).collect();
            Row { k, mu, mult }
        })
        .collect()
}

fn row(k: usize, mu: f64) -> Row {
    oracle_rows()
        .into_iter()
        .find(|r| r.k == k && (r.mu - mu).abs() < 1e-12)
        .unwrap()
}

// ---------------------------------------------------------------------------------------
// Tests 1 and 2: the oracle rows and the trivial multiplier.

/// `(ntst, ncol, nhist)` of tests 1 and 2 for `K = 0` and `K = 1`.
const MESH: [(usize, usize, usize); 2] = [(40, 4, 16), (40, 4, 24)];

/// Bounds on the error of the three leading nontrivial multipliers, per `K`. Each is 2.5 to 3
/// times the largest error measured over the rows of that `K` (the tables in SPEC 8m). The
/// oracle values have ten decimals, so 1.5e-10 is near their resolution.
const ORACLE_BOUND: [[f64; 3]; 2] = [[1.5e-10, 1.5e-9, 1.5e-7], [2e-6, 2e-6, 1.2e-4]];

/// Bounds on `trivial_error`. The measured values are rounding level (1e-13 for `K = 0`,
/// 5e-13 for `K = 1`), so the bound leaves room for a different eigenvalue routine on
/// another platform. A coarse history or mesh gives 1e-3 or more.
const TRIVIAL_BOUND: [f64; 2] = [1e-11, 1e-11];

fn oracle_errors(r: &Row, which: Evaluator) -> ([f64; 3], f64) {
    let (ntst, ncol, nhist) = MESH[r.k];
    let mut m = dde(&osc_source(tau_k(r.k)), which);
    let (st, cyc) = osc_orbit(&mut m, r.mu, r.k, ntst, ncol, &opts());
    let f = floquet(&mut m, &st, &cyc, nhist, nhist + 1).unwrap();
    assert_eq!(f.multipliers.len(), nhist + 1);
    let lead = leading(&f.multipliers);
    let mut want = r.mult.clone();
    want.sort_by(|a, b| modulus(*b).total_cmp(&modulus(*a)));
    let mut e = [0.0; 3];
    for i in 0..3 {
        e[i] = dist(lead[i], want[i]);
    }
    (e, f.trivial_error)
}

#[test]
fn oracle_rows_and_trivial_multiplier() {
    let rows = oracle_rows();
    assert_eq!(rows.len(), 7);
    for r in &rows {
        let (e, triv) = oracle_errors(r, Evaluator::Jit);
        for i in 0..3 {
            assert!(
                e[i] <= ORACLE_BOUND[r.k][i],
                "K={} mu={}: multiplier {i} error {:e}",
                r.k,
                r.mu,
                e[i]
            );
        }
        assert!(
            triv <= TRIVIAL_BOUND[r.k],
            "K={} mu={}: trivial_error {triv:e}",
            r.k,
            r.mu
        );
    }
}

// ---------------------------------------------------------------------------------------
// Test 3: convergence in `nhist` and in the mesh, `mu = 0.3`.

/// `nhist` at the mesh `(40, 4)`, per `K`.
const NHIST_SEQ: [[usize; 4]; 2] = [[4, 6, 8, 12], [8, 12, 16, 24]];

/// `(ntst, ncol)` at the `nhist` of `MESH`.
const MESH_SEQ: [(usize, usize); 4] = [(10, 3), (20, 3), (20, 4), (40, 4)];

/// Below this the error of the leading multiplier no longer falls, because the other
/// discretization (the history for a mesh sequence, the mesh for a history sequence) and the
/// oracle's own accuracy set the floor. The measured floor for `K = 1` is 8.5e-8.
const ERROR_FLOOR: f64 = 2e-7;

fn lead_error(k: usize, ntst: usize, ncol: usize, nhist: usize) -> f64 {
    let r = row(k, 0.3);
    let mut m = dde(&osc_source(tau_k(k)), Evaluator::Jit);
    let (st, cyc) = osc_orbit(&mut m, 0.3, k, ntst, ncol, &opts());
    let f = floquet(&mut m, &st, &cyc, nhist, nhist + 1).unwrap();
    dist(leading(&f.multipliers)[0], r.mult[0])
}

#[test]
fn error_falls_with_nhist_and_with_the_mesh() {
    for k in 0..2 {
        let by_nhist: Vec<f64> = NHIST_SEQ[k]
            .iter()
            .map(|&nh| lead_error(k, 40, 4, nh))
            .collect();
        for w in by_nhist.windows(2) {
            assert!(
                w[1] <= 0.5 * w[0] || w[0] <= ERROR_FLOOR,
                "K={k}: nhist sequence {by_nhist:?}"
            );
        }
        let (_, _, nh) = MESH[k];
        let by_mesh: Vec<f64> = MESH_SEQ
            .iter()
            .map(|&(n, m)| lead_error(k, n, m, nh))
            .collect();
        for w in by_mesh.windows(2) {
            assert!(
                w[1] <= 0.5 * w[0] || w[0] <= ERROR_FLOOR,
                "K={k}: mesh sequence {by_mesh:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// Test 4: `count` takes a prefix.

#[test]
fn count_is_a_prefix() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let (st, cyc) = osc_orbit(&mut m, 0.3, 0, 20, 4, &opts());
    let full = floquet(&mut m, &st, &cyc, 12, 13).unwrap();
    for c in [1, 3, 10] {
        let part = floquet(&mut m, &st, &cyc, 12, c).unwrap();
        assert_eq!(part.multipliers.len(), c);
        assert_eq!(part.multipliers[..], full.multipliers[..c]);
        assert_eq!(part.trivial_error, full.trivial_error);
    }
}

// ---------------------------------------------------------------------------------------
// Test 5: the block triangular model.

/// Bound on `|mu - exp(-0.1 T)|` for the closest multiplier (SPEC 8m, test 5). The measured
/// error is 1e-15: the block is triangular, so that multiplier does not depend on the orbit.
const FORCED_BOUND: f64 = 1e-9;

fn forced_errors(which: Evaluator, o: &Options) -> (f64, [f64; 3], Vec<(f64, f64)>) {
    let mut m = dde(FORCED, which);
    let st = setup(&[0.3, tau_k(0)], 1, 40, 4);
    let cancel = AtomicBool::new(false);
    let start = profile(
        2,
        40,
        4,
        |c, s| {
            if c == 0 { (2.0 * PI * s).sin() } else { 0.0 }
        },
    );
    let c = dde_correct_cycle(
        &mut m,
        &st,
        &DdeCycleStart {
            profile: start,
            period: 2.0 * PI,
        },
        o,
        &cancel,
    )
    .unwrap();
    let cyc = DdeCycleStart {
        profile: c.profile,
        period: c.period,
    };
    let f = floquet(&mut m, &st, &cyc, 16, 34).unwrap();
    let want = (-0.1 * c.period).exp();
    let near = (0..f.multipliers.len())
        .min_by(|&i, &j| {
            dist(f.multipliers[i], (want, 0.0)).total_cmp(&dist(f.multipliers[j], (want, 0.0)))
        })
        .unwrap();
    let e_exp = dist(f.multipliers[near], (want, 0.0));
    let mut rest = f.multipliers.clone();
    rest.remove(near);
    let lead = leading(&rest);
    let r = row(0, 0.3);
    let mut e = [0.0; 3];
    for i in 0..3 {
        e[i] = dist(lead[i], r.mult[i]);
    }
    (e_exp, e, f.multipliers)
}

#[test]
fn block_triangular_model_has_the_closed_form_multiplier() {
    let (e_exp, e, _) = forced_errors(Evaluator::Jit, &opts());
    assert!(e_exp <= FORCED_BOUND, "exp(-0.1 T) error {e_exp:e}");
    for i in 0..3 {
        assert!(
            e[i] <= ORACLE_BOUND[0][i],
            "multiplier {i} error {:e}",
            e[i]
        );
    }
}

// ---------------------------------------------------------------------------------------
// Test 6: a literal delay.

#[test]
fn literal_delay_agrees_with_a_parameter_delay() {
    let mut a = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let (sta, cyca) = osc_orbit(&mut a, 0.3, 0, 20, 4, &tight());
    let fa = floquet(&mut a, &sta, &cyca, 12, 6).unwrap();
    let mut b = dde(OSCLIT, Evaluator::Jit);
    let stb = setup(&[0.3], 0, 20, 4);
    let cancel = AtomicBool::new(false);
    let c = dde_correct_cycle(
        &mut b,
        &stb,
        &DdeCycleStart {
            profile: perturbed(1, 20, 4),
            period: 6.0,
        },
        &tight(),
        &cancel,
    )
    .unwrap();
    let cycb = DdeCycleStart {
        profile: c.profile,
        period: c.period,
    };
    let fb = floquet(&mut b, &stb, &cycb, 12, 6).unwrap();
    for (x, y) in fa.multipliers.iter().zip(&fb.multipliers) {
        assert!(dist(*x, *y) <= 1e-10, "{x:?} against {y:?}");
    }
}

// ---------------------------------------------------------------------------------------
// Test 7: the two evaluators.

#[test]
fn both_evaluators_agree() {
    for (k, mu) in [(0, 0.1), (1, 0.1)] {
        let (ntst, ncol, nhist) = MESH[k];
        let mut out = Vec::new();
        for which in [Evaluator::Jit, Evaluator::Fallback] {
            let mut m = dde(&osc_source(tau_k(k)), which);
            let (st, cyc) = osc_orbit(&mut m, mu, k, ntst, ncol, &tight());
            out.push(floquet(&mut m, &st, &cyc, nhist, 6).unwrap());
        }
        for (x, y) in out[0].multipliers.iter().zip(&out[1].multipliers) {
            assert!(dist(*x, *y) <= 1e-9, "K={k}: {x:?} against {y:?}");
        }
    }
    let a = forced_errors(Evaluator::Jit, &tight()).2;
    let b = forced_errors(Evaluator::Fallback, &tight()).2;
    for (x, y) in a.iter().zip(&b).take(6) {
        assert!(dist(*x, *y) <= 1e-9, "FORCED: {x:?} against {y:?}");
    }
}

// ---------------------------------------------------------------------------------------
// Test 8: errors.

#[test]
fn bad_input_is_bad_problem() {
    let mut m = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    let (st, cyc) = osc_orbit(&mut m, 0.3, 0, 10, 3, &opts());
    let full = 4; // n (nhist + 1) for nhist = 3
    bad(floquet(&mut m, &st, &cyc, 0, 1), "nhist 0");
    bad(floquet(&mut m, &st, &cyc, 3, 0), "count 0");
    bad(floquet(&mut m, &st, &cyc, 3, full + 1), "count too large");
    bad(
        floquet(&mut m, &st, &cyc, 400, 1),
        "stage system over the limit",
    );
    bad(floquet(&mut m, &st, &cyc, usize::MAX, 1), "nhist overflows");

    let mut short = cyc.profile.clone();
    short.pop();
    bad(
        floquet(
            &mut m,
            &st,
            &DdeCycleStart {
                profile: short,
                period: cyc.period,
            },
            3,
            1,
        ),
        "profile length",
    );
    let mut nan = cyc.profile.clone();
    nan[2] = f64::NAN;
    bad(
        floquet(
            &mut m,
            &st,
            &DdeCycleStart {
                profile: nan,
                period: cyc.period,
            },
            3,
            1,
        ),
        "NaN in the profile",
    );
    for period in [0.0, -1.0, f64::NAN] {
        bad(
            floquet(
                &mut m,
                &st,
                &DdeCycleStart {
                    profile: cyc.profile.clone(),
                    period,
                },
                3,
                1,
            ),
            "period",
        );
    }
    for (ntst, ncol) in [(0, 3), (10, 1), (10, 8)] {
        let s = setup(&[0.3, tau_k(0)], 1, ntst, ncol);
        bad(floquet(&mut m, &s, &cyc, 3, 1), "mesh");
    }
    // `build_dde` refuses a model without a delay call, so the test empties the list of a
    // built model.
    let mut nodelay = dde(&osc_source(tau_k(0)), Evaluator::Jit);
    nodelay.delays.clear();
    bad(floquet(&mut nodelay, &st, &cyc, 3, 1), "no delay slot");

    let cancel = AtomicBool::new(true);
    let r = dde_floquet_multipliers(
        &mut m,
        &st,
        &cyc,
        &DdeFloquetOptions { nhist: 3, count: 1 },
        &cancel,
    );
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
    assert!(cancel.load(Ordering::Relaxed));
}

// ---------------------------------------------------------------------------------------
// The measured tables of SPEC section 8m.

#[test]
#[ignore = "prints the tables of SPEC section 8m"]
fn print_tables() {
    println!("oracle rows (K, mu): errors of the three leading multipliers, trivial_error");
    for r in oracle_rows() {
        let (e, t) = oracle_errors(&r, Evaluator::Jit);
        println!(
            "K={} mu={:.1}  {:.2e} {:.2e} {:.2e}  trivial {:.2e}",
            r.k, r.mu, e[0], e[1], e[2], t
        );
    }
    for k in 0..2 {
        println!("K={k} by nhist at (40,4):");
        for &nh in &NHIST_SEQ[k] {
            println!("  nhist {nh}: {:.2e}", lead_error(k, 40, 4, nh));
        }
        let (_, _, nh) = MESH[k];
        println!("K={k} by mesh at nhist {nh}:");
        for &(n, m) in &MESH_SEQ {
            println!("  ({n},{m}): {:.2e}", lead_error(k, n, m, nh));
        }
    }
    let (e_exp, e, _) = forced_errors(Evaluator::Jit, &opts());
    println!(
        "FORCED: exp(-0.1 T) error {e_exp:.2e}, block errors {:.2e} {:.2e} {:.2e}",
        e[0], e[1], e[2]
    );
}
