//! Codim-2 labels on the LP, PD and NS curves of fixed points of maps (SPEC section 8x,
//! acceptance tests 2, 3, 5 and 6). Each case states its map and the closed form of the
//! curve and of its tests, and computes the expected values in the test. Every case runs in
//! both directions with both evaluators, and the special points of the two directions
//! together must be exactly the listed ones, located to 1e-8 in the two curve parameters.

mod common;

use common::load;
use pc_engine::{
    Branch, CurveSetup, EngineError, Evaluator, ModelEval, Options, Point, SpecialPoint,
    StopReason, build, continue_map_lp_curve, continue_map_ns_curve, continue_map_pd_curve,
};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

type Follow = fn(
    &mut dyn ModelEval,
    &CurveSetup,
    &Options,
    &AtomicBool,
    &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError>;

#[derive(Clone, Copy)]
enum Curve {
    Lp,
    Pd,
    Ns,
}

impl Curve {
    fn follow(self) -> Follow {
        match self {
            Curve::Lp => continue_map_lp_curve,
            Curve::Pd => continue_map_pd_curve,
            Curve::Ns => continue_map_ns_curve,
        }
    }

    /// The number of codim-2 test values on a point of this curve.
    fn slots(self) -> usize {
        match self {
            Curve::Ns => 8,
            _ => 4,
        }
    }
}

/// The codim-2 slots of `Point.codim2` (SPEC 8x, table of test functions).
const LP_LPPD: usize = 0;
const LP_LPNS: usize = 1;
const LP_CP: usize = 2;
const LP_R1: usize = 3;
const PD_LPPD: usize = 0;
const PD_PDNS: usize = 1;
const PD_GPD: usize = 2;
const PD_R2: usize = 3;
const NS_R1: usize = 0;
const NS_R2: usize = 1;
const NS_R3: usize = 2;
const NS_R4: usize = 3;
const NS_LPNS: usize = 4;
const NS_PDNS: usize = 5;
const NS_NSNS: usize = 6;
const NS_CH: usize = 7;

/// K = R(1) of `map_lpns.ode`, `map_pdns.ode` and `map_nsns.ode`, as (p, q, r).
fn rot1() -> [f64; 3] {
    [1f64.cos(), -1f64.sin(), 1f64.sin()]
}

/// The saddle K = [cosh 0.7, sinh 0.7; sinh 0.7, cosh 0.7], as (p, q, r).
fn saddle() -> [f64; 3] {
    [0.7f64.cosh(), 0.7f64.sinh(), 0.7f64.sinh()]
}

fn model(name: &str, ev: &Evaluator) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    build(&m, &info, ev.clone()).unwrap()
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:e}, want {want:e}, error {:e}",
        (got - want).abs()
    );
}

fn nan(got: f64, what: &str) {
    assert!(got.is_nan(), "{what}: got {got:e}, want NaN");
}

fn has_multipliers(got: &[(f64, f64)], want: &[(f64, f64)], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?}");
    for w in want {
        assert!(
            got.iter()
                .any(|g| (g.0 - w.0).abs() <= tol && (g.1 - w.1).abs() <= tol),
            "{what}: no multiplier near {w:?} in {got:?}"
        );
    }
}

/// Multipliers r e^{+-i th}.
fn pair(r: f64, th: f64) -> [(f64, f64); 2] {
    [(r * th.cos(), r * th.sin()), (r * th.cos(), -r * th.sin())]
}

/// The multipliers of K scaled by g: g (p +- sqrt(q r)).
fn k_pair(k: [f64; 3], g: f64) -> [(f64, f64); 2] {
    let [p, q, r] = k;
    let d = q * r;
    if d >= 0.0 {
        [(g * (p + d.sqrt()), 0.0), (g * (p - d.sqrt()), 0.0)]
    } else {
        [(g * p, g * (-d).sqrt()), (g * p, -g * (-d).sqrt())]
    }
}

/// Checks a point of a curve against its closed form.
type PointCheck = Box<dyn Fn(&Point, &str)>;

struct Case {
    what: String,
    curve: Curve,
    model: &'static str,
    setup: CurveSetup,
    bounds: (f64, f64),
    /// The special points of both directions together: label and the values of the
    /// parameters `setup.free[0]` and `setup.free[1]`.
    want: Vec<(&'static str, f64, f64)>,
    /// Labels of each direction, [default, -1], where the test states them.
    per_direction: Option<[Vec<&'static str>; 2]>,
    /// Prefixes of the warnings the two runs must give together; no other warning is allowed.
    warn: Vec<&'static str>,
    /// Checks a point of the curve against the closed form.
    check: PointCheck,
}

fn sorted(mut v: Vec<&str>) -> Vec<&str> {
    v.sort_unstable();
    v
}

fn only<'a>(specials: &'a [SpecialPoint], label: &str, what: &str) -> &'a SpecialPoint {
    let found: Vec<&SpecialPoint> = specials.iter().filter(|s| s.label == label).collect();
    assert_eq!(
        found.len(),
        1,
        "{what}: {label} points among {}",
        specials.len()
    );
    found[0]
}

fn run(case: &Case) {
    let n = case.setup.u0.len();
    let n_codim2 = case.curve.slots();
    for ev in &BOTH {
        let mut specials: Vec<SpecialPoint> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        for (k, direction) in [1.0, -1.0].into_iter().enumerate() {
            let what = format!("{} {ev:?} direction {direction}", case.what);
            let opts = Options {
                direction,
                bounds: case.bounds,
                ..Options::default()
            };
            let mut m = model(case.model, ev);
            let br = (case.curve.follow())(
                m.as_mut(),
                &case.setup,
                &opts,
                &AtomicBool::new(false),
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(br.stop, StopReason::Bounds, "{what}");
            assert!(br.points.len() > 5, "{what}: {} points", br.points.len());
            let pts = br.points.iter().chain(br.specials.iter().map(|s| &s.point));
            for p in pts {
                let at = format!("{what} at alpha = {}", p.alpha);
                assert_eq!(p.codim2.len(), n_codim2, "{at}");
                assert!(p.tests.iter().all(|t| t.is_nan()), "{at}: {:?}", p.tests);
                (case.check)(p, &at);
            }
            for s in &br.specials {
                assert_eq!(s.extras.coefficient, None, "{what}: {}", s.label);
            }
            if let Some(per) = &case.per_direction {
                let got = sorted(br.specials.iter().map(|s| s.label.as_str()).collect());
                assert_eq!(got, sorted(per[k].clone()), "{what}");
            }
            specials.extend(br.specials);
            warnings.extend(br.warnings);
        }
        let what = format!("{} {ev:?}", case.what);
        let got = sorted(specials.iter().map(|s| s.label.as_str()).collect());
        let want = sorted(case.want.iter().map(|w| w.0).collect());
        assert_eq!(got, want, "{what}: warnings {warnings:?}");
        for &(label, p0, p1) in &case.want {
            let s = only(&specials, label, &what);
            close(s.point.u[n], p0, 1e-8, &format!("{what} {label} free[0]"));
            close(s.point.alpha, p1, 1e-8, &format!("{what} {label} free[1]"));
        }
        for w in &warnings {
            assert!(
                case.warn.iter().any(|p| w.starts_with(p)),
                "{what}: unexpected warning {w:?}"
            );
        }
        for p in &case.warn {
            assert!(
                warnings.iter().any(|w| w.starts_with(p)),
                "{what}: no warning {p:?} in {warnings:?}"
            );
        }
    }
}

// Test 2. The generalized Henon map (x1, x2) -> (x2, alpha - beta x1 - x2^2 + x1 x2 / 2) has
// the fixed points x1 = x2 = x with alpha = (1 + beta) x + x^2 / 2 and
// F_x = [[0, 1], [0.5 x - beta, -1.5 x]], so the multipliers are the roots of
// mu^2 + 1.5 x mu + beta - 0.5 x. The second derivative of the second component is
// [[0, 0.5], [0.5, -2]], and the third is 0.

/// LP curve: beta = -1 - x, alpha = -x^2/2, multipliers 1 and -1 - 1.5 x. With
/// v = (1, 1)/sqrt 2 and w = (1 + 1.5 x, 1)/|.|, B(v, v) = (0, -0.5), so
/// |CP| = 0.5/|w| and |R1| = |2 + 1.5 x|/(sqrt 2 |w|), with |w| = sqrt((1 + 1.5 x)^2 + 1).
/// Points store u = (x1, x2, alpha) and alpha = beta.
#[test]
fn test2_henon_gen_lp_curve() {
    run(&Case {
        what: "henon_gen LP".into(),
        curve: Curve::Lp,
        model: "map_henon_gen",
        setup: CurveSetup::new([0, 1], vec![-0.125, -0.5], vec![-0.5, -0.5]),
        bounds: (-1.5, 1.0),
        want: vec![("LPPD", 0.0, -1.0), ("R1", -8.0 / 9.0, 1.0 / 3.0)],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            let x = p.u[0];
            close(p.u[1], x, 1e-8, &format!("{at} x2"));
            close(p.alpha, -1.0 - x, 1e-8, &format!("{at} beta"));
            close(p.u[2], -x * x / 2.0, 1e-8, &format!("{at} alpha"));
            has_multipliers(
                &p.eigenvalues,
                &[(1.0, 0.0), (-1.0 - 1.5 * x, 0.0)],
                1e-7,
                at,
            );
            let wn = ((1.0 + 1.5 * x).powi(2) + 1.0).sqrt();
            close(p.codim2[LP_LPPD], -1.5 * x, 1e-7, &format!("{at} LPPD"));
            nan(p.codim2[LP_LPNS], &format!("{at} LPNS"));
            close(p.codim2[LP_CP].abs(), 0.5 / wn, 1e-7, &format!("{at} CP"));
            let r1 = (2.0 + 1.5 * x).abs() / (2f64.sqrt() * wn);
            close(p.codim2[LP_R1].abs(), r1, 1e-7, &format!("{at} R1"));
        }),
    });
}

/// PD curve: beta = 2x - 1, alpha = 5x^2/2, multipliers -1 and 1 - 1.5 x. With
/// v = (1, -1)/sqrt 2 and w = (1 - 1.5 x, -1)/|.|, B(v, v) = (0, -1.5) and (I - A) h = B(v, v)
/// gives h = -(1, 1)/(2x), so |GPD| = 1/(2 sqrt 2 |x| |w|) with a pole at the LPPD point
/// x = 0, and |R2| = |2 - 1.5 x|/(sqrt 2 |w|), with |w| = sqrt((1.5 x - 1)^2 + 1).
#[test]
fn test2_henon_gen_pd_curve() {
    run(&Case {
        what: "henon_gen PD".into(),
        curve: Curve::Pd,
        model: "map_henon_gen",
        setup: CurveSetup::new([0, 1], vec![0.625, 0.0], vec![0.5, 0.5]),
        bounds: (-2.0, 2.5),
        want: vec![("LPPD", 0.0, -1.0), ("R2", 40.0 / 9.0, 5.0 / 3.0)],
        per_direction: None,
        warn: vec!["could not locate GPD between points "],
        check: Box::new(|p, at| {
            let x = p.u[0];
            close(p.u[1], x, 1e-8, &format!("{at} x2"));
            close(p.alpha, 2.0 * x - 1.0, 1e-8, &format!("{at} beta"));
            close(p.u[2], 2.5 * x * x, 1e-8, &format!("{at} alpha"));
            has_multipliers(
                &p.eigenvalues,
                &[(-1.0, 0.0), (1.0 - 1.5 * x, 0.0)],
                1e-7,
                at,
            );
            let wn = ((1.5 * x - 1.0).powi(2) + 1.0).sqrt();
            close(p.codim2[PD_LPPD], -1.5 * x, 1e-7, &format!("{at} LPPD"));
            nan(p.codim2[PD_PDNS], &format!("{at} PDNS"));
            let r2 = (2.0 - 1.5 * x).abs() / (2f64.sqrt() * wn);
            close(p.codim2[PD_R2].abs(), r2, 1e-7, &format!("{at} R2"));
            if x.abs() > 1e-3 {
                let g = 1.0 / (2.0 * 2f64.sqrt() * x.abs() * wn);
                close(p.codim2[PD_GPD].abs(), g, 1e-6 * g, &format!("{at} GPD"));
            }
        }),
    });
}

/// NS curve: det F_x = beta - 0.5 x = 1, so beta = 1 + x/2, alpha = 2x + x^2 and
/// kappa = -3x/4. Both multipliers are critical, so the rest set is empty and LPNS, PDNS and
/// NSNS are NaN. Points store u = (x1, x2, alpha, kappa) and alpha = beta.
#[test]
fn test2_henon_gen_ns_curve() {
    run(&Case {
        what: "henon_gen NS".into(),
        curve: Curve::Ns,
        model: "map_henon_gen",
        setup: CurveSetup::new([0, 1], vec![0.69, 1.15], vec![0.3, 0.3]),
        bounds: (0.0, 2.0),
        want: vec![
            ("R1", -8.0 / 9.0, 1.0 / 3.0),
            ("R4", 0.0, 1.0),
            ("R3", 16.0 / 9.0, 4.0 / 3.0),
            ("R2", 40.0 / 9.0, 5.0 / 3.0),
        ],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            let x = p.u[0];
            let k = -0.75 * x;
            close(p.u[1], x, 1e-8, &format!("{at} x2"));
            close(p.alpha, 1.0 + x / 2.0, 1e-8, &format!("{at} beta"));
            close(p.u[2], 2.0 * x + x * x, 1e-8, &format!("{at} alpha"));
            close(p.u[3], k, 1e-8, &format!("{at} kappa"));
            for (i, s) in [(NS_R1, -1.0), (NS_R2, 1.0), (NS_R3, 0.5), (NS_R4, 0.0)] {
                close(p.codim2[i], k + s, 1e-8, &format!("{at} R slot {i}"));
            }
            for i in [NS_LPNS, NS_PDNS, NS_NSNS] {
                nan(p.codim2[i], &format!("{at} slot {i}"));
            }
            let ch = p.codim2[NS_CH];
            assert!(ch.is_nan() || ch > 0.0, "{at}: CH = {ch:e}");
            if k.abs() >= 1.0 {
                nan(ch, &format!("{at} CH"));
            }
        }),
    });
}

// Test 3.

/// x -> x + a1 + a2 x - x^3, y -> y/2: the LP curve is a1 = -2x^3, a2 = 3x^2, with v = w = e1,
/// B(v, v) = (-6x, 0), so CP = +-6x, LPPD = 3/2 and R1 = +-1. Points store u = (x, y, a1).
#[test]
fn test3_cusp() {
    run(&Case {
        what: "cusp LP".into(),
        curve: Curve::Lp,
        model: "map_cusp_cp",
        setup: CurveSetup::new([0, 1], vec![0.25, 0.75], vec![-0.5, 0.0]),
        bounds: (-1.0, 1.0),
        want: vec![("CP", 0.0, 0.0)],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            let x = p.u[0];
            close(p.u[1], 0.0, 1e-8, &format!("{at} y"));
            close(p.u[2], -2.0 * x.powi(3), 1e-8, &format!("{at} a1"));
            close(p.alpha, 3.0 * x * x, 1e-8, &format!("{at} a2"));
            has_multipliers(&p.eigenvalues, &[(1.0, 0.0), (0.5, 0.0)], 1e-7, at);
            close(p.codim2[LP_LPPD], 1.5, 1e-7, &format!("{at} LPPD"));
            nan(p.codim2[LP_LPNS], &format!("{at} LPNS"));
            close(
                p.codim2[LP_CP].abs(),
                6.0 * x.abs(),
                1e-7,
                &format!("{at} CP"),
            );
            close(p.codim2[LP_R1].abs(), 1.0, 1e-8, &format!("{at} R1"));
        }),
    });
}

/// x -> -(1 + a1) x + x^2 + a2 x^3, y -> y/2: the PD curve is x = 0, a1 = 0, and
/// B(e1, e1) = 2, C = 6 a2 and h = e1 give GPD = R2 (a2 + 1), LPPD = -1/2 and R2 = +-1.
#[test]
fn test3_generalized_flip() {
    run(&Case {
        what: "gpd PD".into(),
        curve: Curve::Pd,
        model: "map_gpd",
        setup: CurveSetup::new([0, 1], vec![0.0, -2.0], vec![0.0, 0.0]),
        bounds: (-3.0, 0.0),
        want: vec![("GPD", 0.0, -1.0)],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            for (i, &u) in p.u.iter().enumerate().take(3) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            has_multipliers(&p.eigenvalues, &[(-1.0, 0.0), (0.5, 0.0)], 1e-8, at);
            close(p.codim2[PD_LPPD], -0.5, 1e-8, &format!("{at} LPPD"));
            nan(p.codim2[PD_PDNS], &format!("{at} PDNS"));
            let r2 = p.codim2[PD_R2];
            close(r2.abs(), 1.0, 1e-8, &format!("{at} R2"));
            close(
                p.codim2[PD_GPD],
                r2 * (p.alpha + 1.0),
                1e-7,
                &format!("{at} GPD"),
            );
        }),
    });
}

/// z -> (1 + a1) R(1) z + |z|^2 R(1) (a2 I + J) z: c1 is a positive multiple of
/// e^{i} (a2 + i), so CH = a2/sqrt(a2^2 + 1). Points store u = (x, y, a1, kappa).
#[test]
fn test3_chenciner() {
    run(&Case {
        what: "ch NS".into(),
        curve: Curve::Ns,
        model: "map_ch",
        setup: CurveSetup::new([0, 1], vec![0.0, -0.5], vec![0.0, 0.0]),
        bounds: (-1.0, 1.0),
        want: vec![("CH", 0.0, 0.0)],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            let a2 = p.alpha;
            for (i, &u) in p.u.iter().enumerate().take(3) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            close(p.u[3], 1f64.cos(), 1e-8, &format!("{at} kappa"));
            has_multipliers(&p.eigenvalues, &pair(1.0, 1.0), 1e-8, at);
            let ch = a2 / (a2 * a2 + 1.0).sqrt();
            close(p.codim2[NS_CH], ch, 1e-7, &format!("{at} CH"));
            for i in [NS_LPNS, NS_PDNS, NS_NSNS] {
                nan(p.codim2[i], &format!("{at} slot {i}"));
            }
        }),
    });
}

/// x -> x + a1 - x^2 next to z -> (1 + a2 - |z|^2) K z: the LP curve is x = z = 0, a1 = 0,
/// with the rest multipliers (1 + a2) times those of K, so LPNS = (1 + a2)^2 det K - 1 =
/// (1 + a2)^2 - 1 for both K. Points store u = (x, y, w, a1).
fn lpns_lp(k: [f64; 3], want: Vec<(&'static str, f64, f64)>, what: &str) -> Case {
    let mut params = vec![0.0, -0.2];
    params.extend(k);
    Case {
        what: what.into(),
        curve: Curve::Lp,
        model: "map_lpns",
        setup: CurveSetup::new([0, 1], params, vec![0.0; 3]),
        bounds: (-0.3, 0.3),
        want,
        per_direction: None,
        warn: vec![],
        check: Box::new(move |p, at| {
            let a2 = p.alpha;
            for (i, &u) in p.u.iter().enumerate().take(4) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            let [m1, m2] = k_pair(k, 1.0 + a2);
            has_multipliers(&p.eigenvalues, &[(1.0, 0.0), m1, m2], 1e-8, at);
            let lpns = (1.0 + a2).powi(2) - 1.0;
            close(p.codim2[LP_LPNS], lpns, 1e-8, &format!("{at} LPNS"));
        }),
    }
}

#[test]
fn test3_lpns_on_the_lp_curve() {
    run(&lpns_lp(rot1(), vec![("LPNS", 0.0, 0.0)], "lpns LP"));
    run(&lpns_lp(saddle(), vec![], "lpns LP saddle"));
}

/// The NS curve of `map_lpns.ode` with free [a2, a1] is a1 = x^2, a2 = 0, kappa = cos 1,
/// with the rest multiplier 1 - 2x, so LPNS = -2x, PDNS = 2 - 2x and CH = -1. Points store
/// u = (x, y, w, a2, kappa) and alpha = a1.
#[test]
fn test3_lpns_on_the_ns_curve() {
    let mut params = vec![0.25, 0.0];
    params.extend(rot1());
    run(&Case {
        what: "lpns NS".into(),
        curve: Curve::Ns,
        model: "map_lpns",
        setup: CurveSetup::new([1, 0], params, vec![-0.5, 0.0, 0.0]),
        bounds: (-1.0, 0.5),
        want: vec![("LPNS", 0.0, 0.0)],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            let x = p.u[0];
            close(p.alpha, x * x, 1e-8, &format!("{at} a1"));
            for (i, &u) in p.u.iter().enumerate().take(4).skip(1) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            close(p.u[4], 1f64.cos(), 1e-8, &format!("{at} kappa"));
            let [m1, m2] = pair(1.0, 1.0);
            has_multipliers(&p.eigenvalues, &[(1.0 - 2.0 * x, 0.0), m1, m2], 1e-8, at);
            close(p.codim2[NS_LPNS], -2.0 * x, 1e-8, &format!("{at} LPNS"));
            close(
                p.codim2[NS_PDNS],
                2.0 - 2.0 * x,
                1e-8,
                &format!("{at} PDNS"),
            );
            nan(p.codim2[NS_NSNS], &format!("{at} NSNS"));
            // At x = 0 the rest multiplier is 1 and the solve for h1100 is singular.
            if x.abs() > 1e-6 {
                close(p.codim2[NS_CH], -1.0, 1e-7, &format!("{at} CH"));
            }
        }),
    });
}

/// x -> -(1 + a1) x + x^2 next to the z block of `map_lpns.ode`: the PD curve is x = z = 0,
/// a1 = 0, with PDNS = (1 + a2)^2 - 1 for both K and GPD = +-1.
fn pdns_pd(k: [f64; 3], want: Vec<(&'static str, f64, f64)>, what: &str) -> Case {
    let mut params = vec![0.0, -0.2];
    params.extend(k);
    Case {
        what: what.into(),
        curve: Curve::Pd,
        model: "map_pdns",
        setup: CurveSetup::new([0, 1], params, vec![0.0; 3]),
        bounds: (-0.3, 0.3),
        want,
        per_direction: None,
        warn: vec![],
        check: Box::new(move |p, at| {
            let a2 = p.alpha;
            for (i, &u) in p.u.iter().enumerate().take(4) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            let [m1, m2] = k_pair(k, 1.0 + a2);
            has_multipliers(&p.eigenvalues, &[(-1.0, 0.0), m1, m2], 1e-8, at);
            let pdns = (1.0 + a2).powi(2) - 1.0;
            close(p.codim2[PD_PDNS], pdns, 1e-8, &format!("{at} PDNS"));
            close(p.codim2[PD_GPD].abs(), 1.0, 1e-7, &format!("{at} GPD"));
        }),
    }
}

#[test]
fn test3_pdns_on_the_pd_curve() {
    run(&pdns_pd(rot1(), vec![("PDNS", 0.0, 0.0)], "pdns PD"));
    run(&pdns_pd(saddle(), vec![], "pdns PD saddle"));
}

/// The NS curve of `map_pdns.ode` with free [a2, a1] is x = z = 0, a2 = 0, with the rest
/// multiplier -(1 + a1), so PDNS = -a1, LPNS = -2 - a1 and CH = -1. Points store
/// u = (x, y, w, a2, kappa) and alpha = a1.
#[test]
fn test3_pdns_on_the_ns_curve() {
    let mut params = vec![-0.2, 0.0];
    params.extend(rot1());
    run(&Case {
        what: "pdns NS".into(),
        curve: Curve::Ns,
        model: "map_pdns",
        setup: CurveSetup::new([1, 0], params, vec![0.0; 3]),
        bounds: (-0.3, 0.3),
        want: vec![("PDNS", 0.0, 0.0)],
        per_direction: None,
        warn: vec![],
        check: Box::new(|p, at| {
            let a1 = p.alpha;
            for (i, &u) in p.u.iter().enumerate().take(4) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            close(p.u[4], 1f64.cos(), 1e-8, &format!("{at} kappa"));
            let [m1, m2] = pair(1.0, 1.0);
            has_multipliers(&p.eigenvalues, &[(-1.0 - a1, 0.0), m1, m2], 1e-8, at);
            close(p.codim2[NS_PDNS], -a1, 1e-8, &format!("{at} PDNS"));
            close(p.codim2[NS_LPNS], -2.0 - a1, 1e-8, &format!("{at} LPNS"));
            close(p.codim2[NS_CH], -1.0, 1e-7, &format!("{at} CH"));
        }),
    });
}

/// z -> (1 + a1 - |z|^2) R(2) z next to w -> (1 + a2) K w: the NS curve is z = w = 0,
/// a1 = 0, kappa = cos 2, with the rest pair (1 + a2) times the multipliers of K, so
/// NSNS = (1 + a2)^2 - 1 for both K and CH = -1. Points store u = (x, y, u, v, a1, kappa).
fn nsns(k: [f64; 3], want: Vec<(&'static str, f64, f64)>, what: &str) -> Case {
    let mut params = vec![0.0, -0.2];
    params.extend(k);
    Case {
        what: what.into(),
        curve: Curve::Ns,
        model: "map_nsns",
        setup: CurveSetup::new([0, 1], params, vec![0.0; 4]),
        bounds: (-0.3, 0.3),
        want,
        per_direction: None,
        warn: vec![],
        check: Box::new(move |p, at| {
            let a2 = p.alpha;
            for (i, &u) in p.u.iter().enumerate().take(5) {
                close(u, 0.0, 1e-8, &format!("{at} u[{i}]"));
            }
            close(p.u[5], 2f64.cos(), 1e-8, &format!("{at} kappa"));
            let [m1, m2] = pair(1.0, 2.0);
            let [m3, m4] = k_pair(k, 1.0 + a2);
            has_multipliers(&p.eigenvalues, &[m1, m2, m3, m4], 1e-8, at);
            let nsns = (1.0 + a2).powi(2) - 1.0;
            close(p.codim2[NS_NSNS], nsns, 1e-8, &format!("{at} NSNS"));
            close(p.codim2[NS_CH], -1.0, 1e-7, &format!("{at} CH"));
        }),
    }
}

#[test]
fn test3_nsns() {
    run(&nsns(rot1(), vec![("NSNS", 0.0, 0.0)], "nsns NS"));
    run(&nsns(saddle(), vec![], "nsns NS saddle"));
}

/// x -> x + a1 - x^2 + |z|^2/2 coupled to z -> (1 + a2 + x) R(1) z + |z|^2 R(1) (-I + J) z:
/// the NS curve with free [a2, a1] is a1 = x^2, a2 = -x, kappa = cos 1, with the rest
/// multiplier 1 - 2x, so LPNS = -2x and PDNS = 2 - 2x. The coupling through x adds
/// (1/(2x)) e^{i} to c1, so CH = s/sqrt(s^2 + 4) with s = 0.5/x - 2, which jumps at x = 0
/// and crosses 0 at x = 1/4. Points store u = (x, y, w, a2, kappa) and alpha = a1.
#[test]
fn test3_lpns_and_ch() {
    run(&Case {
        what: "lpns_ch NS".into(),
        curve: Curve::Ns,
        model: "map_lpns_ch",
        setup: CurveSetup::new([1, 0], vec![0.25, 0.5], vec![-0.5, 0.0, 0.0]),
        bounds: (-1.0, 0.3),
        want: vec![("LPNS", 0.0, 0.0), ("CH", -0.25, 1.0 / 16.0)],
        per_direction: None,
        warn: vec!["could not locate CH between points "],
        check: Box::new(|p, at| {
            let x = p.u[0];
            close(p.alpha, x * x, 1e-8, &format!("{at} a1"));
            close(p.u[3], -x, 1e-8, &format!("{at} a2"));
            close(p.u[1], 0.0, 1e-8, &format!("{at} y"));
            close(p.u[2], 0.0, 1e-8, &format!("{at} w"));
            close(p.u[4], 1f64.cos(), 1e-8, &format!("{at} kappa"));
            let [m1, m2] = pair(1.0, 1.0);
            has_multipliers(&p.eigenvalues, &[(1.0 - 2.0 * x, 0.0), m1, m2], 1e-8, at);
            close(p.codim2[NS_LPNS], -2.0 * x, 1e-8, &format!("{at} LPNS"));
            close(
                p.codim2[NS_PDNS],
                2.0 - 2.0 * x,
                1e-8,
                &format!("{at} PDNS"),
            );
            if x.abs() > 1e-3 {
                let s = 0.5 / x - 2.0;
                let ch = s / (s * s + 4.0).sqrt();
                close(p.codim2[NS_CH], ch, 1e-7, &format!("{at} CH"));
            }
        }),
    });
}

/// Test 5. AUTO's Henon form (x, y) -> (y, a - b x - y^2) with free [b, a]: the NS curve is
/// b = 1, a = (x + 1)^2 - 1, kappa = -x. The map is area preserving there, so CH is
/// rounding noise and gives no special and no warning.
#[test]
fn test5_henon_auto_r_points() {
    run(&Case {
        what: "henon_auto NS".into(),
        curve: Curve::Ns,
        model: "map_henon_auto",
        setup: CurveSetup::new([1, 0], vec![0.69, 1.0], vec![0.3, 0.3]),
        bounds: (-1.5, 3.5),
        want: vec![
            ("R3", 1.0, 1.25),
            ("R2", 1.0, 3.0),
            ("R4", 1.0, 0.0),
            ("R1", 1.0, -1.0),
        ],
        per_direction: Some([vec!["R3", "R2"], vec!["R4", "R1"]]),
        warn: vec![],
        check: Box::new(|p, at| {
            let x = p.u[0];
            close(p.u[3], -x, 1e-8, &format!("{at} kappa"));
            close(p.codim2[NS_R3], 0.5 - x, 1e-8, &format!("{at} R3"));
        }),
    });
}

/// Test 6. `map_ns4.ode` with free [a, th]: the NS curve is a = 0, kappa = cos th, with the
/// rest pair 0.95 e^{+-1.6 i}, so LPNS, PDNS and NSNS are constant, and the normal form is
/// z -> e^{i th} z (1 - 0.4 |z|^2) up to a change of coordinates, so CH = -1. Points store
/// u = (x, y, z, w, a, kappa) and alpha = th.
#[test]
fn test6_ns4() {
    // a, s, th, rho, om, k, m with a = 0 on the curve.
    let params = vec![0.0, -0.4, 1.1, 0.95, 1.6, 0.9, 0.3];
    // The rest pair mu = 0.95 e^{+-1.6 i}: LPNS = |mu - 1|^2, PDNS = |mu + 1|^2 and
    // NSNS = |mu|^2 - 1.
    let (rho, om) = (0.95f64, 1.6f64);
    let lpns = rho * rho - 2.0 * rho * om.cos() + 1.0;
    let pdns = rho * rho + 2.0 * rho * om.cos() + 1.0;
    let nsns = rho * rho - 1.0;
    run(&Case {
        what: "ns4 NS".into(),
        curve: Curve::Ns,
        model: "map_ns4",
        setup: CurveSetup::new([0, 2], params, vec![0.0; 4]),
        bounds: (0.5, 2.5),
        want: vec![("R4", 0.0, PI / 2.0), ("R3", 0.0, 2.0 * PI / 3.0)],
        per_direction: Some([vec![], vec!["R4", "R3"]]),
        warn: vec![],
        check: Box::new(move |p, at| {
            close(p.u[5], p.alpha.cos(), 1e-8, &format!("{at} kappa"));
            close(p.codim2[NS_LPNS], lpns, 1e-8, &format!("{at} LPNS"));
            close(p.codim2[NS_PDNS], pdns, 1e-8, &format!("{at} PDNS"));
            close(p.codim2[NS_NSNS], nsns, 1e-8, &format!("{at} NSNS"));
            // CH is refused at the 1:3 point (kappa = -1/2) and finite elsewhere.
            if (p.u[5] + 0.5).abs() > 1e-6 {
                close(p.codim2[NS_CH], -1.0, 1e-6, &format!("{at} CH"));
            }
        }),
    });
}
