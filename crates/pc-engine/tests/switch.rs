//! Switching branches at BP points, on both evaluators.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, EpSetup, Evaluator, ModelEval, Options, Point, SpecialPoint, StopReason,
    build, continue_ep, special_index, switch_bp,
};
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn model_of(name: &str, which: Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which.clone()).unwrap()
}

fn run(
    name: &str,
    which: Evaluator,
    setup: &EpSetup,
    opts: &Options,
) -> Result<Branch, EngineError> {
    let mut m = model_of(name, which.clone());
    continue_ep(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
}

fn first_opts(hi: f64) -> Options {
    Options {
        h0: 0.07,
        hmax: 0.07,
        bounds: (f64::NEG_INFINITY, hi),
        ..Options::default()
    }
}

fn switched_opts(direction: f64) -> Options {
    Options {
        h0: 0.07,
        hmax: 0.07,
        max_steps: 30,
        direction,
        bounds: (-5.0, 5.0),
        ..Options::default()
    }
}

/// The first branch of a model through its BP and the switch setup.
fn bp_setup(name: &str, which: Evaluator, setup: &EpSetup, hi: f64) -> (Branch, usize, EpSetup) {
    let b = run(name, which.clone(), setup, &first_opts(hi)).unwrap();
    let k = special_index(&b, "BP1").unwrap_or_else(|| panic!("{name}: no BP"));
    let mut m = model_of(name, which.clone());
    let next = switch_bp(m.as_mut(), setup, &b, k, &Options::default()).unwrap();
    (b, k, next)
}

fn no_bp(b: &Branch) {
    let labels: Vec<_> = b.specials.iter().map(|s| s.label.as_str()).collect();
    assert!(!labels.contains(&"BP"), "spurious BP: {labels:?}");
}

#[test]
fn transcritical_switches_to_x_equals_alpha() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let (b, k, next) = bp_setup("transcritical", which.clone(), &setup, 1.0);
        let bp = &b.specials[k].point;
        assert!(next.start_is_special && next.tangent0.is_some());
        assert!((next.params[0] - bp.alpha).abs() < 1e-15 && next.u0 == bp.u);
        let nb = run("transcritical", which.clone(), &next, &switched_opts(1.0)).unwrap();
        assert!(nb.points.len() > 20, "{which:?}: {}", nb.points.len());
        for p in &nb.points {
            assert!((p.u[0] - p.alpha).abs() < 1e-10, "{which:?}: {p:?}");
        }
        no_bp(&nb);
    }
}

#[test]
fn pitchfork_switches_to_x_squared_equals_alpha() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let (_, _, next) = bp_setup("pitchfork", which.clone(), &setup, 1.0);
        let t = next.tangent0.clone().unwrap();
        assert!(t[1].abs() < 1e-8 && t[0] > 0.9, "{t:?}");
        let nb = run("pitchfork", which.clone(), &next, &switched_opts(1.0)).unwrap();
        let d = reach(&nb, &next);
        assert!(d > 1.0, "{d}");
        assert!(nb.points.len() > 20);
        for p in &nb.points {
            assert!(
                (p.u[0] * p.u[0] - p.alpha).abs() < 1e-10,
                "{which:?}: {p:?}"
            );
        }
        // The first point sits on the BP, where psi_BP is about 0. That is not reported.
        no_bp(&nb);
    }
}

#[test]
fn lorenz_switches_to_the_convective_branch() {
    let b_par = 8.0 / 3.0;
    for which in BOTH {
        let setup = EpSetup::new(1, vec![10.0, 0.5, b_par], vec![0.0; 3]);
        let (_, _, next) = bp_setup("lorenz", which.clone(), &setup, 1.5);
        let nb = run("lorenz", which.clone(), &next, &switched_opts(1.0)).unwrap();
        let d = reach(&nb, &next);
        assert!(d > 1.0, "{d}");
        assert!(nb.points.len() > 20);
        for p in &nb.points {
            assert!((p.u[0] - p.u[1]).abs() < 1e-9, "{which:?}: {p:?}");
            assert!(
                (p.u[2] - p.u[0] * p.u[0] / b_par).abs() < 1e-9,
                "{which:?}: {p:?}"
            );
        }
        no_bp(&nb);
    }
}

#[test]
fn direction_minus_one_reverses_the_switched_branch() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let (_, _, next) = bp_setup("transcritical", which.clone(), &setup, 1.0);
        let fwd = run("transcritical", which.clone(), &next, &switched_opts(1.0)).unwrap();
        let rev = run("transcritical", which.clone(), &next, &switched_opts(-1.0)).unwrap();
        let (tf, tr) = (&fwd.points[0].tangent, &rev.points[0].tangent);
        for (a, b) in tf.iter().zip(tr) {
            assert!((a + b).abs() < 1e-12, "{tf:?} {tr:?}");
        }
        assert!(fwd.points[5].alpha > 0.0 && rev.points[5].alpha < 0.0);
        for p in &rev.points {
            assert!((p.u[0] - p.alpha).abs() < 1e-10, "{p:?}");
        }
    }
}

#[test]
fn switching_refuses_bad_specials() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let (b, k, _) = bp_setup("transcritical", which.clone(), &setup, 1.0);
        let mut m = model_of("transcritical", which.clone());
        let opts = Options::default();
        let bad = |r: Result<EpSetup, EngineError>| {
            assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
        };
        bad(switch_bp(m.as_mut(), &setup, &b, 99, &opts));
        let mut unrefined = b.clone();
        unrefined.specials[k].extras.refined = false;
        bad(switch_bp(m.as_mut(), &setup, &unrefined, k, &opts));
        let mut lp = b.clone();
        lp.specials[k].label = "LP".into();
        bad(switch_bp(m.as_mut(), &setup, &lp, k, &opts));

        // A real LP from the fold model.
        let fold = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let fb = run("fold", which.clone(), &fold, &first_opts(1.5)).unwrap();
        let i = special_index(&fb, "LP1").expect("fold has an LP");
        assert!(fb.specials[i].extras.refined);
        let mut mf = model_of("fold", which.clone());
        bad(switch_bp(mf.as_mut(), &fold, &fb, i, &opts));
    }
}

#[test]
fn a_start_that_is_not_an_equilibrium_is_refused() {
    for which in BOTH {
        let mut s = EpSetup::new(0, vec![-1.0], vec![0.5]);
        s.tangent0 = Some(vec![1.0, 1.0]);
        let r = run("transcritical", which.clone(), &s, &Options::default());
        assert!(
            matches!(&r, Err(EngineError::BadProblem(m)) if m.contains("not an equilibrium")),
            "{r:?}"
        );
        s.u0 = vec![0.0];
        s.params = vec![0.0];
        s.tangent0 = Some(vec![1.0, 1.0, 1.0]);
        let r = run("transcritical", which.clone(), &s, &Options::default());
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
    }
}

fn named(label: &str) -> SpecialPoint {
    SpecialPoint {
        label: label.into(),
        index: 0,
        point: Point {
            u: vec![0.0],
            alpha: 0.0,
            tangent: vec![0.0, 1.0],
            eigenvalues: Vec::new(),
            tests: [0.0; 3],
            codim2: Vec::new(),
            h: 0.0,
            newton_iters: 0,
        },
        extras: Default::default(),
    }
}

#[test]
fn special_index_counts_per_label() {
    let b = Branch {
        points: Vec::new(),
        specials: ["LP", "BP", "H", "BP", "H", "BP"].map(named).to_vec(),
        stop: StopReason::MaxSteps,
        warnings: Vec::new(),
    };
    assert_eq!(special_index(&b, "LP1"), Some(0));
    assert_eq!(special_index(&b, "BP1"), Some(1));
    assert_eq!(special_index(&b, "BP2"), Some(3));
    assert_eq!(special_index(&b, "BP3"), Some(5));
    assert_eq!(special_index(&b, "H2"), Some(4));
    for name in ["bp1", "Bp2", "h2", "lp1"] {
        let upper = name.to_ascii_uppercase();
        assert_eq!(special_index(&b, name), special_index(&b, &upper), "{name}");
        assert!(special_index(&b, name).is_some(), "{name}");
    }
    for bad in ["BP0", "BP4", "BP", "XX1", "1", "", "BP1x", "BP-1", "H3"] {
        assert_eq!(special_index(&b, bad), None, "{bad:?}");
    }
}

#[test]
fn switching_a_bp_with_a_definite_form_is_not_simple() {
    for which in BOTH {
        // x' = a^2 + x^2 at the origin: F_x = 0 and c = 2 I, so there is no real root pair.
        let setup = EpSetup::new(0, vec![0.0], vec![0.0]);
        let mut sp = named("BP");
        sp.extras.refined = true;
        let b = Branch {
            points: Vec::new(),
            specials: vec![sp],
            stop: StopReason::MaxSteps,
            warnings: Vec::new(),
        };
        let mut m = model_of("isola", which.clone());
        let r = switch_bp(m.as_mut(), &setup, &b, 0, &Options::default());
        assert!(
            matches!(&r, Err(EngineError::BadProblem(m)) if m == "BP is not simple"),
            "{which:?}: {r:?}"
        );
    }
}

#[test]
fn a_one_dimensional_bp_needs_a_zero_state_block() {
    for which in BOTH {
        // At (0,0), x' = a + x^2 - a^2 has F_x = [0, 1]: rank 1, so the kernel is
        // one-dimensional. Its F_xx = diag(2, -2) is indefinite, so the branching
        // equation alone would accept the point; only the rank check refuses it.
        let setup = EpSetup::new(0, vec![0.0], vec![0.0]);
        let mut sp = named("BP");
        sp.extras.refined = true;
        sp.point.tangent = vec![1.0, 0.0];
        let b = Branch {
            points: Vec::new(),
            specials: vec![sp],
            stop: StopReason::MaxSteps,
            warnings: Vec::new(),
        };
        let mut m = model_of("fold_saddle", which.clone());
        let r = switch_bp(m.as_mut(), &setup, &b, 0, &Options::default());
        assert!(
            matches!(&r, Err(EngineError::BadProblem(m)) if m == "BP is not simple"),
            "{which:?}: {r:?}"
        );
    }
}

#[test]
fn a_tangent0_start_is_validated_before_use() {
    for which in BOTH {
        let mut s = EpSetup::new(5, vec![0.0], vec![0.0]);
        s.tangent0 = Some(vec![1.0, 1.0]);
        let r = run("transcritical", which.clone(), &s, &Options::default());
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
        let mut s = EpSetup::new(0, vec![0.0], vec![0.0, 0.0]);
        s.tangent0 = Some(vec![1.0, 1.0]);
        let r = run("transcritical", which.clone(), &s, &Options::default());
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
        let mut s = EpSetup::new(0, vec![], vec![0.0]);
        s.tangent0 = Some(vec![1.0, 1.0]);
        let r = run("transcritical", which.clone(), &s, &Options::default());
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
    }
}

/// Inf-norm distance of the last point of `b` from the start of `s` in (u, alpha).
fn reach(b: &Branch, s: &EpSetup) -> f64 {
    let p = b.points.last().unwrap();
    let mut d = (p.alpha - s.params[s.free]).abs();
    for (a, c) in p.u.iter().zip(&s.u0) {
        d = d.max((a - c).abs());
    }
    d
}
