//! Starting an equilibrium run from an LP, H or BP point (SPEC section 11), on both
//! evaluators.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, EpSetup, Evaluator, ModelEval, Options, SpecialPoint, build, continue_ep,
    special_index,
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

fn first_run(name: &str, which: Evaluator, setup: &EpSetup, lo: f64, hi: f64) -> Branch {
    let opts = Options {
        direction: 1.0,
        hmax: 0.1,
        max_steps: 400,
        bounds: (lo, hi),
        ..Options::default()
    };
    run(name, which.clone(), setup, &opts).unwrap()
}

/// A setup that starts at the special point `label` of `b`.
fn start_at(setup: &EpSetup, b: &Branch, label: &str) -> (EpSetup, SpecialPoint) {
    let k = special_index(b, label).unwrap_or_else(|| panic!("no {label}"));
    let sp = b.specials[k].clone();
    let mut s = setup.clone();
    s.params[s.free] = sp.point.alpha;
    s.u0 = sp.point.u.clone();
    s.start_is_special = true;
    (s, sp)
}

fn opts(direction: f64, lo: f64, hi: f64) -> Options {
    Options {
        direction,
        h0: 0.05,
        hmax: 0.05,
        max_steps: 40,
        bounds: (lo, hi),
        ..Options::default()
    }
}

#[test]
fn lp_start_leaves_on_both_sides_of_the_fold() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let b = first_run("fold", which.clone(), &setup, -5.0, 5.0);
        let (s, lp) = start_at(&setup, &b, "LP1");
        let mut sides = Vec::new();
        for dir in [1.0, -1.0] {
            let o = opts(dir, -5.0, 5.0);
            let nb = run("fold", which.clone(), &s, &o).unwrap();
            assert!(nb.points.len() > 10, "{which:?} dir {dir}");
            let p0 = &nb.points[0];
            assert!((p0.alpha - lp.point.alpha).abs() < 1e-10, "{which:?}");
            assert!((p0.u[0] - lp.point.u[0]).abs() < 1e-10, "{which:?}");
            // Residual of a + x^2 at every point.
            for p in &nb.points {
                let r = p.alpha + p.u[0] * p.u[0];
                assert!(r.abs() <= o.tol_f, "{which:?} dir {dir}: {r:e}");
            }
            let last = nb.points.last().unwrap();
            let dist = (last.u[0] - p0.u[0])
                .abs()
                .max((last.alpha - p0.alpha).abs());
            assert!(dist > o.h0, "{which:?} dir {dir}: {dist}");
            assert!(nb.points[1].u[0] != p0.u[0], "{which:?}");
            sides.push(last.u[0].signum());
        }
        assert!(sides[0] != sides[1], "{which:?}: {sides:?}");
    }
}

#[test]
fn h_start_keeps_the_hopf_equilibrium_branch_going() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5, 1.0], vec![0.0, 0.0]);
        let b = first_run("hopf_c", which.clone(), &setup, -1.0, 1.0);
        let (s, h) = start_at(&setup, &b, "H1");
        for dir in [1.0, -1.0] {
            let o = opts(dir, -1.0, 1.0);
            let nb = run("hopf_c", which.clone(), &s, &o).unwrap();
            assert!(nb.points.len() > 10, "{which:?} dir {dir}");
            assert!((nb.points[0].alpha - h.point.alpha).abs() < 1e-10);
            let last = nb.points.last().unwrap();
            assert_eq!(last.alpha.signum(), dir, "{which:?}");
            for p in &nb.points {
                // x' = a x - y, y' = x + a y style normal form: the equilibrium is the origin.
                assert!(p.u.iter().all(|v| v.abs() <= o.tol_f), "{which:?}: {p:?}");
            }
        }
    }
}

#[test]
fn bp_start_is_refused_and_names_switch() {
    let b_par = 8.0 / 3.0;
    for which in BOTH {
        let setup = EpSetup::new(1, vec![10.0, 0.5, b_par], vec![0.0; 3]);
        let b = first_run("lorenz", which.clone(), &setup, 0.0, 5.0);
        let (s, _) = start_at(&setup, &b, "BP1");
        let r = run("lorenz", which.clone(), &s, &opts(1.0, 0.0, 5.0));
        assert!(
            matches!(&r, Err(EngineError::BadProblem(m)) if m.contains("Switch")),
            "{which:?}: {r:?}"
        );
    }
    // Pitchfork too (one variable).
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
        let b = first_run("pitchfork", which.clone(), &setup, -5.0, 1.0);
        let (s, _) = start_at(&setup, &b, "BP1");
        let r = run("pitchfork", which.clone(), &s, &opts(1.0, -5.0, 5.0));
        assert!(
            matches!(&r, Err(EngineError::BadProblem(m)) if m.contains("Switch")),
            "{which:?}: {r:?}"
        );
    }
}

#[test]
fn a_special_start_that_is_not_an_equilibrium_is_refused() {
    for which in BOTH {
        let mut s = EpSetup::new(0, vec![-1.0], vec![0.5]);
        s.start_is_special = true;
        let r = run("fold", which.clone(), &s, &Options::default());
        assert!(
            matches!(&r, Err(EngineError::BadProblem(m))
                if m == "the special start point is not an equilibrium"),
            "{which:?}: {r:?}"
        );
    }
}

#[test]
fn a_regular_start_still_runs_newton() {
    for which in BOTH {
        // Not special: Newton moves u0 = -0.9 onto x^2 = 1.
        let s = EpSetup::new(0, vec![-1.0], vec![-0.9]);
        let nb = run("fold", which.clone(), &s, &opts(1.0, -5.0, 5.0)).unwrap();
        assert!((nb.points[0].u[0] + 1.0).abs() < 1e-9, "{which:?}");
    }
}
