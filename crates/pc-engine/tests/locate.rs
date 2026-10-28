//! Locating LP, BP, Hopf and neutral saddle points on real branches, on both evaluators.

mod common;

use common::load;
use pc_engine::{
    Branch, EpProblem, EpSetup, Evaluator, Options, StopReason, build, continue_branch, continue_ep,
};
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

fn run(name: &str, which: Evaluator, setup: &EpSetup, opts: &Options) -> Branch {
    let (model, info) = load(name);
    let mut m = build(&model, &info, which).unwrap();
    continue_ep(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

fn line_opts() -> Options {
    Options {
        h0: 0.07,
        hmax: 0.07,
        bounds: (f64::NEG_INFINITY, 1.0),
        ..Options::default()
    }
}

fn labels(b: &Branch) -> Vec<&str> {
    b.specials.iter().map(|s| s.label.as_str()).collect()
}

#[test]
fn fold_lp_is_located_and_refined() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let opts = Options {
            max_steps: 150,
            ..Options::default()
        };
        let b = run("fold", which, &setup, &opts);
        assert_eq!(labels(&b), ["LP"], "{which:?}");
        let s = &b.specials[0];
        assert!(
            s.point.alpha.abs() < 1e-8,
            "{which:?}: alpha {}",
            s.point.alpha
        );
        assert!(s.point.u[0].abs() < 1e-3);
        assert!(s.extras.refined);
        // The sign change is at the fold, not at the start.
        assert!(s.index >= 2, "index {}", s.index);
        assert!(b.points[s.index - 1].tests[0] > 0.0 && b.points[s.index].tests[0] < 0.0);
        assert!(b.points[s.index - 1].u[0] < 0.0 && b.points[s.index].u[0] > 0.0);
        // The special lies between its neighbours and is not stored in `points`.
        assert!(b.points[s.index - 1].alpha < s.point.alpha + 1e-12);
        assert!(s.point.tests[0].abs() < 1e-6);
    }
}

#[test]
fn bp_is_located_on_transcritical_and_pitchfork() {
    for which in BOTH {
        for name in ["transcritical", "pitchfork"] {
            let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
            let b = run(name, which, &setup, &line_opts());
            assert_eq!(b.stop, StopReason::Bounds);
            assert_eq!(labels(&b), ["BP"], "{name} {which:?}");
            let s = &b.specials[0];
            assert!(s.point.alpha.abs() < 1e-8, "{name}: {}", s.point.alpha);
            assert!(b.points[s.index - 1].alpha < s.point.alpha);
            assert!(s.point.alpha < b.points[s.index].alpha);
            assert!(b.warnings.is_empty());
        }
    }
}

#[test]
fn hopf_c_gives_one_h() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5, -1.0], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("hopf_c", which, &setup, &opts);
        assert_eq!(labels(&b), ["H"], "{which:?}");
        let s = &b.specials[0];
        assert!(s.point.alpha.abs() < 1e-8, "alpha {}", s.point.alpha);
        let (re, im) = s.point.eigenvalues[0];
        assert!(re.abs() < 1e-8 && (im.abs() - 1.0).abs() < 1e-8);
    }
}

#[test]
fn neutral_saddle_gives_nsa_and_no_h() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("neutral_saddle", which, &setup, &opts);
        assert_eq!(labels(&b), ["NSA"], "{which:?}");
        assert!(b.specials[0].point.alpha.abs() < 1e-8);
    }
}

#[test]
fn start_at_a_branch_point_works() {
    for which in BOTH {
        let (model, info) = load("transcritical");
        let mut m = build(&model, &info, which).unwrap();
        let setup = EpSetup::new(0, vec![0.0], vec![0.0]);
        let mut p = EpProblem::new(m.as_mut(), &setup).unwrap();
        let b = continue_branch(
            &mut p,
            &[0.0, 0.0],
            &line_opts(),
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(b.points[0].tests[1], 0.0, "{which:?}");
        assert_eq!(b.stop, StopReason::Bounds);
        assert!(b.specials.is_empty());
        assert!(b.points.len() > 3);
    }
}

#[test]
fn start_is_special_skips_the_first_step_only() {
    for which in BOTH {
        // The first step from al = -0.01 crosses the BP at 0.
        let mut setup = EpSetup::new(0, vec![-0.01], vec![0.0]);
        assert!(!setup.start_is_special);
        let b = run("transcritical", which, &setup, &line_opts());
        assert_eq!(labels(&b), ["BP"], "{which:?}");
        assert_eq!(b.specials[0].index, 1);
        setup.start_is_special = true;
        let b = run("transcritical", which, &setup, &line_opts());
        assert!(b.specials.is_empty(), "{which:?}");
    }
}

#[test]
fn a_zero_on_the_step_that_lands_on_a_bound_is_located() {
    for which in BOTH {
        // From al = -0.01 the first step would reach 0.06; the bound at 0.03 cuts it, and
        // the BP at 0 lies inside the cut step.
        let setup = EpSetup::new(0, vec![-0.01], vec![0.0]);
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 0.03),
            ..line_opts()
        };
        let b = run("transcritical", which, &setup, &opts);
        assert_eq!(b.stop, StopReason::Bounds);
        assert_eq!(b.points.len(), 2, "{which:?}");
        assert_eq!(labels(&b), ["BP"]);
        assert_eq!(b.specials[0].index, 1);
        assert!(b.specials[0].point.alpha.abs() < 1e-8);
        assert!((b.points[1].alpha - 0.03).abs() < 1e-10);
    }
}
