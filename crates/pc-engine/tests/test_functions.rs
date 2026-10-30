//! LP, BP and Hopf test functions over real branches, each on both evaluators.

mod common;

use common::load;
use pc_engine::{
    Branch, EpSetup, Evaluator, HopfZero, Options, StopReason, build, classify_h_zero, continue_ep,
};
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];
const LP: usize = 0;
const BP: usize = 1;
const H: usize = 2;

fn run(name: &str, which: Evaluator, setup: &EpSetup, opts: &Options) -> Branch {
    let (model, info) = load(name);
    let mut m = build(&model, &info, which.clone()).unwrap();
    continue_ep(
        m.as_mut(),
        setup,
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

/// Indices k where test `t` has opposite signs at points k and k+1.
fn sign_changes(b: &Branch, t: usize) -> Vec<usize> {
    (0..b.points.len() - 1)
        .filter(|&k| b.points[k].tests[t] * b.points[k + 1].tests[t] < 0.0)
        .collect()
}

fn line_opts() -> Options {
    Options {
        h0: 0.07,
        hmax: 0.07,
        bounds: (f64::NEG_INFINITY, 1.0),
        ..Options::default()
    }
}

#[test]
fn bp_changes_sign_across_transcritical_and_pitchfork() {
    for which in BOTH {
        for name in ["transcritical", "pitchfork"] {
            let setup = EpSetup::new(0, vec![-1.0], vec![0.0]);
            let b = run(name, which.clone(), &setup, &line_opts());
            assert_eq!(b.stop, StopReason::Bounds);
            let ch = sign_changes(&b, BP);
            assert_eq!(ch.len(), 1, "{name} {which:?}: {ch:?}");
            let (a0, a1) = (b.points[ch[0]].alpha, b.points[ch[0] + 1].alpha);
            assert!(a0 < 0.0 && a1 > 0.0, "{name}: BP between {a0} and {a1}");
            // Along x = 0 the tangent is e_alpha, so psi_LP never changes sign.
            assert!(sign_changes(&b, LP).is_empty());
            // One variable: no Hopf pairs.
            assert!(b.points.iter().all(|p| p.tests[H] == 1.0));
        }
    }
}

#[test]
fn lp_changes_sign_across_the_fold() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-1.0], vec![-1.0]);
        let opts = Options {
            max_steps: 150,
            ..Options::default()
        };
        let b = run("fold", which.clone(), &setup, &opts);
        let ch = sign_changes(&b, LP);
        assert_eq!(ch.len(), 1, "{which:?}: {ch:?}");
        let k = ch[0];
        assert!(b.points[k].u[0] < 0.0 && b.points[k + 1].u[0] > 0.0);
        assert!(b.points[k].tests[LP] > 0.0 && b.points[k + 1].tests[LP] < 0.0);
        // A = 2x is zero at the fold; the eigenvalue changes sign there too.
        assert!(b.points[k].eigenvalues[0].0 < 0.0 && b.points[k + 1].eigenvalues[0].0 > 0.0);
    }
}

#[test]
fn h_changes_sign_across_the_hopf_of_hopf_c() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5, -1.0], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("hopf_c", which.clone(), &setup, &opts);
        assert_eq!(b.stop, StopReason::Bounds);
        let ch = sign_changes(&b, H);
        assert_eq!(ch.len(), 1, "{which:?}: {ch:?}");
        let (p0, p1) = (&b.points[ch[0]], &b.points[ch[0] + 1]);
        assert!(p0.alpha < 0.0 && p1.alpha > 0.0);
        assert!(p0.tests[H] < 0.0 && p1.tests[H] > 0.0);
        assert_eq!(classify_h_zero(&p0.eigenvalues), HopfZero::Hopf);
        assert_eq!(classify_h_zero(&p1.eigenvalues), HopfZero::Hopf);
        assert!(sign_changes(&b, BP).is_empty());
        assert!(sign_changes(&b, LP).is_empty());
    }
}

#[test]
fn h_changes_sign_across_the_neutral_saddle() {
    for which in BOTH {
        // Bounds keep the branch points at al = +-1 out of the run (ruling R4).
        let setup = EpSetup::new(0, vec![-0.5], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("neutral_saddle", which.clone(), &setup, &opts);
        assert_eq!(b.stop, StopReason::Bounds);
        let ch = sign_changes(&b, H);
        assert_eq!(ch.len(), 1, "{which:?}: {ch:?}");
        let (p0, p1) = (&b.points[ch[0]], &b.points[ch[0] + 1]);
        assert!(p0.alpha < 0.0 && p1.alpha > 0.0);
        assert_eq!(classify_h_zero(&p0.eigenvalues), HopfZero::NeutralSaddle);
        assert_eq!(classify_h_zero(&p1.eigenvalues), HopfZero::NeutralSaddle);
        assert!(sign_changes(&b, BP).is_empty());
    }
}

#[test]
fn detect_flags_turn_tests_into_nan() {
    for which in BOTH {
        let setup = EpSetup::new(0, vec![-0.5, -1.0], vec![0.0, 0.0]);
        for (lp, bp, h) in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            let opts = Options {
                bounds: (-0.5, 0.5),
                detect_lp: lp,
                detect_bp: bp,
                detect_h: h,
                ..Options::default()
            };
            let b = run("hopf_c", which.clone(), &setup, &opts);
            for p in &b.points {
                for (t, on) in [lp, bp, h].into_iter().enumerate() {
                    assert_eq!(p.tests[t].is_nan(), !on, "test {t}");
                }
                assert_eq!(p.eigenvalues.len(), 2);
            }
            assert!(b.warnings.is_empty());
        }
    }
}

#[test]
fn hopf_is_detected_above_100_variables() {
    for which in BOTH {
        // The planar Hopf normal form of hopf_c plus 99 decoupled stable variables.
        let setup = EpSetup::new(0, vec![-0.5, -1.0], vec![0.0; 101]);
        let opts = Options {
            bounds: (-0.5, 0.5),
            ..Options::default()
        };
        let b = run("hopf101", which.clone(), &setup, &opts);
        assert_eq!(b.stop, StopReason::Bounds);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
        assert!(b.points.iter().all(|p| p.eigenvalues.len() == 101));
        assert!(b.points.iter().all(|p| !p.tests[H].is_nan()));
        let ch = sign_changes(&b, H);
        assert_eq!(ch.len(), 1, "{which:?}: {ch:?}");
        assert!(b.points[ch[0]].alpha < 0.0 && b.points[ch[0] + 1].alpha > 0.0);
        let h: Vec<_> = b.specials.iter().filter(|s| s.label == "H").collect();
        assert_eq!(h.len(), 1, "{which:?}");
        assert!(h[0].point.alpha.abs() < 1e-8, "alpha {}", h[0].point.alpha);
        assert!(h[0].extras.refined);
    }
}

#[test]
fn negative_direction_keeps_the_sign_of_psi_bp() {
    for which in BOTH {
        // From al = 1 down to -1 along x = 0: the tangent is flipped at the start, and
        // psi_BP must not change sign between points 0 and 1 (only at the real BP).
        let setup = EpSetup::new(0, vec![1.0], vec![0.0]);
        let opts = Options {
            direction: -1.0,
            bounds: (-1.0, f64::INFINITY),
            ..line_opts()
        };
        let b = run("transcritical", which.clone(), &setup, &opts);
        assert_eq!(b.stop, StopReason::Bounds);
        assert!(b.points[0].tests[BP] * b.points[1].tests[BP] > 0.0);
        let ch = sign_changes(&b, BP);
        assert_eq!(ch.len(), 1, "{which:?}: {ch:?}");
        assert!(b.points[ch[0]].alpha > 0.0 && b.points[ch[0] + 1].alpha < 0.0);
    }
}
