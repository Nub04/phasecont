//! IFS and IFU on a HOM curve (SPEC §8b-3): the oracle zero of Sandstede's model, the
//! noise models of SPEC §8b tests 2 and 3, and the layout of the two columns.
//!
//! The oracle model is `sandstede.ode` at (a, b, c, beta) = (-0.25, 1, -3, 1) and
//! mu = mut = gamma = 0. The plane z = 0 holds the homoclinic orbit for every alpha, and the
//! IFS of the numpy solve of SPEC §8b-3 has a transversal zero at alpha = 1.8961.
//! Run with `-- --nocapture` to see the measured values.

mod common;

use common::{load, plane_orbit};
use pc_engine::{
    Branch, Evaluator, HOM_CURVE_LABELS, HomStart, HomStartKind, ModelEval, Options, StopReason,
    build, continue_hom_curve,
};
use std::sync::atomic::AtomicBool;

const BOTH: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

/// Parameter indices of `sandstede.ode`.
const MU: usize = 3;
const MUT: usize = 4;
const ALPHA: usize = 5;

/// Indices of the IF tests in `Point::codim2`.
const IFS: usize = 12;
const IFU: usize = 13;

fn model_of(name: &str, which: &Evaluator) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, which.clone()).unwrap()
}

fn opts(bounds: (f64, f64)) -> Options {
    Options {
        max_newton: 15,
        hmax: 0.05,
        bounds,
        ..Options::default()
    }
}

fn run(m: &mut dyn ModelEval, start: &HomStart, bounds: (f64, f64)) -> Branch {
    let cancel = AtomicBool::new(false);
    let b = continue_hom_curve(m, start, &opts(bounds), &cancel, &mut |_| {}).unwrap();
    assert_eq!(b.stop, StopReason::Bounds, "{:?}", b.stop);
    b
}

fn labels_of(b: &Branch) -> Vec<(String, f64)> {
    b.specials
        .iter()
        .map(|s| (s.label.clone(), s.point.alpha))
        .collect()
}

/// The oracle start at alpha = 0 with `ntst = 8 T` intervals of 4 points.
fn oracle_start(period: f64, flips: bool) -> HomStart {
    let (ntst, ncol) = ((8.0 * period) as usize, 4);
    HomStart {
        kind: HomStartKind::Orbit,
        free: [MUT, ALPHA],
        params: vec![-0.25, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        orbit: plane_orbit(-0.25, 1.0, period, ntst, ncol, 1),
        period,
        mesh: None,
        ntst,
        ncol,
        inclination_flips: flips,
    }
}

/// SPEC §8b-3, acceptance 3: from alpha = 0 to 2.1 at T = 20 the run reports one IFS label,
/// within 5e-3 of the oracle zero, and no other label.
#[test]
fn the_oracle_curve_has_one_ifs_label_at_the_zero() {
    for which in BOTH {
        let mut m = model_of("sandstede", &which);
        let b = run(m.as_mut(), &oracle_start(20.0, true), (-0.1, 2.1));
        let labels = labels_of(&b);
        println!("{which:?}: {} points, labels {labels:?}", b.points.len());
        assert_eq!(labels.len(), 1, "{which:?}: {labels:?}");
        assert_eq!(labels[0].0, "IFS", "{which:?}: {labels:?}");
        assert!(
            (labels[0].1 - 1.8961).abs() < 5e-3,
            "{which:?}: IFS at alpha = {}",
            labels[0].1
        );
    }
}

/// With the key on, both IF columns hold a value at every point of the oracle run where the
/// eigenvalue they use is real, which is every point of it.
#[test]
fn the_columns_hold_values_with_the_key_on() {
    let mut m = model_of("sandstede", &Evaluator::Jit);
    let b = run(m.as_mut(), &oracle_start(20.0, true), (-0.1, 0.5));
    assert!(!b.points.is_empty());
    for pt in &b.points {
        assert_eq!(pt.codim2.len(), HOM_CURVE_LABELS.len());
        assert!(pt.codim2[IFS].is_finite(), "IFS {}", pt.codim2[IFS]);
        assert!(pt.codim2[IFU].is_finite(), "IFU {}", pt.codim2[IFU]);
    }
}

/// With the key off both columns are NaN, the labels of SPEC §8b are unchanged and the
/// oracle run has none of them in the same range.
#[test]
fn the_columns_are_nan_with_the_key_off() {
    let mut m = model_of("sandstede", &Evaluator::Jit);
    let b = run(m.as_mut(), &oracle_start(20.0, false), (-0.1, 2.1));
    for pt in &b.points {
        assert_eq!(pt.codim2.len(), HOM_CURVE_LABELS.len());
        assert!(pt.codim2[IFS].is_nan() && pt.codim2[IFU].is_nan());
    }
    assert!(labels_of(&b).is_empty(), "{:?}", labels_of(&b));
}

/// SPEC §8b-3, acceptance 4, test 2 of SPEC §8b: the orbit flip model (a, b, c) = (-0.5,
/// 2.5, -1), where IFS is rounding noise of order e^-T. The only label is OFS.
#[test]
fn the_orbit_flip_model_gets_no_if_label() {
    for period in [20.0, 30.0, 40.0] {
        let (ntst, ncol) = ((8.0 * period) as usize, 4);
        let mut m = model_of("sandstede", &Evaluator::Jit);
        let start = HomStart {
            kind: HomStartKind::Orbit,
            free: [MUT, MU],
            params: vec![-0.5, 2.5, -1.0, -0.1, 0.0, 0.0, 0.0, 0.0],
            orbit: plane_orbit(-0.5, 2.5, period, ntst, ncol, 1),
            period,
            mesh: None,
            ntst,
            ncol,
            inclination_flips: true,
        };
        let b = run(m.as_mut(), &start, (-0.11, 0.1));
        let labels = labels_of(&b);
        let worst = b
            .points
            .iter()
            .fold(0.0_f64, |w, p| w.max(p.codim2[IFS].abs()));
        println!("T = {period}: max |IFS| {worst:e}, labels {labels:?}");
        assert!(
            labels.iter().all(|l| l.0 != "IFS" && l.0 != "IFU"),
            "T = {period}: {labels:?}"
        );
        assert_eq!(labels.len(), 1, "T = {period}: {labels:?}");
        assert_eq!(labels[0].0, "OFS");
    }
}

/// SPEC §8b-3, acceptance 4, test 3 of SPEC §8b: `belyakov4.ode`, where IFS is noise too.
/// The labels are BLS and possibly TLS, as without the key.
#[test]
fn the_belyakov_model_gets_no_if_label() {
    const DELTA: usize = 4;
    const MUT4: usize = 5;
    for period in [20.0, 30.0, 40.0] {
        let (ntst, ncol) = ((8.0 * period) as usize, 4);
        let mut m = model_of("belyakov4", &Evaluator::Jit);
        let start = HomStart {
            kind: HomStartKind::Orbit,
            free: [MUT4, DELTA],
            params: vec![0.0, 1.0, -0.4, 0.5, -0.1, 0.0],
            orbit: plane_orbit(0.0, 1.0, period, ntst, ncol, 2),
            period,
            mesh: None,
            ntst,
            ncol,
            inclination_flips: true,
        };
        let b = run(m.as_mut(), &start, (-0.11, 0.1));
        let labels = labels_of(&b);
        println!("T = {period}: labels {labels:?}");
        assert!(
            labels.iter().all(|l| l.0 == "BLS" || l.0 == "TLS"),
            "T = {period}: {labels:?}"
        );
        assert_eq!(labels.iter().filter(|l| l.0 == "BLS").count(), 1);
    }
}
