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

/// The brackets of a sign change of the IF column `col` between consecutive accepted points
/// of `b`, each as (alpha at the left point, larger end value in absolute value).
fn brackets(b: &Branch, col: usize) -> Vec<(f64, f64)> {
    b.points
        .windows(2)
        .filter_map(|w| {
            let (x, y) = (w[0].codim2[col], w[1].codim2[col]);
            (x * y < 0.0 || x == 0.0 || y == 0.0).then(|| (w[0].alpha, x.abs().max(y.abs())))
        })
        .collect()
}

/// The start of a run of `sandstede.ode` or one of its variants at the given (a, b, c, beta)
/// with the free pair (mut, alpha), from alpha = 0.
fn sandstede_start(par: (f64, f64, f64, f64), period: f64) -> HomStart {
    let (ntst, ncol) = ((8.0 * period) as usize, 4);
    HomStart {
        kind: HomStartKind::Orbit,
        free: [MUT, ALPHA],
        params: vec![par.0, par.1, par.2, 0.0, 0.0, 0.0, par.3, 0.0],
        orbit: plane_orbit(par.0, par.1, period, ntst, ncol, 1),
        period,
        mesh: None,
        ntst,
        ncol,
        inclination_flips: true,
    }
}

/// SPEC §8b-3, acceptance 6, signal side. Every sign change of IFS on three families of
/// runs gets a label at the floor of 1e-8. The bound asserted is the fallback bound of the
/// SPEC, 1e-6 on the larger end value. The target of 1e-4 is missed by one bracket (the
/// last zero of `sandstede_lin` at T = 30), which the SPEC records.
/// The families differ in lambda_S (-1.25 and -2), in T (10 to 40) and in the model (the
/// coupling of `sandstede_lin.ode`). The numpy sweeps of SPEC §8b-3 at T = 20 give the
/// number of zeros each family has in its range of alpha: 3, 4 and 7.
#[test]
fn every_ifs_bracket_of_the_signal_models_clears_the_floor() {
    // (model, (a, b, c, beta), alpha range, zeros of the numpy sweep at T = 20)
    let families = [
        ("sandstede", (-0.25, 1.0, -3.0, 1.0), (-0.1, 3.0), 3),
        ("sandstede", (-0.5, 1.5, -3.0, 1.0), (-0.1, 4.0), 4),
        ("sandstede_lin", (-0.25, 1.0, -3.0, 1.0), (-0.1, 4.0), 7),
    ];
    let mut smallest = f64::INFINITY;
    for (model, par, bounds, zeros) in families {
        for period in [10.0, 20.0, 30.0, 40.0] {
            let mut m = model_of(model, &Evaluator::Jit);
            let b = run(m.as_mut(), &sandstede_start(par, period), bounds);
            let br = brackets(&b, IFS);
            let labels = labels_of(&b);
            let larger: Vec<f64> = br.iter().map(|x| x.1).collect();
            println!(
                "{model} {par:?} T = {period}: {} points, {} brackets, larger ends {larger:?}, {} labels",
                b.points.len(),
                br.len(),
                labels.len()
            );
            assert_eq!(br.len(), zeros, "{model} {par:?} T = {period}: {br:?}");
            assert_eq!(
                labels.len(),
                zeros,
                "{model} {par:?} T = {period}: {labels:?}"
            );
            assert!(labels.iter().all(|l| l.0 == "IFS"), "{labels:?}");
            for (alpha, big) in &br {
                assert!(
                    *big > 1e-6,
                    "{model} {par:?} T = {period}: bracket at alpha {alpha} has larger end {big:e}"
                );
                smallest = smallest.min(*big);
            }
        }
    }
    println!("smallest larger end over all signal brackets: {smallest:e}");
}

/// SPEC §8b-3, acceptance 6, noise side. In the models of SPEC §8b, tests 2 and 3, no
/// bracket of IFS or IFU has a larger end value above 1e-10, at T = 10 to 40.
#[test]
fn the_noise_models_have_no_bracket_above_1e_10() {
    const DELTA: usize = 4;
    const MUT4: usize = 5;
    let mut worst = 0.0_f64;
    for period in [10.0, 20.0, 30.0, 40.0] {
        let (ntst, ncol) = ((8.0 * period) as usize, 4);
        let flip = HomStart {
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
        let bely = HomStart {
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
        for (name, model, start) in [
            ("orbit flip", "sandstede", flip),
            ("belyakov", "belyakov4", bely),
        ] {
            let mut m = model_of(model, &Evaluator::Jit);
            let b = run(m.as_mut(), &start, (-0.11, 0.1));
            for (col, tag) in [(IFS, "IFS"), (IFU, "IFU")] {
                let big = brackets(&b, col).iter().fold(0.0_f64, |w, x| w.max(x.1));
                let all = b
                    .points
                    .iter()
                    .fold(0.0_f64, |w, p| w.max(p.codim2[col].abs()));
                println!(
                    "{name} T = {period} {tag}: largest bracket end {big:e}, max |value| {all:e}"
                );
                assert!(
                    big <= 1e-10,
                    "{name} T = {period} {tag}: bracket end {big:e}"
                );
                worst = worst.max(big);
            }
        }
    }
    println!("largest bracket end over the noise models: {worst:e}");
}

/// SPEC §8b-3, acceptance 7. The inclination flips of Sandstede's model in the form of
/// Giraldo, Krauskopf and Osinga (2022), where `gamma` couples z to x. The published
/// values of alpha at the flip are 0.4664012 for (a, b, c, beta, gamma) = (0.22, 1, -2, 1, 2)
/// (case B, arXiv:2207.13846) and 0.3694818 for (0.7, 1, -2, 1, 2) (case C,
/// arXiv:2207.13854). Case C has lambda_S = -0.3, so its end values are far above the
/// floor. The IFS label must sit within 1e-5 of the published value for T = 20, 30 and 40.
#[test]
fn the_published_flips_of_cases_b_and_c_are_found() {
    let cases = [("B", 0.22, 0.466_401_2), ("C", 0.7, 0.369_481_8)];
    for (name, a, published) in cases {
        for period in [20.0, 30.0, 40.0] {
            let mut start = sandstede_start((a, 1.0, -2.0, 1.0), period);
            start.params[7] = 2.0;
            let mut m = model_of("sandstede", &Evaluator::Jit);
            let b = run(m.as_mut(), &start, (-0.1, 0.8));
            let labels = labels_of(&b);
            let larger: Vec<f64> = brackets(&b, IFS).iter().map(|x| x.1).collect();
            println!("case {name} T = {period}: labels {labels:?}, larger ends {larger:?}");
            let ifs: Vec<f64> = labels
                .iter()
                .filter(|l| l.0 == "IFS")
                .map(|l| l.1)
                .collect();
            assert_eq!(ifs.len(), 1, "case {name} T = {period}: {labels:?}");
            assert!(
                (ifs[0] - published).abs() < 1e-5,
                "case {name} T = {period}: alpha {} against {published}",
                ifs[0]
            );
            assert!(
                larger.iter().all(|v| *v > 1e-6),
                "case {name} T = {period}: larger ends {larger:?}"
            );
        }
    }
}

/// SPEC §8b-3, acceptance 7, mesh part. The sign reference of the adjoint is the whole
/// vector, interpolated onto the new mesh when the mesh moves. Case B at T = 30 gives the
/// published label with the default adaptation, which moves the mesh, and with none.
#[test]
fn the_sign_reference_follows_the_mesh_when_it_moves() {
    let mut start = sandstede_start((0.22, 1.0, -2.0, 1.0), 30.0);
    start.params[7] = 2.0;
    let uniform: Vec<f64> = (0..=start.ntst)
        .map(|i| i as f64 / start.ntst as f64)
        .collect();
    for adapt in [3, 0] {
        let mut m = model_of("sandstede", &Evaluator::Jit);
        let cancel = AtomicBool::new(false);
        let o = Options {
            mesh_adapt: adapt,
            ..opts((-0.1, 0.8))
        };
        let b = continue_hom_curve(m.as_mut(), &start, &o, &cancel, &mut |_| {}).unwrap();
        let last = &b.points.last().unwrap().mesh;
        let moved = last.len() != uniform.len()
            || last.iter().zip(&uniform).any(|(a, c)| (a - c).abs() > 1e-6);
        assert_eq!(moved, adapt > 0, "mesh_adapt {adapt}: moved {moved}");
        let ifs: Vec<f64> = labels_of(&b)
            .iter()
            .filter(|l| l.0 == "IFS")
            .map(|l| l.1)
            .collect();
        assert_eq!(ifs.len(), 1, "mesh_adapt {adapt}: {ifs:?}");
        assert!(
            (ifs[0] - 0.466_401_2).abs() < 1e-5,
            "mesh_adapt {adapt}: {ifs:?}"
        );
    }
}
