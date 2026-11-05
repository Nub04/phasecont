//! Tier A: ODEs with closed form solutions, and a bit-exact cross-check of the adaptive
//! method against `pc_engine::integrate`.

mod common;

use std::sync::atomic::AtomicBool;

use common::*;
use pc_engine::{IvpOptions, IvpSetup, integrate};
use pc_ensemble::{Method, Output, Params, Precision, Status};

#[test]
fn decay_rk4_double_matches_exp() {
    let m = build_model(DECAY);
    let x0 = [1.0, 2.0, -0.5];
    let j = job(
        &m,
        3,
        &x0,
        &[1.0],
        2.0,
        Method::Rk4 { dt: 0.01 },
        Precision::Double,
    );
    let r = run(&j);
    for (i, x) in x0.iter().enumerate() {
        let want = x * (-2.0_f64).exp();
        let got = r.state(i, 0)[0];
        assert!((got - want).abs() < 1e-9, "member {i}: {got} vs {want}");
        assert_eq!(r.status[i], Status::Ok);
        assert_eq!(r.t_reached[i], 2.0);
        assert_eq!(r.accepted[i], 200);
    }
    assert_eq!(r.label.backend, "cpu");
    assert!(!r.label.single_precision());
}

#[test]
fn rk4_converges_with_order_four() {
    let m = build_model(DECAY);
    let want = (-2.0_f64).exp();
    let err = |dt: f64| {
        let j = job(
            &m,
            1,
            &[1.0],
            &[1.0],
            2.0,
            Method::Rk4 { dt },
            Precision::Double,
        );
        (run(&j).state(0, 0)[0] - want).abs()
    };
    let (e1, e2, e3) = (err(0.1), err(0.05), err(0.025));
    let (r1, r2) = (e1 / e2, e2 / e3);
    assert!((14.0..18.0).contains(&r1), "ratio {r1} (errors {e1} {e2})");
    assert!((14.0..18.0).contains(&r2), "ratio {r2} (errors {e2} {e3})");
}

#[test]
fn dp5_decay_is_within_tolerance() {
    let m = build_model(DECAY);
    let j = job(&m, 1, &[1.0], &[1.0], 5.0, DP5_STD, Precision::Double);
    let r = run(&j);
    let err = (r.state(0, 0)[0] - (-5.0_f64).exp()).abs();
    assert!(err < 1e-7, "error {err}");
    assert_eq!(r.status[0], Status::Ok);
    assert!(
        r.accepted[0] > 3 && r.accepted[0] < 200,
        "{}",
        r.accepted[0]
    );
}

#[test]
fn oscillator_samples_follow_cos_and_sin() {
    let m = build_model(OSC);
    let pi = std::f64::consts::PI;
    let times = vec![pi / 2.0, pi, 2.0 * pi];
    let mut j = job(
        &m,
        1,
        &[1.0, 0.0],
        &[],
        2.0 * pi,
        DP5_TIGHT,
        Precision::Double,
    );
    j.output = Output::Samples(times.clone());
    let r = run(&j);
    assert_eq!(r.times, times);
    for (s, &t) in times.iter().enumerate() {
        let x = r.state(0, s);
        assert!((x[0] - t.cos()).abs() < 1e-7, "x at {t}: {}", x[0]);
        assert!((x[1] + t.sin()).abs() < 1e-7, "v at {t}: {}", x[1]);
    }
}

#[test]
fn logistic_matches_the_closed_form() {
    let m = build_model(LOGISTIC);
    let x0 = [0.1, 0.5, 2.0];
    let j = job(&m, 3, &x0, &[], 4.0, DP5_TIGHT, Precision::Double);
    let r = run(&j);
    for (i, &x) in x0.iter().enumerate() {
        let want = 1.0 / (1.0 + (1.0 / x - 1.0) * (-4.0_f64).exp());
        assert!((r.state(i, 0)[0] - want).abs() < 1e-8, "member {i}");
    }
}

#[test]
fn time_enters_the_model() {
    let m = build_model(FORCED);
    for method in [Method::Rk4 { dt: 0.01 }, DP5_TIGHT] {
        let j = job(&m, 1, &[0.0], &[], 3.0, method, Precision::Double);
        let r = run(&j);
        let err = (r.state(0, 0)[0] - 3.0_f64.sin()).abs();
        assert!(err < 1e-8, "{method:?}: {err}");
    }
}

#[test]
fn parameters_can_differ_per_member() {
    let m = build_model(DECAY);
    let a = [0.5, 1.0, 2.0];
    let mut j = job(
        &m,
        3,
        &[1.0, 1.0, 1.0],
        &[],
        2.0,
        DP5_TIGHT,
        Precision::Double,
    );
    j.params = Params::PerMember(&a);
    let r = run(&j);
    for (i, &ai) in a.iter().enumerate() {
        let want = (-2.0 * ai).exp();
        assert!((r.state(i, 0)[0] - want).abs() < 1e-8, "member {i}");
    }
}

#[test]
fn single_precision_is_labeled_and_close() {
    let m = build_model(DECAY);
    let want = (-2.0_f64).exp();
    for (method, tol) in [
        (Method::Rk4 { dt: 0.01 }, 1e-4),
        (
            Method::Dp5 {
                rtol: 1e-5,
                atol: 1e-7,
                hmax: f64::INFINITY,
            },
            1e-3,
        ),
    ] {
        let j = job(&m, 1, &[1.0], &[1.0], 2.0, method, Precision::Single);
        let r = run(&j);
        let err = (r.state(0, 0)[0] - want).abs();
        assert!(err < tol, "{method:?}: {err}");
        assert!(r.label.single_precision());
        assert_eq!(r.label.precision.name(), "single");
        assert_eq!(r.status[0], Status::Ok);
    }
}

/// With the final state as output the adaptive run takes the same steps as
/// `pc_engine::integrate`. The two differ only in `powf`: the ensemble uses `libm::pow` so a CPU
/// result does not depend on the platform C library, the engine uses `f64::powf`. With `powf`
/// swapped for the standard one the final states agree in every bit (checked once by hand), so
/// here they must take the same number of steps and agree to 1e-12 relative.
#[test]
fn dp5_matches_the_engine() {
    for (src, x0, p, t1) in [
        (VDP, vec![2.0, 0.0], vec![1.5], 12.0),
        (MIXED, vec![0.3, -0.4], vec![0.7], 9.0),
        (LOGISTIC, vec![0.05], vec![], 10.0),
    ] {
        let m = build_model(src);
        let j = job(&m, 1, &x0, &p, t1, DP5_STD, Precision::Double);
        let r = run(&j);

        let mut model = engine_model(src);
        let setup = IvpSetup {
            x0: x0.clone(),
            params: p.clone(),
            t0: 0.0,
            t1,
        };
        let opts = IvpOptions {
            rtol: 1e-8,
            atol: 1e-10,
            max_steps: 1_000_000,
            hmax: f64::INFINITY,
        };
        let b = integrate(
            model.as_mut(),
            &setup,
            &opts,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .expect("integrate");
        let last = b.points.last().expect("points");
        assert_eq!(r.accepted[0] as usize, b.points.len() - 1, "{src}");
        assert_eq!(r.status[0], Status::Ok);
        for (g, w) in r.state(0, 0).iter().zip(&last.u) {
            let rel = (g - w).abs() / w.abs().max(1e-300);
            assert!(rel < 1e-12, "{src}: {g} vs {w} (relative {rel})");
        }
    }
}
