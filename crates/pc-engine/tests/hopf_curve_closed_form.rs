//! The ODE Hopf curve against a closed form that does not come from this engine.
//!
//! Strogatz, Nonlinear Dynamics and Chaos, 3rd ed., Section 8.3 (pp. 281-286), works out the
//! Lengyel-Epstein model
//!
//!   x' = a - x - 4xy/(1 + x^2),   y' = b x (1 - y/(1 + x^2))
//!
//! by hand. Its only equilibrium is x* = a/5, y* = 1 + (a/5)^2 (eq. 5). The Jacobian there has
//! determinant Delta = 5 b x*/(1 + x*^2) and trace tau = (3 x*^2 - 5 - b x*)/(1 + x*^2). The
//! Hopf curve is tau = 0 with Delta > 0, which gives b_c = 3a/5 - 25/a (eq. 7), and there the
//! frequency satisfies omega^2 = Delta = (15 a^2 - 625)/(a^2 + 25). This needs a > sqrt(125/3),
//! about 6.455. For a = 10 the book finds b_c = 3.5 and a supercritical bifurcation, so the
//! first Lyapunov coefficient is negative.
//!
//! The book stops at a = 10. Past a = 18.495... the first Lyapunov coefficient changes sign, so
//! the curve has a generalized Hopf point there. That number and the value of l1 at a = 10 come
//! from an independent calculation made while writing this test: exact order-3 Taylor jets of
//! the right-hand side, the planar formula of Guckenheimer and Holmes (eq. 3.4.11) in
//! coordinates where the Jacobian is [[0, -w], [w, 0]], and the factor 4/(w |q|^2) that turns
//! its cubic coefficient into Kuznetsov's l1 for the eigenvector scaling <q,q> = 1. Over the
//! 302 points of the upward run that calculation and the engine agree to 9e-15. At a = 10 it
//! gives l1 = -0.09/sqrt(7). The GH root is a = 18.49507523094392.
//!
//! The curve runs in (a, b). Each point stores the state, a, kappa = omega^2 and, as `alpha`, b.
//! The model file is our own, written from those two equations.

mod common;

use common::load;
use pc_engine::{
    Branch, CurveSetup, EpSetup, Evaluator, ModelEval, Options, build, continue_ep,
    continue_hopf_curve,
};
use std::sync::atomic::AtomicBool;

fn model() -> Box<dyn ModelEval> {
    let (model, info) = load("lengyel_epstein");
    build(&model, &info, Evaluator::Jit).unwrap()
}

/// The curve from the Hopf point at a = 10 in the given direction, with b bounded by `bounds`.
fn curve(direction: f64, bounds: (f64, f64)) -> Branch {
    let mut m = model();
    let cancel = AtomicBool::new(false);
    let a0 = 10.0;
    let u0 = vec![a0 / 5.0, 1.0 + (a0 / 5.0) * (a0 / 5.0)];
    // The equilibrium does not depend on b, so the branch in b is a straight line and the Hopf
    // point is where the trace changes sign.
    let ep_opts = Options {
        bounds: (0.5, 6.0),
        ..Options::default()
    };
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(1, vec![a0, 1.0], u0),
        &ep_opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let h: Vec<_> = ep.specials.iter().filter(|s| s.label == "H").collect();
    assert_eq!(h.len(), 1, "specials {:?}", ep.specials);
    assert!(
        (h[0].point.alpha - 3.5).abs() < 1e-8,
        "Hopf point at b = {}, the book has 3.5",
        h[0].point.alpha
    );
    let setup = CurveSetup::new([0, 1], vec![a0, h[0].point.alpha], h[0].point.u.clone());
    let opts = Options {
        direction,
        bounds,
        ..Options::default()
    };
    continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap()
}

/// The generalized Hopf point of the curve, from the independent calculation in the header.
const A_GH: f64 = 18.49507523094392;

/// Largest deviation from the closed form over the curve, as (state, b, kappa), and the range of
/// a that the curve covers. Also checks the sign of l1 against the position of `A_GH`.
fn check(br: &Branch) -> ([f64; 3], (f64, f64)) {
    let (mut dx, mut db, mut dk) = (0.0f64, 0.0f64, 0.0f64);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &br.points {
        let (x, y, a, kappa, b) = (p.u[0], p.u[1], p.u[2], p.u[3], p.alpha);
        let (xs, ys) = (a / 5.0, 1.0 + (a / 5.0) * (a / 5.0));
        dx = dx.max((x - xs).abs()).max((y - ys).abs());
        db = db.max((b - (3.0 * a / 5.0 - 25.0 / a)).abs());
        dk = dk.max((kappa - (15.0 * a * a - 625.0) / (a * a + 25.0)).abs());
        lo = lo.min(a);
        hi = hi.max(a);
        // codim2[0] is the first Lyapunov coefficient: negative (supercritical) below the GH
        // point and positive above it. Skip the points within 1e-6 of the sign change.
        let l1 = p.codim2[0];
        if a < A_GH - 1e-6 {
            assert!(l1 < 0.0, "l1 = {l1} at a = {a}");
        } else if a > A_GH + 1e-6 {
            assert!(l1 > 0.0, "l1 = {l1} at a = {a}");
        }
    }
    ([dx, db, dk], (lo, hi))
}

#[test]
fn the_lengyel_epstein_hopf_curve_matches_strogatz_towards_larger_a() {
    let br = curve(1.0, (1.2, 14.0));
    let (d, (_, hi)) = check(&br);
    assert!(d[0] < 1e-8 && d[1] < 1e-8 && d[2] < 1e-7, "{d:?}");
    assert!(hi > 24.0, "the curve only reaches a = {hi}");
    // One label, GH, at the sign change of l1.
    assert_eq!(br.specials.len(), 1, "{:?}", br.specials);
    let gh = &br.specials[0];
    assert_eq!(gh.label, "GH");
    let a = gh.point.u[2];
    assert!((a - A_GH).abs() < 1e-9, "GH at a = {a}, expected {A_GH}");
    // b and omega at the GH point follow from the same closed form.
    assert!((gh.point.alpha - (3.0 * A_GH / 5.0 - 25.0 / A_GH)).abs() < 1e-8);
    assert!(
        (gh.extras.omega.unwrap().powi(2) - (15.0 * A_GH * A_GH - 625.0) / (A_GH * A_GH + 25.0))
            .abs()
            < 1e-7
    );
}

#[test]
fn the_lengyel_epstein_hopf_curve_matches_strogatz_towards_smaller_a() {
    let br = curve(-1.0, (1.2, 14.0));
    assert!(br.specials.is_empty(), "{:?}", br.specials);
    let (d, (lo, _)) = check(&br);
    assert!(d[0] < 1e-8 && d[1] < 1e-8 && d[2] < 1e-7, "{d:?}");
    assert!(lo < 7.7, "the curve only reaches a = {lo}");
}

/// The first Lyapunov coefficient at the book's point a = 10, b = 3.5: -0.09/sqrt(7), see the
/// header for where the number comes from. The start point of the curve is that Hopf point.
#[test]
fn the_first_lyapunov_coefficient_at_the_books_hopf_point_has_the_closed_value() {
    let br = curve(1.0, (1.2, 14.0));
    let start = &br.points[0];
    assert!(
        (start.u[2] - 10.0).abs() < 1e-8,
        "starts at a = {}",
        start.u[2]
    );
    let want = -0.09 / 7.0f64.sqrt();
    assert!(
        (start.codim2[0] - want).abs() < 1e-9,
        "l1 = {}, expected {want}",
        start.codim2[0]
    );
}
