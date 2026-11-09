//! The first Lyapunov coefficient under a rescaling of time (SPEC section 8g).
//!
//! With `t = alpha t'`, Wright's equation `u' = -a u(t-1) (1+u)` at its Hopf point
//! `a = pi/2, omega = pi/2, tau = 1` becomes `a = alpha pi/2, tau = 1/alpha,
//! omega = alpha pi/2`. The Hopf point is the same physical point, and `l1 = Re c1 / omega`
//! does not change, so `l1 = -0.21413092244177392` at every `alpha`. The thresholds in
//! `lyap.rs` must not make some `alpha` fail.

use pc_engine::{DdeModel, Evaluator, build_dde, dde_lyapunov};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;

const WRIGHT: &str = "par a=1.5, tau=1, b=0\nu'=-a*delay(u,tau)*(1+u)+b*u^3\ndone\n";
const L1_AT_ZERO: f64 = -0.214_130_922_441_773_92;

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

fn l1_at(alpha: f64, which: Evaluator) -> Result<f64, String> {
    let mut m = dde(WRIGHT, which);
    let w = alpha * PI / 2.0;
    dde_lyapunov(&mut m, &[0.0], &[w, 1.0 / alpha, 0.0], w)
        .map(|l| l.l1)
        .map_err(|e| e.to_string())
}

#[test]
fn l1_does_not_change_when_time_is_rescaled() {
    let mut bad = Vec::new();
    for alpha in [
        1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8, 1e10, 1e12,
    ] {
        match l1_at(alpha, Evaluator::Fallback) {
            Ok(l1) if (l1 - L1_AT_ZERO).abs() <= 1e-6 * L1_AT_ZERO.abs() => {}
            Ok(l1) => bad.push(format!("alpha {alpha:e}: l1 = {l1:e}")),
            Err(e) => bad.push(format!("alpha {alpha:e}: {e}")),
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn the_floor_follows_the_size_of_the_matrices() {
    // A frequency of alpha * 1e-9 is a fixed fraction, 6e-10, of the size alpha pi / 2 of
    // the matrices, so it is below the floor at every alpha.
    for alpha in [1.0, 1e6] {
        let mut m = dde(WRIGHT, Evaluator::Fallback);
        let w = alpha * PI / 2.0;
        let err = dde_lyapunov(&mut m, &[0.0], &[w, 1.0 / alpha, 0.0], alpha * 1e-9)
            .map(|l| l.l1)
            .map_err(|e| e.to_string());
        match err {
            Err(e) if e.contains("times the size") => {}
            other => panic!("alpha {alpha:e}: {other:?}"),
        }
    }
}
