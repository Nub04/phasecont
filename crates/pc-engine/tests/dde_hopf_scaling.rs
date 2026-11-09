//! The Hopf curve of a delay equilibrium under a rescaling of time (SPEC section 8f).
//!
//! Hutchinson's equation `x' = r x (1 - x(t - tau))` has the Hopf curve `r tau = pi/2` with
//! `omega = r`. With `t = alpha t'` the same curve is `r = alpha r'`, `tau = tau'/alpha`,
//! `omega = alpha omega'`. The start and the run must not depend on `alpha`: the floor on
//! `omega` follows the size of the matrices, as the Lyapunov floor does (SPEC section 8g).
//!
//! The run is checked for `alpha` from 1e-3 to 1e3. The step and tolerance options are
//! absolute, so the test scales them with the size of the parameters. Below 1e-3 the Newton
//! corrector no longer reaches the tolerances at this scaling, and that range is not claimed.

use pc_engine::{DdeHopfSetup, Evaluator, Options, StopReason, build_dde, dde_continue_hopf};
use pc_ode::{MapLoader, Severity, SourceMap, load};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const HUTCH: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

fn run(alpha: f64, which: Evaluator) -> Result<pc_engine::DdeHopfBranch, String> {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", HUTCH);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    let mut m = build_dde(&l.expanded, &l.info, which).map_err(|e| e.to_string())?;
    let s = DdeHopfSetup {
        free: [0, 1],
        params: vec![1.57 * alpha, 1.0 / alpha],
        x0: vec![1.001],
        omega: 1.57 * alpha,
    };
    // The tolerances and the steps are absolute, so they follow the size of the parameters:
    // r is of size alpha and tau of size 1/alpha.
    let big = (alpha * alpha).max(1.0 / alpha);
    let o = Options {
        h0: 0.01 * big,
        hmin: 1e-8 * big,
        hmax: 0.05 * big,
        bounds: (0.5 / alpha, 3.0 / alpha),
        direction: 1.0,
        tol_x: 1e-12 * big,
        tol_f: 1e-12 * big,
        tol_refine: 1e-12 * big,
        max_newton: 20,
        ..Options::default()
    };
    dde_continue_hopf(&mut m, &s, &o, &AtomicBool::new(false)).map_err(|e| e.to_string())
}

#[test]
fn the_hopf_curve_does_not_depend_on_the_time_scale() {
    let mut bad = Vec::new();
    for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3] {
        let b = match run(alpha, Evaluator::Fallback) {
            Ok(b) => b,
            Err(e) => {
                bad.push(format!("alpha {alpha:e}: {e}"));
                continue;
            }
        };
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        let worst = b
            .points
            .iter()
            .map(|p| {
                let [r, tau] = p.alpha;
                let w = p.omega.unwrap_or(f64::NAN);
                (r * tau / (PI / 2.0) - 1.0)
                    .abs()
                    .max(rel(p.kappa, r * r))
                    .max(rel(w, r))
            })
            .fold(0.0_f64, f64::max);
        let last = b.points.last().map(|p| p.alpha[1] * alpha);
        if b.points.len() <= 10
            || worst.is_nan()
            || worst > 1e-7
            || !matches!(b.stop, StopReason::Bounds)
        {
            bad.push(format!(
                "alpha {alpha:e}: {} points, worst relative error {worst:e}, stop {:?}, \
                 last tau*alpha {last:?}",
                b.points.len(),
                b.stop
            ));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}
