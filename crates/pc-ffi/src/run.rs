//! Runs behind the C ABI: the handle, the status JSON and the row packing.

use pc_engine::{EngineError, Point, Run, RunState, Status, StopReason};
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::json::problem::{Kind, cycle_stride};
use crate::registry::Handle;

/// A started run, what it follows and the state size of its model.
pub(crate) struct RunHandle {
    pub run: Run,
    pub kind: Kind,
    pub n: usize,
}

impl Handle for RunHandle {
    const KIND: &'static str = "run";
}

/// Values per row.
///
/// An equilibrium row is `[alpha, u(n), eig_re(n), eig_im(n), psi_LP, psi_BP, psi_H, h,
/// newton_iters]`. A curve row is `[a2, x(n), a1, kappa (Hopf only), eig_re(n), eig_im(n),
/// codim2(k), h, newton_iters]`, with `k` the number of codim-2 labels of the kind. A cycle
/// row on N intervals of m points is `[alpha, u(n (N m + 1)), T, mesh(N + 1), mult_re(n),
/// mult_im(n), psi_LPC, psi_PD, psi_NS, h, newton_iters]`, with the orbit point-major. A
/// row of an LPC or PD curve is `[a2, u(n (N m + 1)), T, a1, mesh(N + 1), mult_re(n),
/// mult_im(n), codim2(k), h, newton_iters]`, and an NS curve adds kappa after a1.
pub(crate) fn stride(kind: Kind, n: usize) -> usize {
    match kind {
        Kind::Ep => 3 * n + 6,
        // [t, x(n)]
        Kind::Ivp => n + 1,
        Kind::Lp => 1 + (n + 1) + 2 * n + kind.labels().len() + 2,
        Kind::Hopf => 1 + (n + 2) + 2 * n + kind.labels().len() + 2,
        // The parser rejects sizes that overflow; MAX makes a copy fail rather than wrap.
        Kind::Lc { .. } | Kind::Lpc { .. } | Kind::Pd { .. } | Kind::Ns { .. } => {
            cycle_stride(kind, n).unwrap_or(usize::MAX)
        }
    }
}

/// Writes one row into `out`, which holds `stride(kind, n)` values. Missing state, mesh,
/// eigenvalue or codim-2 entries are NaN.
pub(crate) fn pack_row(kind: Kind, p: &Point, n: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), stride(kind, n));
    let nan = f64::NAN;
    out[0] = p.alpha;
    if kind == Kind::Ivp {
        for i in 0..n {
            out[1 + i] = p.u.get(i).copied().unwrap_or(nan);
        }
        return;
    }
    // The state block: `n` values, plus a1 (and kappa) on a curve, or the orbit, T and
    // what follows T on a curve of cycles.
    let (block, mesh) = match kind {
        Kind::Ep | Kind::Ivp => (n, 0),
        Kind::Lp => (n + 1, 0),
        Kind::Hopf => (n + 2, 0),
        Kind::Lc { ntst, ncol }
        | Kind::Lpc { ntst, ncol }
        | Kind::Pd { ntst, ncol }
        | Kind::Ns { ntst, ncol } => (n * (ntst * ncol + 1) + kind.cycle_tail(), ntst + 1),
    };
    for i in 0..block {
        out[1 + i] = p.u.get(i).copied().unwrap_or(nan);
    }
    let m = 1 + block;
    for i in 0..mesh {
        out[m + i] = p.mesh.get(i).copied().unwrap_or(nan);
    }
    let e = m + mesh;
    for i in 0..n {
        let (re, im) = p.eigenvalues.get(i).copied().unwrap_or((nan, nan));
        out[e + i] = re;
        out[e + n + i] = im;
    }
    let t = e + 2 * n;
    let tail = match kind {
        Kind::Ep => {
            out[t..t + 3].copy_from_slice(&p.tests);
            t + 3
        }
        _ => {
            let k = kind.labels().len();
            for i in 0..k {
                out[t + i] = p.codim2.get(i).copied().unwrap_or(nan);
            }
            t + k
        }
    };
    out[tail] = p.h;
    out[tail + 1] = p.newton_iters as f64;
}

/// The variant name SPEC section 10 uses.
pub(crate) fn stop_name(s: &StopReason) -> &'static str {
    match s {
        StopReason::End => "End",
        StopReason::MaxSteps => "MaxSteps",
        StopReason::Bounds => "Bounds",
        StopReason::StepTooSmall => "StepTooSmall",
        StopReason::Singular => "Singular",
        StopReason::Cancelled | StopReason::Error(EngineError::Cancelled) => "Cancelled",
        StopReason::Error(_) => "Error",
    }
}

/// The state name for a state that `run_poll` reports, or the error for one it does not.
pub(crate) fn state_name(state: &RunState) -> Result<&'static str> {
    match state {
        RunState::Running => Ok("running"),
        RunState::Done => Ok("done"),
        RunState::Cancelled => Ok("cancelled"),
        RunState::Failed(e) => Err(Error::from(e.clone())),
        RunState::Panicked { message, location } => {
            Err(Error::Panic(format!("panic: {message} (at {location})")))
        }
    }
}

fn opt(x: Option<f64>) -> Value {
    x.map_or(Value::Null, |v| json!(v))
}

/// The status JSON of SPEC section 10.
pub(crate) fn poll_json(h: &RunHandle, st: &Status) -> Result<Value> {
    let state = state_name(&st.state)?;
    let last = st.points.len().saturating_sub(1);
    let branches: Vec<Value> = st
        .points
        .iter()
        .enumerate()
        .map(|(b, count)| {
            // The engine's stop is a placeholder while the last branch runs.
            let stop = if b == last && st.state == RunState::Running {
                None
            } else {
                h.run.stop(b)
            };
            let specials: Vec<Value> = h
                .run
                .specials(b)
                .iter()
                .map(|s| {
                    let values: Map<String, Value> = s
                        .extras
                        .values
                        .iter()
                        .map(|(name, v)| (name.clone(), json!(v)))
                        .collect();
                    json!({
                        "label": s.label,
                        "index": s.index,
                        "coefficient": opt(s.extras.coefficient),
                        "coefficient2": opt(s.extras.coefficient2),
                        "coefficients": values,
                        "omega": opt(s.extras.omega),
                        "normalization": s.extras.normalization,
                        "refined": s.extras.refined,
                    })
                })
                .collect();
            json!({
                "points": count,
                "stop": stop.as_ref().map(stop_name),
                "warnings": h.run.warnings(b),
                "specials": specials,
            })
        })
        .collect();
    let mut out = json!({
        "state": state,
        "kind": h.kind.name(),
        "codim2_labels": h.kind.labels(),
        "n": h.n,
        "stride": stride(h.kind, h.n),
        "branches": branches,
    });
    if let Some((ntst, ncol)) = h.kind.mesh_size() {
        out["ntst"] = json!(ntst);
        out["ncol"] = json!(ncol);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point() -> Point {
        Point {
            u: vec![1.0, 2.0],
            alpha: 0.5,
            tangent: vec![0.0; 3],
            eigenvalues: vec![(-1.0, 0.25)],
            tests: [10.0, 11.0, 12.0],
            codim2: Vec::new(),
            h: 0.125,
            newton_iters: 3,
            mesh: Vec::new(),
        }
    }

    #[test]
    fn pack_row_lays_out_the_row_and_fills_gaps_with_nan() {
        let mut row = vec![0.0; stride(Kind::Ep, 2)];
        pack_row(Kind::Ep, &point(), 2, &mut row);
        assert_eq!(row.len(), 12);
        assert_eq!(&row[..4], [0.5, 1.0, 2.0, -1.0]);
        assert!(row[4].is_nan());
        assert_eq!(row[5], 0.25);
        assert!(row[6].is_nan());
        assert_eq!(&row[7..], [10.0, 11.0, 12.0, 0.125, 3.0]);
        let mut p = point();
        p.u.clear();
        p.eigenvalues.clear();
        pack_row(Kind::Ep, &p, 2, &mut row);
        assert_eq!(row[0], 0.5);
        assert!(row[1..7].iter().all(|x| x.is_nan()));
        assert_eq!(&row[7..], [10.0, 11.0, 12.0, 0.125, 3.0]);
    }

    #[test]
    fn strides_follow_the_kind() {
        assert_eq!(stride(Kind::Ep, 2), 12);
        // [a2, x(2), a1, eig_re(2), eig_im(2), codim2(3), h, iters]
        assert_eq!(stride(Kind::Lp, 2), 1 + 2 + 1 + 4 + 3 + 2);
        // The Hopf curve adds kappa and has four codim-2 values.
        assert_eq!(stride(Kind::Hopf, 2), 1 + 2 + 1 + 1 + 4 + 4 + 2);
        assert_eq!(Kind::Ep.labels(), &[] as &[&str]);
        assert_eq!(Kind::Lp.labels(), ["CP", "BT", "ZH"]);
        assert_eq!(Kind::Hopf.labels(), ["GH", "BT", "ZH", "HH"]);
        let lc = Kind::Lc { ntst: 4, ncol: 3 };
        // [alpha, u(2 * 13), T, mesh(5), mult_re(2), mult_im(2), LPC, PD, NS, h, iters]
        assert_eq!(stride(lc, 2), 1 + 26 + 1 + 5 + 4 + 3 + 2);
        assert_eq!(lc.labels(), ["LPC", "PD", "NS"]);
        assert_eq!(
            [
                Kind::Ep.name(),
                Kind::Lp.name(),
                Kind::Hopf.name(),
                lc.name()
            ],
            ["EP", "LP", "H", "LC"]
        );
        // An overflowing size makes the stride MAX, so a copy fails instead of wrapping.
        let huge = Kind::Lc {
            ntst: usize::MAX / 2,
            ncol: 3,
        };
        assert_eq!(stride(huge, 2), usize::MAX);
    }

    #[test]
    fn a_cycle_row_has_the_orbit_the_period_and_the_mesh() {
        let lc = Kind::Lc { ntst: 1, ncol: 2 };
        let mut p = point();
        // n = 1: the orbit at 3 base points, then T; one multiplier; two mesh points.
        p.u = vec![1.0, 2.0, 3.0, 6.5];
        p.mesh = vec![0.0, 1.0];
        p.eigenvalues = vec![(1.0, 0.0)];
        p.codim2 = vec![7.0, 8.0];
        let mut row = vec![0.0; stride(lc, 1)];
        assert_eq!(row.len(), 1 + 3 + 1 + 2 + 2 + 3 + 2);
        pack_row(lc, &p, 1, &mut row);
        assert_eq!(&row[..7], [0.5, 1.0, 2.0, 3.0, 6.5, 0.0, 1.0]);
        assert_eq!(&row[7..9], [1.0, 0.0]);
        assert_eq!(&row[9..11], [7.0, 8.0]);
        assert!(row[11].is_nan());
        assert_eq!(&row[12..], [0.125, 3.0]);
        // A point without a mesh gets NaN there.
        p.mesh.clear();
        pack_row(lc, &p, 1, &mut row);
        assert!(row[5].is_nan() && row[6].is_nan());
    }

    #[test]
    fn a_row_of_a_curve_of_cycles_has_a1_and_kappa_after_the_period() {
        let mut p = point();
        // n = 1 on 1 interval of 2 points: the orbit at 3 base points, T, a1 and kappa.
        p.u = vec![1.0, 2.0, 3.0, 6.5, 4.0, 0.25];
        p.alpha = 0.75;
        p.mesh = vec![0.0, 1.0];
        p.eigenvalues = vec![(-1.0, 0.0)];
        p.codim2 = vec![7.0, 8.0, 9.0, 10.0];
        for kind in [
            Kind::Lpc { ntst: 1, ncol: 2 },
            Kind::Pd { ntst: 1, ncol: 2 },
        ] {
            let mut row = vec![0.0; stride(kind, 1)];
            // [a2, u(3), T, a1, mesh(2), mult_re, mult_im, codim2(4), h, iters]
            assert_eq!(row.len(), 1 + 3 + 2 + 2 + 2 + 4 + 2);
            pack_row(kind, &p, 1, &mut row);
            assert_eq!(
                &row[..10],
                [0.75, 1.0, 2.0, 3.0, 6.5, 4.0, 0.0, 1.0, -1.0, 0.0]
            );
            assert_eq!(&row[10..], [7.0, 8.0, 9.0, 10.0, 0.125, 3.0]);
            assert_eq!(kind.mesh_size(), Some((1, 2)));
        }
        let ns = Kind::Ns { ntst: 1, ncol: 2 };
        let mut row = vec![0.0; stride(ns, 1)];
        // [a2, u(3), T, a1, kappa, mesh(2), mult_re, mult_im, R1 to R4, LPNS, PDNS, CH, h,
        // iters]
        assert_eq!(row.len(), 1 + 3 + 3 + 2 + 2 + 7 + 2);
        pack_row(ns, &p, 1, &mut row);
        assert_eq!(
            &row[..11],
            [0.75, 1.0, 2.0, 3.0, 6.5, 4.0, 0.25, 0.0, 1.0, -1.0, 0.0]
        );
        assert_eq!(&row[11..15], [7.0, 8.0, 9.0, 10.0]);
        assert!(row[15..18].iter().all(|x| x.is_nan()));
        assert_eq!(&row[18..], [0.125, 3.0]);
        assert_eq!(
            [
                Kind::Lpc { ntst: 1, ncol: 2 }.name(),
                Kind::Pd { ntst: 1, ncol: 2 }.name(),
                ns.name()
            ],
            ["LPC", "PD", "NS"]
        );
        let huge = Kind::Ns {
            ntst: usize::MAX / 2,
            ncol: 3,
        };
        assert_eq!(stride(huge, 2), usize::MAX);
    }

    #[test]
    fn a_hopf_row_has_kappa_after_a1_and_codim2_after_the_eigenvalues() {
        let mut p = point();
        // (x, y, a1, kappa) and two of the four codim-2 values.
        p.u = vec![1.0, 2.0, 3.0, 4.0];
        p.eigenvalues = vec![(-1.0, 0.25), (-2.0, 0.5)];
        p.codim2 = vec![7.0, 8.0];
        let mut row = vec![0.0; stride(Kind::Hopf, 2)];
        pack_row(Kind::Hopf, &p, 2, &mut row);
        assert_eq!(&row[..7], [0.5, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0]);
        assert_eq!(&row[7..9], [0.25, 0.5]);
        assert_eq!(&row[9..11], [7.0, 8.0]);
        assert!(row[11].is_nan() && row[12].is_nan());
        assert_eq!(&row[13..], [0.125, 3.0]);
    }

    #[test]
    fn an_lp_row_has_no_kappa() {
        let mut p = point();
        p.u = vec![1.0, 2.0, 3.0];
        p.eigenvalues = vec![(-1.0, 0.25), (-2.0, 0.5)];
        p.codim2 = vec![7.0, 8.0, 9.0];
        let mut row = vec![0.0; stride(Kind::Lp, 2)];
        pack_row(Kind::Lp, &p, 2, &mut row);
        assert_eq!(&row[..6], [0.5, 1.0, 2.0, 3.0, -1.0, -2.0]);
        assert_eq!(&row[6..8], [0.25, 0.5]);
        assert_eq!(&row[8..11], [7.0, 8.0, 9.0]);
        assert_eq!(&row[11..], [0.125, 3.0]);
    }

    #[test]
    fn stop_names_cover_every_reason() {
        let cases = [
            (StopReason::End, "End"),
            (StopReason::MaxSteps, "MaxSteps"),
            (StopReason::Bounds, "Bounds"),
            (StopReason::StepTooSmall, "StepTooSmall"),
            (StopReason::Singular, "Singular"),
            (StopReason::Cancelled, "Cancelled"),
            (StopReason::Error(EngineError::Cancelled), "Cancelled"),
            (
                StopReason::Error(EngineError::NotFinite("f".into())),
                "Error",
            ),
        ];
        for (reason, name) in cases {
            assert_eq!(stop_name(&reason), name);
        }
    }

    #[test]
    fn states_map_to_names_and_errors() {
        assert_eq!(state_name(&RunState::Running).unwrap(), "running");
        assert_eq!(state_name(&RunState::Done).unwrap(), "done");
        assert_eq!(state_name(&RunState::Cancelled).unwrap(), "cancelled");
        let e = state_name(&RunState::Failed(EngineError::Singular("x".into()))).unwrap_err();
        assert_eq!(e.code(), 8);
        let e = state_name(&RunState::Panicked {
            message: "m".into(),
            location: "f.rs:1:2".into(),
        })
        .unwrap_err();
        assert_eq!(e.code(), 1);
        assert_eq!(e.to_string(), "panic: m (at f.rs:1:2)");
    }
}
