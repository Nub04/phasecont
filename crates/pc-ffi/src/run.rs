//! Runs behind the C ABI: the handle, the status JSON and the row packing.

use pc_engine::{EngineError, Point, Run, RunState, Status, StopReason};
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::json::problem::Kind;
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
/// codim2(k), h, newton_iters]`, with `k` the number of codim-2 labels of the kind.
pub(crate) fn stride(kind: Kind, n: usize) -> usize {
    match kind {
        Kind::Ep => 3 * n + 6,
        Kind::Lp => 1 + (n + 1) + 2 * n + kind.labels().len() + 2,
        Kind::Hopf => 1 + (n + 2) + 2 * n + kind.labels().len() + 2,
    }
}

/// Writes one row into `out`, which holds `stride(kind, n)` values. Missing state,
/// eigenvalue or codim-2 entries are NaN.
pub(crate) fn pack_row(kind: Kind, p: &Point, n: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), stride(kind, n));
    let nan = f64::NAN;
    out[0] = p.alpha;
    // The state block: `n` values, plus a1 (and kappa) on a curve.
    let block = match kind {
        Kind::Ep => n,
        Kind::Lp => n + 1,
        Kind::Hopf => n + 2,
    };
    for i in 0..block {
        out[1 + i] = p.u.get(i).copied().unwrap_or(nan);
    }
    let e = 1 + block;
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
    Ok(json!({
        "state": state,
        "kind": h.kind.name(),
        "codim2_labels": h.kind.labels(),
        "n": h.n,
        "stride": stride(h.kind, h.n),
        "branches": branches,
    }))
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
        assert_eq!(
            [Kind::Ep.name(), Kind::Lp.name(), Kind::Hopf.name()],
            ["EP", "LP", "H"]
        );
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
