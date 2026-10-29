//! Runs behind the C ABI: the handle, the status JSON and the row packing.

use pc_engine::{EngineError, Point, Run, RunState, Status, StopReason};
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::registry::Handle;

/// A started run and the state size of its model.
pub(crate) struct RunHandle {
    pub run: Run,
    pub n: usize,
}

impl Handle for RunHandle {
    const KIND: &'static str = "run";
}

/// Values per row: `[alpha, u(n), eig_re(n), eig_im(n), psi_LP, psi_BP, psi_H, h, newton_iters]`.
pub(crate) fn stride(n: usize) -> usize {
    3 * n + 6
}

/// Writes one row into `out`, which holds `stride(n)` values. Missing state or
/// eigenvalue entries are NaN.
pub(crate) fn pack_row(p: &Point, n: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), stride(n));
    out[0] = p.alpha;
    for i in 0..n {
        out[1 + i] = p.u.get(i).copied().unwrap_or(f64::NAN);
        let (re, im) = p
            .eigenvalues
            .get(i)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN));
        out[1 + n + i] = re;
        out[1 + 2 * n + i] = im;
    }
    let t = 1 + 3 * n;
    out[t..t + 3].copy_from_slice(&p.tests);
    out[t + 3] = p.h;
    out[t + 4] = p.newton_iters as f64;
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
                    json!({
                        "label": s.label,
                        "index": s.index,
                        "coefficient": opt(s.extras.coefficient),
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
        "n": h.n,
        "stride": stride(h.n),
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
            h: 0.125,
            newton_iters: 3,
        }
    }

    #[test]
    fn pack_row_lays_out_the_row_and_fills_gaps_with_nan() {
        let mut row = vec![0.0; stride(2)];
        pack_row(&point(), 2, &mut row);
        assert_eq!(row.len(), 12);
        assert_eq!(&row[..4], [0.5, 1.0, 2.0, -1.0]);
        assert!(row[4].is_nan());
        assert_eq!(row[5], 0.25);
        assert!(row[6].is_nan());
        assert_eq!(&row[7..], [10.0, 11.0, 12.0, 0.125, 3.0]);
        let mut p = point();
        p.u.clear();
        p.eigenvalues.clear();
        pack_row(&p, 2, &mut row);
        assert_eq!(row[0], 0.5);
        assert!(row[1..7].iter().all(|x| x.is_nan()));
        assert_eq!(&row[7..], [10.0, 11.0, 12.0, 0.125, 3.0]);
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
