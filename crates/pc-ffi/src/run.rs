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
/// mult_im(n), codim2(k), h, newton_iters]`, and an NS curve adds kappa after a1. A row
/// of a HOM curve is `[a2, u(n (N m + 1)), T, x0(n), a1, mesh(N + 1), eig_re(n),
/// eig_im(n), codim2(14), h, newton_iters]`, with x0 the saddle and the eigenvalues those
/// of the Jacobian at x0. A row of a delay equilibrium branch is `[alpha, u(n), roots]`, with
/// `roots` the number of characteristic roots with a positive real part. A row of a delay Hopf
/// curve is `[a2, x(n), a1, kappa, l1, hh]`, with NaN where `l1` or `hh` is off or undefined.
/// A row of a delay cycle branch on N intervals of m points is `[alpha, u(n (N m + 1)), T, h,
/// newton_iters, mu_re(c), mu_im(c), trivial_error]`, with `c` the multipliers kept and
/// NaN where the engine has no value (no `floquet`, or fewer than `c` multipliers). A run with
/// mesh adaptation appends the mesh, N + 1 values, after `trivial_error` (SPEC 8s, slice 5).
pub(crate) fn stride(kind: Kind, n: usize) -> usize {
    match kind {
        Kind::Ep => 3 * n + 6,
        // [alpha, u(n), root count]
        Kind::DdeEp => n + 2,
        // [alpha2, x(n), alpha1, kappa, l1, hh]
        Kind::DdeHopf => n + 5,
        // [alpha, u(n (N m + 1)), T, h, iters, mu_re(c), mu_im(c), trivial_error]
        Kind::DdeLc { .. } => cycle_stride(kind, n).unwrap_or(usize::MAX),
        // [t, x(n)]
        Kind::Ivp => n + 1,
        Kind::Lp => 1 + (n + 1) + 2 * n + kind.labels().len() + 2,
        Kind::Hopf => 1 + (n + 2) + 2 * n + kind.labels().len() + 2,
        // The parser rejects sizes that overflow; MAX makes a copy fail rather than wrap.
        Kind::Lc { .. }
        | Kind::Lpc { .. }
        | Kind::Pd { .. }
        | Kind::Ns { .. }
        | Kind::Hom { .. } => cycle_stride(kind, n).unwrap_or(usize::MAX),
    }
}

/// Writes one row into `out`, which holds `stride(kind, n)` values. Missing state, mesh,
/// eigenvalue or codim-2 entries are NaN.
pub(crate) fn pack_row(kind: Kind, p: &Point, n: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), stride(kind, n));
    let nan = f64::NAN;
    out[0] = p.alpha;
    if kind == Kind::DdeHopf {
        // `u` is [x(n), alpha1, kappa] and `codim2` is [l1, hh] (SPEC 8s, slice 2).
        for i in 0..n + 2 {
            out[1 + i] = p.u.get(i).copied().unwrap_or(nan);
        }
        out[n + 3] = p.codim2.first().copied().unwrap_or(nan);
        out[n + 4] = p.codim2.get(1).copied().unwrap_or(nan);
        return;
    }
    if let Kind::DdeLc {
        ntst,
        ncol,
        count,
        adapt,
    } = kind
    {
        // `u` is the profile followed by T, `eigenvalues` the leading multipliers and `tests[0]`
        // the trivial multiplier error (SPEC 8s, slice 4). With `adapt` the mesh follows, NaN
        // padded when it does not have N + 1 values (slice 5). `stride` has checked the sizes.
        let m = n * (ntst * ncol + 1) + 1;
        for i in 0..m {
            out[1 + i] = p.u.get(i).copied().unwrap_or(nan);
        }
        out[m + 1] = p.h;
        out[m + 2] = p.newton_iters as f64;
        for i in 0..count {
            let (re, im) = p.eigenvalues.get(i).copied().unwrap_or((nan, nan));
            out[m + 3 + i] = re;
            out[m + 3 + count + i] = im;
        }
        out[m + 3 + 2 * count] = p.tests.first().copied().unwrap_or(nan);
        if adapt {
            let mesh = &mut out[m + 4 + 2 * count..];
            debug_assert_eq!(mesh.len(), ntst + 1);
            let kept = if p.mesh.len() == ntst + 1 {
                &p.mesh[..]
            } else {
                &[]
            };
            for (i, slot) in mesh.iter_mut().enumerate() {
                *slot = kept.get(i).copied().unwrap_or(nan);
            }
        }
        return;
    }
    if matches!(kind, Kind::Ivp | Kind::DdeEp) {
        for i in 0..n {
            out[1 + i] = p.u.get(i).copied().unwrap_or(nan);
        }
        // The IVP row ends with the state. A delay equilibrium row adds the root count that the
        // engine keeps in `tests[0]` (SPEC 8s).
        if kind == Kind::DdeEp {
            out[n + 1] = p.tests[0];
        }
        return;
    }
    // The state block: `n` values, plus a1 (and kappa) on a curve, or the orbit, T and
    // what follows T on a curve of cycles or a HOM curve.
    let (block, mesh) = match kind {
        Kind::Ep | Kind::DdeEp | Kind::DdeHopf | Kind::DdeLc { .. } | Kind::Ivp => (n, 0),
        Kind::Lp => (n + 1, 0),
        Kind::Hopf => (n + 2, 0),
        Kind::Lc { ntst, ncol }
        | Kind::Lpc { ntst, ncol }
        | Kind::Pd { ntst, ncol }
        | Kind::Ns { ntst, ncol }
        | Kind::Hom { ntst, ncol } => (
            // `stride` has checked these sizes for overflow.
            n * (ntst * ncol + 1) + kind.cycle_tail(n).unwrap_or(0),
            ntst + 1,
        ),
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
    fn a_delay_row_ends_with_the_root_count() {
        let mut p = point();
        p.tests = [3.0, f64::NAN, f64::NAN];
        assert_eq!(stride(Kind::DdeEp, 2), 4);
        let mut row = vec![0.0; 4];
        pack_row(Kind::DdeEp, &p, 2, &mut row);
        assert_eq!(row, [0.5, 1.0, 2.0, 3.0]);
        p.u.clear();
        pack_row(Kind::DdeEp, &p, 2, &mut row);
        assert_eq!(row[0], 0.5);
        assert!(row[1].is_nan() && row[2].is_nan());
        assert_eq!(row[3], 3.0);
        assert_eq!(Kind::DdeEp.name(), "EP");
        assert!(Kind::DdeEp.labels().is_empty());
    }

    #[test]
    fn a_delay_hopf_row_is_x_alpha1_kappa_l1_hh() {
        let mut p = point();
        // The engine's delay Hopf point: u = [x(2), alpha1, kappa], alpha = alpha2.
        p.u = vec![1.0, 2.0, 3.0, 4.0];
        p.alpha = 0.5;
        p.codim2 = vec![-0.25, f64::NAN];
        assert_eq!(stride(Kind::DdeHopf, 2), 7);
        assert_eq!(stride(Kind::DdeHopf, 1), 6);
        let mut row = vec![0.0; 7];
        pack_row(Kind::DdeHopf, &p, 2, &mut row);
        assert_eq!(&row[..6], [0.5, 1.0, 2.0, 3.0, 4.0, -0.25]);
        assert!(row[6].is_nan());
        // A point without codim-2 values (a defensive case) leaves both NaN.
        p.codim2.clear();
        pack_row(Kind::DdeHopf, &p, 2, &mut row);
        assert!(row[5].is_nan() && row[6].is_nan());
        assert_eq!(Kind::DdeHopf.name(), "H");
        assert_eq!(Kind::DdeHopf.labels(), ["GH", "HH"]);
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
    fn a_row_of_a_hom_curve_has_the_saddle_and_a1_after_t() {
        let mut p = point();
        // n = 2 on 1 interval of 2 points: the orbit at 3 base points, T, x0(2) and a1.
        p.u = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0, 0.5, -0.5, 0.25];
        p.alpha = 0.75;
        p.mesh = vec![0.0, 1.0];
        p.eigenvalues = vec![(-1.0, 0.0), (2.0, 0.0)];
        p.codim2 = (0..14).map(f64::from).collect();
        let hom = Kind::Hom { ntst: 1, ncol: 2 };
        let mut row = vec![0.0; stride(hom, 2)];
        // [a2, u(6), T, x0(2), a1, mesh(2), eig_re(2), eig_im(2), codim2(14), h, iters]
        assert_eq!(row.len(), 1 + 6 + 1 + 2 + 1 + 2 + 4 + 14 + 2);
        pack_row(hom, &p, 2, &mut row);
        assert_eq!(
            &row[..17],
            [
                0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0, 0.5, -0.5, 0.25, 0.0, 1.0, -1.0, 2.0,
                0.0, 0.0
            ]
        );
        assert_eq!(&row[17..31], p.codim2.as_slice());
        assert_eq!(&row[31..], [0.125, 3.0]);
        assert_eq!(hom.name(), "HOM");
        assert_eq!(hom.labels(), pc_engine::HOM_CURVE_LABELS);
        assert_eq!(hom.mesh_size(), Some((1, 2)));
        assert_eq!(hom.cycle_tail(2), Some(4));
        assert_eq!(hom.cycle_tail(usize::MAX), None);
        assert_eq!(cycle_stride(hom, usize::MAX), None);
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

    #[test]
    fn a_delay_cycle_row_is_the_orbit_the_period_and_the_multipliers() {
        // n = 1, ntst 2, ncol 2: five profile values, then T.
        let profile = [0.0, 0.1, 0.2, 0.3, 0.4];
        let mut p = point();
        p.u = profile.iter().copied().chain([6.25]).collect();
        p.alpha = 1.5;
        p.h = 0.5;
        p.newton_iters = 4;
        p.eigenvalues = vec![(1.5, 0.0), (0.25, -0.5)];
        p.tests = [1e-12, f64::NAN, f64::NAN];
        let kind = Kind::DdeLc {
            ntst: 2,
            ncol: 2,
            count: 2,
            adapt: false,
        };
        // [alpha, profile(5), T, h, iters, mu_re(2), mu_im(2), trivial_error]
        assert_eq!(stride(kind, 1), 1 + 5 + 1 + 2 + 4 + 1);
        let mut row = vec![0.0; stride(kind, 1)];
        pack_row(kind, &p, 1, &mut row);
        assert_eq!(&row[..7], [1.5, 0.0, 0.1, 0.2, 0.3, 0.4, 6.25]);
        assert_eq!(&row[7..9], [0.5, 4.0]);
        assert_eq!(&row[9..13], [1.5, 0.25, 0.0, -0.5]);
        assert_eq!(row[13], 1e-12);
        // Fewer multipliers than `count` pad with NaN, and a missing orbit is NaN.
        p.eigenvalues = vec![(1.5, 0.0)];
        pack_row(kind, &p, 1, &mut row);
        assert_eq!(row[9], 1.5);
        assert!(row[10].is_nan() && row[12].is_nan());
        assert_eq!(row[11], 0.0);
        p.u.truncate(3);
        pack_row(kind, &p, 1, &mut row);
        assert!(row[4].is_nan() && row[6].is_nan());
        // Without `floquet` the row has no multiplier columns and ends with the trivial error.
        let kind = Kind::DdeLc {
            ntst: 2,
            ncol: 2,
            count: 0,
            adapt: false,
        };
        assert_eq!(stride(kind, 1), 1 + 5 + 1 + 2 + 1);
        let mut row = vec![0.0; stride(kind, 1)];
        p.u = profile.iter().copied().chain([6.25]).collect();
        p.tests = [f64::NAN; 3];
        pack_row(kind, &p, 1, &mut row);
        assert_eq!(&row[..7], [1.5, 0.0, 0.1, 0.2, 0.3, 0.4, 6.25]);
        assert_eq!(&row[7..9], [0.5, 4.0]);
        assert!(row[9].is_nan());
        assert_eq!(kind.name(), "LC");
        assert_eq!(kind.labels(), ["LPC", "PD", "NS", "BPC"]);
        assert_eq!(kind.mesh_size(), None);
    }

    #[test]
    fn a_delay_cycle_row_with_adaptation_ends_with_the_mesh() {
        // n = 1, ntst 2, ncol 2, two multipliers: the row of slice 4 plus N + 1 mesh values.
        let profile = [0.0, 0.1, 0.2, 0.3, 0.4];
        let mut p = point();
        p.u = profile.iter().copied().chain([6.25]).collect();
        p.alpha = 1.5;
        p.h = 0.5;
        p.newton_iters = 4;
        p.eigenvalues = vec![(1.5, 0.0), (0.25, -0.5)];
        p.tests = [1e-12, f64::NAN, f64::NAN];
        p.mesh = vec![0.0, 0.375, 1.0];
        let plain = Kind::DdeLc {
            ntst: 2,
            ncol: 2,
            count: 2,
            adapt: false,
        };
        let kind = Kind::DdeLc {
            ntst: 2,
            ncol: 2,
            count: 2,
            adapt: true,
        };
        assert_eq!(stride(kind, 1), stride(plain, 1) + 3);
        let mut slice4 = vec![0.0; stride(plain, 1)];
        pack_row(plain, &p, 1, &mut slice4);
        let mut row = vec![0.0; stride(kind, 1)];
        pack_row(kind, &p, 1, &mut row);
        // The part before the mesh is the row of a run without adaptation, bit for bit.
        let head = stride(plain, 1);
        assert!(
            row[..head]
                .iter()
                .zip(&slice4)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(&row[head..], [0.0, 0.375, 1.0]);
        // A mesh of the wrong length is NaN, and so is a missing one.
        p.mesh = vec![0.0, 1.0];
        pack_row(kind, &p, 1, &mut row);
        assert!(row[head..].iter().all(|v| v.is_nan()));
        p.mesh.clear();
        pack_row(kind, &p, 1, &mut row);
        assert!(row[head..].iter().all(|v| v.is_nan()));
        // Without multipliers the mesh follows the trivial error directly.
        let kind = Kind::DdeLc {
            ntst: 2,
            ncol: 2,
            count: 0,
            adapt: true,
        };
        p.mesh = vec![0.0, 0.5, 1.0];
        let mut row = vec![0.0; stride(kind, 1)];
        pack_row(kind, &p, 1, &mut row);
        assert_eq!(stride(kind, 1), 1 + 5 + 1 + 2 + 1 + 3);
        assert_eq!(row[9], 1e-12);
        assert_eq!(&row[10..], [0.0, 0.5, 1.0]);
    }
}
