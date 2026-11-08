//! Floquet multipliers of a periodic orbit of a delay equation (SPEC section 8m).
//!
//! The variational equation `v' = A(t) v + sum_k J_k(t) v(t - tau_k)` is written for the
//! history segment on `[-tau_max, 0]`, which is stored by its values at Chebyshev extreme
//! points. That turns it into an ODE `W' = G(t) W` of dimension `n (nhist + 1)`. The matrix
//! `Phi(T)` of that ODE is built with the Gauss collocation Runge-Kutta method on the mesh
//! intervals of the orbit, and the multipliers are its eigenvalues.
//!
//! The formulation is my own derivation; it is checked against a numerical oracle in the
//! tests, not against a published formula.

use super::cycle::{DdeCycleSetup, DdeCycleStart, check_mesh, check_profile, check_setup, locate};
use super::eq::time_of;
use super::nodes::Nodes;
use crate::cycle::{CollocationBasis, try_zeros};
use crate::error::EngineError;
use crate::model::DdeModel;
use pc_ir::EntryKind;
use pc_linalg::Lu;
use pc_linalg::eigenvalues;
use std::sync::atomic::{AtomicBool, Ordering};

/// Largest order `m n (nhist + 1)` of the stage system of one mesh interval. The system is
/// dense, so this bounds its memory at 8 MB.
const MAX_STAGE: usize = 1000;

/// How many history nodes to use and how many multipliers to keep. There is no default for
/// either.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeFloquetOptions {
    /// The history is stored at `nhist + 1` Chebyshev points, `nhist >= 1`.
    pub nhist: usize,
    /// How many multipliers to return, `1 <= count <= n (nhist + 1)`.
    pub count: usize,
}

/// The multipliers of one orbit.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeFloquet {
    /// `(re, im)` pairs, largest modulus first, `count` of them. A cut can split a
    /// conjugate pair.
    pub multipliers: Vec<(f64, f64)>,
    /// `min |mu - 1|` over all `n (nhist + 1)` multipliers. The multiplier 1 always exists
    /// for a periodic orbit, so this measures the discretization error.
    pub trivial_error: f64,
}

/// The checks on `options` that do not need the orbit: the model has a delay, `nhist >= 1`,
/// the stage system fits in [`MAX_STAGE`] unknowns and `1 <= count <= d`. `n` is the number of
/// states. Returns `d = n (nhist + 1)`.
pub(super) fn check_floquet(
    model: &DdeModel,
    setup: &DdeCycleSetup,
    n: usize,
    options: &DdeFloquetOptions,
) -> Result<usize, EngineError> {
    if model.delays.is_empty() {
        return Err(EngineError::BadProblem(
            "the model has no delay; a periodic orbit of an ODE has no history to follow".into(),
        ));
    }
    let nhist = options.nhist;
    if nhist < 1 {
        return Err(EngineError::BadProblem("nhist must be at least 1".into()));
    }
    let mm = setup.ncol;
    let d = nhist
        .checked_add(1)
        .and_then(|p| p.checked_mul(n))
        .filter(|&d| d.checked_mul(mm).is_some_and(|s| s <= MAX_STAGE))
        .ok_or_else(|| {
            EngineError::BadProblem(format!(
                "ncol = {mm}, nhist = {nhist} and n = {n} give a stage system of more than \
                 {MAX_STAGE} unknowns"
            ))
        })?;
    if options.count < 1 || options.count > d {
        return Err(EngineError::BadProblem(format!(
            "count = {}; it must be between 1 and {d} = n (nhist + 1)",
            options.count
        )));
    }
    Ok(d)
}

/// The summary of all `d` multipliers: `trivial_error` over all of them, then the leading
/// `count` by modulus. Empty input (an eigenvalue computation that failed along a branch) gives
/// no multipliers and a NaN `trivial_error`.
pub(super) fn summarize(mut all: Vec<(f64, f64)>, count: usize) -> DdeFloquet {
    if all.is_empty() {
        return DdeFloquet {
            multipliers: all,
            trivial_error: f64::NAN,
        };
    }
    let trivial_error = all
        .iter()
        .map(|&(re, im)| (re - 1.0).hypot(im))
        .fold(f64::INFINITY, f64::min);
    all.sort_by(|a, b| b.0.hypot(b.1).total_cmp(&a.0.hypot(a.1)));
    all.truncate(count);
    DdeFloquet {
        multipliers: all,
        trivial_error,
    }
}

/// The Floquet multipliers of the orbit `cycle` of the model at `setup.params`. The profile
/// need not solve the collocation system exactly, but the result only means something for a
/// solution.
pub fn floquet_multipliers(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    cycle: &DdeCycleStart,
    options: &DdeFloquetOptions,
    cancel: &AtomicBool,
) -> Result<DdeFloquet, EngineError> {
    let (phi, d) = monodromy(model, setup, cycle, options, cancel)?;
    let all = eigenvalues(&phi, d)?;
    Ok(summarize(all, options.count))
}

/// The matrix `Phi(T)` of the history ODE, column-major `d` by `d` with `d = n (nhist + 1)`,
/// after all the checks of [`floquet_multipliers`].
pub(super) fn monodromy(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    cycle: &DdeCycleStart,
    options: &DdeFloquetOptions,
    cancel: &AtomicBool,
) -> Result<(Vec<f64>, usize), EngineError> {
    monodromy_on(model, setup, cycle, options, None, cancel)
}

/// [`monodromy`] on the mesh `mesh` (`ntst + 1` points from 0 to 1) of the profile, or on the
/// uniform mesh for `None`. Interval `j` has the step `h_j = T (s_{j+1} - s_j)`, its Gauss
/// points sit at `s_j + c_i (s_{j+1} - s_j)`, and the delayed points are found on the mesh
/// (SPEC section 8p). `None` takes the formulas of the uniform mesh unchanged.
pub(super) fn monodromy_on(
    model: &mut DdeModel,
    setup: &DdeCycleSetup,
    cycle: &DdeCycleStart,
    options: &DdeFloquetOptions,
    mesh: Option<&[f64]>,
    cancel: &AtomicBool,
) -> Result<(Vec<f64>, usize), EngineError> {
    let (n, _) = check_setup(model, setup)?;
    check_profile(&cycle.profile, n, setup)?;
    if let Some(mesh) = mesh {
        check_mesh(mesh, setup.ntst)?;
    }
    if !(cycle.period.is_finite() && cycle.period > 0.0) {
        return Err(EngineError::BadProblem(format!(
            "the period is {}; it must be finite and positive",
            cycle.period
        )));
    }
    let d = check_floquet(model, setup, n, options)?;
    let n_s = model.delays.len();
    let nhist = options.nhist;
    let mm = setup.ncol;

    let basis = CollocationBasis::new(mm)?;
    let ntst = setup.ntst;
    let t = cycle.period;
    let params = setup.params.clone();
    let taus: Vec<f64> = model.delays.iter().map(|s| time_of(s, &params)).collect();
    let tau_max = taus.iter().fold(0.0_f64, |a, v| a.max(*v));
    let nodes = Nodes::chebyshev(nhist, tau_max);
    // The delayed value of the history at -tau_k as a combination of the nodal values.
    let lag: Vec<Vec<f64>> = taus.iter().map(|&tau| nodes.lagrange_at(-tau)).collect();
    let a_irk = irk_matrix(&basis);
    let (c, bw) = (basis.nodes(), basis.weights());
    // The step of interval `j`: `T / N` on the uniform mesh.
    let h_of = |j: usize| match mesh {
        None => t / ntst as f64,
        Some(mesh) => t * (mesh[j + 1] - mesh[j]),
    };

    let n_user = model.n_user_params;
    let n_p = model.eval.n_p();
    let mut full = params;
    full.resize(n_user + n_s + 1, 0.0);
    let states: Vec<usize> = model.delays.iter().map(|s| s.state).collect();
    let nn = ntst as f64;
    let md = mm * d;

    let mut u = vec![0.0; n];
    let mut ju = vec![0.0; n * n];
    let mut jp = vec![0.0; n * n_p];
    let mut lv = vec![0.0; mm + 1];
    // One G per Gauss point of the interval, column-major d by d, and the stage system.
    let mut gs: Vec<Vec<f64>> = (0..mm).map(|_| vec![0.0; d * d]).collect();
    let mut stage = try_zeros(md.checked_mul(md))?;
    let mut phi = try_zeros(d.checked_mul(d))?;
    for i in 0..d {
        phi[i + d * i] = 1.0;
    }
    let mut rhs = vec![0.0; md];
    let mut gy = vec![0.0; d];
    for j in 0..ntst {
        if cancel.load(Ordering::Relaxed) {
            return Err(EngineError::Cancelled);
        }
        let at = j * mm * n;
        let h = h_of(j);
        for i in 0..mm {
            for (cc, uc) in u.iter_mut().enumerate() {
                *uc = (0..=mm)
                    .map(|l| basis.value_at_node(i, l) * cycle.profile[at + l * n + cc])
                    .sum();
            }
            let s = match mesh {
                None => (j as f64 + c[i]) / nn,
                Some(mesh) => mesh[j] + c[i] * (mesh[j + 1] - mesh[j]),
            };
            for k in 0..n_s {
                let sp = (s - taus[k] / t).rem_euclid(1.0);
                let (jk, zeta) = match mesh {
                    None => {
                        let jk = ((nn * sp).floor() as usize).min(ntst - 1);
                        (jk, nn * sp - jk as f64)
                    }
                    Some(mesh) => locate(mesh, sp),
                };
                basis.values(zeta, &mut lv);
                let base = jk * mm * n;
                full[n_user + k] = (0..=mm)
                    .map(|l| lv[l] * cycle.profile[base + l * n + states[k]])
                    .sum();
            }
            model.eval.eval(&EntryKind::JacU, &u, &full, &[], &mut ju)?;
            model.eval.eval(&EntryKind::JacP, &u, &full, &[], &mut jp)?;
            let g = &mut gs[i];
            g.fill(0.0);
            // Rows 1..=nhist: the history moves along the segment.
            for jj in 1..=nhist {
                for l in 0..=nhist {
                    let dv = nodes.diff(jj, l);
                    for cc in 0..n {
                        g[(jj * n + cc) + d * (l * n + cc)] = dv;
                    }
                }
            }
            // Row 0: the equation, with the delayed values read from the interpolant.
            for cc in 0..n {
                for c2 in 0..n {
                    g[cc + d * c2] += ju[cc + n * c2];
                }
                for k in 0..n_s {
                    let jk = jp[cc + n * (n_user + k)];
                    for (l, w) in lag[k].iter().enumerate() {
                        g[cc + d * (l * n + states[k])] += jk * w;
                    }
                }
            }
            if g.iter().any(|v| !v.is_finite()) {
                return Err(EngineError::NotFinite("a model derivative".into()));
            }
        }
        // The stage system: Y_i - h sum_l a_il G_l Y_l = Phi_a.
        stage.fill(0.0);
        for i in 0..md {
            stage[i + md * i] = 1.0;
        }
        for i in 0..mm {
            for l in 0..mm {
                let f = h * a_irk[i * mm + l];
                let g = &gs[l];
                for col in 0..d {
                    for row in 0..d {
                        stage[(i * d + row) + md * (l * d + col)] -= f * g[row + d * col];
                    }
                }
            }
        }
        let lu = Lu::new(&stage, md)?;
        for q in 0..d {
            for i in 0..mm {
                rhs[i * d..(i + 1) * d].copy_from_slice(&phi[d * q..d * (q + 1)]);
            }
            lu.solve(&mut rhs)?;
            // Phi_b = Phi_a + h sum_i b_i G_i Y_i, one column at a time.
            for i in 0..mm {
                let y = &rhs[i * d..(i + 1) * d];
                let g = &gs[i];
                gy.fill(0.0);
                for (col, yc) in y.iter().enumerate() {
                    for row in 0..d {
                        gy[row] += g[row + d * col] * yc;
                    }
                }
                for row in 0..d {
                    phi[row + d * q] += h * bw[i] * gy[row];
                }
            }
        }
        if phi.iter().any(|v| !v.is_finite()) {
            return Err(EngineError::NotFinite("the monodromy matrix".into()));
        }
    }

    Ok((phi, d))
}

/// The Gauss Runge-Kutta matrix `a_il = integral_0^{c_i} ell_l(z) dz` with `ell_l` the
/// Lagrange basis on the Gauss nodes, row-major `m` by `m`. Each `ell_l` has degree `m - 1`,
/// so the Gauss rule with `m` nodes integrates it exactly.
fn irk_matrix(basis: &CollocationBasis) -> Vec<f64> {
    let (c, b) = (basis.nodes(), basis.weights());
    let m = c.len();
    let ell = |l: usize, z: f64| -> f64 {
        (0..m)
            .filter(|&k| k != l)
            .map(|k| (z - c[k]) / (c[l] - c[k]))
            .product()
    };
    let mut a = vec![0.0; m * m];
    for i in 0..m {
        for l in 0..m {
            a[i * m + l] = c[i] * (0..m).map(|q| b[q] * ell(l, c[i] * c[q])).sum::<f64>();
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build_dde};
    use pc_ode::{MapLoader, SourceMap, load};
    use std::f64::consts::{FRAC_PI_2, PI};

    const NTST: usize = 20;
    const NCOL: usize = 4;

    fn osc() -> DdeModel {
        let src = format!(
            "par mu=0.3, tau={FRAC_PI_2:?}\nx'=-delay(x,tau)+mu*x*(1-x^2-delay(x,tau)^2)\ndone\n"
        );
        let mut map = SourceMap::new();
        let top = map.add("t.ode", &src);
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, Evaluator::Fallback).expect("model builds")
    }

    fn hand_mesh() -> Vec<f64> {
        let mut mesh = vec![0.0];
        for j in 0..NTST {
            let w = if j % 2 == 0 { 0.5 } else { 1.5 };
            mesh.push(mesh[j] + w / NTST as f64);
        }
        mesh[NTST] = 1.0;
        mesh
    }

    fn uniform_mesh() -> Vec<f64> {
        (0..=NTST).map(|j| j as f64 / NTST as f64).collect()
    }

    /// The exact orbit `sin(2 pi s)` of OSC at its base points on `mesh`.
    fn sine_start(mesh: &[f64]) -> DdeCycleStart {
        let mut profile = Vec::new();
        for j in 0..NTST {
            for l in 0..NCOL {
                let s = mesh[j] + (l as f64 / NCOL as f64) * (mesh[j + 1] - mesh[j]);
                profile.push((2.0 * PI * s).sin());
            }
        }
        profile.push(0.0);
        DdeCycleStart {
            profile,
            period: 2.0 * PI,
        }
    }

    fn setup() -> DdeCycleSetup {
        DdeCycleSetup {
            params: vec![0.3, FRAC_PI_2],
            free: 0,
            ntst: NTST,
            ncol: NCOL,
        }
    }

    fn options() -> DdeFloquetOptions {
        DdeFloquetOptions {
            nhist: 16,
            count: 2,
        }
    }

    fn leading(phi: &[f64], d: usize) -> Vec<(f64, f64)> {
        summarize(eigenvalues(phi, d).expect("eigenvalues"), 2).multipliers
    }

    fn max_gap(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold(0.0, |w, (x, y)| w.max((x.0 - y.0).hypot(x.1 - y.1)))
    }

    #[test]
    fn an_explicit_uniform_mesh_gives_the_same_eigenvalues() {
        let mut m = osc();
        let cancel = AtomicBool::new(false);
        let start = sine_start(&uniform_mesh());
        let (a, d) = monodromy(&mut m, &setup(), &start, &options(), &cancel).unwrap();
        let mesh = uniform_mesh();
        let (b, _) =
            monodromy_on(&mut m, &setup(), &start, &options(), Some(&mesh), &cancel).unwrap();
        let gap = max_gap(&leading(&a, d), &leading(&b, d));
        assert!(gap < 1e-10, "gap {gap:e}");
    }

    /// The same orbit sampled on a mesh with intervals of 0.5 / N and 1.5 / N gives the same
    /// leading multipliers as the uniform mesh, up to the discretization error.
    #[test]
    fn a_hand_made_mesh_gives_the_leading_multipliers_of_the_uniform_mesh() {
        let mut m = osc();
        let cancel = AtomicBool::new(false);
        let uniform = floquet_multipliers(
            &mut m,
            &setup(),
            &sine_start(&uniform_mesh()),
            &options(),
            &cancel,
        )
        .unwrap();
        let mesh = hand_mesh();
        let (phi, d) = monodromy_on(
            &mut m,
            &setup(),
            &sine_start(&mesh),
            &options(),
            Some(&mesh),
            &cancel,
        )
        .unwrap();
        let hand = leading(&phi, d);
        let gap = max_gap(&uniform.multipliers, &hand);
        assert!(
            gap < 1.2e-7,
            "gap {gap:e}: {:?} vs {hand:?}",
            uniform.multipliers
        );
    }

    #[test]
    fn a_bad_mesh_is_refused() {
        let mut m = osc();
        let cancel = AtomicBool::new(false);
        let start = sine_start(&uniform_mesh());
        let mut bad = uniform_mesh();
        bad.pop();
        let r = monodromy_on(&mut m, &setup(), &start, &options(), Some(&bad), &cancel);
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{r:?}");
    }

    #[test]
    fn irk_matrix_has_the_gauss_row_sums_and_order_conditions() {
        for m in 2..=7usize {
            let basis = CollocationBasis::new(m).unwrap();
            let a = irk_matrix(&basis);
            let (c, b) = (basis.nodes(), basis.weights());
            // sum_l a_il = c_i, and sum_l a_il c_l^(k-1) = c_i^k / k for k <= m.
            for i in 0..m {
                for k in 1..=m {
                    let got: f64 = (0..m).map(|l| a[i * m + l] * c[l].powi(k as i32 - 1)).sum();
                    let want = c[i].powi(k as i32) / k as f64;
                    assert!(
                        (got - want).abs() <= 1e-13,
                        "m={m} i={i} k={k}: {got} vs {want}"
                    );
                }
            }
            // sum_i b_i c_i^(k-1) = 1/k for k <= 2m.
            for k in 1..=2 * m {
                let got: f64 = (0..m).map(|i| b[i] * c[i].powi(k as i32 - 1)).sum();
                assert!((got - 1.0 / k as f64).abs() <= 1e-13, "m={m} k={k}");
            }
        }
    }
}
