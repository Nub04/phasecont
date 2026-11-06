//! Normal form coefficients of cycle bifurcations by periodic normalization (SPEC §8a).
//!
//! The coefficient at a bifurcation of a limit cycle comes from linear boundary value
//! problems along the cycle, solved on the collocation mesh of the cycle itself, and from
//! integrals of the second and third derivatives of f over one period. No Poincaré map is
//! formed. The formulas are in SPEC §8a with their sources.
//!
//! Every problem is the same kind: g' = T (M(tau) g + r(tau) - s q(tau)) on tau in [0, 1],
//! with a boundary condition, one border row and one scalar unknown s. [`Grid::solve`]
//! assembles it in the block layout of `pc-linalg` and solves it with the condensation
//! solver, as the continuation does for the cycle itself.

use super::basis::{CollocationBasis, NCOL_RANGE};
use crate::branch::{Branch, Point};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::{At, Cv};
use pc_ir::EntryKind;
use pc_linalg::{
    BlockSystem, CondenseShape, Condenser, eigenvalues, eigvec, null_left, null_right,
};
use std::f64::consts::PI;

/// A bifurcation of a limit cycle that has a normal form coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleBifurcation {
    /// A fold of cycles (limit point of cycles).
    Lpc,
    /// A period doubling (flip).
    Pd,
    /// A torus bifurcation (Neimark-Sacker).
    Ns,
}

/// The normal form coefficient of a cycle bifurcation.
#[derive(Clone, Debug, PartialEq)]
pub struct CycleNormalForm {
    /// `b` of an LPC point, `c` of a PD point, the real part of `d` of an NS point.
    pub coefficient: f64,
    /// The imaginary part of the coefficient of an NS point; `None` otherwise.
    pub coefficient2: Option<f64>,
    /// The normalization the coefficient used.
    pub normalization: &'static str,
}

/// The normal form coefficient at `point`, a special point of a cycle branch (as in
/// [`continue_cycle`](super::continue_cycle)) in parameter `free`, with the other
/// parameters from `params` (all `n_p` values, without t). `ncol` is the number of
/// collocation points per interval of the point's mesh.
pub fn lc_point_normal_form(
    model: &mut dyn ModelEval,
    kind: CycleBifurcation,
    free: usize,
    params: &[f64],
    point: &Point,
    ncol: usize,
) -> Result<CycleNormalForm, EngineError> {
    let bad = |m: String| Err(EngineError::BadProblem(m));
    let (n, n_p) = (model.n(), model.n_p());
    if params.len() != n_p || free >= n_p {
        return bad("normal form: parameters or free index do not fit the model".into());
    }
    let Some((period, orbit)) = point.u.split_last() else {
        return bad("normal form: the point has no orbit".into());
    };
    let nint = point.mesh.len().saturating_sub(1);
    if nint == 0 || !NCOL_RANGE.contains(&ncol) || orbit.len() != n * (nint * ncol + 1) {
        return bad("normal form: the point does not match the mesh and NCOL".into());
    }
    let mut p = params.to_vec();
    p[free] = point.alpha;
    p.push(0.0);
    normal_form(model, kind, &p, orbit, *period, &point.mesh, ncol)
}

/// The normal form coefficient at `point` of a curve of cycles in the parameters `free` =
/// [a1, a2] (as in [`continue_lpc_curve`](super::continue_lpc_curve)), with the other
/// parameters from `params`. The point holds the orbit, T and a1 in `Point::u`, followed by
/// kappa on an NS curve, and a2 in `Point::alpha`. `kind` is the bifurcation the curve
/// follows. `ncol` is the number of collocation points per interval of the point's mesh.
pub fn curve_point_normal_form(
    model: &mut dyn ModelEval,
    kind: CycleBifurcation,
    free: [usize; 2],
    params: &[f64],
    point: &Point,
    ncol: usize,
) -> Result<CycleNormalForm, EngineError> {
    let bad = |m: String| Err(EngineError::BadProblem(m));
    let (n, n_p) = (model.n(), model.n_p());
    if params.len() != n_p || free[0] >= n_p || free[1] >= n_p || free[0] == free[1] {
        return bad("normal form: parameters or free indices do not fit the model".into());
    }
    // An NS curve point carries kappa after a1, which the coefficient does not use.
    let x = match kind {
        CycleBifurcation::Ns => point.u.split_last().map_or(&[][..], |(_, x)| x),
        _ => &point.u[..],
    };
    let [orbit @ .., period, a1] = x else {
        return bad("normal form: the curve point has no orbit".into());
    };
    let nint = point.mesh.len().saturating_sub(1);
    if nint == 0 || !NCOL_RANGE.contains(&ncol) || orbit.len() != n * (nint * ncol + 1) {
        return bad("normal form: the point does not match the mesh and NCOL".into());
    }
    let mut p = params.to_vec();
    p[free[0]] = *a1;
    p[free[1]] = point.alpha;
    p.push(0.0);
    normal_form(model, kind, &p, orbit, *period, &point.mesh, ncol)
}

/// Gives every LPC, PD and NS special point of the cycle branch `branch` its normal form
/// coefficient: `Extras::coefficient` is b, c or Re d, `coefficient2` is Im d of an NS
/// point and `normalization` names the conditions. A point whose coefficient cannot be
/// computed keeps empty extras and adds a warning, the run itself is not affected.
pub(crate) fn attach_coefficients(
    model: &mut dyn ModelEval,
    branch: &mut Branch,
    free: usize,
    params: &[f64],
    ncol: usize,
) {
    for sp in branch.specials.iter_mut() {
        let kind = match sp.label.as_str() {
            "LPC" => CycleBifurcation::Lpc,
            "PD" => CycleBifurcation::Pd,
            "NS" => CycleBifurcation::Ns,
            _ => continue,
        };
        match lc_point_normal_form(model, kind, free, params, &sp.point, ncol) {
            Ok(nf) => {
                sp.extras.coefficient = Some(nf.coefficient);
                sp.extras.coefficient2 = nf.coefficient2;
                sp.extras.normalization = Some(nf.normalization.to_string());
            }
            Err(e) => branch.warnings.push(format!(
                "{} point at alpha = {}: no normal form coefficient: {e}",
                sp.label, sp.point.alpha
            )),
        }
    }
}

/// The coefficient for the orbit at the base points of `mesh`, with `p` the parameters
/// and t = 0.
pub(crate) fn normal_form(
    model: &mut dyn ModelEval,
    kind: CycleBifurcation,
    p: &[f64],
    orbit: &[f64],
    period: f64,
    mesh: &[f64],
    ncol: usize,
) -> Result<CycleNormalForm, EngineError> {
    let n = model.n();
    let nint = mesh.len().saturating_sub(1);
    let ok = nint > 0
        && mesh[0] == 0.0
        && mesh[nint] == 1.0
        && mesh.windows(2).all(|w| w[1] > w[0])
        && orbit.len() == n * (nint * ncol + 1)
        && orbit.iter().all(|v| v.is_finite())
        && period.is_finite()
        && period > 0.0;
    if !ok {
        return Err(EngineError::BadProblem(
            "normal form: the mesh, orbit or period is not valid".into(),
        ));
    }
    let grid = Grid {
        n,
        basis: CollocationBasis::new(ncol)?,
        mesh: mesh.to_vec(),
        period,
    };
    let orb = grid.orbit(model, p, orbit)?;
    match kind {
        CycleBifurcation::Lpc => lpc(&grid, model, p, &orb),
        CycleBifurcation::Pd => pd(&grid, model, p, &orb),
        CycleBifurcation::Ns => ns(&grid, model, p, &orb),
    }
}

/// The mesh of the cycle and its collocation basis.
struct Grid {
    n: usize,
    basis: CollocationBasis,
    mesh: Vec<f64>,
    period: f64,
}

/// F and F_u of the cycle at the Gauss nodes, node q = j m + i for interval j.
struct Orbit {
    u: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    a: Vec<Vec<f64>>,
}

/// How a problem closes at tau = 0 and 1.
enum Bc {
    /// g(1) = sigma g(0), with sigma = 1 or -1.
    Periodic(f64),
    /// g(0) = w.
    Initial(Vec<f64>),
}

/// A linear problem g' = T (M g + r - s q) with one border row.
struct Lin<'a> {
    /// M at the Gauss nodes, nn by nn column-major.
    mat: &'a [Vec<f64>],
    rhs: Option<&'a [Vec<f64>]>,
    col: Option<&'a [Vec<f64>]>,
    bc: Bc,
    /// The border row over the profile, and its weight on s.
    border: Vec<f64>,
    border_s: f64,
    border_rhs: f64,
}

impl Grid {
    fn m(&self) -> usize {
        self.basis.m()
    }

    fn nint(&self) -> usize {
        self.mesh.len() - 1
    }

    /// Number of base points N m + 1.
    fn points(&self) -> usize {
        self.nint() * self.m() + 1
    }

    /// Number of Gauss nodes N m.
    fn nodes(&self) -> usize {
        self.nint() * self.m()
    }

    fn h(&self, j: usize) -> f64 {
        self.mesh[j + 1] - self.mesh[j]
    }

    /// The profile of nn values per base point at the Gauss nodes.
    fn at_nodes(&self, nn: usize, profile: &[f64]) -> Vec<Vec<f64>> {
        let m = self.m();
        let mut out = Vec::with_capacity(self.nodes());
        for j in 0..self.nint() {
            for i in 0..m {
                out.push(
                    (0..nn)
                        .map(|c| {
                            (0..=m)
                                .map(|l| {
                                    self.basis.value_at_node(i, l) * profile[(j * m + l) * nn + c]
                                })
                                .sum()
                        })
                        .collect(),
                );
            }
        }
        out
    }

    fn orbit(
        &self,
        model: &mut dyn ModelEval,
        p: &[f64],
        profile: &[f64],
    ) -> Result<Orbit, EngineError> {
        let n = self.n;
        let u = self.at_nodes(n, profile);
        let (mut f, mut a) = (Vec::new(), Vec::new());
        for x in &u {
            let mut fx = vec![0.0; n];
            let mut ax = vec![0.0; n * n];
            model.eval(&EntryKind::F, x, p, &[], &mut fx)?;
            model.eval(&EntryKind::JacU, x, p, &[], &mut ax)?;
            f.push(fx);
            a.push(ax);
        }
        Ok(Orbit { u, f, a })
    }

    /// The integral over [0, T] of a function given at the Gauss nodes.
    fn integrate(&self, vals: &[f64]) -> f64 {
        let (m, w) = (self.m(), self.basis.weights());
        let mut s = 0.0;
        for j in 0..self.nint() {
            for i in 0..m {
                s += self.h(j) * w[i] * vals[j * m + i];
            }
        }
        self.period * s
    }

    /// The row ell over a profile of nn values per base point with ell . g equal to the
    /// integral over [0, T] of psi(t)^T g(t), for psi at the Gauss nodes.
    fn functional(&self, nn: usize, psi: &[Vec<f64>]) -> Vec<f64> {
        let (m, w) = (self.m(), self.basis.weights());
        let mut row = vec![0.0; nn * self.points()];
        for j in 0..self.nint() {
            for i in 0..m {
                let wt = self.period * self.h(j) * w[i];
                for l in 0..=m {
                    let c = wt * self.basis.value_at_node(i, l);
                    let at = (j * m + l) * nn;
                    for (r, v) in row[at..at + nn].iter_mut().zip(&psi[j * m + i]) {
                        *r += c * v;
                    }
                }
            }
        }
        row
    }

    /// The interval blocks of `lin` and the right-hand side of the interval rows.
    fn blocks(
        &self,
        nn: usize,
        lin_mat: &[Vec<f64>],
        col: Option<&[Vec<f64>]>,
        rhs: Option<&[Vec<f64>]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let m = self.m();
        let nm = nn * m;
        let tcol = nn * (m + 1);
        let size = nm * (tcol + 1);
        let mut blocks = vec![0.0; self.nint() * size];
        let mut r = vec![0.0; self.nint() * nm];
        for j in 0..self.nint() {
            let ht = self.h(j) * self.period;
            let blk = &mut blocks[j * size..(j + 1) * size];
            for i in 0..m {
                let mat = &lin_mat[j * m + i];
                for c in 0..nn {
                    let row = i * nn + c;
                    for l in 0..=m {
                        let d = self.basis.derivative_at_node(i, l);
                        let v = ht * self.basis.value_at_node(i, l);
                        for c2 in 0..nn {
                            let e = -v * mat[c + nn * c2];
                            blk[row + nm * (l * nn + c2)] = if c2 == c { e + d } else { e };
                        }
                    }
                    if let Some(q) = col {
                        blk[row + nm * tcol] = ht * q[j * m + i][c];
                    }
                    if let Some(src) = rhs {
                        r[j * nm + row] = ht * src[j * m + i][c];
                    }
                }
            }
        }
        (blocks, r)
    }

    /// The boundary rows, nn by 2 nn + 1 column-major.
    fn boundary(nn: usize, sigma: f64) -> Vec<f64> {
        let mut bc = vec![0.0; nn * (2 * nn + 1)];
        for c in 0..nn {
            bc[c + nn * c] = 1.0;
            bc[c + nn * (nn + c)] = -sigma;
        }
        bc
    }

    /// The monodromy matrix of g' = T M g, n by n column-major, mapping g(0) to g(1).
    fn monodromy(&self, nn: usize, mat: &[Vec<f64>]) -> Result<Vec<f64>, EngineError> {
        let (blocks, _) = self.blocks(nn, mat, None, None);
        let shape = CondenseShape::new(nn, self.m(), self.nint(), 1)?;
        let bc = Grid::boundary(nn, 1.0);
        let border = vec![0.0; shape.dim()];
        let sys = BlockSystem {
            blocks: &blocks,
            bc: &bc,
            border: &border,
        };
        let mut cond = Condenser::new(shape)?;
        cond.condense_intervals(&sys, shape.parallel_pays())?;
        Ok(cond.monodromy()?)
    }

    /// Solves `lin` for the profile of nn values per base point and the scalar s.
    fn solve(&self, nn: usize, lin: &Lin<'_>) -> Result<(Vec<f64>, f64), EngineError> {
        let (blocks, mut rhs) = self.blocks(nn, lin.mat, lin.col, lin.rhs);
        let shape = CondenseShape::new(nn, self.m(), self.nint(), 1)?;
        let (bc, bc_rhs) = match &lin.bc {
            Bc::Periodic(sigma) => (Grid::boundary(nn, *sigma), vec![0.0; nn]),
            Bc::Initial(w) => {
                let mut bc = vec![0.0; nn * (2 * nn + 1)];
                for c in 0..nn {
                    bc[c + nn * c] = 1.0;
                }
                (bc, w.clone())
            }
        };
        let ext = nn * self.points();
        let mut border = vec![0.0; shape.dim()];
        border[..ext].copy_from_slice(&lin.border);
        border[ext] = lin.border_s;
        rhs.extend(bc_rhs);
        rhs.push(lin.border_rhs);
        let sys = BlockSystem {
            blocks: &blocks,
            bc: &bc,
            border: &border,
        };
        let mut cond = Condenser::new(shape)?;
        let mut sol = vec![0.0; shape.dim()];
        cond.solve(&sys, &rhs, &mut sol, shape.parallel_pays())?;
        let s = sol[ext];
        sol.truncate(ext);
        Ok((sol, s))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `-A^T` for each node.
fn minus_transpose(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    a.iter()
        .map(|m| {
            let mut t = vec![0.0; n * n];
            for c in 0..n {
                for c2 in 0..n {
                    t[c + n * c2] = -m[c2 + n * c];
                }
            }
            t
        })
        .collect()
}

/// The multilinear form of the real directions `dirs` at the state u: B(a, b) for two,
/// C(a, b, c) for three.
fn real_form(
    model: &mut dyn ModelEval,
    p: &[f64],
    u: &[f64],
    dirs: &[&[f64]],
) -> Result<Vec<f64>, EngineError> {
    let z = vec![0.0; u.len()];
    let cv: Vec<Cv> = dirs.iter().map(|d| (d.to_vec(), z.clone())).collect();
    let refs: Vec<&Cv> = cv.iter().collect();
    Ok(At::new(model, u.len(), u, p.to_vec()).form(&refs)?.0)
}

/// `A x` for A n by n column-major.
fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|r| (0..n).map(|c| a[r + n * c] * x[c]).sum())
        .collect()
}

/// `b` of a fold of cycles, from BLSK eq. 60 (SPEC §8a): with F = u0', phi1 the periodic
/// solution of phi1' - A phi1 = -F with int <phi1, F> = 0, and phi* the periodic solution
/// of phi*' = -A^T phi* with int <phi*, phi1> = 1, b = 1/2 int <phi*, B(phi1, phi1) -
/// 2 A phi1>.
fn lpc(
    g: &Grid,
    model: &mut dyn ModelEval,
    p: &[f64],
    orb: &Orbit,
) -> Result<CycleNormalForm, EngineError> {
    let n = g.n;
    let pts = g.points();
    // phi*(0) spans the left null space of M - I, M the monodromy matrix: at the fold the
    // multiplier 1 is double with one eigenvector, so M - I has rank n - 1.
    let mut mono = g.monodromy(n, &orb.a)?;
    for c in 0..n {
        mono[c + n * c] -= 1.0;
    }
    let w = null_left(&mono, n, n, 1)?.swap_remove(0);
    let adj = minus_transpose(&orb.a, n);
    let (ps, _) = g.solve(
        n,
        &Lin {
            mat: &adj,
            rhs: None,
            col: None,
            bc: Bc::Initial(w),
            border: vec![0.0; n * pts],
            border_s: 1.0,
            border_rhs: 0.0,
        },
    )?;
    let ps_nodes = g.at_nodes(n, &ps);
    let neg_f: Vec<Vec<f64>> = orb
        .f
        .iter()
        .map(|f| f.iter().map(|v| -v).collect())
        .collect();
    let (phi1, _) = g.solve(
        n,
        &Lin {
            mat: &orb.a,
            rhs: Some(&neg_f),
            col: Some(&ps_nodes),
            bc: Bc::Periodic(1.0),
            border: g.functional(n, &orb.f),
            border_s: 0.0,
            border_rhs: 0.0,
        },
    )?;
    let phi1_nodes = g.at_nodes(n, &phi1);
    let overlap: Vec<f64> = (0..g.nodes())
        .map(|q| dot(&ps_nodes[q], &phi1_nodes[q]))
        .collect();
    let scale = g.integrate(&overlap);
    if !scale.is_finite() || scale == 0.0 {
        return Err(EngineError::Singular(
            "the adjoint eigenfunction is orthogonal to phi1".into(),
        ));
    }
    let mut total = Vec::with_capacity(g.nodes());
    for q in 0..g.nodes() {
        let bb = real_form(model, p, &orb.u[q], &[&phi1_nodes[q], &phi1_nodes[q]])?;
        let a_phi = matvec(&orb.a[q], n, &phi1_nodes[q]);
        let rhs: Vec<f64> = (0..n).map(|c| bb[c] - 2.0 * a_phi[c]).collect();
        total.push(dot(&ps_nodes[q], &rhs));
    }
    let b = 0.5 * g.integrate(&total) / scale;
    if !b.is_finite() {
        return Err(EngineError::NotFinite("the fold coefficient b".into()));
    }
    Ok(CycleNormalForm {
        coefficient: b,
        coefficient2: None,
        normalization: "periodic: int <phi*, phi1> = 1 over [0, T]",
    })
}

/// `c` of a period doubling, from DWDGK eq. 28-35 (SPEC §8a): v is the antiperiodic
/// solution of v' = A v with int <v, v> = 1, v* the antiperiodic solution of v*' = -A^T v*
/// with int <v*, v> = 1, phi* the periodic adjoint with int <phi*, F> = 1, alpha1 =
/// 1/2 int <phi*, B(v, v)>, h2 the periodic solution of h2' = A h2 + B(v, v) - 2 alpha1 F
/// with int <phi*, h2> = 0, and c = 1/6 int <v*, C(v, v, v) + 3 B(v, h2) - 6 alpha1 A v>.
fn pd(
    g: &Grid,
    model: &mut dyn ModelEval,
    p: &[f64],
    orb: &Orbit,
) -> Result<CycleNormalForm, EngineError> {
    let n = g.n;
    let nodes = g.nodes();
    let zero = vec![0.0; n * g.points()];
    // v(0) and v*(0) span the right and left null spaces of M + I, M the monodromy matrix.
    let mut mono = g.monodromy(n, &orb.a)?;
    for c in 0..n {
        mono[c + n * c] += 1.0;
    }
    let v0 = null_right(&mono, n, n, 1)?.swap_remove(0);
    let w0 = null_left(&mono, n, n, 1)?.swap_remove(0);
    let initial = |mat: &[Vec<f64>], w: Vec<f64>| {
        g.solve(
            n,
            &Lin {
                mat,
                rhs: None,
                col: None,
                bc: Bc::Initial(w),
                border: zero.clone(),
                border_s: 1.0,
                border_rhs: 0.0,
            },
        )
    };
    let adj = minus_transpose(&orb.a, n);
    let mut v = g.at_nodes(n, &initial(&orb.a, v0)?.0);
    let mut vs = g.at_nodes(n, &initial(&adj, w0)?.0);
    let norm: Vec<f64> = v.iter().map(|x| dot(x, x)).collect();
    let nv = g.integrate(&norm).sqrt();
    let overlap: Vec<f64> = (0..nodes).map(|q| dot(&vs[q], &v[q])).collect();
    let ov = g.integrate(&overlap);
    if !(nv.is_finite() && nv > 0.0 && ov.is_finite() && ov != 0.0) {
        return Err(EngineError::Singular(
            "the flip eigenfunctions are degenerate".into(),
        ));
    }
    v.iter_mut().flatten().for_each(|x| *x /= nv);
    vs.iter_mut().flatten().for_each(|x| *x *= nv / ov);
    // phi*: the periodic adjoint, with the column F and the border int <F, g> = 1.
    let (phi, _) = g.solve(
        n,
        &Lin {
            mat: &adj,
            rhs: None,
            col: Some(&orb.f),
            bc: Bc::Periodic(1.0),
            border: g.functional(n, &orb.f),
            border_s: 0.0,
            border_rhs: 1.0,
        },
    )?;
    let phi = g.at_nodes(n, &phi);
    let bvv = (0..nodes)
        .map(|q| real_form(model, p, &orb.u[q], &[&v[q], &v[q]]))
        .collect::<Result<Vec<_>, _>>()?;
    let weighted: Vec<f64> = (0..nodes).map(|q| dot(&phi[q], &bvv[q])).collect();
    let alpha1 = 0.5 * g.integrate(&weighted);
    let (h2, _) = g.solve(
        n,
        &Lin {
            mat: &orb.a,
            rhs: Some(&bvv),
            col: Some(&orb.f),
            bc: Bc::Periodic(1.0),
            border: g.functional(n, &phi),
            border_s: 0.0,
            border_rhs: 0.0,
        },
    )?;
    let h2 = g.at_nodes(n, &h2);
    let mut total = Vec::with_capacity(nodes);
    for q in 0..nodes {
        let cvvv = real_form(model, p, &orb.u[q], &[&v[q], &v[q], &v[q]])?;
        let bvh = real_form(model, p, &orb.u[q], &[&v[q], &h2[q]])?;
        let av = matvec(&orb.a[q], n, &v[q]);
        let rhs: Vec<f64> = (0..n)
            .map(|c| cvvv[c] + 3.0 * bvh[c] - 6.0 * alpha1 * av[c])
            .collect();
        total.push(dot(&vs[q], &rhs));
    }
    let c = g.integrate(&total) / 6.0;
    if !c.is_finite() {
        return Err(EngineError::NotFinite("the flip coefficient c".into()));
    }
    Ok(CycleNormalForm {
        coefficient: c,
        coefficient2: None,
        normalization: "periodic: int <v, v> = 1, int <v*, v> = 1, int <phi*, F> = 1 over [0, T]",
    })
}

/// The complex vector of a doubled profile value, (re, im) with n entries each.
fn split(x: &[f64]) -> Cv {
    let n = x.len() / 2;
    (x[..n].to_vec(), x[n..].to_vec())
}

/// The doubled vector (re, im) of a complex vector.
fn join(c: &Cv) -> Vec<f64> {
    c.0.iter().chain(&c.1).copied().collect()
}

/// `<a, b> = conj(a)^T b` of two complex vectors.
fn herm(a: &Cv, b: &Cv) -> (f64, f64) {
    (
        dot(&a.0, &b.0) + dot(&a.1, &b.1),
        dot(&a.0, &b.1) - dot(&a.1, &b.0),
    )
}

/// The conjugate of a complex vector.
fn conj(a: &Cv) -> Cv {
    (a.0.clone(), a.1.iter().map(|x| -x).collect())
}

/// The real form of the complex matrix `Mr + i c I` at each node, 2n by 2n column-major:
/// `[[Mr, -c I], [c I, Mr]]` acting on (re; im).
fn doubled(mats: &[Vec<f64>], n: usize, c: f64) -> Vec<Vec<f64>> {
    let nn = 2 * n;
    mats.iter()
        .map(|m| {
            let mut d = vec![0.0; nn * nn];
            for col in 0..n {
                for row in 0..n {
                    let v = m[row + n * col];
                    d[row + nn * col] = v;
                    d[(row + n) + nn * (col + n)] = v;
                }
            }
            for k in 0..n {
                d[k + nn * (k + n)] = -c;
                d[(k + n) + nn * k] = c;
            }
            d
        })
        .collect()
}

/// The complex multilinear form of the complex directions `dirs` at the state u.
fn complex_form(
    model: &mut dyn ModelEval,
    p: &[f64],
    u: &[f64],
    dirs: &[&Cv],
) -> Result<Cv, EngineError> {
    At::new(model, u.len(), u, p.to_vec()).form(dirs)
}

/// `d` of a torus bifurcation, from DWDGK eq. 42-45 (SPEC §8a). With the multiplier
/// exp(i theta) of the monodromy matrix, theta in (0, pi), and omega = theta / T: v and v*
/// are the periodic solutions of v' = (A - i omega) v and v*' = -(A^T + i omega) v* with
/// int <v, v> = 1 and int <v*, v> = 1, phi* is the periodic adjoint with int <phi*, F> = 1,
/// alpha1 = int <phi*, B(v, conj v)>, h11 is the periodic solution of h11' = A h11 +
/// B(v, conj v) - alpha1 F with int <phi*, h11> = 0, h20 the periodic solution of
/// h20' = (A - 2 i omega) h20 + B(v, v), and d = 1/2 int <v*, B(h20, conj v) + 2 B(h11, v) +
/// C(v, v, conj v) - 2 alpha1 A v> + i alpha1 omega. The complex problems are solved as
/// real ones of twice the size.
fn ns(
    g: &Grid,
    model: &mut dyn ModelEval,
    p: &[f64],
    orb: &Orbit,
) -> Result<CycleNormalForm, EngineError> {
    let (n, nn) = (g.n, 2 * g.n);
    let (nodes, pts) = (g.nodes(), g.points());
    let mono = g.monodromy(n, &orb.a)?;
    // The multiplier of the pair with the argument in (0, pi) that is closest to the unit
    // circle.
    let mu = eigenvalues(&mono, n)?
        .into_iter()
        .filter(|&(_, im)| im > 0.0)
        .min_by(|a, b| {
            (a.0.hypot(a.1) - 1.0)
                .abs()
                .total_cmp(&(b.0.hypot(b.1) - 1.0).abs())
        })
        .ok_or_else(|| EngineError::Singular("the monodromy matrix has no complex pair".into()))?;
    let theta = mu.1.atan2(mu.0);
    if !(theta > 1e-6 && theta < PI - 1e-6) {
        return Err(EngineError::Singular(
            "the multiplier is too close to 1 or -1 (strong resonance R1 or R2)".into(),
        ));
    }
    let omega = theta / g.period;
    // v(0) is the right eigenvector of the monodromy matrix for exp(i theta), and v*(0) the
    // eigenvector of its transpose for exp(-i theta), as then <v*, v> is the complex
    // pairing that stays constant along the orbit.
    let v0 = eigvec(&mono, n, mu)?;
    let mut mono_t = vec![0.0; n * n];
    for c in 0..n {
        for r in 0..n {
            mono_t[r + n * c] = mono[c + n * r];
        }
    }
    let w0 = eigvec(&mono_t, n, (mu.0, -mu.1))?;
    let zero2 = vec![0.0; nn * pts];
    let initial = |mat: &[Vec<f64>], w: &Cv| {
        g.solve(
            nn,
            &Lin {
                mat,
                rhs: None,
                col: None,
                bc: Bc::Initial(join(w)),
                border: zero2.clone(),
                border_s: 1.0,
                border_rhs: 0.0,
            },
        )
    };
    let adj = minus_transpose(&orb.a, n);
    let mut v = g.at_nodes(nn, &initial(&doubled(&orb.a, n, -omega), &v0)?.0);
    let mut vs = g.at_nodes(nn, &initial(&doubled(&adj, n, -omega), &w0)?.0);
    let norm: Vec<f64> = v.iter().map(|x| dot(x, x)).collect();
    let nv = g.integrate(&norm).sqrt();
    if !(nv.is_finite() && nv > 0.0) {
        return Err(EngineError::Singular(
            "the torus eigenfunction is degenerate".into(),
        ));
    }
    v.iter_mut().flatten().for_each(|x| *x /= nv);
    let (mut ov_re, mut ov_im) = (Vec::with_capacity(nodes), Vec::with_capacity(nodes));
    for q in 0..nodes {
        let (re, im) = herm(&split(&vs[q]), &split(&v[q]));
        ov_re.push(re);
        ov_im.push(im);
    }
    let (ov_re, ov_im) = (g.integrate(&ov_re), g.integrate(&ov_im));
    let ov2 = ov_re * ov_re + ov_im * ov_im;
    if !(ov2.is_finite() && ov2 > 0.0) {
        return Err(EngineError::Singular(
            "the torus eigenfunctions are degenerate".into(),
        ));
    }
    // <a v*, v> = conj(a) <v*, v> = 1 for a = <v*, v> / |<v*, v>|^2.
    let (ar, ai) = (ov_re / ov2, ov_im / ov2);
    for x in &mut vs {
        let c = split(x);
        let (re, im): (Vec<f64>, Vec<f64>) = (0..n)
            .map(|k| (ar * c.0[k] - ai * c.1[k], ar * c.1[k] + ai * c.0[k]))
            .unzip();
        *x = join(&(re, im));
    }
    let vc: Vec<Cv> = v.iter().map(|x| split(x)).collect();
    let vsc: Vec<Cv> = vs.iter().map(|x| split(x)).collect();
    let vbar: Vec<Cv> = vc.iter().map(conj).collect();
    // phi*: the periodic adjoint, with the column F and the border int <F, g> = 1.
    let (phi, _) = g.solve(
        n,
        &Lin {
            mat: &adj,
            rhs: None,
            col: Some(&orb.f),
            bc: Bc::Periodic(1.0),
            border: g.functional(n, &orb.f),
            border_s: 0.0,
            border_rhs: 1.0,
        },
    )?;
    let phi = g.at_nodes(n, &phi);
    let mut b11 = Vec::with_capacity(nodes);
    let mut b20 = Vec::with_capacity(nodes);
    for q in 0..nodes {
        b11.push(complex_form(model, p, &orb.u[q], &[&vc[q], &vbar[q]])?.0);
        b20.push(join(&complex_form(model, p, &orb.u[q], &[&vc[q], &vc[q]])?));
    }
    let weighted: Vec<f64> = (0..nodes).map(|q| dot(&phi[q], &b11[q])).collect();
    let alpha1 = g.integrate(&weighted);
    let (h11, _) = g.solve(
        n,
        &Lin {
            mat: &orb.a,
            rhs: Some(&b11),
            col: Some(&orb.f),
            bc: Bc::Periodic(1.0),
            border: g.functional(n, &phi),
            border_s: 0.0,
            border_rhs: 0.0,
        },
    )?;
    let h11 = g.at_nodes(n, &h11);
    let (h20, _) = g.solve(
        nn,
        &Lin {
            mat: &doubled(&orb.a, n, -2.0 * omega),
            rhs: Some(&b20),
            col: None,
            bc: Bc::Periodic(1.0),
            border: zero2.clone(),
            border_s: 1.0,
            border_rhs: 0.0,
        },
    )?;
    let h20 = g.at_nodes(nn, &h20);
    let (mut tot_re, mut tot_im) = (Vec::with_capacity(nodes), Vec::with_capacity(nodes));
    let zero = vec![0.0; n];
    for q in 0..nodes {
        let u = &orb.u[q];
        let h20q = split(&h20[q]);
        let h11q = (h11[q].clone(), zero.clone());
        let b_h20 = complex_form(model, p, u, &[&h20q, &vbar[q]])?;
        let b_h11 = complex_form(model, p, u, &[&h11q, &vc[q]])?;
        let c_vvv = complex_form(model, p, u, &[&vc[q], &vc[q], &vbar[q]])?;
        let (av_re, av_im) = (
            matvec(&orb.a[q], n, &vc[q].0),
            matvec(&orb.a[q], n, &vc[q].1),
        );
        let rhs: Cv = (
            (0..n)
                .map(|k| b_h20.0[k] + 2.0 * b_h11.0[k] + c_vvv.0[k] - 2.0 * alpha1 * av_re[k])
                .collect(),
            (0..n)
                .map(|k| b_h20.1[k] + 2.0 * b_h11.1[k] + c_vvv.1[k] - 2.0 * alpha1 * av_im[k])
                .collect(),
        );
        let (re, im) = herm(&vsc[q], &rhs);
        tot_re.push(re);
        tot_im.push(im);
    }
    let d_re = 0.5 * g.integrate(&tot_re);
    let d_im = 0.5 * g.integrate(&tot_im) + alpha1 * omega;
    if !(d_re.is_finite() && d_im.is_finite()) {
        return Err(EngineError::NotFinite("the torus coefficient d".into()));
    }
    Ok(CycleNormalForm {
        coefficient: d_re,
        coefficient2: Some(d_im),
        normalization: "periodic: int <v, v> = 1, int <v*, v> = 1, int <phi*, F> = 1 over [0, T], theta in (0, pi)",
    })
}
