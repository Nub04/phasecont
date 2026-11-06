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
use crate::branch::Point;
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::At;
use pc_ir::EntryKind;
use pc_linalg::{BlockSystem, CondenseShape, Condenser, null_left};

/// A bifurcation of a limit cycle that has a normal form coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CycleBifurcation {
    /// A fold of cycles (limit point of cycles).
    Lpc,
}

/// The normal form coefficient of a cycle bifurcation.
#[derive(Clone, Debug, PartialEq)]
pub struct CycleNormalForm {
    /// `b` of an LPC point.
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

/// B(a, b) of the real directions a and b at the state u.
fn form2(
    model: &mut dyn ModelEval,
    p: &[f64],
    u: &[f64],
    a: &[f64],
    b: &[f64],
) -> Result<Vec<f64>, EngineError> {
    let z = vec![0.0; u.len()];
    let mut at = At::new(model, u.len(), u, p.to_vec());
    Ok(at.form(&[&(a.to_vec(), z.clone()), &(b.to_vec(), z)])?.0)
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
        let bb = form2(model, p, &orb.u[q], &phi1_nodes[q], &phi1_nodes[q])?;
        let a_phi: Vec<f64> = (0..n)
            .map(|c| {
                (0..n)
                    .map(|c2| orb.a[q][c + n * c2] * phi1_nodes[q][c2])
                    .sum()
            })
            .collect();
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
