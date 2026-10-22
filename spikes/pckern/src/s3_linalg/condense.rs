//! Method (b): condensation of the interior unknowns of each interval, the
//! reduced system on the mesh values, T and alpha, and back-substitution
//! (NOTES.md, "S3 design").

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::Solve;
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_unit_lower_triangular_in_place,
    solve_upper_triangular_in_place,
};
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat, Triplet};
use faer::{Accum, Mat, MatMut, Par};
use rayon::prelude::*;

use super::matrix::Problem;
use crate::ffi::{Result, internal};

/// One condensed interval. `lu` holds P B_int = [L11; L21] U. `y` holds Y1
/// (top ni rows) and Y2 (bottom n rows), with the right-hand side in its
/// last column. `border[k]` is Y1ᵀ g for the phase (k = 0) and PALC rows.
pub struct Condensed {
    lu: Mat<f64>,
    y: Mat<f64>,
    border: [Vec<f64>; 2],
}

/// Workspace for one thread: LU scratch and the permutation arrays.
struct Work {
    buf: MemBuffer,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl Work {
    fn new(p: &Problem) -> Work {
        let ni = p.n * (p.m - 1);
        let req = lu_in_place_scratch::<usize, f64>(p.rows, ni, Par::Seq, Default::default());
        Work {
            buf: MemBuffer::new(req),
            perm: vec![0; p.rows],
            perm_inv: vec![0; p.rows],
        }
    }
}

/// Block column of remaining column q: u_{jm}, then u_{(j+1)m}, then T and
/// alpha.
fn rest_col(n: usize, m: usize, q: usize) -> usize {
    if q < n {
        q
    } else if q < 2 * n {
        n * m + (q - n)
    } else {
        n * (m + 1) + (q - 2 * n)
    }
}

/// Reduced column of remaining column q of interval j. Mesh j and mesh j + 1
/// are adjacent there, so both map to j·n + q.
fn reduced_col(n: usize, nmesh: usize, j: usize, q: usize) -> usize {
    if q < 2 * n {
        j * n + q
    } else {
        nmesh + (q - 2 * n)
    }
}

/// Condenses interval j: LU of the interior columns, Y1 and Y2, and the
/// border contributions.
fn condense(p: &Problem, j: usize, w: &mut Work) -> Condensed {
    let (n, m, rows, cols) = (p.n, p.m, p.rows, p.cols);
    let (ni, ne) = (n * (m - 1), 2 * n + 2);
    let blk = p.block(j);
    let mut lu = Mat::from_fn(rows, ni, |r, c| blk[r * cols + n + c]);
    let stack = MemStack::new(&mut w.buf);
    lu_in_place(
        lu.as_mut(),
        &mut w.perm,
        &mut w.perm_inv,
        Par::Seq,
        stack,
        Default::default(),
    );
    let perm = &w.perm;
    let mut y = Mat::from_fn(rows, ne + 1, |r, q| {
        if q < ne {
            blk[perm[r] * cols + rest_col(n, m, q)]
        } else {
            p.b[j * rows + perm[r]]
        }
    });
    let l11 = lu.as_ref().submatrix(0, 0, ni, ni);
    let (mut y1, y2) = y.as_mut().split_at_row_mut(ni);
    solve_unit_lower_triangular_in_place(l11, y1.as_mut(), Par::Seq);
    let l21 = lu.as_ref().submatrix(ni, 0, n, ni);
    matmul(y2, Accum::Add, l21, y1.as_ref(), -1.0, Par::Seq);
    // Solve Uᵀ g = r_int for both border rows; U is the upper part of l11.
    let base = j * m * n + n;
    let mut g = Mat::from_fn(ni, 2, |c, k| p.border(k, base + c));
    solve_lower_triangular_in_place(l11.transpose(), g.as_mut(), Par::Seq);
    let border = [0, 1].map(|k| {
        let gk = g.col_as_slice(k);
        (0..=ne)
            .map(|q| {
                let yq = &y.col_as_slice(q)[..ni];
                yq.iter().zip(gk).fold(0.0, |s, (a, b)| s + a * b)
            })
            .collect()
    });
    Condensed { lu, y, border }
}

/// Condenses every interval and returns them in interval order. With
/// `parallel` set it runs on the current rayon pool, one workspace per job.
pub fn condense_all(p: &Problem, parallel: bool) -> Vec<Condensed> {
    if parallel {
        (0..p.big_n)
            .into_par_iter()
            .map_init(|| Work::new(p), |w, j| condense(p, j, w))
            .collect()
    } else {
        let mut w = Work::new(p);
        (0..p.big_n).map(|j| condense(p, j, &mut w)).collect()
    }
}

/// The phase and PALC rows of the reduced system with their right-hand
/// sides.
pub struct Border {
    rows: [Vec<f64>; 2],
    rhs: [f64; 2],
}

/// Moves the mesh, T and alpha entries of the two border rows into reduced
/// columns and subtracts each interval's Y1ᵀ g. The subtraction runs in
/// interval order, so the result is the same at every thread count.
pub fn border_rows(p: &Problem, cond: &[Condensed]) -> Border {
    let (n, m, nmesh) = (p.n, p.m, p.n * (p.big_n + 1));
    let ne = 2 * n + 2;
    let mut rows = [0, 1].map(|k| {
        let mut row = vec![0.0; nmesh + 2];
        for jj in 0..=p.big_n {
            for c in 0..n {
                row[jj * n + c] = p.border(k, jj * m * n + c);
            }
        }
        row[nmesh] = p.border(k, p.nu);
        row[nmesh + 1] = p.border(k, p.nu + 1);
        row
    });
    let mut rhs = [p.b[p.nr - 2], p.b[p.nr - 1]];
    for (j, cj) in cond.iter().enumerate() {
        for k in 0..2 {
            for q in 0..ne {
                rows[k][reduced_col(n, nmesh, j, q)] -= cj.border[k][q];
            }
            rhs[k] -= cj.border[k][ne];
        }
    }
    Border { rows, rhs }
}

/// Visits the reduced entries in a fixed order: the condensed rows (Y2) of
/// each interval, the periodicity rows, then the two dense border rows.
fn for_each_reduced(
    p: &Problem,
    cond: &[Condensed],
    bd: &Border,
    mut f: impl FnMut(usize, usize, f64),
) {
    let (n, nmesh) = (p.n, p.n * (p.big_n + 1));
    let (ni, ne) = (n * (p.m - 1), 2 * n + 2);
    for (j, cj) in cond.iter().enumerate() {
        for i in 0..n {
            for q in 0..ne {
                f(j * n + i, reduced_col(n, nmesh, j, q), cj.y[(ni + i, q)]);
            }
        }
    }
    let last = p.big_n * n;
    for c in 0..n {
        f(last + c, c, 1.0);
        f(last + c, last + c, -1.0);
    }
    for (k, row) in bd.rows.iter().enumerate() {
        for (col, &v) in row.iter().enumerate() {
            f(nmesh + k, col, v);
        }
    }
}

/// Right-hand side of the reduced system.
fn reduced_rhs(p: &Problem, cond: &[Condensed], bd: &Border) -> Vec<f64> {
    let (n, nmesh) = (p.n, p.n * (p.big_n + 1));
    let (ni, ne) = (n * (p.m - 1), 2 * n + 2);
    let mut r = vec![0.0; nmesh + 2];
    for (j, cj) in cond.iter().enumerate() {
        for i in 0..n {
            r[j * n + i] = cj.y[(ni + i, ne)];
        }
    }
    for c in 0..n {
        r[p.big_n * n + c] = p.b[p.big_n * p.m * n + c];
    }
    r[nmesh] = bd.rhs[0];
    r[nmesh + 1] = bd.rhs[1];
    r
}

/// The reduced matrix in CSC form with its right-hand side, for the LU
/// threshold probe. It has the same entries as `pattern`, explicit zeros of
/// the border rows included.
pub(super) fn reduced_system(
    p: &Problem,
    cond: &[Condensed],
    bd: &Border,
) -> Result<(SparseColMat<usize, f64>, Vec<f64>)> {
    let nred = p.nred();
    let mut t = Vec::new();
    for_each_reduced(p, cond, bd, |r, c, v| t.push(Triplet::new(r, c, v)));
    let a = SparseColMat::try_new_from_triplets(nred, nred, &t)
        .map_err(|e| internal(format!("s3: cannot build the reduced matrix: {e:?}")))?;
    Ok((a, reduced_rhs(p, cond, bd)))
}

/// The reduced CSC pattern, its symbolic LU, and for each entry in visiting
/// order its position in the values array.
pub struct Pattern {
    symbolic: SymbolicSparseColMat<usize>,
    lu: SymbolicLu<usize>,
    pos: Vec<usize>,
}

pub fn pattern(p: &Problem, cond: &[Condensed], bd: &Border) -> Result<Pattern> {
    let nred = p.nred();
    let mut t = Vec::new();
    for_each_reduced(p, cond, bd, |r, c, _| t.push(Triplet::new(r, c, 1.0)));
    let (symbolic, _) = SparseColMat::<usize, f64>::try_new_from_triplets(nred, nred, &t)
        .map_err(|e| internal(format!("s3: cannot build the reduced pattern: {e:?}")))?
        .into_parts();
    let (cp, ri) = (symbolic.col_ptr(), symbolic.row_idx());
    let pos = t
        .iter()
        .map(|e| {
            let found = ri[cp[e.col]..cp[e.col + 1]].binary_search(&e.row);
            found
                .map(|k| cp[e.col] + k)
                .map_err(|_| internal("s3: reduced entry missing from its pattern"))
        })
        .collect::<Result<Vec<usize>>>()?;
    let lu = SymbolicLu::try_new(symbolic.as_ref())
        .map_err(|e| internal(format!("s3: reduced symbolic LU failed: {e:?}")))?;
    Ok(Pattern { symbolic, lu, pos })
}

/// Fills the reduced values, runs the numeric LU with the stored symbolic
/// part and solves for the mesh values, T and alpha.
pub fn solve_sparse(
    p: &Problem,
    cond: &[Condensed],
    bd: &Border,
    pat: &Pattern,
) -> Result<Vec<f64>> {
    let mut vals = vec![0.0; pat.pos.len()];
    let mut e = 0;
    for_each_reduced(p, cond, bd, |_, _, v| {
        vals[pat.pos[e]] = v;
        e += 1;
    });
    let a = SparseColMatRef::new(pat.symbolic.as_ref(), &vals);
    let lu = Lu::try_new_with_symbolic(pat.lu.clone(), a)
        .map_err(|e| internal(format!("s3: reduced numeric LU failed: {e:?}")))?;
    let mut x = reduced_rhs(p, cond, bd);
    let nred = x.len();
    lu.solve_in_place(MatMut::from_column_major_slice_mut(&mut x, nred, 1));
    Ok(x)
}

/// Dense LU of the reduced matrix with partial pivoting, then the two
/// triangular solves.
pub fn solve_dense(p: &Problem, cond: &[Condensed], bd: &Border, par: Par) -> Vec<f64> {
    let nred = p.nred();
    let mut a = Mat::<f64>::zeros(nred, nred);
    for_each_reduced(p, cond, bd, |r, c, v| a[(r, c)] = v);
    let (mut perm, mut perm_inv) = (vec![0usize; nred], vec![0usize; nred]);
    let req = lu_in_place_scratch::<usize, f64>(nred, nred, par, Default::default());
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);
    lu_in_place(
        a.as_mut(),
        &mut perm,
        &mut perm_inv,
        par,
        stack,
        Default::default(),
    );
    let rhs = reduced_rhs(p, cond, bd);
    let mut x: Vec<f64> = perm.iter().map(|&i| rhs[i]).collect();
    let mut xm = MatMut::from_column_major_slice_mut(&mut x, nred, 1);
    solve_unit_lower_triangular_in_place(a.as_ref(), xm.as_mut(), par);
    solve_upper_triangular_in_place(a.as_ref(), xm.as_mut(), par);
    x
}

/// Interior values of interval j: U z = Y1_rhs - Y1_rest x_rest.
fn backsub(p: &Problem, j: usize, cj: &Condensed, xr: &[f64]) -> Vec<f64> {
    let (n, nmesh) = (p.n, p.n * (p.big_n + 1));
    let (ni, ne) = (n * (p.m - 1), 2 * n + 2);
    let rest: Vec<f64> = (0..ne).map(|q| xr[reduced_col(n, nmesh, j, q)]).collect();
    let mut z: Vec<f64> = (0..ni)
        .map(|i| (0..ne).fold(cj.y[(i, ne)], |s, q| s - cj.y[(i, q)] * rest[q]))
        .collect();
    let u = cj.lu.as_ref().submatrix(0, 0, ni, ni);
    solve_upper_triangular_in_place(
        u,
        MatMut::from_column_major_slice_mut(&mut z, ni, 1),
        Par::Seq,
    );
    z
}

/// Back-substitution for every interval, then the full solution: mesh
/// values, T and alpha from the reduced solution, interiors from `backsub`.
pub fn backsub_all(p: &Problem, cond: &[Condensed], xr: &[f64], parallel: bool) -> Vec<f64> {
    let interiors: Vec<Vec<f64>> = if parallel {
        cond.par_iter()
            .enumerate()
            .map(|(j, cj)| backsub(p, j, cj, xr))
            .collect()
    } else {
        cond.iter()
            .enumerate()
            .map(|(j, cj)| backsub(p, j, cj, xr))
            .collect()
    };
    let (n, m, nmesh) = (p.n, p.m, p.n * (p.big_n + 1));
    let mut x = vec![0.0; p.nr];
    for jj in 0..=p.big_n {
        x[jj * m * n..jj * m * n + n].copy_from_slice(&xr[jj * n..(jj + 1) * n]);
    }
    for (j, z) in interiors.iter().enumerate() {
        x[j * m * n + n..j * m * n + n + z.len()].copy_from_slice(z);
    }
    x[p.nu] = xr[nmesh];
    x[p.nu + 1] = xr[nmesh + 1];
    x
}
