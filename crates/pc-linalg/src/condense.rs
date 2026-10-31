//! Condensation of parameters for the bordered block systems of orthogonal collocation
//! (Doedel, Keller and Kernévez 1991, Part II).
//!
//! The unknowns are u_0, ..., u_{Nm} (n values each) followed by k extras, so the
//! dimension is n(Nm+1)+k. The rows are N interval blocks of nm rows, then n boundary
//! rows, then k dense border rows. Interval j couples u_{jm}, ..., u_{jm+m} and the
//! extras. A dense LU of each interval's interior columns removes u_{jm+1}, ...,
//! u_{jm+m-1}. What is left is a sparse system in the mesh values u_{jm} and the extras,
//! of dimension n(N+1)+k, which faer's simplicial sparse LU solves. A back substitution
//! per interval then recovers the interior values.
//!
//! The intervals can be condensed in parallel on the current rayon pool. Each interval
//! works only on its own slot of the store, and the border rows sum the interval
//! contributions in interval order, so the result has the same bits at every thread
//! count.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_unit_lower_triangular_in_place,
    solve_upper_triangular_in_place,
};
use faer::perm::PermRef;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::sparse::linalg::lu::simplicial::{
    SimplicialLu, factorize_simplicial_numeric_lu, factorize_simplicial_numeric_lu_scratch,
    solve_in_place_scratch,
};
use faer::sparse::linalg::lu::{LuSymbolicParams, SymbolicLu, factorize_symbolic_lu};
use faer::sparse::linalg::{LuError, SupernodalThreshold};
use faer::sparse::{
    CreationError, FaerError, SparseColMat, SparseColMatRef, SymbolicSparseColMat, Triplet,
};
use faer::{Accum, Conj, MatMut, MatRef, Par};
use rayon::prelude::*;

use crate::dense::{LinalgError, Lu};

/// The interval work above which [`CondenseShape::parallel_pays`] holds. Measured with
/// `timing_sequential_and_parallel` in release mode on 12 logical CPUs (m = 4, k = 2):
/// on the default pool of 12 threads a parallel solve took 1.2 to 1.5 times less time
/// from n = 10 and N = 320 (work 2.0e7), and more time below about 1e7, by up to 50
/// times at n = 2 and N = 20.
pub const PARALLEL_WORK: usize = 20_000_000;

/// Sizes of a block system: n states, m collocation points per interval, N intervals
/// and k extra unknowns with one border row each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CondenseShape {
    n: usize,
    m: usize,
    nint: usize,
    k: usize,
}

impl CondenseShape {
    /// `BadShape` when n is 0, m is below 2, N is 0, or a size the solver forms does not
    /// fit in `usize`.
    pub fn new(n: usize, m: usize, nint: usize, k: usize) -> Result<Self, LinalgError> {
        if n == 0 || m < 2 || nint == 0 || sizes(n, m, nint, k).is_none() {
            return Err(LinalgError::BadShape);
        }
        Ok(CondenseShape { n, m, nint, k })
    }

    /// Number of states n.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Collocation points per interval m.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of intervals N.
    pub fn nint(&self) -> usize {
        self.nint
    }

    /// Number of extra unknowns and border rows k.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Dimension of the full system, n(Nm+1)+k.
    pub fn dim(&self) -> usize {
        self.n * (self.nint * self.m + 1) + self.k
    }

    /// Rows of one interval block, nm.
    pub fn block_rows(&self) -> usize {
        self.n * self.m
    }

    /// Columns of one interval block, n(m+1)+k.
    pub fn block_cols(&self) -> usize {
        self.n * (self.m + 1) + self.k
    }

    /// Dimension of the reduced system, n(N+1)+k.
    pub fn reduced_dim(&self) -> usize {
        self.n * (self.nint + 1) + self.k
    }

    /// Whether condensing the intervals in parallel pays: when the interval
    /// eliminations take about N nm ni (ni + 2n + k + 1) multiply-adds, at least
    /// [`PARALLEL_WORK`].
    pub fn parallel_pays(&self) -> bool {
        let (nm, ni, ne) = (self.block_rows(), self.ni(), self.ne());
        let work = self
            .nint
            .saturating_mul(nm)
            .saturating_mul(ni)
            .saturating_mul(ni + ne + 1);
        work >= PARALLEL_WORK
    }

    /// Interior columns of one interval, n(m-1).
    fn ni(&self) -> usize {
        self.n * (self.m - 1)
    }

    /// Columns left after condensation: u_{jm}, u_{jm+m} and the extras, 2n+k.
    fn ne(&self) -> usize {
        2 * self.n + self.k
    }

    /// Length of one interval's slot in the store: the LU of the interior columns
    /// (nm by ni), Y (nm by ne+1), the border contribution (k by ne+1) and g (ni by k).
    fn stride(&self) -> usize {
        let (nm, ni, ne, k) = (self.block_rows(), self.ni(), self.ne(), self.k);
        nm * ni + nm * (ne + 1) + k * (ne + 1) + ni * k
    }

    /// Structural nonzeros of the reduced matrix.
    fn reduced_nnz(&self) -> usize {
        let (n, k) = (self.n, self.k);
        self.nint * n * self.ne() + n * (2 * n + k) + k * self.reduced_dim()
    }
}

/// Checks that every size the solver forms fits in `usize`.
fn sizes(n: usize, m: usize, nint: usize, k: usize) -> Option<()> {
    let nm = n.checked_mul(m)?;
    let bcols = n.checked_mul(m.checked_add(1)?)?.checked_add(k)?;
    let dim = n
        .checked_mul(nint.checked_mul(m)?.checked_add(1)?)?
        .checked_add(k)?;
    nint.checked_mul(nm)?.checked_mul(bcols)?;
    k.checked_mul(dim)?;
    let ni = n.checked_mul(m - 1)?;
    let ne1 = n.checked_mul(2)?.checked_add(k)?.checked_add(1)?;
    let stride = nm
        .checked_mul(ni)?
        .checked_add(nm.checked_mul(ne1)?)?
        .checked_add(k.checked_mul(ne1)?)?
        .checked_add(ni.checked_mul(k)?)?;
    nint.checked_mul(stride)?;
    let rdim = n.checked_mul(nint.checked_add(1)?)?.checked_add(k)?;
    let rb = k.checked_mul(rdim)?;
    nint.checked_mul(n)?
        .checked_mul(ne1 - 1)?
        .checked_add(n.checked_mul(ne1 - 1)?)?
        .checked_add(rb)?;
    Some(())
}

/// The matrix of a block system, all parts column-major.
///
/// `blocks` holds the N interval blocks one after another, block j at offset
/// j·nm·(n(m+1)+k). Its columns are u_{jm}, ..., u_{jm+m}, then the extras. `bc` is the
/// n by 2n+k boundary block with columns u_0, u_{Nm}, then the extras. `border` is the
/// k by n(Nm+1)+k matrix of the border rows over all unknowns. The right-hand side has
/// the same row order: the interval rows, the boundary rows, then the border rows.
#[derive(Clone, Copy, Debug)]
pub struct BlockSystem<'a> {
    pub blocks: &'a [f64],
    pub bc: &'a [f64],
    pub border: &'a [f64],
}

/// Workspace of one condensation job.
struct Work {
    buf: MemBuffer,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl Work {
    fn try_new(s: &CondenseShape) -> Result<Work, LinalgError> {
        let nm = s.block_rows();
        let req = lu_in_place_scratch::<usize, f64>(nm, s.ni(), Par::Seq, Default::default());
        Ok(Work {
            buf: MemBuffer::try_new(req).map_err(|_| LinalgError::OutOfMemory)?,
            perm: try_filled(nm, 0)?,
            perm_inv: try_filled(nm, 0)?,
        })
    }
}

/// Sparsity pattern, symbolic factorization and buffers of the reduced system. They
/// depend on the shape only, so `Condenser::new` builds them once.
struct Reduced {
    symbolic: SymbolicSparseColMat<usize>,
    sym: SymbolicLu<usize>,
    /// Index into `vals` of each entry in the order `for_each_entry` visits them.
    pos: Vec<usize>,
    vals: Vec<f64>,
    /// Right-hand side, then the solution.
    x: Vec<f64>,
    /// The k border rows over the reduced unknowns, row-major.
    rb: Vec<f64>,
    row_perm: Vec<usize>,
    row_perm_inv: Vec<usize>,
    lu: SimplicialLu<usize, f64>,
    buf: MemBuffer,
}

/// Where the value of a reduced entry comes from.
#[derive(Clone, Copy)]
enum Entry {
    /// Row i of the condensed rows of interval j, remaining column q.
    Y { j: usize, i: usize, q: usize },
    /// Boundary row c, column q of `bc`.
    Bc { c: usize, q: usize },
    /// Border row r, reduced column `col`.
    Border { r: usize, col: usize },
}

/// Reduced column of remaining column q of interval j. The end point of interval j is
/// the start of interval j+1, so its columns follow those of u_{jm}.
fn reduced_col(s: &CondenseShape, j: usize, q: usize) -> usize {
    let n = s.n;
    if q < 2 * n {
        j * n + q
    } else {
        n * (s.nint + 1) + (q - 2 * n)
    }
}

/// Reduced column of column q of the boundary block.
fn bc_col(s: &CondenseShape, q: usize) -> usize {
    let n = s.n;
    if q < n {
        q
    } else if q < 2 * n {
        s.nint * n + (q - n)
    } else {
        n * (s.nint + 1) + (q - 2 * n)
    }
}

/// Block column of remaining column q: u_{jm} is block columns 0..n, and u_{jm+m} and
/// the extras follow the ni interior columns.
fn rest_col(s: &CondenseShape, q: usize) -> usize {
    if q < s.n { q } else { q + s.ni() }
}

/// Visits the reduced entries in a fixed order: the condensed rows of each interval,
/// the boundary rows, then the dense border rows. No (row, column) pair repeats.
fn for_each_entry(s: &CondenseShape, mut f: impl FnMut(usize, usize, Entry)) {
    let (n, nint, k) = (s.n, s.nint, s.k);
    for j in 0..nint {
        for i in 0..n {
            for q in 0..s.ne() {
                f(j * n + i, reduced_col(s, j, q), Entry::Y { j, i, q });
            }
        }
    }
    for c in 0..n {
        for q in 0..2 * n + k {
            f(nint * n + c, bc_col(s, q), Entry::Bc { c, q });
        }
    }
    let nmesh = n * (nint + 1);
    for r in 0..k {
        for col in 0..s.reduced_dim() {
            f(nmesh + r, col, Entry::Border { r, col });
        }
    }
}

/// A vector of `len` copies of `v`, or `OutOfMemory` instead of an abort.
fn try_filled<T: Clone>(len: usize, v: T) -> Result<Vec<T>, LinalgError> {
    let mut x = Vec::new();
    x.try_reserve_exact(len)
        .map_err(|_| LinalgError::OutOfMemory)?;
    x.resize(len, v);
    Ok(x)
}

fn faer_err(e: FaerError) -> LinalgError {
    match e {
        FaerError::OutOfMemory => LinalgError::OutOfMemory,
        _ => LinalgError::BadShape,
    }
}

/// Relative pivot test on the diagonal of U, taken in elimination order. A pivot that
/// is 0 or at most dim·EPSILON times the largest pivot before the first non-finite one
/// means `Singular`. A non-finite pivot after that (an overflow, since the input is
/// finite) means `NotFinite`.
fn check_pivots(diag: impl Iterator<Item = f64>, dim: usize) -> Result<(), LinalgError> {
    let mut umin = f64::INFINITY;
    let mut umax = 0.0_f64;
    let mut finite = true;
    for d in diag {
        if !d.is_finite() {
            finite = false;
            break;
        }
        umin = umin.min(d.abs());
        umax = umax.max(d.abs());
    }
    if umin == 0.0 || umin <= dim as f64 * f64::EPSILON * umax {
        return Err(LinalgError::Singular);
    }
    if finite {
        Ok(())
    } else {
        Err(LinalgError::NotFinite)
    }
}

/// Solver for block systems of one shape. It keeps the condensed intervals of the last
/// successful solve, from which [`Condenser::monodromy`] forms the monodromy matrix.
pub struct Condenser {
    shape: CondenseShape,
    store: Vec<f64>,
    work: Work,
    red: Reduced,
    valid: bool,
}

impl Condenser {
    /// Builds the reduced pattern and allocates every buffer the solves need, so a
    /// later solve allocates little. `OutOfMemory` when an allocation fails.
    pub fn new(shape: CondenseShape) -> Result<Condenser, LinalgError> {
        let s = shape;
        let rdim = s.reduced_dim();
        let nnz = s.reduced_nnz();
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        triplets
            .try_reserve_exact(nnz)
            .map_err(|_| LinalgError::OutOfMemory)?;
        for_each_entry(&s, |row, col, _| triplets.push(Triplet::new(row, col, 1.0)));
        let (symbolic, _) =
            SparseColMat::<usize, f64>::try_new_from_triplets(rdim, rdim, &triplets)
                .map_err(|e| match e {
                    CreationError::Generic(e) => faer_err(e),
                    CreationError::OutOfBounds { .. } => LinalgError::BadShape,
                })?
                .into_parts();
        let mut pos = try_filled(nnz, 0usize)?;
        {
            let (cp, ri) = (symbolic.col_ptr(), symbolic.row_idx());
            for (p, t) in pos.iter_mut().zip(&triplets) {
                let (lo, hi) = (cp[t.col], cp[t.col + 1]);
                let off = ri[lo..hi]
                    .binary_search(&t.row)
                    .map_err(|_| LinalgError::BadShape)?;
                *p = lo + off;
            }
        }
        drop(triplets);
        let sym = factorize_symbolic_lu(
            symbolic.as_ref(),
            LuSymbolicParams {
                supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
                ..Default::default()
            },
        )
        .map_err(faer_err)?;
        let req = StackReq::any_of(&[
            factorize_simplicial_numeric_lu_scratch::<usize, f64>(rdim, rdim),
            solve_in_place_scratch::<usize, f64>(rdim, 1, Par::Seq),
        ]);
        let red = Reduced {
            symbolic,
            sym,
            pos,
            vals: try_filled(nnz, 0.0)?,
            x: try_filled(rdim, 0.0)?,
            rb: try_filled(s.k * rdim, 0.0)?,
            row_perm: try_filled(rdim, 0)?,
            row_perm_inv: try_filled(rdim, 0)?,
            lu: SimplicialLu::new(),
            buf: MemBuffer::try_new(req).map_err(|_| LinalgError::OutOfMemory)?,
        };
        Ok(Condenser {
            shape,
            store: try_filled(s.nint * s.stride(), 0.0)?,
            work: Work::try_new(&s)?,
            red,
            valid: false,
        })
    }

    /// The shape this solver was built for.
    pub fn shape(&self) -> CondenseShape {
        self.shape
    }

    /// Solves the system `sys` with right-hand side `rhs` into `sol`. With `parallel`
    /// set, the intervals are condensed on the current rayon pool.
    ///
    /// `BadShape` when a length does not match the shape, `NotFinite` for NaN or
    /// infinity in the input or the result, `Singular` when an interval's interior
    /// columns or the reduced system are singular to working precision, and
    /// `OutOfMemory` when a workspace cannot be allocated.
    pub fn solve(
        &mut self,
        sys: &BlockSystem<'_>,
        rhs: &[f64],
        sol: &mut [f64],
        parallel: bool,
    ) -> Result<(), LinalgError> {
        let dim = self.shape.dim();
        if rhs.len() != dim || sol.len() != dim {
            return Err(LinalgError::BadShape);
        }
        self.check_system(sys, rhs)?;
        self.valid = false;
        self.condense_all(sys, rhs, parallel)?;
        self.solve_reduced(sys, rhs)?;
        self.back_substitute(sol, parallel);
        self.valid = true;
        Ok(())
    }

    /// Condenses the intervals of `sys` without solving the system, so that
    /// [`Condenser::monodromy`] then gives the monodromy matrix of `sys`. The errors are
    /// those of [`Condenser::solve`] for the interval blocks.
    pub fn condense_intervals(
        &mut self,
        sys: &BlockSystem<'_>,
        parallel: bool,
    ) -> Result<(), LinalgError> {
        self.check_system(sys, &[])?;
        self.valid = false;
        let zeros = try_filled(self.shape.nint * self.shape.block_rows(), 0.0)?;
        self.condense_all(sys, &zeros, parallel)?;
        self.valid = true;
        Ok(())
    }

    /// `BadShape` when a part of `sys` has the wrong length, `NotFinite` for NaN or
    /// infinity in `sys` or `rhs`. Finiteness is checked here so that the error kind does
    /// not depend on which interval a parallel run reaches first.
    fn check_system(&self, sys: &BlockSystem<'_>, rhs: &[f64]) -> Result<(), LinalgError> {
        let s = self.shape;
        if sys.blocks.len() != s.nint * s.block_rows() * s.block_cols()
            || sys.bc.len() != s.n * (2 * s.n + s.k)
            || sys.border.len() != s.k * s.dim()
        {
            return Err(LinalgError::BadShape);
        }
        let all = [sys.blocks, sys.bc, sys.border, rhs];
        if all.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
            return Err(LinalgError::NotFinite);
        }
        Ok(())
    }

    /// Condenses every interval into its slot of the store. `rhs` needs only the
    /// interval rows.
    fn condense_all(
        &mut self,
        sys: &BlockSystem<'_>,
        rhs: &[f64],
        parallel: bool,
    ) -> Result<(), LinalgError> {
        let s = self.shape;
        let stride = s.stride();
        if parallel {
            self.store
                .par_chunks_mut(stride)
                .enumerate()
                .try_for_each_init(
                    || Work::try_new(&s),
                    |w, (j, slot)| {
                        let w = w.as_mut().map_err(|e| *e)?;
                        condense(&s, sys, rhs, j, slot, w)
                    },
                )
        } else {
            for (j, slot) in self.store.chunks_mut(stride).enumerate() {
                condense(&s, sys, rhs, j, slot, &mut self.work)?;
            }
            Ok(())
        }
    }

    /// Assembles the reduced system from the condensed intervals and solves it into
    /// `red.x`.
    fn solve_reduced(&mut self, sys: &BlockSystem<'_>, rhs: &[f64]) -> Result<(), LinalgError> {
        let s = self.shape;
        let (n, m, nint, k) = (s.n, s.m, s.nint, s.k);
        let (nm, ni, ne) = (s.block_rows(), s.ni(), s.ne());
        let (rdim, nmesh, stride) = (s.reduced_dim(), n * (nint + 1), s.stride());
        let red = &mut self.red;
        // Border rows: the mesh and extra entries move over unchanged, then each
        // interval's contribution is subtracted in interval order.
        for r in 0..k {
            let row = &mut red.rb[r * rdim..(r + 1) * rdim];
            for jj in 0..=nint {
                for c in 0..n {
                    row[jj * n + c] = sys.border[r + k * (jj * m * n + c)];
                }
            }
            for e in 0..k {
                row[nmesh + e] = sys.border[r + k * (n * (nint * m + 1) + e)];
            }
            let mut b = rhs[nint * nm + n + r];
            for (j, slot) in self.store.chunks(stride).enumerate() {
                let contrib = &slot[nm * ni + nm * (ne + 1)..][..k * (ne + 1)];
                for q in 0..ne {
                    row[reduced_col(&s, j, q)] -= contrib[r + k * q];
                }
                b -= contrib[r + k * ne];
            }
            red.x[nmesh + r] = b;
        }
        for (j, slot) in self.store.chunks(stride).enumerate() {
            let y = &slot[nm * ni..];
            for i in 0..n {
                red.x[j * n + i] = y[ni + i + nm * ne];
            }
        }
        for c in 0..n {
            red.x[nint * n + c] = rhs[nint * nm + c];
        }
        let store = &self.store;
        let (vals, pos, rb) = (&mut red.vals, &red.pos, &red.rb);
        let mut t = 0;
        for_each_entry(&s, |_, _, e| {
            vals[pos[t]] = match e {
                Entry::Y { j, i, q } => store[j * stride + nm * ni + ni + i + nm * q],
                Entry::Bc { c, q } => sys.bc[c + n * q],
                Entry::Border { r, col } => rb[r * rdim + col],
            };
            t += 1;
        });
        let a = SparseColMatRef::new(red.symbolic.as_ref(), &red.vals);
        factorize_simplicial_numeric_lu(
            &mut red.row_perm,
            &mut red.row_perm_inv,
            &mut red.lu,
            a,
            red.sym.col_perm(),
            MemStack::new(&mut red.buf),
        )
        .map_err(|e| match e {
            LuError::SymbolicSingular { .. } => LinalgError::Singular,
            LuError::Generic(e) => faer_err(e),
        })?;
        let u = red.lu.u_factor_unsorted();
        let (cp, uv) = (u.symbolic().col_ptr(), u.val());
        check_pivots((0..rdim).map(|j| uv[cp[j + 1] - 1]), rdim)?;
        red.lu.solve_in_place_with_conj(
            PermRef::new_checked(&red.row_perm, &red.row_perm_inv, rdim),
            red.sym.col_perm(),
            Conj::No,
            MatMut::from_column_major_slice_mut(&mut red.x, rdim, 1),
            Par::Seq,
            MemStack::new(&mut red.buf),
        );
        if red.x.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(())
    }

    /// Recovers the interior values of each interval from the reduced solution.
    fn back_substitute(&self, sol: &mut [f64], parallel: bool) {
        let s = self.shape;
        let (n, m, nint, k) = (s.n, s.m, s.nint, s.k);
        let (nm, ni, ne) = (s.block_rows(), s.ni(), s.ne());
        let (nmesh, stride) = (n * (nint + 1), s.stride());
        let xr = &self.red.x;
        // Slot j of `sol` is u_{jm} followed by the interior values z of interval j:
        // U z = Y1_rhs - Y1_rest w, with w = (u_{jm}, u_{jm+m}, extras).
        let body = |(j, (out, slot)): (usize, (&mut [f64], &[f64]))| {
            out[..n].copy_from_slice(&xr[j * n..(j + 1) * n]);
            let z = &mut out[n..];
            let (lu, y) = slot.split_at(nm * ni);
            z.copy_from_slice(&y[nm * ne..nm * ne + ni]);
            for q in 0..ne {
                let wq = if q < 2 * n {
                    xr[j * n + q]
                } else {
                    xr[nmesh + q - 2 * n]
                };
                for (zc, yc) in z.iter_mut().zip(&y[nm * q..nm * q + ni]) {
                    *zc -= yc * wq;
                }
            }
            let u = MatRef::from_column_major_slice(lu, nm, ni).submatrix(0, 0, ni, ni);
            solve_upper_triangular_in_place(
                u,
                MatMut::from_column_major_slice_mut(z, ni, 1),
                Par::Seq,
            );
        };
        let (inner, tail) = sol.split_at_mut(nint * m * n);
        if parallel {
            inner
                .par_chunks_mut(m * n)
                .zip(self.store.par_chunks(stride))
                .enumerate()
                .for_each(body);
        } else {
            inner
                .chunks_mut(m * n)
                .zip(self.store.chunks(stride))
                .enumerate()
                .for_each(body);
        }
        tail[..n].copy_from_slice(&xr[nint * n..nmesh]);
        tail[n..].copy_from_slice(&xr[nmesh..nmesh + k]);
    }

    /// The monodromy matrix of the last successful solve, n by n column-major: the
    /// product over the intervals of -C_R^-1 C_L, where C_L and C_R are the condensed
    /// columns of u_{jm} and u_{jm+m}. `BadShape` before a successful solve, and
    /// `Singular` when some C_R is singular.
    pub fn monodromy(&self) -> Result<Vec<f64>, LinalgError> {
        if !self.valid {
            return Err(LinalgError::BadShape);
        }
        let s = self.shape;
        let n = s.n;
        let (nm, ni) = (s.block_rows(), s.ni());
        let mut mono = vec![0.0; n * n];
        for i in 0..n {
            mono[i + n * i] = 1.0;
        }
        let mut cl = vec![0.0; n * n];
        let mut cr = vec![0.0; n * n];
        let mut next = vec![0.0; n * n];
        for slot in self.store.chunks(s.stride()) {
            let y = &slot[nm * ni..];
            for q in 0..n {
                for i in 0..n {
                    cl[i + n * q] = y[ni + i + nm * q];
                    cr[i + n * q] = y[ni + i + nm * (n + q)];
                }
            }
            let lu = Lu::new(&cr, n)?;
            matmul(
                MatMut::from_column_major_slice_mut(&mut next, n, n),
                Accum::Replace,
                MatRef::from_column_major_slice(&cl, n, n),
                MatRef::from_column_major_slice(&mono, n, n),
                -1.0,
                Par::Seq,
            );
            for col in next.chunks_mut(n) {
                lu.solve(col)?;
            }
            std::mem::swap(&mut mono, &mut next);
        }
        if mono.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(mono)
    }
}

/// Condenses interval j into `slot`: the LU of its interior columns, Y = P (rest | rhs)
/// with Y1 <- L11^-1 Y1 and Y2 <- Y2 - L21 Y1, and the border contribution g^T Y1 with
/// U^T g = the border entries of the interior columns.
fn condense(
    s: &CondenseShape,
    sys: &BlockSystem<'_>,
    rhs: &[f64],
    j: usize,
    slot: &mut [f64],
    w: &mut Work,
) -> Result<(), LinalgError> {
    let (n, m, k) = (s.n, s.m, s.k);
    let (nm, ni, ne, bcols) = (s.block_rows(), s.ni(), s.ne(), s.block_cols());
    let blk = &sys.blocks[j * nm * bcols..(j + 1) * nm * bcols];
    let (lu, rest) = slot.split_at_mut(nm * ni);
    let (y, rest) = rest.split_at_mut(nm * (ne + 1));
    let (contrib, g) = rest.split_at_mut(k * (ne + 1));
    lu.copy_from_slice(&blk[nm * n..nm * (n + ni)]);
    let _ = lu_in_place(
        MatMut::from_column_major_slice_mut(lu, nm, ni),
        &mut w.perm,
        &mut w.perm_inv,
        Par::Seq,
        MemStack::new(&mut w.buf),
        Default::default(),
    );
    let l = MatRef::from_column_major_slice(lu, nm, ni);
    check_pivots((0..ni).map(|i| l[(i, i)]), nm)?;
    let perm = &w.perm;
    for q in 0..ne {
        let src = &blk[nm * rest_col(s, q)..][..nm];
        for (r, d) in y[nm * q..nm * (q + 1)].iter_mut().enumerate() {
            *d = src[perm[r]];
        }
    }
    let rj = &rhs[j * nm..(j + 1) * nm];
    for (r, d) in y[nm * ne..].iter_mut().enumerate() {
        *d = rj[perm[r]];
    }
    let l11 = l.submatrix(0, 0, ni, ni);
    let l21 = l.submatrix(ni, 0, n, ni);
    let mut ym = MatMut::from_column_major_slice_mut(y, nm, ne + 1);
    let (mut y1, y2) = ym.rb_mut().split_at_row_mut(ni);
    solve_unit_lower_triangular_in_place(l11, y1.rb_mut(), Par::Seq);
    matmul(y2, Accum::Add, l21, y1.rb(), -1.0, Par::Seq);
    if k > 0 {
        let base = j * m * n + n;
        for r in 0..k {
            for c in 0..ni {
                g[c + ni * r] = sys.border[r + k * (base + c)];
            }
        }
        let mut gm = MatMut::from_column_major_slice_mut(g, ni, k);
        solve_lower_triangular_in_place(l11.transpose(), gm.rb_mut(), Par::Seq);
        matmul(
            MatMut::from_column_major_slice_mut(contrib, k, ne + 1),
            Accum::Replace,
            gm.rb().transpose(),
            y1.rb(),
            1.0,
            Par::Seq,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::testutil::Rng;

    /// A random system with a heavy diagonal, so the full matrix, the interior blocks
    /// and the reduced system are well conditioned.
    struct Sys {
        blocks: Vec<f64>,
        bc: Vec<f64>,
        border: Vec<f64>,
        rhs: Vec<f64>,
    }

    impl Sys {
        fn random(s: &CondenseShape, rng: &mut Rng) -> Sys {
            let (n, k, nint) = (s.n(), s.k(), s.nint());
            let (nm, bcols, dim) = (s.block_rows(), s.block_cols(), s.dim());
            let mut blocks: Vec<f64> = (0..nint * nm * bcols).map(|_| rng.unit()).collect();
            for j in 0..nint {
                for r in 0..nm {
                    blocks[j * nm * bcols + r + nm * r] += bcols as f64;
                }
            }
            let mut bc: Vec<f64> = (0..n * (2 * n + k)).map(|_| rng.unit()).collect();
            for c in 0..n {
                bc[c + n * (n + c)] += (2 * n + k) as f64;
            }
            let mut border: Vec<f64> = (0..k * dim).map(|_| rng.unit()).collect();
            for r in 0..k {
                border[r + k * (n * (nint * s.m() + 1) + r)] += dim as f64;
            }
            let rhs = (0..dim).map(|_| rng.unit()).collect();
            Sys {
                blocks,
                bc,
                border,
                rhs,
            }
        }

        fn view(&self) -> BlockSystem<'_> {
            BlockSystem {
                blocks: &self.blocks,
                bc: &self.bc,
                border: &self.border,
            }
        }

        /// The full matrix as (row, column, value) entries.
        fn entries(&self, s: &CondenseShape) -> Vec<(usize, usize, f64)> {
            let (n, m, k, nint) = (s.n(), s.m(), s.k(), s.nint());
            let (nm, bcols, dim) = (s.block_rows(), s.block_cols(), s.dim());
            let ext = n * (nint * m + 1);
            let mut e = Vec::new();
            for j in 0..nint {
                for c in 0..bcols {
                    let col = if c < n * (m + 1) {
                        j * m * n + c
                    } else {
                        ext + c - n * (m + 1)
                    };
                    for r in 0..nm {
                        e.push((j * nm + r, col, self.blocks[j * nm * bcols + r + nm * c]));
                    }
                }
            }
            for q in 0..2 * n + k {
                let col = if q < n {
                    q
                } else if q < 2 * n {
                    nint * m * n + q - n
                } else {
                    ext + q - 2 * n
                };
                for c in 0..n {
                    e.push((nint * nm + c, col, self.bc[c + n * q]));
                }
            }
            for col in 0..dim {
                for r in 0..k {
                    e.push((nint * nm + n + r, col, self.border[r + k * col]));
                }
            }
            e
        }

        fn dense(&self, s: &CondenseShape) -> Vec<f64> {
            let dim = s.dim();
            let mut a = vec![0.0; dim * dim];
            for (r, c, v) in self.entries(s) {
                a[r + dim * c] += v;
            }
            a
        }
    }

    const SHAPES: [(usize, usize, usize, usize); 5] = [
        (1, 2, 1, 0),
        (2, 3, 4, 2),
        (3, 4, 5, 2),
        (2, 7, 3, 3),
        (4, 2, 6, 1),
    ];

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()))
    }

    fn solve(s: &CondenseShape, sys: &Sys, parallel: bool) -> Vec<f64> {
        let mut c = Condenser::new(*s).unwrap();
        let mut x = vec![0.0; s.dim()];
        c.solve(&sys.view(), &sys.rhs, &mut x, parallel).unwrap();
        x
    }

    #[test]
    fn matches_dense_lu_with_small_backward_error() {
        let mut rng = Rng(7);
        for (n, m, nint, k) in SHAPES {
            let s = CondenseShape::new(n, m, nint, k).unwrap();
            let sys = Sys::random(&s, &mut rng);
            let x = solve(&s, &sys, false);
            let dim = s.dim();
            let a = sys.dense(&s);
            let mut xd = sys.rhs.clone();
            Lu::new(&a, dim).unwrap().solve(&mut xd).unwrap();
            let err = max_diff(&x, &xd);
            assert!(err <= 1e-10 * max_abs(&xd), "{n} {m} {nint} {k}: {err}");
            let anorm = (0..dim)
                .map(|i| (0..dim).map(|j| a[i + dim * j].abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            let res: Vec<f64> = (0..dim)
                .map(|i| (0..dim).map(|j| a[i + dim * j] * x[j]).sum::<f64>() - sys.rhs[i])
                .collect();
            let back = max_abs(&res) / (anorm * max_abs(&x) + max_abs(&sys.rhs));
            assert!(back <= 1e-13, "{n} {m} {nint} {k}: backward error {back}");
        }
    }

    #[test]
    fn matches_sparse_lu_of_the_full_matrix() {
        use faer::linalg::solvers::Solve;
        let mut rng = Rng(11);
        for (n, m, nint, k) in SHAPES {
            let s = CondenseShape::new(n, m, nint, k).unwrap();
            let sys = Sys::random(&s, &mut rng);
            let x = solve(&s, &sys, false);
            let t: Vec<_> = sys
                .entries(&s)
                .into_iter()
                .map(|(r, c, v)| Triplet::new(r, c, v))
                .collect();
            let dim = s.dim();
            let a = SparseColMat::<usize, f64>::try_new_from_triplets(dim, dim, &t).unwrap();
            let lu = a.as_ref().sp_lu().unwrap();
            let mut xs = sys.rhs.clone();
            lu.solve_in_place(MatMut::from_column_major_slice_mut(&mut xs, dim, 1));
            let err = max_diff(&x, &xs);
            assert!(err <= 1e-10 * max_abs(&xs), "{n} {m} {nint} {k}: {err}");
        }
    }

    #[test]
    fn parallel_pays_only_for_large_systems() {
        let pays = |n, nint| CondenseShape::new(n, 4, nint, 2).unwrap().parallel_pays();
        assert!(!pays(2, 640));
        assert!(!pays(4, 640));
        assert!(!pays(10, 160));
        assert!(pays(10, 320));
        assert!(pays(20, 50));
    }

    #[test]
    fn same_bits_sequential_and_in_a_pool() {
        let s = CondenseShape::new(2, 3, 37, 2).unwrap();
        let sys = Sys::random(&s, &mut Rng(3));
        let seq = solve(&s, &sys, false);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (par, mono_par) = pool.install(|| {
            let mut c = Condenser::new(s).unwrap();
            let mut x = vec![0.0; s.dim()];
            c.solve(&sys.view(), &sys.rhs, &mut x, true).unwrap();
            (x, c.monodromy().unwrap())
        });
        let mut c = Condenser::new(s).unwrap();
        let mut x = vec![0.0; s.dim()];
        c.solve(&sys.view(), &sys.rhs, &mut x, false).unwrap();
        let mono_seq = c.monodromy().unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&seq), bits(&par));
        assert_eq!(bits(&x), bits(&par));
        assert_eq!(bits(&mono_seq), bits(&mono_par));
    }

    /// The monodromy matrix from per-interval transfer maps of the full blocks: with
    /// the extras fixed, [B_int | B_R] [z; u_R] = -B_L u_L, and u_R is the last n rows.
    fn transfer_product(s: &CondenseShape, sys: &Sys) -> Vec<f64> {
        let (n, nint) = (s.n(), s.nint());
        let (nm, bcols) = (s.block_rows(), s.block_cols());
        let mut mono = vec![0.0; n * n];
        for i in 0..n {
            mono[i + n * i] = 1.0;
        }
        for j in 0..nint {
            let blk = &sys.blocks[j * nm * bcols..(j + 1) * nm * bcols];
            let lu = Lu::new(&blk[nm * n..nm * (n + nm)], nm).unwrap();
            let mut next = vec![0.0; n * n];
            for q in 0..n {
                let mut b: Vec<f64> = (0..nm)
                    .map(|r| {
                        -(0..n)
                            .map(|c| blk[r + nm * c] * mono[c + n * q])
                            .sum::<f64>()
                    })
                    .collect();
                lu.solve(&mut b).unwrap();
                next[n * q..n * (q + 1)].copy_from_slice(&b[nm - n..]);
            }
            mono = next;
        }
        mono
    }

    #[test]
    fn monodromy_matches_the_product_of_transfer_maps() {
        let mut rng = Rng(5);
        for (n, m, nint, k) in SHAPES {
            let s = CondenseShape::new(n, m, nint, k).unwrap();
            let sys = Sys::random(&s, &mut rng);
            let mut c = Condenser::new(s).unwrap();
            assert_eq!(c.monodromy(), Err(LinalgError::BadShape));
            let mut x = vec![0.0; s.dim()];
            c.solve(&sys.view(), &sys.rhs, &mut x, false).unwrap();
            let mono = c.monodromy().unwrap();
            let want = transfer_product(&s, &sys);
            let err = max_diff(&mono, &want);
            assert!(err <= 1e-9 * max_abs(&want), "{n} {m} {nint} {k}: {err}");
        }
    }

    #[test]
    fn condense_intervals_gives_the_monodromy_of_solve() {
        let mut rng = Rng(6);
        for (n, m, nint, k) in SHAPES {
            let s = CondenseShape::new(n, m, nint, k).unwrap();
            let sys = Sys::random(&s, &mut rng);
            let mut c = Condenser::new(s).unwrap();
            let mut x = vec![0.0; s.dim()];
            c.solve(&sys.view(), &sys.rhs, &mut x, false).unwrap();
            let want = c.monodromy().unwrap();
            for parallel in [false, true] {
                let mut d = Condenser::new(s).unwrap();
                d.condense_intervals(&sys.view(), parallel).unwrap();
                assert_eq!(d.monodromy().unwrap(), want, "{n} {m} {nint} {k}");
            }
        }
        let s = CondenseShape::new(2, 3, 4, 2).unwrap();
        let sys = Sys::random(&s, &mut Rng(2));
        let mut c = Condenser::new(s).unwrap();
        let short = BlockSystem {
            bc: &sys.bc[1..],
            ..sys.view()
        };
        assert_eq!(
            c.condense_intervals(&short, false),
            Err(LinalgError::BadShape)
        );
        let mut blocks = sys.blocks.clone();
        blocks[0] = f64::INFINITY;
        let bad = BlockSystem {
            blocks: &blocks,
            ..sys.view()
        };
        assert_eq!(
            c.condense_intervals(&bad, false),
            Err(LinalgError::NotFinite)
        );
        assert_eq!(c.monodromy(), Err(LinalgError::BadShape));
    }

    #[test]
    fn reuse_across_solves() {
        let s = CondenseShape::new(3, 4, 5, 2).unwrap();
        let mut rng = Rng(9);
        let mut c = Condenser::new(s).unwrap();
        for _ in 0..2 {
            let sys = Sys::random(&s, &mut rng);
            let mut x = vec![0.0; s.dim()];
            c.solve(&sys.view(), &sys.rhs, &mut x, false).unwrap();
            assert_eq!(x, solve(&s, &sys, false));
        }
    }

    #[test]
    fn bad_shapes_and_lengths() {
        for (n, m, nint, k) in [
            (0, 2, 1, 0),
            (1, 1, 1, 0),
            (1, 2, 0, 0),
            (usize::MAX / 2, 2, 2, 0),
        ] {
            assert_eq!(
                CondenseShape::new(n, m, nint, k),
                Err(LinalgError::BadShape)
            );
        }
        let s = CondenseShape::new(2, 3, 4, 2).unwrap();
        let sys = Sys::random(&s, &mut Rng(1));
        let mut c = Condenser::new(s).unwrap();
        let mut x = vec![0.0; s.dim()];
        let short = BlockSystem {
            blocks: &sys.blocks[1..],
            ..sys.view()
        };
        assert_eq!(
            c.solve(&short, &sys.rhs, &mut x, false),
            Err(LinalgError::BadShape)
        );
        assert_eq!(
            c.solve(&sys.view(), &sys.rhs[1..], &mut x, false),
            Err(LinalgError::BadShape)
        );
        assert_eq!(
            c.solve(&sys.view(), &sys.rhs, &mut x[1..], false),
            Err(LinalgError::BadShape)
        );
        let mut rhs = sys.rhs.clone();
        rhs[3] = f64::NAN;
        assert_eq!(
            c.solve(&sys.view(), &rhs, &mut x, false),
            Err(LinalgError::NotFinite)
        );
        assert_eq!(c.monodromy(), Err(LinalgError::BadShape));
    }

    #[test]
    fn singular_interval_and_singular_reduced_system() {
        let s = CondenseShape::new(2, 3, 4, 2).unwrap();
        let (n, nm, k) = (s.n(), s.block_rows(), s.k());
        let mut x = vec![0.0; s.dim()];
        // Unknown u_1 (the first interior column of interval 0) appears nowhere.
        let mut sys = Sys::random(&s, &mut Rng(2));
        sys.blocks[nm * n..nm * (n + 1)].fill(0.0);
        for r in 0..k {
            sys.border[r + k * n] = 0.0;
        }
        for parallel in [false, true] {
            let mut c = Condenser::new(s).unwrap();
            assert_eq!(
                c.solve(&sys.view(), &sys.rhs, &mut x, parallel),
                Err(LinalgError::Singular)
            );
        }
        // A zero border row.
        let mut sys = Sys::random(&s, &mut Rng(2));
        for col in 0..s.dim() {
            sys.border[1 + k * col] = 0.0;
        }
        let mut c = Condenser::new(s).unwrap();
        assert_eq!(
            c.solve(&sys.view(), &sys.rhs, &mut x, false),
            Err(LinalgError::Singular)
        );
        assert_eq!(c.monodromy(), Err(LinalgError::BadShape));
    }

    /// The timings behind the parallel threshold of the cycle problem. Run with
    /// `cargo test --release -p pc-linalg timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn timing_sequential_and_parallel() {
        use std::time::Instant;
        let median = |c: &mut Condenser, sys: &Sys, x: &mut [f64], par: bool| {
            let mut t: Vec<f64> = (0..15)
                .map(|_| {
                    let start = Instant::now();
                    c.solve(&sys.view(), &sys.rhs, x, par).unwrap();
                    start.elapsed().as_secs_f64()
                })
                .collect();
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        };
        for threads in [2, 4, 6, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for n in [2, 4, 10] {
                for nint in [20, 40, 80, 160, 320, 640] {
                    let s = CondenseShape::new(n, 4, nint, 2).unwrap();
                    let sys = Sys::random(&s, &mut Rng(5));
                    let mut c = Condenser::new(s).unwrap();
                    let mut x = vec![0.0; s.dim()];
                    let seq = median(&mut c, &sys, &mut x, false);
                    let par = pool.install(|| median(&mut c, &sys, &mut x, true));
                    println!(
                        "threads {threads:2} n {n:2} N {nint:3}: seq {:8.1} us, par {:8.1} us, \
                         speedup {:.2}",
                        seq * 1e6,
                        par * 1e6,
                        seq / par
                    );
                }
            }
        }
    }
}
