# Spike notes

The state of each spike and the next steps are in `docs/STATUS.md`. The approved Phase 0 plan is kept outside the repo; `docs/STATUS.md` summarizes it. API recipes for Cranelift, faer and cudarc are in `API_NOTES.md`.

## S3 design

### Arguments and sizes

- Matrix arguments: `n=4 m=4 N=40 T=1 seed=2026`. `Args` has no f64 helper, so parse T from `str_or`.
- Require n ≥ 1, m ≥ 2 and N ≥ 1, with caps n ≤ 32, m ≤ 10 and N ≤ 20000. Run the dense reduced solve only when nred ≤ 5000.
- Unknowns: u_0 .. u_{Nm} (n each), then T, then α. nu = n(Nm+1) and nr = nu + 2. Component c of u_i is column i·n + c, T is column nu and α is column nu + 1.
- h = 1/N.

### Rows

- Collocation, interval j, Gauss point k, component r: row (j·m + k)·n + r.
- Periodicity: row N·m·n + c, with +1 in column c and -1 in column N·m·n + c.
- Phase row nu, PALC row nu + 1.

### Gauss-Legendre points on [0, 1]

- Newton on the Legendre recurrence, starting from cos(π(i + 0.75)/(m + 0.5)), with P_m' = m(P_{m-1} - x P_m)/(1 - x²).
- Nodes z = (1 + x)/2 and weights w = 1/((1 - x²) P_m'²), sorted ascending. The weights sum to 1, and the rule is exact for degree ≤ 2m - 1.
- Test for m = 4 against the [-1, 1] values x = ±0.861136311594053, ±0.339981043584856 with weights 0.347854845137454 and 0.652145154862546. On [0, 1] the weights are half of those.

### Lagrange basis

- Representation points s_l = l/m for l = 0..m.
- L[k][l] = ℓ_l(z_k) and D[k][l] = ℓ_l'(z_k). Start from val = 1 and der = 0, then for each i ≠ l take f = (z - s_i)/(s_l - s_i), set der = der·f + val/(s_l - s_i), then val *= f.
- Tests: Σ_l L = 1, Σ_l D = 0, Σ_l D·s_l = 1 and Σ_l D·s_l^m = m z^{m-1}.

### Blocks

- One block per interval, row-major, with n·m rows and n(m+1) + 2 columns. Columns 0 .. n(m+1) hold u_{jm} .. u_{(j+1)m}; then come T and α.
- Entry in row k·n + r, column l·n + c: D[k][l]·δ_rc - h·T·L[k][l]·A_jk[r][c].
- A_jk = -I + 0.3 R, with R uniform in [-1, 1]. It is drawn per Gauss point; the plan said per interval, so `docs/spikes.md` must note the change.
- The T and α entries are h·rng.range(-1, 1).

### Phase and PALC rows, right-hand side

- Phase row: for each (j, k), draw udot_c uniform in [-1, 1] for c = 0..n, then add h·w_k·L[k][l]·udot_c into phase[(j·m + l)·n + c] for l = 0..m. Its T and α entries are zero.
- PALC row: a random unit vector of length nr.
- Right-hand side: uniform in [-1, 1].

### Entries

- `for_each_entry(f(row, col, val))` visits rows in order, with no duplicate pairs. The triplets, `s3_copy` (1-based), the matvec, the row-sum norm and the backward error all use it.
- nnz = N·n·m·(n(m+1) + 2) + 2n + nu + nr, which is 153,618 at N = 400.

### Method (a): sparse LU of the full matrix

- Build the CSC matrix once from triplets.
- "first" times symbolic + numeric + solve; "repeat" times numeric + solve, with the symbolic factorization reused.

### Method (b): condensation, per interval j

R = n·m rows, ni = n(m-1) interior columns, ne = 2n + 2 remaining columns.

1. `lu = Mat::from_fn(R, ni, |r, c| blk[r*cols + n + c])`, then `lu_in_place` with `Par::Seq`.
2. `y` is R × (ne + 1), with its rows permuted by perm: row r takes block row perm[r]. Remaining column q maps to block column q if q < n, n·m + (q - n) if q < 2n, and n(m+1) + (q - 2n) otherwise. The last column is b[j·R + perm[r]].
3. Y1, the top ni rows of y: `solve_unit_lower_triangular_in_place` with L11.
4. Y2, the bottom n rows: Y2 = Y2 - L21·Y1 through `matmul` with `Accum::Add` and alpha -1.
5. Border rows (phase, PALC): r_int is the full row's entries in columns j·m·n + n .. j·m·n + n + ni. Solve Uᵀg = r_int with `solve_lower_triangular_in_place(U.transpose(), g)`. The contribution is Y1ᵀg, of length ne + 1.

Keep the result as `struct Condensed { lu, y, border: [Vec<f64>; 2] }`. The permutation is only needed while building y.

### Reduced system

- nmesh = n(N+1) and nred = nmesh + 2. Its unknowns are the mesh values u_{jm} (reduced column j·n + c), then T and α.
- Condensed rows j·n + i hold row i of Y2: mesh j, mesh j + 1, then T and α. Their right-hand side is the last column of Y2.
- Periodicity rows N·n + c, with right-hand side b[N·m·n + c].
- Phase and PALC rows nmesh and nmesh + 1 start from the full row's mesh values and its T and α entries, then subtract every interval's border contribution. Their right-hand side is b[nr - 2 + k] minus the sum over intervals of g[ne].
- Sparse path: compute the CSC pattern, a position map (binary search in row_idx) and the reduced `SymbolicLu` once. Each solve fills a values array and uses `SparseColMatRef::new`.
- Dense path: zero a `Mat`, fill it, run `lu_in_place` with par, permute the right-hand side, then do the unit-lower and upper solves.

### Back-substitution, per interval

- z = Y1_rhs - Y1_rest·x_rest, then `solve_upper_triangular_in_place` with U.
- The interior goes to x[j·m·n + n + c]. Mesh values, T and α come from the reduced solution.

### Parallel variant

- A local `rayon::ThreadPoolBuilder` sized by `threads=`. Condensation uses `map_init` with one `MemBuffer` per thread, and back-substitution uses a parallel map. The border sums are added up sequentially.
- Never use rayon's global pool. Call `faer::set_global_parallelism(Par::Seq)`, because the sparse solves read the global setting. Inside `pool.install`, `Par::rayon(0)` means the pool's size.

### Checks

- Backward error ‖Ax - b‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞) ≤ 1e-13 for every solution.
- Pairwise ‖x_i - x_j‖₂ / ‖x_sparse‖₂ ≤ 1e-10.
- A Hager/Higham 1-norm condition estimate built on the LU solve and transpose solve. MATLAB reports `condest(A)`.

### Timing

One warm-up, then 20 repeats with median, min and max, plus a breakdown into condense, reduced and backsub.

### Interfaces

The gateway contract is already in `mex/pckern_gateway.c`, so the Rust side must match it.

- `s3_build(args, &h, &n, &nnz)`.
- `s3_copy(h, n, nnz, I, J, V, b)`: MATLAB allocates nnz × 1 doubles for I, J and V and n × 1 for b.
- `s3_solve(h, n, args, x, &json)`, with args `method=sparse_lu|condensed_sparse|condensed_dense threads=1 reps=20`.
- `run` (native, through `run_json "s3"` and `spikecli s3`) builds the matrix and runs every method for one N.
- MATLAB side: `A = sparse(I, J, V, nr, nr)`, `timeit` on `decomposition(A)` plus a solve and on `A\b`, `condest(A)` and `maxNumCompThreads`. The timing loop goes in a function file.

## faer 0.24.4 calls checked for S3

- `faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch}`:
  - `lu_in_place(A: MatMut, perm: &mut [I], perm_inv: &mut [I], par: Par, stack: &mut MemStack, params: Spec<PartialPivLuParams, T>) -> (PartialPivLuInfo, PermRef<'_, I>)`.
  - `lu_in_place_scratch::<I, T>(nrows, ncols, par, params) -> StackReq`.
  - Convention: (PA)[i] = A[perm[i]]. `PermRef::arrays(self)` returns (fwd, inv).
- `faer::linalg::triangular_solve::{solve_unit_lower_triangular_in_place, solve_upper_triangular_in_place, solve_lower_triangular_in_place}`, each taking `(tri: MatRef, rhs: MatMut, par: Par)`.
- `faer::linalg::matmul::matmul(dst, beta: Accum, lhs, rhs, alpha: T, par)`, with `faer::Accum::{Replace, Add}`.
- `Par::rayon(n)`: n = 0 means `rayon::current_num_threads()`, which inside `pool.install` is the pool's size.
- `faer::dyn_stack::{MemBuffer, MemStack}` (dyn-stack 0.13.2): `MemStack::new(&mut MemBuffer::new(req))`.
- Sparse matrices:
  - `faer::sparse::Triplet { row, col, val }`.
  - `SparseColMat::<usize, f64>::try_new_from_triplets(nr, nc, &[Triplet]) -> Result<_, CreationError>`. Duplicates are summed.
  - `SparseColMat::new(SymbolicSparseColMat, Vec<T>)` and `SparseColMatRef::new(SymbolicSparseColMatRef, &[T])`.
  - Accessors `.symbolic()`, `.col_ptr()`, `.row_idx()` and `.val()`.
- `faer::sparse::linalg::solvers::{SymbolicLu, Lu}` (source file `src/sparse/solvers.rs`):
  - `SymbolicLu::try_new(SymbolicSparseColMatRef) -> Result<_, FaerError>`. It wraps an Arc, so cloning is cheap.
  - `Lu::try_new_with_symbolic(SymbolicLu, SparseColMatRef) -> Result<_, LuError>`.
  - `SolveCore::solve_in_place_with_conj(&self, Conj, MatMut)` and `solve_transpose_in_place_with_conj`. `Solve` is implemented for every `SolveCore`.
  - The sparse solves use `get_global_parallelism()`.
- `Mat` and `MatMut`: `submatrix(row, col, nrows, ncols)`, `submatrix_mut`, `split_at_row_mut`, `transpose()` and `as_mut()`.
