//! Method (a): sparse LU of the full matrix, and the 1-norm condition
//! estimate built on its solves.

use faer::MatMut;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};

use super::matrix::Problem;
use crate::ffi::{Result, internal};

/// The full matrix in CSC form, built from `for_each_entry`.
pub fn csc(p: &Problem) -> Result<SparseColMat<usize, f64>> {
    let mut t = Vec::new();
    t.try_reserve_exact(p.nnz)
        .map_err(|_| internal(format!("s3: cannot allocate {} triplets", p.nnz)))?;
    p.for_each_entry(|row, col, val| t.push(Triplet::new(row, col, val)));
    SparseColMat::try_new_from_triplets(p.nr, p.nr, &t)
        .map_err(|e| internal(format!("s3: cannot build the sparse matrix: {e:?}")))
}

/// Symbolic LU of a CSC matrix.
pub fn symbolic(a: &SparseColMat<usize, f64>) -> Result<SymbolicLu<usize>> {
    SymbolicLu::try_new(a.symbolic())
        .map_err(|e| internal(format!("s3: symbolic LU failed: {e:?}")))
}

/// Numeric LU that reuses a symbolic factorization.
pub fn numeric(sym: &SymbolicLu<usize>, a: &SparseColMat<usize, f64>) -> Result<Lu<usize, f64>> {
    Lu::try_new_with_symbolic(sym.clone(), a.as_ref())
        .map_err(|e| internal(format!("s3: numeric LU failed: {e:?}")))
}

/// Solves A x = v (or Aᵀ x = v) in place.
pub fn solve_in_place(lu: &Lu<usize, f64>, v: &mut [f64], transpose: bool) {
    let n = v.len();
    let rhs = MatMut::from_column_major_slice_mut(v, n, 1);
    if transpose {
        lu.solve_transpose_in_place(rhs);
    } else {
        lu.solve_in_place(rhs);
    }
}

/// Estimates ‖A⁻¹‖₁ from solves with A and Aᵀ. This is the algorithm of
/// Hager (1984) with the refinements of Higham (1988), in the form LAPACK's
/// DLACN2 uses: at most five steps on sign vectors, then the test vector
/// with alternating signs and magnitudes 1 to 2 as a lower bound.
pub fn inv_norm1_est(lu: &Lu<usize, f64>, n: usize) -> f64 {
    let norm1 = |v: &[f64]| v.iter().map(|x| x.abs()).sum::<f64>();
    let signs = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .map(|&x| if x >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    };
    // First index of the largest magnitude, as IDAMAX returns it.
    let argmax = |v: &[f64]| (1..n).fold(0, |j, i| if v[i].abs() > v[j].abs() { i } else { j });
    let mut x = vec![1.0 / n as f64; n];
    solve_in_place(lu, &mut x, false);
    if n == 1 {
        return x[0].abs();
    }
    let mut est = norm1(&x);
    let mut sgn = signs(&x);
    let mut z = sgn.clone();
    solve_in_place(lu, &mut z, true);
    let mut j = argmax(&z);
    let mut iter = 2;
    loop {
        let mut v = vec![0.0; n];
        v[j] = 1.0;
        solve_in_place(lu, &mut v, false);
        let est_old = est;
        est = norm1(&v);
        let new_sgn = signs(&v);
        // A repeated sign vector means convergence; no growth means cycling.
        if new_sgn == sgn || est <= est_old {
            break;
        }
        sgn = new_sgn;
        z.copy_from_slice(&sgn);
        solve_in_place(lu, &mut z, true);
        let j_last = j;
        j = argmax(&z);
        if z[j_last] == z[j].abs() || iter >= 5 {
            break;
        }
        iter += 1;
    }
    let mut alt: Vec<f64> = (0..n)
        .map(|i| {
            let mag = 1.0 + i as f64 / (n - 1) as f64;
            if i % 2 == 0 { mag } else { -mag }
        })
        .collect();
    solve_in_place(lu, &mut alt, false);
    est.max(2.0 * norm1(&alt) / (3.0 * n as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s3_linalg::matrix::Params;

    fn factor(n: usize, m: usize, big_n: usize) -> (Problem, Lu<usize, f64>) {
        let p = Problem::build(&Params {
            n,
            m,
            big_n,
            t: 1.0,
            seed: 7,
        })
        .unwrap();
        let a = csc(&p).unwrap();
        let lu = numeric(&symbolic(&a).unwrap(), &a).unwrap();
        (p, lu)
    }

    #[test]
    fn transpose_solve_satisfies_the_transposed_system() {
        let (p, lu) = factor(3, 4, 5);
        let v: Vec<f64> = (0..p.nr).map(|i| (i as f64).sin()).collect();
        let mut x = v.clone();
        solve_in_place(&lu, &mut x, true);
        // r = Aᵀx - v, accumulated entry by entry.
        let mut r: Vec<f64> = v.iter().map(|t| -t).collect();
        p.for_each_entry(|i, j, a| r[j] += a * x[i]);
        let worst = r.iter().fold(0.0f64, |w, t| w.max(t.abs()));
        assert!(worst < 1e-10, "max |Aᵀx - v| = {worst:e}");
    }

    #[test]
    fn inv_norm1_est_is_a_close_lower_bound() {
        for (n, m, big_n) in [(1, 2, 1), (2, 3, 4), (3, 4, 5), (4, 4, 10)] {
            let (p, lu) = factor(n, m, big_n);
            // ‖A⁻¹‖₁ is the largest column sum of A⁻¹, one solve per column.
            let mut exact = 0.0f64;
            for j in 0..p.nr {
                let mut e = vec![0.0; p.nr];
                e[j] = 1.0;
                solve_in_place(&lu, &mut e, false);
                exact = exact.max(e.iter().map(|t| t.abs()).sum());
            }
            let est = inv_norm1_est(&lu, p.nr);
            assert!(
                est <= exact * (1.0 + 1e-12) && est >= exact / 3.0,
                "n={n} m={m} N={big_n}: est {est:e}, exact {exact:e}"
            );
        }
    }
}
