//! A flat store of rows of one width for the IVP and ITER runs (SPEC section 8af-1).
//!
//! Row j is `[alpha, x, aux]` at `data[j * stride..(j + 1) * stride]`. The buffer doubles when
//! it is full, starting from one row, so it never holds more than twice the rows it has.

/// Rows of `stride` values each, stored end to end.
#[derive(Debug, Default)]
pub(crate) struct Rows {
    data: Vec<f64>,
    stride: usize,
}

impl Rows {
    /// An empty store for rows of `stride` values. `stride` is at least 1.
    pub(crate) fn new(stride: usize) -> Rows {
        assert!(stride > 0, "a row has at least one value");
        Rows {
            data: Vec::new(),
            stride,
        }
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Number of whole rows.
    pub(crate) fn len(&self) -> usize {
        self.data.len() / self.stride
    }

    /// Appends the row `[alpha, x, aux]`, whose length must be the stride.
    pub(crate) fn push(&mut self, alpha: f64, x: &[f64], aux: &[f64]) {
        debug_assert_eq!(1 + x.len() + aux.len(), self.stride);
        let len = self.data.len();
        if self.data.capacity() - len < self.stride {
            // Double the rows the buffer holds, from one row; `reserve_exact` keeps the
            // growth at that size instead of the allocator's own rounding up.
            let rows = (self.data.capacity() / self.stride).max(1);
            let want = rows.saturating_mul(2).saturating_mul(self.stride);
            let target = want.max(len + self.stride);
            self.data.reserve_exact(target - len);
        }
        self.data.push(alpha);
        self.data.extend_from_slice(x);
        self.data.extend_from_slice(aux);
    }

    /// Copies rows `start..start + count`, or as many of them as exist, into the front of
    /// `out` and returns how many it copied. `out` must hold `count * stride` values.
    pub(crate) fn copy(&self, start: usize, count: usize, out: &mut [f64]) -> usize {
        let rows = self.len().saturating_sub(start).min(count);
        if rows == 0 {
            return 0;
        }
        let from = start * self.stride;
        let n = rows * self.stride;
        out[..n].copy_from_slice(&self.data[from..from + n]);
        rows
    }

    /// Drops rows by the rule of SPEC section 8af-2 and returns how many it kept. Every kept
    /// row keeps its bits. The kept rows move to the front of the same buffer, which then
    /// shrinks to fit, so the only extra memory is a mask of one byte a row.
    pub(crate) fn thin(&mut self, tol: f64) -> usize {
        let keep = thin_mask(&self.data, self.stride, tol);
        let w = self.stride;
        let mut out = 0;
        for (k, &kp) in keep.iter().enumerate() {
            if kp {
                if out != k {
                    self.data.copy_within(k * w..(k + 1) * w, out * w);
                }
                out += 1;
            }
        }
        self.data.truncate(out * w);
        self.data.shrink_to_fit();
        out
    }

    #[cfg(test)]
    fn capacity_rows(&self) -> usize {
        self.data.capacity() / self.stride
    }
}

/// The span of each column for the thinning rule: largest minus smallest finite value, and 1
/// for a column whose span is 0 or that has no finite value.
fn spans(data: &[f64], w: usize) -> Vec<f64> {
    let mut lo = vec![f64::INFINITY; w];
    let mut hi = vec![f64::NEG_INFINITY; w];
    for r in data.chunks_exact(w) {
        for j in 0..w {
            if r[j].is_finite() {
                lo[j] = lo[j].min(r[j]);
                hi[j] = hi[j].max(r[j]);
            }
        }
    }
    (0..w)
        .map(|j| {
            let s = hi[j] - lo[j];
            if hi[j] < lo[j] || s == 0.0 { 1.0 } else { s }
        })
        .collect()
}

/// Which rows of `data` (rows of `w` values, column 0 is t) the thinning rule of SPEC section
/// 8af-2 keeps. The first and last rows are kept, and so is every row with a non-finite value
/// together with the rows next to it. Between those, a Ramer-Douglas-Peucker search on an
/// explicit stack drops every row of a segment that the linear interpolation in t between the
/// segment ends reproduces within `tol` times the span of each column, and otherwise keeps
/// the worst row and searches the two halves.
fn thin_mask(data: &[f64], w: usize, tol: f64) -> Vec<bool> {
    let n = data.len() / w;
    let mut keep = vec![false; n];
    if n == 0 {
        return keep;
    }
    keep[0] = true;
    keep[n - 1] = true;
    for (k, r) in data.chunks_exact(w).enumerate() {
        if r.iter().any(|v| !v.is_finite()) {
            keep[k.saturating_sub(1)..(k + 2).min(n)].fill(true);
        }
    }
    let span = spans(data, w);
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut a = 0;
    for (b, &kp) in keep.iter().enumerate().skip(1) {
        if kp {
            if b > a + 1 {
                stack.push((a, b));
            }
            a = b;
        }
    }
    let row = |k: usize| &data[k * w..(k + 1) * w];
    while let Some((a, b)) = stack.pop() {
        let (ra, rb) = (row(a), row(b));
        let dt = rb[0] - ra[0];
        let mut pass = true;
        let (mut worst, mut worst_k) = (f64::NEG_INFINITY, a + 1);
        for k in a + 1..b {
            let rk = row(k);
            let f = (rk[0] - ra[0]) / dt;
            let mut r = f64::NEG_INFINITY;
            for j in 1..w {
                let v = ra[j] + f * (rb[j] - ra[j]);
                let e = (v - rk[j]).abs();
                // A NaN error fails. The bound is never NaN: tol is finite and every span
                // is positive.
                if e.is_nan() || e > tol * span[j] {
                    pass = false;
                }
                let q = e / span[j];
                if q > r || q.is_nan() {
                    r = if q.is_nan() { f64::INFINITY } else { q };
                }
            }
            if r > worst {
                worst = r;
                worst_k = k;
            }
        }
        if !pass {
            keep[worst_k] = true;
            if worst_k > a + 1 {
                stack.push((a, worst_k));
            }
            if b > worst_k + 1 {
                stack.push((worst_k, b));
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of `w` values from a list of rows.
    fn store(rows: &[Vec<f64>]) -> Rows {
        let mut r = Rows::new(rows[0].len());
        for x in rows {
            r.push(x[0], &x[1..], &[]);
        }
        r
    }

    /// Asserts the guarantee of SPEC 8af-2: every kept row is an unthinned row bit for bit,
    /// in order, the ends and the non-finite rows with their neighbours are kept, and every
    /// dropped row is reproduced within tol times the span by its two kept neighbours.
    fn assert_guarantee(full: &[f64], w: usize, tol: f64) -> usize {
        let keep = thin_mask(full, w, tol);
        let mut r = Rows::new(w);
        r.data = full.to_vec();
        let kept = r.thin(tol);
        assert_eq!(kept, keep.iter().filter(|&&k| k).count());
        let idx: Vec<usize> = (0..keep.len()).filter(|&k| keep[k]).collect();
        for (i, &k) in idx.iter().enumerate() {
            let (got, want) = (&r.data[i * w..(i + 1) * w], &full[k * w..(k + 1) * w]);
            assert!(
                got.iter()
                    .zip(want)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            );
        }
        let n = keep.len();
        assert!(keep[0] && keep[n - 1]);
        let span = spans(full, w);
        let row = |k: usize| &full[k * w..(k + 1) * w];
        for k in 0..n {
            if !row(k).iter().all(|v| v.is_finite()) {
                assert!(keep[k.saturating_sub(1)] && keep[k] && keep[(k + 1).min(n - 1)]);
            }
        }
        for p in idx.windows(2) {
            let (a, b) = (p[0], p[1]);
            for k in a + 1..b {
                let (ra, rb, rk) = (row(a), row(b), row(k));
                let f = (rk[0] - ra[0]) / (rb[0] - ra[0]);
                for j in 1..w {
                    let v = ra[j] + f * (rb[j] - ra[j]);
                    assert!((v - rk[j]).abs() <= tol * span[j], "row {k}, column {j}");
                }
            }
        }
        kept
    }

    #[test]
    fn thinning_keeps_the_guarantee_and_the_bits_of_kept_rows() {
        // A damped spiral with a slow third column and a column of one value.
        let mut full = Vec::new();
        for k in 0..4000 {
            let t = k as f64 * 0.01 + (k % 7) as f64 * 1e-4;
            full.extend([t, (-0.05 * t).exp() * t.cos(), (-0.05 * t).exp() * t.sin()]);
            full.extend([t * t, 2.5]);
        }
        for tol in [1e-2, 1e-3, 1e-6] {
            let kept = assert_guarantee(&full, 5, tol);
            // At 1e-6 the curvature error of a step, about 1e-5, keeps every row.
            assert!(tol < 1e-4 || kept < 1000, "tol {tol} keeps {kept}");
        }
        // A tolerance that drops nothing still gives the same rows.
        assert_eq!(assert_guarantee(&full, 5, 1e-300), 4000);
    }

    #[test]
    fn non_finite_rows_and_their_neighbours_are_kept() {
        let mut rows: Vec<Vec<f64>> = (0..200).map(|k| vec![k as f64, 3.0, k as f64]).collect();
        rows[50][1] = f64::NAN;
        rows[120][2] = f64::INFINITY;
        rows[121][1] = f64::NEG_INFINITY;
        let r = store(&rows);
        let keep = thin_mask(&r.data, 3, 1e-3);
        let kept: Vec<usize> = (0..200).filter(|&k| keep[k]).collect();
        assert_eq!(kept, vec![0, 49, 50, 51, 119, 120, 121, 122, 199]);
        assert_guarantee(&r.data, 3, 1e-3);
    }

    #[test]
    fn short_runs_keep_every_row() {
        for n in 0..3 {
            let rows: Vec<Vec<f64>> = (0..n).map(|k| vec![k as f64, 1.0]).collect();
            let mut r = Rows::new(2);
            for x in &rows {
                r.push(x[0], &x[1..], &[]);
            }
            assert_eq!(r.thin(0.5), n);
        }
    }

    #[test]
    fn a_million_rows_with_many_splits_do_not_overflow_the_stack() {
        // A chirp: the frequency grows along the run, so the search splits deep and often.
        let n = 1_000_000;
        let mut r = Rows::new(3);
        for k in 0..n {
            let t = k as f64 * 1e-3;
            r.push(t, &[(t * t * 0.05).sin(), (t * 0.37).cos()], &[]);
        }
        let kept = r.thin(1e-6);
        assert!(kept > 10_000 && kept < n, "kept {kept}");
        assert_eq!(r.len(), kept);
        assert!(r.data.capacity() == kept * 3);
    }

    #[test]
    fn the_buffer_holds_at_most_twice_the_rows() {
        for stride in [2, 3, 7] {
            let mut r = Rows::new(stride);
            let x = vec![1.5; stride - 1];
            for j in 1..=5000usize {
                r.push(j as f64, &x, &[]);
                assert_eq!(r.len(), j);
                // 8 bytes a value: at most 16 * stride bytes a row.
                assert!(
                    8 * stride * r.capacity_rows() <= 16 * stride * j,
                    "stride {stride}, rows {j}, capacity {}",
                    r.capacity_rows()
                );
            }
        }
    }

    #[test]
    fn rows_come_back_in_order_and_a_copy_past_the_end_is_short() {
        let mut r = Rows::new(4);
        for j in 0..10 {
            let v = j as f64;
            r.push(v, &[v + 0.25, v + 0.5], &[-v]);
        }
        let mut out = vec![f64::NAN; 3 * 4];
        assert_eq!(r.copy(8, 3, &mut out), 2);
        assert_eq!(&out[..8], &[8.0, 8.25, 8.5, -8.0, 9.0, 9.25, 9.5, -9.0]);
        assert!(out[8..].iter().all(|v| v.is_nan()));
        assert_eq!(r.copy(10, 3, &mut out), 0);
        assert_eq!(r.copy(usize::MAX, 3, &mut out), 0);
    }
}
