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

    #[cfg(test)]
    fn capacity_rows(&self) -> usize {
        self.data.capacity() / self.stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
