//! The k-th iterate of a map (SPEC section 8y).
//!
//! A period-k point of `x -> F(x, p)` is a fixed point of `F^k`, so every fixed point
//! problem of SPEC sections 8v to 8x runs on `F^k` in place of `F`. With `x_0 = x`,
//! `x_{j+1} = F(x_j, p)` and `A_j = F_x(x_j)`:
//!
//! - `F` is `x_k`;
//! - `JacU` is `A_{k-1} ... A_1 A_0`;
//! - `JacP` is `M_k` with `M_0 = 0` and `M_{j+1} = A_j M_j + F_p(x_j)`;
//! - `Aux` is the auxiliary values at `x_0`;
//! - `Dir` with slots `z_1 .. z_m` is the derivative of `x_k` along them, by the chain rule
//!   for mixed derivatives of a composition (Faa di Bruno's formula over set partitions).
//!
//! The recursions for the Jacobian and the parameter derivative are eqs. (4.1), (4.2) and
//! (4.7) of Govaerts, Khoshsiar Ghaziani, Kuznetsov and Meijer, SIAM J. Sci. Comput.
//! 29(6):2644-2667 (2007), and the set partition form gives their eqs. (4.3) and (4.4) for
//! two and three state slots. Every step gets the same `p`, `t` included.

use crate::error::EngineError;
use crate::model::{ModelEval, check_finite};
use pc_ir::{EntryKind, Slot};

/// `F^k` over the model of a map `F`. `k = 1` gives `F` itself.
pub struct IteratedMap<'a> {
    inner: &'a mut dyn ModelEval,
    k: usize,
}

impl<'a> IteratedMap<'a> {
    /// `F^k`; a `k` of 0 is taken as 1.
    pub fn new(inner: &'a mut dyn ModelEval, k: usize) -> Self {
        IteratedMap { inner, k: k.max(1) }
    }

    /// The `k` of `F^k`.
    pub fn k(&self) -> usize {
        self.k
    }
}

/// The set partitions of the bit set `mask`, each as the list of its blocks. The block
/// that holds the lowest element of `mask` comes first.
fn partitions(mask: u32) -> Vec<Vec<u32>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let low = mask.isolate_lowest_one();
    let rest = mask & !low;
    let mut out = Vec::new();
    // Every subset `sub` of `rest`, including the empty set, joins `low` in one block.
    let mut sub = rest;
    loop {
        let block = sub | low;
        for mut tail in partitions(mask & !block) {
            tail.insert(0, block);
            out.push(tail);
        }
        if sub == 0 {
            break;
        }
        sub = (sub - 1) & rest;
    }
    out
}

impl IteratedMap<'_> {
    fn step(&mut self, x: &[f64], p: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.inner.eval(&EntryKind::F, x, p, &[], out)
    }

    /// The orbit `x_0 .. x_{k-1}` of `x`, which every derivative needs.
    fn orbit(&mut self, x: &[f64], p: &[f64]) -> Result<Vec<Vec<f64>>, EngineError> {
        let n = x.len();
        let mut xs = Vec::with_capacity(self.k);
        xs.push(x.to_vec());
        for j in 1..self.k {
            let mut next = vec![0.0; n];
            self.step(&xs[j - 1], p, &mut next)?;
            xs.push(next);
        }
        Ok(xs)
    }

    fn dir(
        &mut self,
        slots: &[Slot],
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        let (n, n_p) = (self.inner.n(), self.inner.n_p());
        let m = slots.len();
        if m == 0 || m > 5 {
            return Err(EngineError::Model(format!(
                "bad entry: a directional entry has 1 to 5 slots, got {m}"
            )));
        }
        // The state and parameter parts of each slot direction.
        let mut xs0 = Vec::with_capacity(m);
        let mut ps = Vec::with_capacity(m);
        let mut at = 0;
        for &s in slots {
            let len = s.len(n, n_p);
            let d = &v[at..at + len];
            at += len;
            let (dx, dp) = match s {
                Slot::U => (d.to_vec(), vec![0.0; n_p]),
                Slot::P => (vec![0.0; n], d.to_vec()),
                Slot::Full => (d[..n].to_vec(), d[n..].to_vec()),
            };
            xs0.push(dx);
            ps.push(dp);
        }
        let all_u = slots.iter().all(|s| *s == Slot::U);
        let full = (1u32 << m) - 1;
        // X(S) for every nonempty subset S, indexed by its bit set. The parameter part of a
        // subset is that of its slot for a single slot and zero otherwise.
        let mut cur = vec![vec![0.0; n]; full as usize + 1];
        for s in 0..m {
            cur[1 << s] = xs0[s].clone();
        }
        let p_of = |b: u32| -> Option<&Vec<f64>> {
            if b.count_ones() == 1 {
                Some(&ps[b.trailing_zeros() as usize])
            } else {
                None
            }
        };
        let is_zero = |x: &[f64], b: u32| {
            x.iter().all(|&a| a == 0.0) && p_of(b).is_none_or(|q| q.iter().all(|&a| a == 0.0))
        };
        let parts: Vec<Vec<Vec<u32>>> = (0..=full).map(partitions).collect();
        let orbit = self.orbit(x, p)?;
        let mut term = vec![0.0; n];
        for (j, xj) in orbit.iter().enumerate() {
            let last = j + 1 == self.k;
            let mut next = vec![vec![0.0; n]; full as usize + 1];
            let targets: Vec<u32> = if last {
                vec![full]
            } else {
                (1..=full).collect()
            };
            for sset in targets {
                let mut acc = vec![0.0; n];
                for part in &parts[sset as usize] {
                    if part.iter().any(|&b| is_zero(&cur[b as usize], b)) {
                        continue;
                    }
                    let r = part.len();
                    let mut dv = Vec::new();
                    let kind = if all_u {
                        for &b in part {
                            dv.extend_from_slice(&cur[b as usize]);
                        }
                        EntryKind::Dir(vec![Slot::U; r])
                    } else {
                        for &b in part {
                            dv.extend_from_slice(&cur[b as usize]);
                            match p_of(b) {
                                Some(q) => dv.extend_from_slice(q),
                                None => dv.extend(std::iter::repeat_n(0.0, n_p)),
                            }
                        }
                        EntryKind::Dir(vec![Slot::Full; r])
                    };
                    self.inner.eval(&kind, xj, p, &dv, &mut term)?;
                    for (a, t) in acc.iter_mut().zip(&term) {
                        *a += t;
                    }
                }
                next[sset as usize] = acc;
            }
            cur = next;
        }
        out.copy_from_slice(&cur[full as usize]);
        Ok(())
    }
}

impl ModelEval for IteratedMap<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }

    fn n_p(&self) -> usize {
        self.inner.n_p()
    }

    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        if self.k == 1 || matches!(kind, EntryKind::Aux) {
            return self.inner.eval(kind, x, p, v, out);
        }
        let (n, n_p) = (self.inner.n(), self.inner.n_p());
        let (out_len, v_len) = match kind {
            EntryKind::F => (n, 0),
            EntryKind::JacU => (n * n, 0),
            EntryKind::JacP => (n * n_p, 0),
            EntryKind::Dir(slots) => (n, slots.iter().map(|s| s.len(n, n_p)).sum()),
            EntryKind::Aux => unreachable!("Aux is delegated above"),
        };
        for (name, got, want) in [
            ("x", x.len(), n),
            ("p", p.len(), n_p + 1),
            ("v", v.len(), v_len),
            ("out", out.len(), out_len),
        ] {
            if got != want {
                return Err(EngineError::Model(format!(
                    "bad length: {name} has {got} values, expected {want}"
                )));
            }
        }
        match kind {
            EntryKind::F => {
                let mut cur = x.to_vec();
                for _ in 1..self.k {
                    let mut next = vec![0.0; n];
                    self.step(&cur, p, &mut next)?;
                    cur = next;
                }
                self.step(&cur, p, out)?;
            }
            EntryKind::JacU | EntryKind::JacP => {
                let orbit = self.orbit(x, p)?;
                let jac_p = matches!(kind, EntryKind::JacP);
                let cols = if jac_p { n_p } else { n };
                // Q_0 = I for JacU and M_0 = 0 for JacP, column-major n by cols.
                let mut q = vec![0.0; n * cols];
                if !jac_p {
                    for i in 0..n {
                        q[i + n * i] = 1.0;
                    }
                }
                let mut a = vec![0.0; n * n];
                let mut fp = vec![0.0; n * n_p];
                for xj in &orbit {
                    self.inner.eval(&EntryKind::JacU, xj, p, &[], &mut a)?;
                    let mut next = vec![0.0; n * cols];
                    if jac_p {
                        self.inner.eval(&EntryKind::JacP, xj, p, &[], &mut fp)?;
                        next.copy_from_slice(&fp);
                    }
                    for c in 0..cols {
                        for l in 0..n {
                            let ql = q[l + n * c];
                            if ql != 0.0 {
                                for r in 0..n {
                                    next[r + n * c] += a[r + n * l] * ql;
                                }
                            }
                        }
                    }
                    q = next;
                }
                out.copy_from_slice(&q);
            }
            EntryKind::Dir(slots) => self.dir(slots, x, p, v, out)?,
            EntryKind::Aux => unreachable!("Aux is delegated above"),
        }
        check_finite(kind, out)
    }
}

#[cfg(test)]
mod tests {
    use super::partitions;

    #[test]
    fn partitions_count_the_bell_numbers() {
        let bell = [1, 1, 2, 5, 15, 52];
        for m in 0..=5u32 {
            let parts = partitions((1 << m) - 1);
            assert_eq!(parts.len(), bell[m as usize], "m = {m}");
            for p in &parts {
                let union = p.iter().fold(0, |a, b| {
                    assert_eq!(a & b, 0, "blocks overlap");
                    a | b
                });
                assert_eq!(union, (1 << m) - 1);
            }
        }
        // 202 (subset, partition) pairs for five slots.
        let pairs: usize = (1..32u32).map(|s| partitions(s).len()).sum();
        assert_eq!(pairs, 202);
    }
}
