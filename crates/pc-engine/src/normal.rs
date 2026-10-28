//! Normal form coefficients of refined LP and H points.
//!
//! * LP: the quadratic coefficient `a = (1/2) p^T B(q, q)` of `w' = beta + a w^2`.
//! * H: the first Lyapunov coefficient `l1`.
//!
//! `B` and `C` are the second and third derivatives of `f` in `u`, taken from the `Dir`
//! entries of the model, so there are no finite differences. The coefficients depend on
//! the scaling of `q`, so the normalization is stored with them (see [`LP_NORMALIZATION`]
//! and [`H_NORMALIZATION`]).
//!
//! Sources: Yu. A. Kuznetsov, "Explicit normal form coefficients for all codim 2
//! singularities of equilibria", lecture notes (1998), slides 3, 4 and 15; Kuznetsov,
//! Elements of Applied Bifurcation Theory (2004), p. 180, through Kuehn, arXiv:1201.6595,
//! eq. (9). See `.superpowers/sdd/phase3-pc-engine/sources.md`, items 1 and 2.

use crate::branch::SpecialPoint;
use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::complex::dir_complex;
use pc_ir::{EntryKind, Slot};
use pc_linalg::{Lu, eigvec, null_left, null_right, solve_shifted};

/// Normalization string stored with an LP coefficient.
pub const LP_NORMALIZATION: &str = "|q|=1, p'q=1";
/// Normalization string stored with an H coefficient.
pub const H_NORMALIZATION: &str = "<q,q>=1, <p,q>=1";

/// A complex vector as (real part, imaginary part).
pub type Cv = (Vec<f64>, Vec<f64>);

/// Model access at one point: parameters with t appended and alpha set.
struct At<'a> {
    model: &'a mut dyn ModelEval,
    n: usize,
    u: &'a [f64],
    p: Vec<f64>,
}

impl At<'_> {
    fn jac_u(&mut self) -> Result<Vec<f64>, EngineError> {
        let mut a = vec![0.0; self.n * self.n];
        self.model
            .eval(&EntryKind::JacU, self.u, &self.p, &[], &mut a)?;
        Ok(a)
    }

    /// Column of `JacP` for parameter `free`.
    fn f_alpha(&mut self, free: usize) -> Result<Vec<f64>, EngineError> {
        let mut all = vec![0.0; self.n * (self.p.len() - 1)];
        self.model
            .eval(&EntryKind::JacP, self.u, &self.p, &[], &mut all)?;
        Ok(all[free * self.n..(free + 1) * self.n].to_vec())
    }

    /// The complex multilinear form with all slots `U`: `B(a, b)` for two directions,
    /// `C(a, b, c)` for three.
    fn form(&mut self, dirs: &[&Cv]) -> Result<Cv, EngineError> {
        let n = self.n;
        let kind = EntryKind::Dir(vec![Slot::U; dirs.len()]);
        let re: Vec<&[f64]> = dirs.iter().map(|d| d.0.as_slice()).collect();
        let im: Vec<&[f64]> = dirs.iter().map(|d| d.1.as_slice()).collect();
        let (mut out_re, mut out_im) = (vec![0.0; n], vec![0.0; n]);
        let mut err: Option<EngineError> = None;
        let (model, u, p) = (&mut *self.model, self.u, &self.p);
        dir_complex(
            |v, out| {
                if err.is_none()
                    && let Err(e) = model.eval(&kind, u, p, v, out)
                {
                    err = Some(e);
                }
            },
            &re,
            &im,
            &mut out_re,
            &mut out_im,
        );
        if let Some(e) = err {
            return Err(e);
        }
        if out_re.iter().chain(&out_im).any(|v| !v.is_finite()) {
            return Err(EngineError::NotFinite("second or third derivative".into()));
        }
        Ok((out_re, out_im))
    }
}

/// `<a, b> = conj(a)^T b`.
fn herm(a: &Cv, b: &Cv) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..a.0.len() {
        re += a.0[i] * b.0[i] + a.1[i] * b.1[i];
        im += a.0[i] * b.1[i] - a.1[i] * b.0[i];
    }
    (re, im)
}

fn conj(a: &Cv) -> Cv {
    (a.0.clone(), a.1.iter().map(|x| -x).collect())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Quadratic coefficient `a = (1/2) p^T B(q, q)` of the fold at `(u, alpha)`.
///
/// `q` is the unit right null vector of A and `p` the left null vector with `p^T q = 1`.
/// `a` changes sign with `q`, so the sign is fixed by `p^T f_alpha > 0` (the unfolding
/// parameter then enters `w'` with a plus sign and the sign of `a` gives the side of the
/// fold). When `p^T f_alpha` is zero to rounding, `q` keeps the sign it came with.
pub fn fold_coefficient(
    model: &mut dyn ModelEval,
    params: &[f64],
    free: usize,
    u: &[f64],
    alpha: f64,
) -> Result<f64, EngineError> {
    let (n, mut at) = at_point(model, params, free, u, alpha)?;
    let a = at.jac_u()?;
    let mut q = first(null_right(&a, n, n, 1)?)?;
    let mut p = first(null_left(&a, n, n, 1)?)?;
    let pq = dot(&p, &q);
    if pq == 0.0 || !pq.is_finite() {
        return Err(EngineError::Singular(
            "left and right null vectors are orthogonal".into(),
        ));
    }
    p.iter_mut().for_each(|x| *x /= pq);
    let pf = dot(&p, &at.f_alpha(free)?);
    if pf < 0.0 {
        q.iter_mut().for_each(|x| *x = -*x);
        p.iter_mut().for_each(|x| *x = -*x);
    }
    let cq: Cv = (q, Vec::new());
    let b = at.form(&[&cq, &cq])?;
    let val = 0.5 * dot(&p, &b.0);
    if val.is_finite() {
        Ok(val)
    } else {
        Err(EngineError::NotFinite("fold coefficient".into()))
    }
}

/// Right and left eigenvectors of `a` (n by n, column-major) for the Hopf pair
/// `+-i omega`: `A q = i omega q`, `A^T p = -i omega p`, `<q,q> = 1`, `<p,q> = 1` with
/// `<x,y> = conj(x)^T y`. Returned as `(q, p)`, each as (real part, imaginary part).
pub fn hopf_eigenvectors(a: &[f64], n: usize, omega: f64) -> Result<(Cv, Cv), EngineError> {
    let q = eigvec(a, n, (0.0, omega))?;
    let mut at_mat = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            at_mat[i + n * j] = a[j + n * i];
        }
    }
    let p0 = eigvec(&at_mat, n, (0.0, -omega))?;
    let (dr, di) = herm(&p0, &q);
    let d2 = dr * dr + di * di;
    if !(d2.is_finite() && d2 > 0.0) {
        return Err(EngineError::Singular("<p,q> is zero".into()));
    }
    // p = s p0 gives <p,q> = conj(s) d, so conj(s) = 1/d and s = d / |d|^2.
    let (sr, si) = (dr / d2, di / d2);
    let p: Cv = (
        (0..n).map(|i| p0.0[i] * sr - p0.1[i] * si).collect(),
        (0..n).map(|i| p0.0[i] * si + p0.1[i] * sr).collect(),
    );
    Ok((q, p))
}

/// First Lyapunov coefficient at the Hopf point `(u, alpha)` with frequency `omega`.
///
/// ```text
/// G21 = <p, C(q,q,conj q)> - 2 <p, B(q, A^-1 B(q, conj q))>
///       + <p, B(conj q, (2 i omega I - A)^-1 B(q, q))>,
/// l1  = Re G21 / (2 omega)
/// ```
///
/// with `A q = i omega q`, `A^T p = -i omega p`, `<q,q> = 1`, `<p,q> = 1`. This is
/// Kuznetsov's convention (1998 notes, slide 3; 2004 book p. 180). The 1999 SIAM paper and
/// MatCont leave out the factor `1/omega`, so their value is `omega` times ours.
pub fn first_lyapunov(
    model: &mut dyn ModelEval,
    params: &[f64],
    free: usize,
    u: &[f64],
    alpha: f64,
    omega: f64,
) -> Result<f64, EngineError> {
    if !(omega.is_finite() && omega > 0.0) {
        return Err(EngineError::Singular(
            "Hopf frequency is not positive".into(),
        ));
    }
    let (n, mut at) = at_point(model, params, free, u, alpha)?;
    let a = at.jac_u()?;
    let (q, p) = hopf_eigenvectors(&a, n, omega)?;
    let qc = conj(&q);

    let cqqq = at.form(&[&q, &q, &qc])?;
    let bqq = at.form(&[&q, &q])?;
    let bqqc = at.form(&[&q, &qc])?;
    // A^-1 B(q, conj q); B(q, conj q) is real.
    let lu = Lu::new(&a, n)?;
    let mut w1 = bqqc.0.clone();
    lu.solve(&mut w1)?;
    let w1: Cv = (w1, vec![0.0; n]);
    // (2 i omega I - A)^-1 = -(A - 2 i omega I)^-1.
    let (mut w2r, mut w2i) = solve_shifted(&a, n, (0.0, 2.0 * omega), &bqq.0, &bqq.1)?;
    w2r.iter_mut().for_each(|x| *x = -*x);
    w2i.iter_mut().for_each(|x| *x = -*x);
    let w2: Cv = (w2r, w2i);

    let t1 = herm(&p, &cqqq);
    let t2 = herm(&p, &at.form(&[&q, &w1])?);
    let t3 = herm(&p, &at.form(&[&qc, &w2])?);
    let l1 = (t1.0 - 2.0 * t2.0 + t3.0) / (2.0 * omega);
    if l1.is_finite() {
        Ok(l1)
    } else {
        Err(EngineError::NotFinite("first Lyapunov coefficient".into()))
    }
}

/// The coefficient and normalization string for a refined LP or H point; `Ok(None)` for
/// any other label or a point that is not refined. H takes `omega` from `extras.omega`.
pub fn special_coefficient(
    model: &mut dyn ModelEval,
    params: &[f64],
    free: usize,
    sp: &SpecialPoint,
) -> Result<Option<(f64, &'static str)>, EngineError> {
    if !sp.extras.refined {
        return Ok(None);
    }
    let (u, alpha) = (&sp.point.u, sp.point.alpha);
    match sp.label.as_str() {
        "LP" => {
            let a = fold_coefficient(model, params, free, u, alpha)?;
            Ok(Some((a, LP_NORMALIZATION)))
        }
        "H" => {
            let omega = sp
                .extras
                .omega
                .ok_or_else(|| EngineError::Singular("Hopf point has no frequency".into()))?;
            let l1 = first_lyapunov(model, params, free, u, alpha, omega)?;
            Ok(Some((l1, H_NORMALIZATION)))
        }
        _ => Ok(None),
    }
}

fn at_point<'a>(
    model: &'a mut dyn ModelEval,
    params: &[f64],
    free: usize,
    u: &'a [f64],
    alpha: f64,
) -> Result<(usize, At<'a>), EngineError> {
    let (n, n_p) = (model.n(), model.n_p());
    if n == 0 || params.len() != n_p || free >= n_p || u.len() != n {
        return Err(EngineError::BadProblem(
            "normal form: parameters, free index or state do not fit the model".into(),
        ));
    }
    let mut p = params.to_vec();
    p[free] = alpha;
    p.push(0.0);
    Ok((n, At { model, n, u, p }))
}

fn first(mut v: Vec<Vec<f64>>) -> Result<Vec<f64>, EngineError> {
    if v.is_empty() {
        Err(EngineError::Singular("no null vector".into()))
    } else {
        Ok(v.swap_remove(0))
    }
}
