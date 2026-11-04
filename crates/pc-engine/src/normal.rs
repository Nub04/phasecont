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
//! eq. (9).

use crate::branch::{Extras, SpecialPoint};
use crate::error::EngineError;
use crate::model::ModelEval;
use pc_ir::complex::dir_complex;
use pc_ir::{EntryKind, Slot};
use pc_linalg::{Lu, eigvec, null_left, null_right, solve_shifted};

/// Normalization string stored with an LP coefficient.
pub const LP_NORMALIZATION: &str = "|q|=1, p'q=1, p'F_alpha>0";
/// Normalization string stored with an H coefficient.
pub const H_NORMALIZATION: &str = "<q,q>=1, <p,q>=1";

/// A complex vector as (real part, imaginary part).
pub type Cv = (Vec<f64>, Vec<f64>);

/// Model access at one point: parameters with t appended and alpha set.
pub(crate) struct At<'a> {
    model: &'a mut dyn ModelEval,
    n: usize,
    u: &'a [f64],
    p: Vec<f64>,
}

impl<'a> At<'a> {
    /// `p` holds all parameters with t appended.
    pub(crate) fn new(model: &'a mut dyn ModelEval, n: usize, u: &'a [f64], p: Vec<f64>) -> Self {
        At { model, n, u, p }
    }

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
    /// `C(a, b, c)` for three, and so on up to five.
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
            return Err(EngineError::NotFinite(
                "derivative of F in the state".into(),
            ));
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
/// `a` changes sign with `q`, so the sign is fixed by `p^T F_alpha > 0`: the fold then has
/// the normal form `w' = beta + a w^2` with `beta` increasing in alpha (Kuznetsov, Elements
/// of Applied Bifurcation Theory, Thm 3.1), and the sign of `a` tells on which side of the
/// fold the equilibria exist. When `p^T F_alpha` is exactly 0, `q` keeps the sign that
/// `null_right` returned.
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
    // Normalize explicitly to <q,q> = 1 instead of relying on eigvec.
    let nq = q.0.iter().chain(&q.1).map(|v| v * v).sum::<f64>().sqrt();
    if !(nq.is_finite() && nq > 0.0) {
        return Err(EngineError::Singular("eigenvector has zero norm".into()));
    }
    let q: Cv = (
        q.0.iter().map(|v| v / nq).collect(),
        q.1.iter().map(|v| v / nq).collect(),
    );
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
    let (_, mut at) = at_point(model, params, free, u, alpha)?;
    first_lyapunov_at(&mut at, omega)
}

/// `first_lyapunov` at a prepared point.
pub(crate) fn first_lyapunov_at(at: &mut At, omega: f64) -> Result<f64, EngineError> {
    let n = at.n;
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
    let w2 = inv_shift(&a, n, 2.0 * omega, &bqq)?;

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

/// Normalization string stored with a second Lyapunov coefficient.
pub const GH_NORMALIZATION: &str = "<q,q>=1, <p,q>=1, <p,h21>=0";

/// `sum c_k v_k` with complex coefficients `c_k` as (re, im).
fn lin(terms: &[((f64, f64), &Cv)]) -> Cv {
    let n = terms.first().map_or(0, |t| t.1.0.len());
    let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
    for &((cr, ci), v) in terms {
        for i in 0..n {
            re[i] += cr * v.0[i] - ci * v.1[i];
            im[i] += cr * v.1[i] + ci * v.0[i];
        }
    }
    (re, im)
}

/// `(i w I - A)^-1 rhs` for the n by n column-major `a`.
fn inv_shift(a: &[f64], n: usize, w: f64, rhs: &Cv) -> Result<Cv, EngineError> {
    let (mut re, mut im) = solve_shifted(a, n, (0.0, w), &rhs.0, &rhs.1)?;
    re.iter_mut().for_each(|x| *x = -*x);
    im.iter_mut().for_each(|x| *x = -*x);
    Ok((re, im))
}

/// `A^-1 rhs` for a complex `rhs`, from the LU factors of the real matrix `A`.
fn inv_real(lu: &Lu, rhs: &Cv) -> Result<Cv, EngineError> {
    let (mut re, mut im) = (rhs.0.clone(), rhs.1.clone());
    lu.solve(&mut re)?;
    lu.solve(&mut im)?;
    Ok((re, im))
}

/// Solves `(i w I - A) h = r` with `<p, h> = 0` for the singular matrix at `w = omega`, where
/// `r` is orthogonal to `p`. The complex bordered system `[[i w I - A, q], [p^*, 0]]` is
/// solved in its real embedding.
fn solve_bordered(a: &[f64], n: usize, w: f64, q: &Cv, p: &Cv, r: &Cv) -> Result<Cv, EngineError> {
    let m = n + 1;
    let s = 2 * m;
    let mut e = vec![0.0; s * s];
    // Entry (i, j) of Z = Zr + i Zi goes to the four blocks of [[Zr, -Zi], [Zi, Zr]].
    let mut put = |i: usize, j: usize, zr: f64, zi: f64| {
        e[i + s * j] = zr;
        e[i + s * (m + j)] = -zi;
        e[(m + i) + s * j] = zi;
        e[(m + i) + s * (m + j)] = zr;
    };
    for j in 0..n {
        for i in 0..n {
            put(i, j, -a[i + n * j], if i == j { w } else { 0.0 });
        }
        put(n, j, p.0[j], -p.1[j]);
    }
    for i in 0..n {
        put(i, n, q.0[i], q.1[i]);
    }
    let mut rhs = vec![0.0; s];
    rhs[..n].copy_from_slice(&r.0);
    rhs[m..m + n].copy_from_slice(&r.1);
    Lu::new(&e, s)?.solve(&mut rhs)?;
    Ok((rhs[..n].to_vec(), rhs[m..m + n].to_vec()))
}

/// Second Lyapunov coefficient `l2 = Re G32 / (12 omega)` at a Hopf point with `l1 = 0`.
///
/// The recursion is Kuznetsov's (1998 lecture notes on codim 2 bifurcations, Bautin section;
/// Elements of Applied Bifurcation Theory, Sec. 8.3): with `A q = i omega q`,
/// `A^T p = -i omega p`, `<q,q> = 1`, `<p,q> = 1` and `B..E` the second to fifth derivatives
/// of `F` in the state,
///
/// ```text
/// h20 = (2 i w - A)^-1 B(q,q)                h11 = -A^-1 B(q,cq)
/// h30 = (3 i w - A)^-1 [C(q,q,q) + 3 B(q,h20)]
/// G21 = <p, C(q,q,cq) + B(cq,h20) + 2 B(q,h11)>
/// (i w - A) h21 = C(q,q,cq) + B(cq,h20) + 2 B(q,h11) - G21 q,   <p,h21> = 0
/// ```
///
/// and `h31`, `h22`, `G32` as in the code below (`cq` is the conjugate of `q`). Unlike the
/// Hopf point test, `l1 = Re G21 / (2 w)` is not assumed to vanish here.
pub(crate) fn second_lyapunov_at(at: &mut At, omega: f64) -> Result<f64, EngineError> {
    let n = at.n;
    let a = at.jac_u()?;
    let (q, p) = hopf_eigenvectors(&a, n, omega)?;
    let qc = conj(&q);
    let lu = Lu::new(&a, n)?;
    let one = (1.0, 0.0);

    let bqq = at.form(&[&q, &q])?;
    let h20 = inv_shift(&a, n, 2.0 * omega, &bqq)?;
    let h20c = conj(&h20);
    let bqqc = at.form(&[&q, &qc])?;
    let mut h11 = inv_real(&lu, &bqqc)?;
    h11.0.iter_mut().for_each(|x| *x = -*x);
    h11.1.iter_mut().for_each(|x| *x = -*x);

    let cqqq = at.form(&[&q, &q, &q])?;
    let bqh20 = at.form(&[&q, &h20])?;
    let h30 = inv_shift(
        &a,
        n,
        3.0 * omega,
        &lin(&[(one, &cqqq), ((3.0, 0.0), &bqh20)]),
    )?;

    let cqqqc = at.form(&[&q, &q, &qc])?;
    let bqch20 = at.form(&[&qc, &h20])?;
    let bqh11 = at.form(&[&q, &h11])?;
    let v21 = lin(&[(one, &cqqqc), (one, &bqch20), ((2.0, 0.0), &bqh11)]);
    let g21 = herm(&p, &v21);
    let r21 = lin(&[(one, &v21), ((-g21.0, -g21.1), &q)]);
    let h21 = solve_bordered(&a, n, omega, &q, &p, &r21)?;
    let h21c = conj(&h21);

    let dqqqqc = at.form(&[&q, &q, &q, &qc])?;
    let cqqh11 = at.form(&[&q, &q, &h11])?;
    let cqqch20 = at.form(&[&q, &qc, &h20])?;
    let bh20h11 = at.form(&[&h20, &h11])?;
    let bqch30 = at.form(&[&qc, &h30])?;
    let bqh21 = at.form(&[&q, &h21])?;
    let rhs31 = lin(&[
        (one, &dqqqqc),
        ((3.0, 0.0), &cqqh11),
        ((3.0, 0.0), &cqqch20),
        ((3.0, 0.0), &bh20h11),
        (one, &bqch30),
        ((3.0, 0.0), &bqh21),
        ((-3.0 * g21.0, -3.0 * g21.1), &h20),
    ]);
    let h31 = inv_shift(&a, n, 2.0 * omega, &rhs31)?;

    let dqqqcqc = at.form(&[&q, &q, &qc, &qc])?;
    let cqcqh11 = at.form(&[&q, &qc, &h11])?;
    let cqcqch20 = at.form(&[&qc, &qc, &h20])?;
    let cqqh20c = at.form(&[&q, &q, &h20c])?;
    let bh11h11 = at.form(&[&h11, &h11])?;
    let bqh21c = at.form(&[&q, &h21c])?;
    let bqch21 = at.form(&[&qc, &h21])?;
    let bh20ch20 = at.form(&[&h20c, &h20])?;
    let rhs22 = lin(&[
        (one, &dqqqcqc),
        ((4.0, 0.0), &cqcqh11),
        (one, &cqcqch20),
        (one, &cqqh20c),
        ((2.0, 0.0), &bh11h11),
        ((2.0, 0.0), &bqh21c),
        ((2.0, 0.0), &bqch21),
        (one, &bh20ch20),
        ((-4.0 * g21.0, 0.0), &h11),
    ]);
    let mut h22 = inv_real(&lu, &rhs22)?;
    h22.0.iter_mut().for_each(|x| *x = -*x);
    h22.1.iter_mut().for_each(|x| *x = -*x);

    let e5 = at.form(&[&q, &q, &q, &qc, &qc])?;
    let d1 = at.form(&[&q, &q, &q, &h20c])?;
    let d2 = at.form(&[&q, &qc, &qc, &h20])?;
    let d3 = at.form(&[&q, &q, &qc, &h11])?;
    let c1 = at.form(&[&qc, &qc, &h30])?;
    let c2 = at.form(&[&q, &q, &h21c])?;
    let c3 = at.form(&[&q, &qc, &h21])?;
    let c4 = at.form(&[&q, &h20c, &h20])?;
    let c5 = at.form(&[&q, &h11, &h11])?;
    let c6 = at.form(&[&qc, &h20, &h11])?;
    let b1 = at.form(&[&qc, &h31])?;
    let b2 = at.form(&[&q, &h22])?;
    let b3 = at.form(&[&h20c, &h30])?;
    let b4 = at.form(&[&h21c, &h20])?;
    let b5 = at.form(&[&h11, &h21])?;
    let v32 = lin(&[
        (one, &e5),
        (one, &d1),
        ((3.0, 0.0), &d2),
        ((6.0, 0.0), &d3),
        (one, &c1),
        ((3.0, 0.0), &c2),
        ((6.0, 0.0), &c3),
        ((3.0, 0.0), &c4),
        ((6.0, 0.0), &c5),
        ((6.0, 0.0), &c6),
        ((2.0, 0.0), &b1),
        ((3.0, 0.0), &b2),
        (one, &b3),
        ((3.0, 0.0), &b4),
        ((6.0, 0.0), &b5),
    ]);
    let g32 = herm(&p, &v32);
    let l2 = g32.0 / (12.0 * omega);
    if l2.is_finite() {
        Ok(l2)
    } else {
        Err(EngineError::NotFinite("second Lyapunov coefficient".into()))
    }
}

/// Normalization string stored with the coefficients of a zero-Hopf point. The sign of `q0`
/// is fixed by `b > 0`, like `a > 0` at a Bogdanov-Takens point.
pub const ZH_NORMALIZATION: &str = "b>0, |q0|=1, <p0,q0>=1, <q1,q1>=1, <p1,q1>=1";

/// The coefficients of the zero-Hopf normal form
/// `u' = beta1 + b u^2 + c |z|^2`, `z' = (beta2 + i omega + d u + e u^2) z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ZeroHopf {
    pub b: f64,
    pub c: f64,
    /// Real and imaginary part of `d`.
    pub d: (f64, f64),
    pub e: f64,
}

/// Coefficients `b, c, d, e` at a zero-Hopf point with `A q0 = 0`, `A q1 = i omega q1`.
///
/// Kuznetsov, "Numerical normalization techniques for all codim 2 bifurcations of
/// equilibria in ODE's", SIAM J. Numer. Anal. 36 (1999), section 8, with
/// `<p0,q0> = <p1,q1> = <q1,q1> = 1` and `|q0| = 1`:
///
/// ```text
/// G200 = <p0, B(q0,q0)>   G110 = <p1, B(q0,q1)>   G011 = <p0, B(q1,cq1)>
/// -A h200 = B(q0,q0) - G200 q0,   -A h011 = B(q1,cq1) - G011 q0,   <p0,h> = 0
/// (i w - A) h110 = B(q0,q1) - G110 q1,   <p1,h110> = 0,   h020 = (2 i w - A)^-1 B(q1,q1)
/// G300 = <p0, C(q0,q0,q0) + 3 B(q0,h200)>
/// G111 = <p0, C(q0,q1,cq1) + B(q0,h011) + B(q1,ch110) + B(cq1,h110)>
/// G210 = <p1, C(q0,q0,q1) + 2 B(q0,h110) + B(q1,h200)>
/// G021 = <p1, C(q1,q1,cq1) + 2 B(q1,h011) + B(cq1,h020)>
/// b = G200 / 2,   c = G011,   d = G110 - i w G300 / (3 G200)
/// e = Re[G210 + G110 (Re G021 / G011 - G300 / G200 + G111 / G011)
///        - G021 G200 / (2 G011)] / 2
/// ```
///
/// The coefficients depend on the scale of `q0`, so `|q0| = 1` is part of the normalization.
/// Flipping `q0` and `p0` flips `b`, `c` and `d` and leaves `e` alone, so the sign is fixed
/// by `b > 0`. `G200 = 0` or `G011 = 0` is an error: the normal form is degenerate there.
pub(crate) fn zero_hopf_at(at: &mut At, omega: f64) -> Result<ZeroHopf, EngineError> {
    let n = at.n;
    if !(omega.is_finite() && omega > 0.0) {
        return Err(EngineError::Singular(
            "zero-Hopf frequency is not positive".into(),
        ));
    }
    let a = at.jac_u()?;
    let (q1, p1) = hopf_eigenvectors(&a, n, omega)?;
    let q1c = conj(&q1);
    let mut q0 = first(null_right(&a, n, n, 1)?)?;
    let mut p0 = first(null_left(&a, n, n, 1)?)?;
    let nq = dot(&q0, &q0).sqrt();
    q0.iter_mut().for_each(|x| *x /= nq);
    let pq = dot(&p0, &q0);
    if !(pq.is_finite() && pq != 0.0 && nq.is_finite() && nq > 0.0) {
        return Err(EngineError::Singular(
            "left and right null vectors are orthogonal".into(),
        ));
    }
    p0.iter_mut().for_each(|x| *x /= pq);
    let real = |v: &[f64]| -> Cv { (v.to_vec(), vec![0.0; n]) };
    let one = (1.0, 0.0);

    let mut q0c = real(&q0);
    let b00 = at.form(&[&q0c, &q0c])?;
    let mut g200 = dot(&p0, &b00.0);
    if g200 < 0.0 {
        q0.iter_mut().for_each(|x| *x = -*x);
        p0.iter_mut().for_each(|x| *x = -*x);
        q0c = real(&q0);
        g200 = -g200;
    }
    let p0c = real(&p0);
    let b00 = real(&b00.0);
    let b11 = at.form(&[&q1, &q1c])?;
    let g011 = dot(&p0, &b11.0);
    if !(g200.is_finite() && g200 != 0.0 && g011.is_finite() && g011 != 0.0) {
        return Err(EngineError::Singular(
            "G200 or G011 is zero at the zero-Hopf point".into(),
        ));
    }
    let b11 = real(&b11.0);
    let b01 = at.form(&[&q0c, &q1])?;
    let g110 = herm(&p1, &b01);

    let h200 = solve_bordered(
        &a,
        n,
        0.0,
        &q0c,
        &p0c,
        &lin(&[(one, &b00), ((-g200, 0.0), &q0c)]),
    )?;
    let h011 = solve_bordered(
        &a,
        n,
        0.0,
        &q0c,
        &p0c,
        &lin(&[(one, &b11), ((-g011, 0.0), &q0c)]),
    )?;
    let h110 = solve_bordered(
        &a,
        n,
        omega,
        &q1,
        &p1,
        &lin(&[(one, &b01), ((-g110.0, -g110.1), &q1)]),
    )?;
    let h110c = conj(&h110);
    let h020 = inv_shift(&a, n, 2.0 * omega, &at.form(&[&q1, &q1])?)?;

    let two = (2.0, 0.0);
    let c000 = at.form(&[&q0c, &q0c, &q0c])?;
    let b0h200 = at.form(&[&q0c, &h200])?;
    let g300 = dot(&p0, &lin(&[(one, &c000), ((3.0, 0.0), &b0h200)]).0);

    let c011 = at.form(&[&q0c, &q1, &q1c])?;
    let b0h011 = at.form(&[&q0c, &h011])?;
    let b1h110c = at.form(&[&q1, &h110c])?;
    let b1ch110 = at.form(&[&q1c, &h110])?;
    let g111 = dot(
        &p0,
        &lin(&[
            (one, &c011),
            (one, &b0h011),
            (one, &b1h110c),
            (one, &b1ch110),
        ])
        .0,
    );

    let c001 = at.form(&[&q0c, &q0c, &q1])?;
    let b0h110 = at.form(&[&q0c, &h110])?;
    let b1h200 = at.form(&[&q1, &h200])?;
    let g210 = herm(&p1, &lin(&[(one, &c001), (two, &b0h110), (one, &b1h200)]));

    let c111 = at.form(&[&q1, &q1, &q1c])?;
    let b1h011 = at.form(&[&q1, &h011])?;
    let b1ch020 = at.form(&[&q1c, &h020])?;
    let g021 = herm(&p1, &lin(&[(one, &c111), (two, &b1h011), (one, &b1ch020)]));

    let s = g021.0 / g011 - g300 / g200 + g111 / g011;
    let out = ZeroHopf {
        b: 0.5 * g200,
        c: g011,
        d: (g110.0, g110.1 - omega * g300 / (3.0 * g200)),
        e: 0.5 * (g210.0 + g110.0 * s - g021.0 * g200 / (2.0 * g011)),
    };
    if [out.b, out.c, out.d.0, out.d.1, out.e]
        .iter()
        .all(|v| v.is_finite())
    {
        Ok(out)
    } else {
        Err(EngineError::NotFinite("zero-Hopf coefficients".into()))
    }
}

/// Stores the zero-Hopf coefficients and the frequency with a special point.
pub(crate) fn store_zero_hopf(extras: &mut Extras, z: &ZeroHopf, omega: f64) {
    extras.coefficient = Some(z.b);
    extras.coefficient2 = Some(z.c);
    extras.omega = Some(omega);
    extras.normalization = Some(ZH_NORMALIZATION.to_string());
    extras.values = vec![
        ("re_d".to_string(), z.d.0),
        ("im_d".to_string(), z.d.1),
        ("e".to_string(), z.e),
    ];
}

/// Normalization string stored with the coefficients of a double-Hopf point. The pairs are
/// ordered by frequency, so pair 1 is the faster one.
pub const HH_NORMALIZATION: &str = "<q1,q1>=1, <p1,q1>=1, <q2,q2>=1, <p2,q2>=1, w1>w2";

/// The cubic coefficients of a double-Hopf point, each as (re, im). Only the real parts
/// `P11 = Re G2100`, `P12 = Re G1011`, `P21 = Re G1110` and `P22 = Re G0021` enter the
/// normal form amplitude equations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DoubleHopf {
    pub g2100: (f64, f64),
    pub g1011: (f64, f64),
    pub g1110: (f64, f64),
    pub g0021: (f64, f64),
}

/// Cubic coefficients at a double-Hopf point with `A q1 = i w1 q1`, `A q2 = i w2 q2` and
/// `w1 > w2`.
///
/// Kuznetsov, "Numerical normalization techniques for all codim 2 bifurcations of equilibria
/// in ODE's", SIAM J. Numer. Anal. 36 (1999), section 9, with `<q_k,q_k> = <p_k,q_k> = 1`:
///
/// ```text
/// h1100 = -A^-1 B(q1,cq1)               h2000 = (2 i w1 - A)^-1 B(q1,q1)
/// h0011 = -A^-1 B(q2,cq2)               h0020 = (2 i w2 - A)^-1 B(q2,q2)
/// h1010 = (i (w1 + w2) - A)^-1 B(q1,q2) h1001 = (i (w1 - w2) - A)^-1 B(q1,cq2)
/// G2100 = <p1, C(q1,q1,cq1) + B(h2000,cq1) + 2 B(h1100,q1)>
/// G1011 = <p1, C(q1,q2,cq2) + B(h1010,cq2) + B(h1001,q2) + B(h0011,q1)>
/// G1110 = <p2, C(q1,cq1,q2) + B(h1100,q2) + B(h1010,cq1) + B(ch1001,q1)>
/// G0021 = <p2, C(q2,q2,cq2) + B(h0020,cq2) + 2 B(h0011,q2)>
/// ```
///
/// The fifth-order coefficients of the paper are not computed. `w1 = w2` and `w1 = 2 w2` are
/// errors, since a shifted matrix is singular there.
pub(crate) fn double_hopf_at(at: &mut At, w1: f64, w2: f64) -> Result<DoubleHopf, EngineError> {
    let n = at.n;
    if !(w1.is_finite() && w2.is_finite() && w2 > 0.0 && w1 > w2) {
        return Err(EngineError::Singular(
            "double-Hopf frequencies are not ordered w1 > w2 > 0".into(),
        ));
    }
    if (w1 - w2) < 1e-6 * w1 || (w1 - 2.0 * w2).abs() < 1e-6 * w1 {
        return Err(EngineError::Singular(
            "double-Hopf frequencies are in 1:1 or 1:2 resonance".into(),
        ));
    }
    let a = at.jac_u()?;
    let (q1, p1) = hopf_eigenvectors(&a, n, w1)?;
    let (q2, p2) = hopf_eigenvectors(&a, n, w2)?;
    let (q1c, q2c) = (conj(&q1), conj(&q2));
    let lu = Lu::new(&a, n)?;
    let neg = |v: Cv| -> Cv { lin(&[((-1.0, 0.0), &v)]) };
    let one = (1.0, 0.0);
    let two = (2.0, 0.0);

    let h1100 = neg(inv_real(&lu, &at.form(&[&q1, &q1c])?)?);
    let h0011 = neg(inv_real(&lu, &at.form(&[&q2, &q2c])?)?);
    let h2000 = inv_shift(&a, n, 2.0 * w1, &at.form(&[&q1, &q1])?)?;
    let h0020 = inv_shift(&a, n, 2.0 * w2, &at.form(&[&q2, &q2])?)?;
    let h1010 = inv_shift(&a, n, w1 + w2, &at.form(&[&q1, &q2])?)?;
    let h1001 = inv_shift(&a, n, w1 - w2, &at.form(&[&q1, &q2c])?)?;
    let h1001c = conj(&h1001);

    let c = at.form(&[&q1, &q1, &q1c])?;
    let t1 = at.form(&[&h2000, &q1c])?;
    let t2 = at.form(&[&h1100, &q1])?;
    let g2100 = herm(&p1, &lin(&[(one, &c), (one, &t1), (two, &t2)]));

    let c = at.form(&[&q1, &q2, &q2c])?;
    let t1 = at.form(&[&h1010, &q2c])?;
    let t2 = at.form(&[&h1001, &q2])?;
    let t3 = at.form(&[&h0011, &q1])?;
    let g1011 = herm(&p1, &lin(&[(one, &c), (one, &t1), (one, &t2), (one, &t3)]));

    let c = at.form(&[&q1, &q1c, &q2])?;
    let t1 = at.form(&[&h1100, &q2])?;
    let t2 = at.form(&[&h1010, &q1c])?;
    let t3 = at.form(&[&h1001c, &q1])?;
    let g1110 = herm(&p2, &lin(&[(one, &c), (one, &t1), (one, &t2), (one, &t3)]));

    let c = at.form(&[&q2, &q2, &q2c])?;
    let t1 = at.form(&[&h0020, &q2c])?;
    let t2 = at.form(&[&h0011, &q2])?;
    let g0021 = herm(&p2, &lin(&[(one, &c), (one, &t1), (two, &t2)]));

    let out = DoubleHopf {
        g2100,
        g1011,
        g1110,
        g0021,
    };
    if [g2100, g1011, g1110, g0021]
        .iter()
        .all(|g| g.0.is_finite() && g.1.is_finite())
    {
        Ok(out)
    } else {
        Err(EngineError::NotFinite("double-Hopf coefficients".into()))
    }
}

/// Stores the real parts of the cubic double-Hopf coefficients and both frequencies with a
/// special point: `coefficient` is `Re P11`, `omega` is `w1`.
pub(crate) fn store_double_hopf(extras: &mut Extras, d: &DoubleHopf, w1: f64, w2: f64) {
    extras.coefficient = Some(d.g2100.0);
    extras.omega = Some(w1);
    extras.normalization = Some(HH_NORMALIZATION.to_string());
    extras.values = vec![
        ("re_p12".to_string(), d.g1011.0),
        ("re_p21".to_string(), d.g1110.0),
        ("re_p22".to_string(), d.g0021.0),
        ("omega2".to_string(), w2),
    ];
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
