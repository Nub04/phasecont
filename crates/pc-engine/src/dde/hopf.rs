//! The Hopf curve of delay equilibria in two parameters (SPEC section 8f).
//!
//! A Hopf point of a delay equilibrium `u` with `G(u, p) = f(u, u, p) = 0` is a root
//! `lambda = i omega` of `Delta(lambda) = lambda I - A_0 - sum_j A_j exp(-lambda tau_j)`. The
//! system below is the one of the DDE-BIFTOOL manual (Engelborghs, Luzyanina and Samaey;
//! arXiv 1406.7144, BSD-2-Clause, Sec. 10.1, eq. 14) in a rewritten form, which is our own:
//! with `kappa = omega^2` and the eigenvector `v = a + i omega b`, the equation
//! `Delta(i omega) v = 0` splits into two real rows,
//!
//! ```text
//! R(kappa) a - kappa S(kappa) b = 0,     S(kappa) a + R(kappa) b = 0,
//! R = -A_0 - sum_j A_j cosk(tau_j),      S = I + sum_j A_j sink(tau_j),
//! ```
//!
//! where `cosk(tau)` is `cos(sqrt(kappa) tau)` for `kappa > 0`, `cosh(sqrt(-kappa) tau)` for
//! `kappa < 0` and 1 at 0, and `sink(tau)` is `sin(sqrt(kappa) tau) / sqrt(kappa)`, the same
//! with `sinh`, and `tau` at 0. Both are entire functions of `kappa`, so the curve continues
//! through `kappa = 0`, where the Hopf curve meets the neutral saddle curve at a
//! Bogdanov-Takens point. For `kappa < 0` both `+mu` and `-mu` are roots with
//! `kappa = -mu^2`, so that part of the curve is the continuation of the system, not a
//! Hopf bifurcation.
//!
//! Unknowns `x = (u, a, b, a1, kappa, a2)` (`3n + 3` values). The `3n + 2` rows are `G`, the
//! two blocks above and the normalization `c.a = 1`, `c.b = 0` with a fixed vector `c`.
//! Complex multiples of `v` give `(a, b) -> (s a - kappa t b, t a + s b)`, and the
//! normalization removes that freedom. `c` is fixed at the start, so `c.v` can reach zero
//! on a long curve; the run then ends with `StepTooSmall` or `Singular`.
//!
//! The residual is exact. The Jacobian is by central differences in every column, with
//! step `1e-6 (1 + |x_j|)`. `Point.u` of the underlying branch holds `(u, a, b, a1, kappa)`
//! and `Point.alpha` holds `a2`; `Options::bounds` apply to `a2`.
//!
//! Codim-2 tests, in the order of `LABELS`:
//! * `BT`: `kappa`. It changes sign where the Hopf pair meets zero.
//! * `ZH`: `det Delta(0) = det(-(A_0 + sum_j A_j))`, for `kappa > 0` only. A sign change is
//!   a real root through zero while the Hopf pair is at `+-i omega`.
//! * `GH`, only when the run is asked for it (`continue_hopf_with_lyapunov`, SPEC section
//!   8g-2): the first Lyapunov coefficient `l1` of `lyapunov`, for `kappa > 0` only, NaN where
//!   `lyapunov` refuses the point. It has poles at BT, ZH and at a 1:2 resonance (a root at
//!   `2 i omega`), and the locator drops a sign change through a pole (`POLE_TOL`).
//! * `HH`, only when the run is asked for it (`HopfExtras::hh`, SPEC section 8k): the real
//!   part of the largest root with a positive imaginary part that is not the critical root.
//!   It is NaN where the roots are not available and where the critical root is not single.
//!   Below `hh_omega_floor`, near a Bogdanov-Takens point, the pair at `+-i omega` can be
//!   lost to rounding or to the root finder, so there a missing pair is not counted as lost.

use super::eq::{check_params, check_state, field, linearize};
use super::lyap::lyapunov;
use super::lyap2::lyapunov2;
use super::roots::{CharRoot, LinearDde, RootOptions, RootSet, SHIFT, characteristic_roots};
use crate::branch::{Options, Point, StopReason};
use crate::curve2::start_tangent;
use crate::error::EngineError;
use crate::model::DdeModel;
use crate::palc::run_branch;
use crate::problem::Problem;
use pc_ir::DelayTime;
use pc_linalg::{Lu, null_right};
use std::sync::atomic::{AtomicBool, Ordering};

/// Codim-2 labels of the Hopf curve of a delay equilibrium, in the order of the tests.
const LABELS: [&str; 2] = ["BT", "ZH"];

/// `LABELS` and the generalized Hopf test, for a run that computes `l1`.
const LABELS_L1: [&str; 3] = ["BT", "ZH", "GH"];

/// `LABELS` and the Hopf-Hopf test, for a run with the `hh` option (SPEC section 8k).
const LABELS_HH: [&str; 3] = ["BT", "ZH", "HH"];

/// `LABELS_L1` and the Hopf-Hopf test.
const LABELS_L1_HH: [&str; 4] = ["BT", "ZH", "GH", "HH"];

/// A root within `CRIT_TOL (1 + omega)` of `i omega` is the critical root of the curve point.
/// The constant has no source (SPEC section 8k).
const CRIT_TOL: f64 = 1e-6;

/// Smallest `omega` that a start may have. Below it the split into `a` and `b` is badly
/// scaled.
const OMEGA_MIN: f64 = 1e-8;

/// Margin on the sizes below which the pair at `+-i omega` is lost near a Bogdanov-Takens
/// point. The constant has no source (SPEC section 8k).
const COLLAPSE_MARGIN: f64 = 10.0;

/// The `omega` below which a missing pair at `+-i omega` is not counted as lost for the
/// linearization `lin` (SPEC section 8k). The floor never switches the test off: a pair that
/// the roots do show is used at any `omega`. Near a Bogdanov-Takens point the matrices have a nilpotent part of size `s`, the
/// largest entry of `A_0` and of every `A_j` (at least 1). Two things lose the pair there:
///
/// - One rounding of a matrix entry shifts `omega^2` by about `eps s^2`, so a pair with `omega`
///   near `sqrt(eps) s` is read as two real roots.
/// - The root finder starts its Newton iteration from a vector found by inverse iteration at a
///   point `SHIFT` from the root. The matrix there has a condition number near `s / omega`, so
///   the factorization is singular once `SHIFT omega / s^2` is near `eps`, that is for `omega`
///   below about `eps s^2 / SHIFT`, and the pair is reported as unrefined.
///
/// The floor is `COLLAPSE_MARGIN` times the larger of the two sizes.
fn hh_omega_floor(lin: &LinearDde) -> f64 {
    let s = lin
        .a0
        .iter()
        .chain(lin.delays.iter().flat_map(|(_, a)| a.iter()))
        .fold(1.0_f64, |m, v| m.max(v.abs()));
    let rounding = f64::EPSILON.sqrt() * s;
    let inverse_iteration = f64::EPSILON * s * s / SHIFT;
    COLLAPSE_MARGIN * rounding.max(inverse_iteration)
}

/// Below this size of `kappa tau^2` the series of `cosk` and `sink` replace the closed forms.
const SERIES: f64 = 1e-8;

/// A start on the Hopf curve: a Hopf point of a delay equilibrium, approximately.
#[derive(Clone, Debug)]
pub struct HopfSetup {
    /// The two declared parameters of the curve, as indices into the declared parameters.
    pub free: [usize; 2],
    /// The declared parameters; the entries at `free` are the start values.
    pub params: Vec<f64>,
    /// The state at the start, near the equilibrium.
    pub x0: Vec<f64>,
    /// `omega` at the start, near the frequency of the critical pair.
    pub omega: f64,
}

/// One point of the curve.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeHopfPoint {
    /// The values of the two free parameters.
    pub alpha: [f64; 2],
    /// The equilibrium.
    pub x: Vec<f64>,
    /// `kappa = omega^2`; negative on the neutral saddle side of a BT point.
    pub kappa: f64,
    /// `sqrt(kappa)` for `kappa > 0`, else `None`.
    pub omega: Option<f64>,
    /// The first Lyapunov coefficient, `None` unless the run was asked for it
    /// (`continue_hopf_with_lyapunov`) and `lyapunov` accepts the point.
    pub l1: Option<f64>,
    /// The Hopf-Hopf test `h`, the real part of the second pair of roots (SPEC section 8k):
    /// `None` unless the run has the `hh` option, and where `h` is not defined.
    pub hh: Option<f64>,
}

/// A codim-2 point of the curve.
#[derive(Clone, Debug, PartialEq)]
pub struct DdeHopfSpecial {
    /// `BT` or `ZH`, and `GH` in a run that computes `l1`.
    pub label: &'static str,
    pub alpha: [f64; 2],
    pub x: Vec<f64>,
    pub kappa: f64,
    /// Index of the first point after it.
    pub index: usize,
    /// The second Lyapunov coefficient at the point (`lyapunov2`, SPEC section 8h): `Some`
    /// only for `GH`, and `None` there when `lyapunov2` returns an error (a refusal is the usual case).
    pub l2: Option<f64>,
    /// `Im` of the second critical root at an `HH` special (SPEC section 8k), `None` for the
    /// other labels and where the roots of the special have no single critical root.
    pub omega2: Option<f64>,
}

/// What the run computes along the curve besides the curve itself. There are no defaults:
/// every field says what to add.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HopfExtras {
    /// The first Lyapunov coefficient and the `GH` label (SPEC section 8g-2).
    pub lyapunov: bool,
    /// The characteristic roots at every point with `kappa > 0` and the `HH` label (SPEC
    /// section 8k), with the window and the degree of the roots.
    pub hh: Option<RootOptions>,
}

/// The critical root and the second pair among a set of characteristic roots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HopfPairs {
    /// Roots with `Im > 0` within `CRIT_TOL (1 + omega)` of `i omega`.
    pub critical: usize,
    /// The other root with `Im > 0` of the largest real part (the larger `Im` on a tie).
    pub second: Option<CharRoot>,
}

/// Counts the roots at `i omega` and picks the second pair (SPEC section 8k).
pub fn hopf_pairs(set: &RootSet, omega: f64) -> HopfPairs {
    let tol = CRIT_TOL * (1.0 + omega);
    let mut critical = 0;
    let mut second: Option<CharRoot> = None;
    for r in set.roots.iter().filter(|r| r.im > 0.0) {
        if r.re.hypot(r.im - omega) <= tol {
            critical += 1;
            continue;
        }
        let better = match &second {
            None => true,
            Some(s) => r.re > s.re || (r.re == s.re && r.im > s.im),
        };
        if better {
            second = Some(*r);
        }
    }
    HopfPairs { critical, second }
}

/// A Hopf curve of a delay equilibrium.
#[derive(Clone, Debug)]
pub struct DdeHopfBranch {
    pub points: Vec<DdeHopfPoint>,
    pub specials: Vec<DdeHopfSpecial>,
    pub stop: StopReason,
    pub warnings: Vec<String>,
}

/// `cosk(kappa, tau)`, see the module comment.
fn cos_k(kappa: f64, tau: f64) -> f64 {
    let z = kappa * tau * tau;
    if z.abs() < SERIES {
        1.0 - z / 2.0 + z * z / 24.0
    } else if kappa > 0.0 {
        (kappa.sqrt() * tau).cos()
    } else {
        ((-kappa).sqrt() * tau).cosh()
    }
}

/// `sink(kappa, tau)`, see the module comment.
fn sin_k(kappa: f64, tau: f64) -> f64 {
    let z = kappa * tau * tau;
    if z.abs() < SERIES {
        tau * (1.0 - z / 6.0 + z * z / 120.0)
    } else if kappa > 0.0 {
        let w = kappa.sqrt();
        (w * tau).sin() / w
    } else {
        let w = (-kappa).sqrt();
        (w * tau).sinh() / w
    }
}

/// `R(kappa)` and `S(kappa)` of the linear equation, n by n row major.
fn rs_matrices(lin: &LinearDde, kappa: f64) -> (Vec<f64>, Vec<f64>) {
    let n = lin.dim;
    let mut r: Vec<f64> = lin.a0.iter().map(|v| -v).collect();
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        s[i * n + i] = 1.0;
    }
    for (tau, a) in &lin.delays {
        let (c, sn) = (cos_k(kappa, *tau), sin_k(kappa, *tau));
        for k in 0..n * n {
            r[k] -= a[k] * c;
            s[k] += a[k] * sn;
        }
    }
    (r, s)
}

/// `a x` for an n by n row-major matrix.
fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|k| a[i * n + k] * x[k]).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// A delay time that is not positive during the run is a failed step, so the corrector
/// halves the step instead of the run ending with an error.
fn step_failure(e: EngineError) -> EngineError {
    match e {
        EngineError::BadProblem(m) => EngineError::NotFinite(m),
        e => e,
    }
}

struct HopfProblem<'a> {
    m: &'a mut DdeModel,
    n: usize,
    free: [usize; 2],
    params: Vec<f64>,
    /// The fixed vector of the normalization.
    c: Vec<f64>,
    /// Whether the codim-2 tests include `l1`.
    lyapunov: bool,
    /// The roots of the Hopf-Hopf test, `None` without the option.
    hh: Option<RootOptions>,
    /// Evaluations of the test whose root set has candidates that did not refine.
    hh_unrefined: usize,
    /// Evaluations at `kappa > 0` where the roots failed or there was not exactly one
    /// critical root.
    hh_lost: usize,
}

impl HopfProblem<'_> {
    /// The declared parameters with the two free values set.
    fn declared(&self, a1: f64, a2: f64) -> Vec<f64> {
        let mut p = self.params.clone();
        p[self.free[0]] = a1;
        p[self.free[1]] = a2;
        p
    }

    /// The linearization at `x` with the free parameters taken from `x`.
    fn lin_at(&mut self, x: &[f64]) -> Result<(Vec<f64>, LinearDde), EngineError> {
        let n = self.n;
        let p = self.declared(x[3 * n], x[3 * n + 2]);
        check_params(self.m, &p).map_err(step_failure)?;
        let lin = linearize(self.m, &x[..n], &p)?;
        Ok((p, lin))
    }

    /// `l1` at `x`; NaN for `kappa <= 0` and where `lyapunov` refuses the point: `omega`
    /// near 0, a null pair that does not normalise, or a singular `Delta(2 i omega)` or
    /// `Delta(0)`.
    fn l1_at(&mut self, x: &[f64]) -> f64 {
        let n = self.n;
        let kappa = x[3 * n + 1];
        if kappa.is_nan() || kappa <= 0.0 {
            return f64::NAN;
        }
        let p = self.declared(x[3 * n], x[3 * n + 2]);
        match lyapunov(self.m, &x[..n], &p, kappa.sqrt()) {
            Ok(l) => l.l1,
            Err(_) => f64::NAN,
        }
    }

    /// `l2` at a located `GH` point; `None` for `kappa <= 0` and where `lyapunov2` returns an
    /// error (a refusal is the usual case, for example a 1:3 resonance).
    fn l2_at(&mut self, alpha: [f64; 2], x: &[f64], kappa: f64) -> Option<f64> {
        if kappa.is_nan() || kappa <= 0.0 {
            return None;
        }
        let p = self.declared(alpha[0], alpha[1]);
        lyapunov2(self.m, x, &p, kappa.sqrt())
            .ok()
            .map(|l| l.l2)
            .filter(|v| v.is_finite())
    }

    /// The Hopf-Hopf test `h` at `x` (SPEC section 8k): the real part of the second pair when
    /// there is exactly one critical root, NaN for `kappa <= 0` and without the option, and NaN
    /// where the roots fail or the critical root is not single (both counted as lost). At
    /// `sqrt(kappa)` below `hh_omega_floor` (near a Bogdanov-Takens point) those two cases are
    /// NaN and not counted, because the pair is lost to rounding or to the root finder there.
    fn hh_at(&mut self, x: &[f64]) -> f64 {
        let Some(ro) = self.hh else {
            return f64::NAN;
        };
        let kappa = x[3 * self.n + 1];
        // The locator probes kappa near 1e-17 at a BT point, where nothing can be linearized.
        if kappa.is_nan() || kappa <= 0.0 || kappa.sqrt() <= OMEGA_MIN {
            return f64::NAN;
        }
        let lin = match self.lin_at(x) {
            Ok((_, lin)) => lin,
            Err(_) => {
                self.hh_lost += 1;
                return f64::NAN;
            }
        };
        let omega = kappa.sqrt();
        // Near BT the pair at +-i omega is lost to rounding or to the root finder, which says
        // nothing about a second crossing. The floor only excuses a missing pair: a pair that
        // the roots show is used at any omega, because a large entry that is not part of the
        // pair also raises the floor.
        let near_bt = omega <= hh_omega_floor(&lin);
        let set = match characteristic_roots(&lin, &ro) {
            Ok(s) => s,
            Err(_) => {
                if !near_bt {
                    self.hh_lost += 1;
                }
                return f64::NAN;
            }
        };
        let pairs = hopf_pairs(&set, omega);
        if near_bt && pairs.critical != 1 {
            return f64::NAN;
        }
        if !set.unrefined.is_empty() {
            self.hh_unrefined += 1;
        }
        if pairs.critical != 1 {
            self.hh_lost += 1;
            return f64::NAN;
        }
        pairs.second.map_or(f64::NAN, |r| r.re)
    }

    /// `Im` of the second critical root at a located `HH` point; `None` where the roots fail,
    /// the critical root is not single or there is no second root.
    fn omega2_at(&mut self, alpha: [f64; 2], x: &[f64], kappa: f64) -> Option<f64> {
        let ro = self.hh?;
        if kappa.is_nan() || kappa <= 0.0 {
            return None;
        }
        let p = self.declared(alpha[0], alpha[1]);
        let lin = linearize(self.m, x, &p).ok()?;
        let set = characteristic_roots(&lin, &ro).ok()?;
        let pairs = hopf_pairs(&set, kappa.sqrt());
        if pairs.critical != 1 {
            return None;
        }
        pairs.second.map(|r| r.im).filter(|v| v.is_finite())
    }

    /// `det Delta(0)`; 0 when the matrix is numerically singular.
    fn det_zero(&mut self, x: &[f64]) -> Result<f64, EngineError> {
        let n = self.n;
        let (_, lin) = self.lin_at(x)?;
        let mut sum = lin.a0.clone();
        for (_, a) in &lin.delays {
            for (s, v) in sum.iter_mut().zip(a) {
                *s += v;
            }
        }
        for s in sum.iter_mut() {
            *s = -*s;
        }
        match Lu::new(&sum, n) {
            Ok(lu) => Ok(lu.det()),
            Err(pc_linalg::LinalgError::Singular) => Ok(0.0),
            Err(e) => Err(e.into()),
        }
    }
}

impl Problem for HopfProblem<'_> {
    fn dim(&self) -> usize {
        3 * self.n + 3
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.n;
        let (p, lin) = self.lin_at(x)?;
        let g = field(self.m, &x[..n], &p)?;
        let (a, b) = (&x[n..2 * n], &x[2 * n..3 * n]);
        let kappa = x[3 * n + 1];
        let (r, s) = rs_matrices(&lin, kappa);
        let (ra, sb) = (matvec(&r, a, n), matvec(&s, b, n));
        let (sa, rb) = (matvec(&s, a, n), matvec(&r, b, n));
        out[..n].copy_from_slice(&g);
        for i in 0..n {
            out[n + i] = ra[i] - kappa * sb[i];
            out[2 * n + i] = sa[i] + rb[i];
        }
        out[3 * n] = dot(&self.c, a) - 1.0;
        out[3 * n + 1] = dot(&self.c, b);
        if out[..3 * n + 2].iter().any(|v| !v.is_finite()) {
            return Err(EngineError::NotFinite("the Hopf system".into()));
        }
        Ok(())
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let cols = x.len();
        let rows = cols - 1;
        let mut xp = x.to_vec();
        let mut fp = vec![0.0; rows];
        let mut fm = vec![0.0; rows];
        for j in 0..cols {
            let h = 1e-6 * (1.0 + x[j].abs());
            xp[j] = x[j] + h;
            self.residual(&xp, &mut fp)?;
            xp[j] = x[j] - h;
            self.residual(&xp, &mut fm)?;
            xp[j] = x[j];
            for i in 0..rows {
                out[i + rows * j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        Ok(())
    }

    fn ep_tests(&self) -> bool {
        false
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        labels_of(self.lyapunov, self.hh.is_some())
    }

    fn codim2_tests(
        &mut self,
        x: &[f64],
        _tangent: &[f64],
        _eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        let kappa = x[3 * self.n + 1];
        let zh = if kappa > 0.0 {
            self.det_zero(x)?
        } else {
            f64::NAN
        };
        let mut tests = vec![kappa, zh];
        if self.lyapunov {
            tests.push(self.l1_at(x));
        }
        if self.hh.is_some() {
            tests.push(self.hh_at(x));
        }
        Ok(tests)
    }
}

/// The codim-2 labels of a run, in the order of its tests.
fn labels_of(lyapunov: bool, hh: bool) -> &'static [&'static str] {
    match (lyapunov, hh) {
        (false, false) => &LABELS,
        (true, false) => &LABELS_L1,
        (false, true) => &LABELS_HH,
        (true, true) => &LABELS_L1_HH,
    }
}

/// Checks the start against the model and the options.
fn validate(m: &DdeModel, s: &HopfSetup, opts: &Options) -> Result<(), EngineError> {
    let bad = |msg: String| Err(EngineError::BadProblem(msg));
    let n_user = m.n_user_params;
    let [f0, f1] = s.free;
    if f0 >= n_user || f1 >= n_user {
        return bad(format!(
            "free parameter index out of range: the model has {n_user} declared parameters"
        ));
    }
    if f0 == f1 {
        return bad("the two free parameters must differ".into());
    }
    check_params(m, &s.params)?;
    check_state(m, &s.x0)?;
    if !(s.omega.is_finite() && s.omega > OMEGA_MIN) {
        return bad(format!(
            "omega is {}; the start needs a finite omega above {OMEGA_MIN:e}",
            s.omega
        ));
    }
    let second_is_delay = m
        .delays
        .iter()
        .any(|d| matches!(d.time, DelayTime::Param(j) if j == f1));
    if second_is_delay && (opts.bounds.0 <= 0.0 || opts.bounds.0.is_nan()) {
        return bad(
            "the second free parameter is a delay time, so bounds must have a lower bound \
             above 0"
                .into(),
        );
    }
    Ok(())
}

/// The start vector `(u, a, b, a1, kappa, a2)` and `c`, from the null vector of the real
/// `2n` embedding `[[R, -omega S], [omega S, R]]` of `Delta(i omega)`.
fn start_vector(m: &mut DdeModel, s: &HopfSetup) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
    let n = s.x0.len();
    let omega = s.omega;
    let kappa = omega * omega;
    let lin = linearize(m, &s.x0, &s.params)?;
    let (r, sm) = rs_matrices(&lin, kappa);
    let nn = 2 * n;
    let mut e = vec![0.0; nn * nn];
    for i in 0..n {
        for k in 0..n {
            let (rv, sv) = (r[i * n + k], omega * sm[i * n + k]);
            e[i + nn * k] = rv;
            e[i + nn * (n + k)] = -sv;
            e[n + i + nn * k] = sv;
            e[n + i + nn * (n + k)] = rv;
        }
    }
    let v = null_right(&e, nn, nn, 1)?.remove(0);
    let a0 = &v[..n];
    let b0: Vec<f64> = v[n..].iter().map(|t| t / omega).collect();
    // `c` is the real or the imaginary part of the null vector, whichever is larger.
    let (a_norm, bb_norm) = (norm(a0), norm(&v[n..]));
    let c: Vec<f64> = if a_norm >= bb_norm {
        a0.iter().map(|t| t / a_norm).collect()
    } else {
        v[n..].iter().map(|t| t / bb_norm).collect()
    };
    // Complex multiple of v with c.a = 1 and c.b = 0.
    let (ca, cb) = (dot(&c, a0), dot(&c, &b0));
    let det = ca * ca + kappa * cb * cb;
    let (sr, sig) = (ca / det, -cb / det);
    let mut x = Vec::with_capacity(3 * n + 3);
    x.extend_from_slice(&s.x0);
    x.extend((0..n).map(|i| sr * a0[i] - kappa * sig * b0[i]));
    x.extend((0..n).map(|i| sig * a0[i] + sr * b0[i]));
    x.push(s.params[s.free[0]]);
    x.push(kappa);
    x.push(s.params[s.free[1]]);
    Ok((x, c))
}

/// Follows the Hopf curve of a delay equilibrium through the start of `setup`.
///
/// The start is polished by Newton's method at the fixed second parameter, on the square
/// system in `(u, a, b, a1, kappa)`, with the tolerances and iteration cap of `opts`;
/// `NewtonFailed` when it does not converge. The first tangent points to increasing second
/// parameter, and `Options::direction` -1 reverses it. `Options::bounds` apply to the
/// second parameter, and `detect` and `mesh_adapt` are not used. A cancelled run is
/// `Err(EngineError::Cancelled)`.
pub fn continue_hopf(
    model: &mut DdeModel,
    setup: &HopfSetup,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeHopfBranch, EngineError> {
    let extras = HopfExtras {
        lyapunov: false,
        hh: None,
    };
    continue_hopf_with(model, setup, opts, cancel, &extras)
}

/// `continue_hopf` that also computes the first Lyapunov coefficient at every point with
/// `kappa > 0` (`DdeHopfPoint::l1`, `None` elsewhere) and labels a zero of it `GH`, a
/// generalized Hopf point (SPEC section 8g-2). A `GH` special also carries the second
/// coefficient `l2` (SPEC section 8h). The curve, `BT` and `ZH` are those of
/// `continue_hopf`.
pub fn continue_hopf_with_lyapunov(
    model: &mut DdeModel,
    setup: &HopfSetup,
    opts: &Options,
    cancel: &AtomicBool,
) -> Result<DdeHopfBranch, EngineError> {
    let extras = HopfExtras {
        lyapunov: true,
        hh: None,
    };
    continue_hopf_with(model, setup, opts, cancel, &extras)
}

/// `continue_hopf` with the additions of `extras`: `lyapunov` is
/// `continue_hopf_with_lyapunov`, and `hh` computes the characteristic roots at every point
/// with `kappa > 0` and labels a sign change of the real part of the second pair `HH`, a
/// Hopf-Hopf point (SPEC section 8k). An `HH` special carries `omega2`. `hh.re_min` must be
/// finite and negative, and the roots at the polished start must have exactly one root at
/// `i omega`, else `BadProblem`.
pub fn continue_hopf_with(
    model: &mut DdeModel,
    setup: &HopfSetup,
    opts: &Options,
    cancel: &AtomicBool,
    extras: &HopfExtras,
) -> Result<DdeHopfBranch, EngineError> {
    run_hopf(model, setup, opts, cancel, extras)
}

/// Checks the roots of the `hh` option at the polished start: one root at `i omega`.
fn check_hh_start(prob: &mut HopfProblem<'_>, x: &[f64]) -> Result<(), EngineError> {
    let Some(ro) = prob.hh else {
        return Ok(());
    };
    let kappa = x[3 * prob.n + 1];
    let (_, lin) = prob.lin_at(x)?;
    let set = characteristic_roots(&lin, &ro)?;
    let critical = if kappa > 0.0 {
        hopf_pairs(&set, kappa.sqrt()).critical
    } else {
        0
    };
    if critical != 1 {
        return Err(EngineError::BadProblem(format!(
            "hh: the roots at the start have {critical} roots at i omega instead of one; \
             raise n ({}) or check the start",
            ro.n
        )));
    }
    Ok(())
}

fn run_hopf(
    model: &mut DdeModel,
    setup: &HopfSetup,
    opts: &Options,
    cancel: &AtomicBool,
    extras: &HopfExtras,
) -> Result<DdeHopfBranch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    validate(model, setup, opts)?;
    if let Some(ro) = &extras.hh
        && !(ro.re_min.is_finite() && ro.re_min < 0.0)
    {
        return Err(EngineError::BadProblem(format!(
            "hh: re_min is {}; it must be finite and negative",
            ro.re_min
        )));
    }
    let lyapunov = extras.lyapunov;
    let n = setup.x0.len();
    let (mut x, c) = start_vector(model, setup)?;
    let mut prob = HopfProblem {
        m: model,
        n,
        free: setup.free,
        params: setup.params.clone(),
        c,
        lyapunov,
        hh: extras.hh,
        hh_unrefined: 0,
        hh_lost: 0,
    };
    // Newton at fixed a2 on the square system: the Jacobian without its last column.
    let ns = 3 * n + 2;
    let mut f = vec![0.0; ns];
    let mut jac = vec![0.0; ns * (ns + 1)];
    let mut res = f64::INFINITY;
    let mut converged = false;
    for _ in 0..opts.max_newton {
        prob.residual(&x, &mut f)?;
        prob.jacobian(&x, &mut jac)?;
        let lu = Lu::new(&jac[..ns * ns], ns)?;
        for v in f.iter_mut() {
            *v = -*v;
        }
        lu.solve(&mut f)?;
        let mut dxn = 0.0_f64;
        for (xi, d) in x.iter_mut().zip(&f) {
            *xi += d;
            dxn = dxn.max(d.abs());
        }
        prob.residual(&x, &mut f)?;
        res = f.iter().fold(0.0, |m, v| m.max(v.abs()));
        if dxn <= opts.tol_x && res <= opts.tol_f {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(EngineError::NewtonFailed {
            iters: opts.max_newton,
            residual: res,
        });
    }
    check_hh_start(&mut prob, &x)?;
    let mut tangent = start_tangent(&mut prob, &x)?;
    let last = tangent[3 * n + 2];
    if last.abs() > 1e-8 && last < 0.0 {
        for t in tangent.iter_mut() {
            *t = -*t;
        }
    }
    let branch = run_branch(
        &mut prob,
        &x,
        Some(&tangent),
        opts,
        false,
        cancel,
        &mut |_| {},
    )?;
    let hh_on = extras.hh.is_some();
    let point_of = |p: &Point| DdeHopfPoint {
        alpha: [p.u[3 * n], p.alpha],
        x: p.u[..n].to_vec(),
        kappa: p.u[3 * n + 1],
        omega: (p.u[3 * n + 1] > 0.0).then(|| p.u[3 * n + 1].sqrt()),
        l1: if lyapunov {
            p.codim2.get(2).copied().filter(|v| v.is_finite())
        } else {
            None
        },
        hh: if hh_on {
            p.codim2
                .get(2 + usize::from(lyapunov))
                .copied()
                .filter(|v| v.is_finite())
        } else {
            None
        },
    };
    let labels = labels_of(lyapunov, hh_on);
    let mut specials: Vec<DdeHopfSpecial> = branch
        .specials
        .iter()
        .filter_map(|s| {
            let label = labels.iter().copied().find(|l| *l == s.label)?;
            let p = point_of(&s.point);
            Some(DdeHopfSpecial {
                label,
                alpha: p.alpha,
                x: p.x,
                kappa: p.kappa,
                index: s.index,
                l2: None,
                omega2: None,
            })
        })
        .collect();
    // The second coefficient is computed once the run is over, at the point each `GH`
    // special carries, because the model is borrowed by the run until then.
    for s in specials.iter_mut().filter(|s| s.label == "GH") {
        s.l2 = prob.l2_at(s.alpha, &s.x, s.kappa);
    }
    for s in specials.iter_mut().filter(|s| s.label == "HH") {
        s.omega2 = prob.omega2_at(s.alpha, &s.x, s.kappa);
    }
    let mut warnings = branch.warnings.clone();
    if prob.hh_unrefined > 0 {
        warnings.push(format!(
            "hh_unrefined: {} evaluations of the Hopf-Hopf test had roots that did not \
             refine; a second pair may have been missed, raise n",
            prob.hh_unrefined
        ));
    }
    if prob.hh_lost > 0 {
        warnings.push(format!(
            "hh_lost: {} evaluations of the Hopf-Hopf test at kappa > 0 had no single root \
             at i omega or no roots, and the test was NaN there",
            prob.hh_lost
        ));
    }
    Ok(DdeHopfBranch {
        points: branch.points.iter().map(point_of).collect(),
        specials,
        stop: branch.stop.clone(),
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Evaluator, build_dde};
    use pc_ode::{MapLoader, SourceMap, load};

    fn model(src: &str) -> DdeModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", src);
        let l = load(top, &mut map, &mut MapLoader::new());
        build_dde(&l.expanded, &l.info, Evaluator::Fallback).expect("model builds")
    }

    #[test]
    fn cosk_and_sink_match_the_closed_forms() {
        for tau in [1.0, 2.5] {
            for kappa in [0.7, -0.7] {
                let (c, s) = (cos_k(kappa, tau), sin_k(kappa, tau));
                let (c0, s0) = if kappa > 0.0 {
                    let w = kappa.sqrt();
                    ((w * tau).cos(), (w * tau).sin() / w)
                } else {
                    let w = (-kappa).sqrt();
                    ((w * tau).cosh(), (w * tau).sinh() / w)
                };
                assert!(
                    (c - c0).abs() <= 1e-15 * c0.abs().max(1.0),
                    "cos {kappa} {tau}"
                );
                assert!(
                    (s - s0).abs() <= 1e-15 * s0.abs().max(1.0),
                    "sin {kappa} {tau}"
                );
            }
            assert_eq!(cos_k(0.0, tau), 1.0);
            assert_eq!(sin_k(0.0, tau), tau);
        }
    }

    #[test]
    fn the_series_and_the_closed_forms_agree_at_the_switch() {
        for tau in [0.5, 1.0, 3.0] {
            for sign in [1.0, -1.0] {
                let kappa = sign * SERIES / (tau * tau);
                // The series are used strictly below the switch, the closed forms above.
                for scale in [0.999_999, 1.000_001] {
                    let k = kappa * scale;
                    let (c, s) = (cos_k(k, tau), sin_k(k, tau));
                    let w = k.abs().sqrt();
                    let (c0, s0) = if k > 0.0 {
                        ((w * tau).cos(), (w * tau).sin() / w)
                    } else {
                        ((w * tau).cosh(), (w * tau).sinh() / w)
                    };
                    assert!((c - c0).abs() <= 1e-15, "cos {k} {tau}: {c} {c0}");
                    assert!((s - s0).abs() <= 1e-15 * tau, "sin {k} {tau}: {s} {s0}");
                }
            }
        }
    }

    #[test]
    fn the_hutchinson_residual_vanishes_at_the_exact_point() {
        // x' = r x (1 - x(t - tau)) has a Hopf point at r tau = pi / 2, x = 1, omega = r,
        // with the eigenvector a = 1, b = 0.
        let mut m = model("par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n");
        let r = std::f64::consts::FRAC_PI_2;
        let mut p = HopfProblem {
            m: &mut m,
            n: 1,
            free: [0, 1],
            params: vec![r, 1.0],
            c: vec![1.0],
            lyapunov: false,
            hh: None,
            hh_unrefined: 0,
            hh_lost: 0,
        };
        let x = [1.0, 1.0, 0.0, r, r * r, 1.0];
        let mut f = [0.0; 5];
        p.residual(&x, &mut f).unwrap();
        for (i, v) in f.iter().enumerate() {
            assert!(v.abs() < 1e-13, "row {i}: {v}");
        }
        // The Jacobian agrees with the exact one in the rows that are easy to write down:
        // d(row 3)/da = c, d(row 4)/db = c.
        let mut jac = [0.0; 5 * 6];
        p.jacobian(&x, &mut jac).unwrap();
        assert!((jac[3 + 5] - 1.0).abs() < 1e-9);
        assert!((jac[4 + 5 * 2] - 1.0).abs() < 1e-9);
    }

    /// The pair at `+-i omega` of `x'' = -omega^2 x` with entries of size `s`, where the
    /// `(2, 1)` entry carries an error of `k` times `eps s`, the size of one rounding of a sum
    /// of terms of size `s`.
    fn pair_with_rounding(s: f64, omega: f64, k: f64) -> LinearDde {
        LinearDde {
            dim: 2,
            a0: vec![0.0, s, -(omega * omega) / s + k * f64::EPSILON * s, 0.0],
            delays: vec![(1.0, vec![0.0; 4])],
        }
    }

    #[test]
    fn the_hh_floor_is_above_the_sizes_where_the_pair_is_lost() {
        let ro = RootOptions {
            n: 30,
            re_min: -3.0,
        };
        for s in [1.0, 10.0, 1e3] {
            let floor = hh_omega_floor(&pair_with_rounding(s, 1.0, 0.0));
            assert!(floor > OMEGA_MIN, "s = {s}: floor {floor:e}");
            // Above the floor, up to a factor 10, a rounding error of up to 4 units in one
            // entry neither merges the pair nor stops the root finder from refining it.
            for e in 0..20 {
                let w = floor * 1.000_1 * 10f64.powf(e as f64 * 0.05);
                for k in [-4.0, -1.0, 1.0, 4.0] {
                    let set = characteristic_roots(&pair_with_rounding(s, w, k), &ro).unwrap();
                    assert_eq!(
                        hopf_pairs(&set, w).critical,
                        1,
                        "s = {s}, omega = {w:e}, k = {k}"
                    );
                }
            }
            // Below it the pair is lost for some omega, so the floor is not vacuous.
            let lost = (0..120).any(|e| {
                let w = floor * 10f64.powf(-(e as f64) * 0.1);
                [1.0, 4.0].iter().any(|&k| {
                    let set = characteristic_roots(&pair_with_rounding(s, w, k), &ro).unwrap();
                    hopf_pairs(&set, w).critical != 1
                })
            });
            assert!(lost, "s = {s}: the pair is never lost below {floor:e}");
        }
        // The floor follows the largest entry of a0 and of every delayed matrix.
        let mut lin = pair_with_rounding(1.0, 1.0, 0.0);
        lin.delays[0].1[3] = 100.0;
        let expected = hh_omega_floor(&pair_with_rounding(100.0, 1.0, 0.0));
        assert_eq!(hh_omega_floor(&lin), expected);
    }
}
