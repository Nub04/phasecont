//! Shared tolerance and checker for derivative entries.
//!
//! [`close_err`] is the SPEC section 3 test and the only one the DAG, JIT and corpus tests use
//! to compare two evaluators. [`check_model`] runs the DAG entries against the f64
//! fallback and against central finite differences, and returns a [`Report`] that
//! `pc-cli check-derivs` prints.

use crate::entry::{EntryKind, Slot};
use crate::eval::{eval, eval_err};
use crate::interp::Fallback;
use crate::lower::IrModel;
use std::fmt;

/// Relative error allowed for `A` and `f_alpha` against finite differences.
pub const FD_TOL_FIRST: f64 = 1e-6;
/// Relative error allowed for `B` and `C` against nested finite differences.
pub const FD_TOL_HIGHER: f64 = 1e-4;

/// Spacing of f64 values at `|x|`; NaN for a non-finite `x`.
pub fn ulp(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let a = x.abs();
    f64::from_bits(a.to_bits() + 1) - a
}

/// Largest magnitude among the finite entries of `v` (0 when there are none).
pub fn scale(v: &[f64]) -> f64 {
    v.iter()
        .filter(|x| x.is_finite())
        .fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Absolute floor of [`close`]: values near zero carry rounding from larger intermediate terms.
pub const ABS_FLOOR: f64 = 1e-15;

/// Factor of the running-error term of [`close_err`]: `8 eps E`, which is `16 u` for `u = eps / 2`.
pub const ERR_FACTOR: f64 = 8.0;

/// SPEC section 3 without a running error bound: both NaN, or equal, or
/// `|a-b| <= 4 ulp(max(|a|,|b|))`, or `|a-b| <= 1e-13 * scale` (the largest magnitude of the
/// output vector), or `|a-b| <= ABS_FLOOR`.
pub fn close(a: f64, b: f64, scale: f64) -> bool {
    close_with(a, b, scale, None)
}

/// SPEC section 3: [`close`], or `|a-b| <= ERR_FACTOR * eps * err`, where `err` is the running
/// error bound of the DAG evaluation of this output ([`crate::eval::eval_err`]). A NaN or
/// infinite `err` adds nothing.
pub fn close_err(a: f64, b: f64, scale: f64, err: f64) -> bool {
    close_with(a, b, scale, Some(err))
}

fn close_with(a: f64, b: f64, scale: f64, err: Option<f64>) -> bool {
    if a.is_nan() && b.is_nan() || a == b {
        return true;
    }
    (a - b).abs() <= allowed_diff(a, b, scale, err)
}

/// The largest `|a-b|` that [`close`] (`err` is `None`) or [`close_err`] (`err` is `Some`)
/// accepts. The 4 ulp term ignores a non-finite value, so the result is always finite and at
/// least [`ABS_FLOOR`]. A `Some` bound that is NaN or infinite adds nothing.
pub fn allowed_diff(a: f64, b: f64, scale: f64, err: Option<f64>) -> f64 {
    let mut m = (4.0 * ulp(a.abs().max(b.abs())))
        .max(1e-13 * scale)
        .max(ABS_FLOOR);
    if let Some(e) = err.filter(|e| e.is_finite()) {
        m = m.max(ERR_FACTOR * f64::EPSILON * e);
    }
    m
}

/// `|a-b| <= tol * scale`, for comparisons against finite differences or closed forms.
pub fn within(a: f64, b: f64, scale: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() || a == b {
        return true;
    }
    (a - b).abs() <= tol * scale
}

/// How far `a` is from passing [`close_err`]: the ratio of `|a-b|` to the allowed difference.
/// At most 1 means a pass; infinite when exactly one of the two is NaN or non-finite.
fn ratio(a: f64, b: f64, scale: f64, err: f64) -> f64 {
    if a.is_nan() && b.is_nan() || a == b {
        return 0.0;
    }
    let diff = (a - b).abs();
    if diff.is_nan() {
        // Exactly one of the two is NaN.
        return f64::INFINITY;
    }
    diff / allowed_diff(a, b, scale, Some(err))
}

/// SplitMix64: a small deterministic generator, so a seed reproduces a check run.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// Settings for [`check_model_with`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Random points per entry.
    pub points: usize,
    pub seed: u64,
    /// Highest directional order compared against the fallback (1 to 5).
    pub max_order: usize,
    /// Points are drawn as `init + half_width * r` with `r` in [-1, 1).
    pub half_width: f64,
}

impl Options {
    pub fn new(points: usize, seed: u64) -> Self {
        Options {
            points,
            seed,
            max_order: 5,
            half_width: 0.5,
        }
    }
}

/// The result for one entry kind (or one finite-difference check).
#[derive(Clone, Debug)]
pub struct EntryReport {
    /// `F`, `Aux`, `JacU`, `JacP`, `Dir[U,P,Full]`, or `A (FD)`, `f_alpha (FD)`,
    /// `B (FD)`, `C (FD)`.
    pub label: String,
    pub pass: bool,
    /// Largest absolute error seen.
    pub worst_err: f64,
    /// Output scale `s` at the point of the worst error.
    pub scale: f64,
    /// Worst error divided by the allowed error (at most 1 passes).
    pub ratio: f64,
    /// `x` where the worst error happened.
    pub x: Vec<f64>,
    /// `p` (with `t` last) where the worst error happened.
    pub p: Vec<f64>,
    pub points: usize,
    /// Set when the entry could not be built.
    pub note: Option<String>,
}

impl EntryReport {
    fn new(label: String) -> Self {
        EntryReport {
            label,
            pass: true,
            worst_err: 0.0,
            scale: 0.0,
            ratio: 0.0,
            x: Vec::new(),
            p: Vec::new(),
            points: 0,
            note: None,
        }
    }

    /// Records one output vector. `errs[i]` and `ratios[i]` are per element.
    fn record(
        &mut self,
        errs_ratios: impl Iterator<Item = (f64, f64)>,
        s: f64,
        x: &[f64],
        p: &[f64],
    ) {
        self.points += 1;
        for (err, r) in errs_ratios {
            let r = if r.is_nan() { f64::INFINITY } else { r };
            if r > 1.0 {
                self.pass = false;
            }
            if self.x.is_empty() || r > self.ratio {
                self.ratio = r;
                self.worst_err = if err.is_nan() { f64::INFINITY } else { err };
                self.scale = s;
                self.x = x.to_vec();
                self.p = p.to_vec();
            }
        }
    }
}

/// All entries checked for one model.
#[derive(Clone, Debug)]
pub struct Report {
    pub entries: Vec<EntryReport>,
    pub points: usize,
    pub seed: u64,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.entries.iter().all(|e| e.pass)
    }

    pub fn failures(&self) -> impl Iterator<Item = &EntryReport> {
        self.entries.iter().filter(|e| !e.pass)
    }

    pub fn find(&self, label: &str) -> Option<&EntryReport> {
        self.entries.iter().find(|e| e.label == label)
    }
}

/// A vector as `[1.000000e0, ...]`, the form the report lines use.
pub fn fmt_vec(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.6e}")).collect();
    format!("[{}]", parts.join(", "))
}

impl fmt::Display for EntryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let verdict = if self.pass { "pass" } else { "FAIL" };
        write!(
            f,
            "{:<22} {verdict}  worst {:.3e}  s {:.3e}  ratio {:.3e}",
            self.label, self.worst_err, self.scale, self.ratio
        )?;
        if let Some(n) = &self.note {
            write!(f, "  ({n})")?;
        } else if !self.pass {
            write!(
                f,
                "\n    at x = {}, p = {}",
                fmt_vec(&self.x),
                fmt_vec(&self.p)
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} points per entry, seed {}", self.points, self.seed)?;
        for e in &self.entries {
            writeln!(f, "{e}")?;
        }
        let bad = self.failures().count();
        if bad == 0 {
            write!(f, "all {} checks pass", self.entries.len())
        } else {
            write!(f, "{bad} of {} checks FAIL", self.entries.len())
        }
    }
}

/// Every slot mix of length `k`.
pub fn slot_mixes(k: usize) -> Vec<Vec<Slot>> {
    let mut all = vec![vec![]];
    for _ in 0..k {
        let mut next = Vec::new();
        for m in &all {
            for s in [Slot::U, Slot::P, Slot::Full] {
                let mut m2: Vec<Slot> = m.clone();
                m2.push(s);
                next.push(m2);
            }
        }
        all = next;
    }
    all
}

/// The label of a `Dir` entry, such as `Dir[U,P,Full]`.
pub fn slot_label(slots: &[Slot]) -> String {
    let names: Vec<&str> = slots
        .iter()
        .map(|s| match s {
            Slot::U => "U",
            Slot::P => "P",
            Slot::Full => "Full",
        })
        .collect();
    format!("Dir[{}]", names.join(","))
}

/// A random point in the box around the model's init and parameter values. `p` has `t`
/// last, drawn in [0, 1). `hw` is the half-width of the box around `init` (`Options::half_width`).
pub fn draw_point(ir: &IrModel, rng: &mut Rng, hw: f64) -> (Vec<f64>, Vec<f64>) {
    let (n, n_p) = (ir.n(), ir.n_p());
    let x = (0..n)
        .map(|j| ir.init.get(j).copied().unwrap_or(0.0) + hw * rng.range(-1.0, 1.0))
        .collect();
    let mut p: Vec<f64> = (0..n_p)
        .map(|j| {
            let b = ir.par_values.get(j).copied().unwrap_or(0.0);
            if b == 0.0 {
                0.2 * rng.range(-1.0, 1.0)
            } else {
                b * rng.range(0.9, 1.1)
            }
        })
        .collect();
    p.push(rng.unit());
    (x, p)
}

fn run(ir: &IrModel, kind: &EntryKind, x: &[f64], p: &[f64], v: &[f64]) -> Vec<f64> {
    match ir.entries.get(kind) {
        Some(e) => {
            let mut out = vec![f64::NAN; e.outputs.len()];
            eval(&ir.dag, &e.outputs, x, p, v, &mut out);
            out
        }
        None => Vec::new(),
    }
}

fn compare_kind(
    ir: &mut IrModel,
    fb: &Fallback,
    kind: &EntryKind,
    label: String,
    rng: &mut Rng,
    opt: &Options,
) -> EntryReport {
    let mut rep = EntryReport::new(label);
    let entry = match ir.entry(kind) {
        Ok(e) => e,
        Err(e) => {
            rep.pass = false;
            rep.note = Some(format!("entry not built: {e}"));
            return rep;
        }
    };
    let (n, n_p) = (ir.n(), ir.n_p());
    let vlen = match kind {
        EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
        _ => 0,
    };
    for _ in 0..opt.points {
        let (x, p) = draw_point(ir, rng, opt.half_width);
        let v: Vec<f64> = (0..vlen).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut got = vec![f64::NAN; entry.outputs.len()];
        let mut err = vec![f64::NAN; entry.outputs.len()];
        eval_err(&ir.dag, &entry.outputs, &x, &p, &v, &mut got, &mut err);
        let mut want = vec![f64::NAN; entry.outputs.len()];
        fb.eval(kind, &x, &p, &v, &mut want);
        let s = scale(&want);
        let it = got
            .iter()
            .zip(&want)
            .zip(&err)
            .map(|((&a, &b), &e)| ((a - b).abs(), ratio(a, b, s, e)));
        rep.record(it, s, &x, &p);
    }
    rep
}

/// Central difference of `g` along unit direction `dir` (length matches `base`), step
/// `h = eps^(1/3) * max(1, |base_j|)` taken per component `j`.
fn step(base: f64) -> f64 {
    f64::EPSILON.cbrt() * base.abs().max(1.0)
}

fn fd_report(
    label: &str,
    tol: f64,
    rep_pts: impl Iterator<Item = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64)>,
) -> EntryReport {
    // Each item: (x, p, exact, fd, extra scale from a differenced lower-order quantity).
    let mut rep = EntryReport::new(label.to_string());
    for (x, p, exact, fd, extra) in rep_pts {
        let s = scale(&exact).max(extra).max(f64::MIN_POSITIVE);
        let it = exact.iter().zip(&fd).map(|(&a, &b)| {
            let err = (a - b).abs();
            let r = if a.is_nan() && b.is_nan() || a == b {
                0.0
            } else {
                err / (tol * s)
            };
            (err, r)
        });
        rep.record(it, s, &x, &p);
    }
    rep
}

/// DAG output of `kind` at (x, p, v); the entry must already be built.
fn dag(ir: &IrModel, kind: &EntryKind, x: &[f64], p: &[f64], v: &[f64]) -> Vec<f64> {
    run(ir, kind, x, p, v)
}

/// Compares the DAG entries with the f64 fallback and with finite differences at
/// `points` random points. Never panics: an entry that cannot be built is reported as a
/// failure with a note.
pub fn check_model(ir: &mut IrModel, fb: &Fallback, points: usize, seed: u64) -> Report {
    check_model_with(ir, fb, &Options::new(points, seed))
}

pub fn check_model_with(ir: &mut IrModel, fb: &Fallback, opt: &Options) -> Report {
    let mut rng = Rng(opt.seed);
    let n_p = ir.n_p();
    let mut entries = Vec::new();

    let mut kinds: Vec<(EntryKind, String)> =
        vec![(EntryKind::F, "F".into()), (EntryKind::JacU, "JacU".into())];
    if !ir.aux.is_empty() {
        kinds.push((EntryKind::Aux, "Aux".into()));
    }
    if n_p > 0 {
        kinds.push((EntryKind::JacP, "JacP".into()));
    }
    for (k, l) in kinds {
        entries.push(compare_kind(ir, fb, &k, l, &mut rng, opt));
    }
    for k in 1..=opt.max_order.min(5) {
        for slots in slot_mixes(k) {
            if n_p == 0 && slots.iter().any(|s| *s != Slot::U) {
                continue;
            }
            let label = slot_label(&slots);
            entries.push(compare_kind(
                ir,
                fb,
                &EntryKind::Dir(slots),
                label,
                &mut rng,
                opt,
            ));
        }
    }

    entries.extend(fd_checks(ir, opt, &mut rng));
    Report {
        entries,
        points: opt.points,
        seed: opt.seed,
    }
}

fn fd_checks(ir: &mut IrModel, opt: &Options, rng: &mut Rng) -> Vec<EntryReport> {
    let (n, n_p) = (ir.n(), ir.n_p());
    let u1 = EntryKind::Dir(vec![Slot::U]);
    let u2 = EntryKind::Dir(vec![Slot::U, Slot::U]);
    let u3 = EntryKind::Dir(vec![Slot::U, Slot::U, Slot::U]);
    let mut need = vec![
        EntryKind::F,
        EntryKind::JacU,
        u1.clone(),
        u2.clone(),
        u3.clone(),
    ];
    if n_p > 0 {
        need.push(EntryKind::JacP);
    }
    for k in &need {
        if let Err(e) = ir.entry(k) {
            let mut r = EntryReport::new("FD".into());
            r.pass = false;
            r.note = Some(format!("entry not built: {e}"));
            return vec![r];
        }
    }
    let ir: &IrModel = ir;
    let f_at = |x: &[f64], p: &[f64]| dag(ir, &EntryKind::F, x, p, &[]);
    let mut a_pts = Vec::new();
    let mut fa_pts = Vec::new();
    let mut b_pts = Vec::new();
    let mut c_pts = Vec::new();
    for _ in 0..opt.points {
        let (x, p) = draw_point(ir, rng, opt.half_width);
        // A: column j by central differences of f.
        let exact = dag(ir, &EntryKind::JacU, &x, &p, &[]);
        let mut fd = vec![0.0; n * n];
        for j in 0..n {
            let h = step(x[j]);
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[j] += h;
            xm[j] -= h;
            let (fp, fm) = (f_at(&xp, &p), f_at(&xm, &p));
            for (i, (a, b)) in fp.iter().zip(&fm).enumerate() {
                fd[i + n * j] = (a - b) / (2.0 * h);
            }
        }
        a_pts.push((x.clone(), p.clone(), exact, fd, 0.0));
        // f_alpha.
        if n_p > 0 {
            let exact = dag(ir, &EntryKind::JacP, &x, &p, &[]);
            let mut fd = vec![0.0; n * n_p];
            for j in 0..n_p {
                let h = step(p[j]);
                let (mut pp, mut pm) = (p.clone(), p.clone());
                pp[j] += h;
                pm[j] -= h;
                let (fp, fm) = (f_at(&x, &pp), f_at(&x, &pm));
                for (i, (a, b)) in fp.iter().zip(&fm).enumerate() {
                    fd[i + n * j] = (a - b) / (2.0 * h);
                }
            }
            fa_pts.push((x.clone(), p.clone(), exact, fd, 0.0));
        }
        // B(v, w) by differencing A v = Dir[U](v) along w.
        let v: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
        let w: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
        let z: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
        let vw: Vec<f64> = v.iter().chain(&w).copied().collect();
        let exact_b = dag(ir, &u2, &x, &p, &vw);
        let h = f64::EPSILON.cbrt();
        let shift = |x: &[f64], d: &[f64], s: f64| -> Vec<f64> {
            x.iter().zip(d).map(|(a, b)| a + s * b).collect()
        };
        let av_p = dag(ir, &u1, &shift(&x, &w, h), &p, &v);
        let av_m = dag(ir, &u1, &shift(&x, &w, -h), &p, &v);
        let fd_b: Vec<f64> = av_p
            .iter()
            .zip(&av_m)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect();
        // Rounding in A v is about 2e-11 of its size after division by h, so s keeps a
        // 1e-6 share of |A v|; this matters only where B is much smaller than A.
        let extra_b = scale(&av_p).max(scale(&av_m)) * 1e-6;
        b_pts.push((x.clone(), p.clone(), exact_b, fd_b, extra_b));
        // C(v, w, z) by differencing B(v, w) along z.
        let vwz: Vec<f64> = v.iter().chain(&w).chain(&z).copied().collect();
        let exact_c = dag(ir, &u3, &x, &p, &vwz);
        let bp = dag(ir, &u2, &shift(&x, &z, h), &p, &vw);
        let bm = dag(ir, &u2, &shift(&x, &z, -h), &p, &vw);
        let fd_c: Vec<f64> = bp
            .iter()
            .zip(&bm)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect();
        let extra_c = scale(&bp).max(scale(&bm)) * 1e-6;
        c_pts.push((x.clone(), p.clone(), exact_c, fd_c, extra_c));
    }
    let mut out = vec![fd_report("A (FD)", FD_TOL_FIRST, a_pts.into_iter())];
    if n_p > 0 {
        out.push(fd_report("f_alpha (FD)", FD_TOL_FIRST, fa_pts.into_iter()));
    }
    out.push(fd_report("B (FD)", FD_TOL_HIGHER, b_pts.into_iter()));
    out.push(fd_report("C (FD)", FD_TOL_HIGHER, c_pts.into_iter()));
    out
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod tests;
