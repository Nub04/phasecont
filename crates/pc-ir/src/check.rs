//! Shared tolerance and checker for derivative entries.
//!
//! [`close_err`] is the SPEC section 3 test and the only one the DAG, JIT and corpus tests use
//! to compare two evaluators. [`check_model`] runs the DAG entries against the f64
//! fallback and against central finite differences, and returns a [`Report`] that
//! `pc-cli check-derivs` prints.

use crate::entry::{EntryKind, Slot};
use crate::error::{IrError, IrErrorKind};
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

/// SPEC section 4 for `f`: both NaN, or equal (so `+0` equals `-0`), or
/// `|a-b| <= 4 ulp(max(|a|,|b|))`. There is no scale term and no absolute floor.
pub fn close_ulps(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() || a == b {
        return true;
    }
    (a - b).abs() <= 4.0 * ulp(a.abs().max(b.abs()))
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
    /// False when a counted check failed. A skipped entry is not a failure, so `pass` is
    /// true for it.
    pub pass: bool,
    /// Set when the entry was not built because it is too large. It was not checked and
    /// counts neither as a pass nor as a failure; `note` gives the reason.
    pub skip: bool,
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
    /// Usable sample points that the row ran on. A point is left out only when `f(x, p)` is
    /// not finite in the fallback and the DAG gives the same value there.
    pub points: usize,
    /// Output values that were compared. An element that is NaN on both sides is an
    /// agreement but not a compared value, so it is not counted here.
    pub compared: usize,
    /// Sample points that were drawn and left out because the model value `f(x, p)` there
    /// is not finite in the fallback and the DAG gives the same. The row drew
    /// `points + skipped_points` points in all.
    pub skipped_points: usize,
    /// Set when the entry could not be built, or when the row compared nothing.
    pub note: Option<String>,
}

impl EntryReport {
    fn new(label: String) -> Self {
        EntryReport {
            label,
            pass: true,
            skip: false,
            worst_err: 0.0,
            scale: 0.0,
            ratio: 0.0,
            x: Vec::new(),
            p: Vec::new(),
            points: 0,
            compared: 0,
            skipped_points: 0,
            note: None,
        }
    }

    /// The row for an entry that `IrModel::entry` refused. An entry over the node limit is
    /// skipped, since nothing was built to check; any other refusal is a failure.
    pub(crate) fn unbuilt(label: String, e: &IrError) -> Self {
        let mut r = EntryReport::new(label);
        r.skip = e.kind == IrErrorKind::TooLarge;
        r.pass = r.skip;
        r.note = Some(format!("entry not built: {e}"));
        r
    }

    /// Records the elements of one output vector at one sample point. An item is
    /// `Some((err, ratio))` for a compared element and `None` for an element that is NaN on
    /// both sides: that is an agreement, but nothing was compared, so it neither counts as
    /// a compared value nor sets the worst error. The first compared element sets the worst
    /// error even when it is zero; a later one replaces it only with a larger ratio.
    fn record(
        &mut self,
        items: impl Iterator<Item = Option<(f64, f64)>>,
        s: f64,
        x: &[f64],
        p: &[f64],
    ) {
        self.points += 1;
        for (err, r) in items.flatten() {
            let r = if r.is_nan() { f64::INFINITY } else { r };
            if r > 1.0 {
                self.pass = false;
            }
            let first = self.compared == 0;
            self.compared += 1;
            if first || r > self.ratio {
                self.ratio = r;
                self.worst_err = if err.is_nan() { f64::INFINITY } else { err };
                self.scale = s;
                self.x = x.to_vec();
                self.p = p.to_vec();
            }
        }
    }

    /// Ends a row. A row that compared nothing fails and says so: a pass would claim a
    /// check that did not happen.
    fn finish(mut self) -> Self {
        if self.compared > 0 || self.skip {
            return self;
        }
        self.pass = false;
        let drawn = self.points + self.skipped_points;
        let why = if drawn == 0 {
            "no point was drawn".to_string()
        } else if self.points == 0 {
            format!("no usable point, all {drawn} drawn points have a non-finite model value")
        } else {
            format!(
                "every value is NaN on both sides at the {} usable points",
                self.points
            )
        };
        self.note = Some(format!("nothing was compared: {why}"));
        self
    }
}

/// The comparison of one element for [`EntryReport::record`]: `None` when both sides are
/// NaN, else the absolute error and its ratio to `allowed`. Equal values, infinities
/// included, have error and ratio 0. When exactly one side is NaN the error is NaN and
/// the ratio is infinite.
fn element(a: f64, b: f64, allowed: f64) -> Option<(f64, f64)> {
    if a.is_nan() && b.is_nan() {
        return None;
    }
    if a == b {
        return Some((0.0, 0.0));
    }
    let err = (a - b).abs();
    if err.is_nan() {
        return Some((err, f64::INFINITY));
    }
    Some((err, err / allowed))
}

/// The comparison of one element of a DAG entry against the fallback, the one every row
/// against the fallback uses: `a` is the DAG value, `b` the fallback value, `s` the scale of
/// the fallback's vector and `e` the running error bound of `a`. See [`element`] for the
/// result.
fn compare_with_bound(a: f64, b: f64, s: f64, e: f64) -> Option<(f64, f64)> {
    element(a, b, allowed_diff(a, b, s, Some(e)))
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
        let verdict = if self.skip {
            "SKIP"
        } else if self.pass {
            "pass"
        } else {
            "FAIL"
        };
        write!(
            f,
            "{:<22} {verdict}  worst {:.3e}  s {:.3e}  ratio {:.3e}",
            self.label, self.worst_err, self.scale, self.ratio
        )?;
        if !self.skip {
            write!(f, "  {} points, {} values", self.points, self.compared)?;
        }
        if self.skipped_points > 0 && self.points > 0 {
            write!(
                f,
                "  ({} usable of {} drawn, non-finite model value)",
                self.points,
                self.points + self.skipped_points
            )?;
        }
        if let Some(n) = &self.note {
            write!(f, "  ({n})")?;
        }
        if !self.pass && self.compared > 0 {
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
        let skipped = self.entries.iter().filter(|e| e.skip).count();
        let counted = self.entries.len() - skipped;
        if bad == 0 {
            write!(f, "all {counted} checks pass")?;
        } else {
            write!(f, "{bad} of {counted} checks FAIL")?;
        }
        if skipped > 0 {
            write!(f, "; {skipped} skipped (entry too large)")?;
        }
        Ok(())
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
    let entry = match ir.entry(kind) {
        Ok(e) => e,
        Err(e) => return EntryReport::unbuilt(label, &e),
    };
    let mut rep = EntryReport::new(label);
    let (n, n_p) = (ir.n(), ir.n_p());
    let vlen = match kind {
        EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
        _ => 0,
    };
    let (samples, drawn) = usable_points(ir, fb, rng, opt, &[vlen]);
    rep.skipped_points = drawn - samples.len();
    for sample in &samples {
        let (x, p, v) = (&sample.x, &sample.p, &sample.dirs[0]);
        let mut got = vec![f64::NAN; entry.outputs.len()];
        let mut err = vec![f64::NAN; entry.outputs.len()];
        eval_err(&ir.dag, &entry.outputs, x, p, v, &mut got, &mut err);
        let mut want = vec![f64::NAN; entry.outputs.len()];
        fb.eval(kind, x, p, v, &mut want);
        let s = scale(&want);
        let it = got
            .iter()
            .zip(&want)
            .zip(&err)
            .map(|((&a, &b), &e)| compare_with_bound(a, b, s, e));
        rep.record(it, s, x, p);
    }
    rep.finish()
}

/// Sample points drawn per requested point, at most, when the model value is not finite at
/// some of them.
pub const MAX_DRAW_FACTOR: usize = 10;

/// One drawn sample point and the direction vectors a row draws with it.
struct Sample {
    x: Vec<f64>,
    p: Vec<f64>,
    /// One vector per length in the `dir_lens` the row asked for, each entry in [-1, 1).
    dirs: Vec<Vec<f64>>,
}

/// Whether the model value `f(x, p)` is not finite and both evaluators say so. The fallback
/// gives a value that is not finite at some element, and the DAG's `F` agrees with the
/// fallback at every element under the rule of the `F` row ([`compare_with_bound`]): finite
/// values within its allowance, NaN with NaN, `+inf` with `+inf`, `-inf` with `-inf`.
///
/// A point where the DAG and the fallback disagree is not one of these, even when the DAG's
/// own value is not finite: the `F` row has to compare it and report the mismatch.
/// Without a built `F` entry there is nothing to compare, and the answer is false.
fn non_finite_in_both(ir: &IrModel, fb: &Fallback, x: &[f64], p: &[f64]) -> bool {
    let Some(entry) = ir.entries.get(&EntryKind::F) else {
        return false;
    };
    let mut want = vec![f64::NAN; entry.outputs.len()];
    fb.eval(&EntryKind::F, x, p, &[], &mut want);
    if want.iter().all(|y| y.is_finite()) {
        return false;
    }
    let mut got = vec![f64::NAN; entry.outputs.len()];
    let mut err = vec![f64::NAN; entry.outputs.len()];
    eval_err(&ir.dag, &entry.outputs, x, p, &[], &mut got, &mut err);
    let s = scale(&want);
    got.iter()
        .zip(&want)
        .zip(&err)
        .all(|((&a, &b), &e)| compare_with_bound(a, b, s, e).is_none_or(|(_, ratio)| ratio <= 1.0))
}

/// Draws sample points until `opt.points` of them are usable, or until
/// `MAX_DRAW_FACTOR * opt.points` points have been drawn. A point is not usable when the
/// model value `f(x, p)` is not finite in the fallback and the DAG gives the same value
/// ([`non_finite_in_both`]): there is no finite derivative to compare with, so the point is
/// dropped and the next one is drawn in its place. A point where the DAG's `f` is not finite
/// and the fallback's is, or where the two differ in any other way, is usable, so the `F`
/// row compares it and fails. Each draw takes `x` and `p` (as [`draw_point`]) and then one
/// vector of each length in `dir_lens`, in that order, so a seed reproduces the same points
/// and when every point is usable the draws are the first `opt.points` ones, as before.
/// Returns the usable points and the number of points drawn. The `F` entry must already be
/// built (the `F` row builds it first); without it no point can be tested and none is dropped.
fn usable_points(
    ir: &IrModel,
    fb: &Fallback,
    rng: &mut Rng,
    opt: &Options,
    dir_lens: &[usize],
) -> (Vec<Sample>, usize) {
    let cap = opt.points.saturating_mul(MAX_DRAW_FACTOR);
    let mut found = Vec::with_capacity(opt.points);
    let mut drawn = 0_usize;
    while found.len() < opt.points && drawn < cap {
        let (x, p) = draw_point(ir, rng, opt.half_width);
        let dirs = dir_lens
            .iter()
            .map(|&len| (0..len).map(|_| rng.range(-1.0, 1.0)).collect())
            .collect();
        drawn += 1;
        if !non_finite_in_both(ir, fb, &x, &p) {
            found.push(Sample { x, p, dirs });
        }
    }
    (found, drawn)
}

/// Step of the central differences of `A` and `f_alpha`: `h = eps^(1/3) * max(1, |base|)`,
/// with `base` the component that is shifted.
fn step(base: f64) -> f64 {
    f64::EPSILON.cbrt() * base.abs().max(1.0)
}

/// Values and running error bounds of the outputs of `kind` at (x, p, v), from
/// [`eval_err`]; the entry must already be built.
fn val_err(
    ir: &IrModel,
    kind: &EntryKind,
    x: &[f64],
    p: &[f64],
    v: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    match ir.entries.get(kind) {
        Some(e) => {
            let mut got = vec![f64::NAN; e.outputs.len()];
            let mut err = vec![f64::NAN; e.outputs.len()];
            eval_err(&ir.dag, &e.outputs, x, p, v, &mut got, &mut err);
            (got, err)
        }
        None => (Vec::new(), Vec::new()),
    }
}

/// A bound that is NaN or infinite says nothing, so it adds nothing (the rule of
/// [`close_err`]).
fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() { v } else { 0.0 }
}

/// The central difference `D = (g+ - g-) / (2 h)` of one output, and the bound `E_D` on its
/// rounding error in the units of `eval_err`: the error is at most `u E_D` for `u = eps / 2`.
///
/// `eval_err` gives `E+` and `E-` with `|computed g(+h) - g(+h)| <= u E+` and the same for
/// `g(-h)` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 3.3).
/// The computed difference is `(g+ - g- + d+ - d-) (1 + t1) (1 + t2) / (2 h)` with
/// `|d+| <= u E+`, `|d-| <= u E-`, and `|t1|, |t2| <= u` for the subtraction and the
/// division. To first order in `u` the error is therefore at most
/// `u (E+ + E-) / (2 h) + 2 u |D|`, so
/// `E_D = (E+ + E- + |g+ - g-|) / (2 h) + |D|`, since `|g+ - g-| / (2 h) = |D|`.
///
/// The shifted point `x + h` is itself rounded, which changes the step actually taken by
/// at most `u |x + h|`: a relative change of about `eps^(2/3) / 2 = 1e-11` in `D`. That is
/// five orders below `FD_TOL_FIRST`, and `D` is at most about the output scale `s` that the
/// tolerance multiplies, so this shift error is left to `tol * s` and is not in `E_D`.
fn central(gp: f64, gm: f64, ep: f64, em: f64, h: f64) -> (f64, f64) {
    let d = (gp - gm) / (2.0 * h);
    let e = (ep + em + (gp - gm).abs()) / (2.0 * h) + d.abs();
    (d, e)
}

/// The Richardson value of the central difference of every output of `at(s)` (the values
/// and `eval_err` bounds of the outputs at the point shifted by `s`), the allowance that
/// goes with it. Rows `B` and `C` take the same allowance: there is no separate share of the
/// differenced quantity, since the `eval_err` bounds already follow a cancellation between
/// `A v` and its shifted value (tests `a_model_with_tiny_second_and_third_derivatives_*`).
///
/// `D(h) = g' + c2 h^2 + c4 h^4 + ...` has only even powers of `h`, so
/// `R = (4 D(h/2) - D(h)) / 3 = D(h/2) + (D(h/2) - D(h)) / 3` has no `h^2` term and the
/// error `-c4 h^4 / 4`. With `c4 = g^(5) / 5!` and the step `h/2` this is the leading term
/// `-4 g^(5) (h/2)^4 / 5!` of eq. (21) below.
///
/// Sources:
/// - Lecture notes on numerical differentiation, AMSC466, University of Maryland
///   (www.math.umd.edu/~mariakc/AMSC466/LectureNotes/differentiation.pdf), section 4:
///   eq. (15) is the central difference `F(h)`, and eq. (21) is
///   `F(h, 2h) = (4 F(h) - F(2h)) / 3 = F(h) + (F(h) - F(2h)) / 3 = g' - 4 g^(5) h^4 / 5! + O(h^6)`.
///   Those notes refer to Dahlquist and Bjorck, *Numerical Methods in Scientific Computing*,
///   vol. I, SIAM 2008, section 3.4.6.
/// - Wikipedia, "Richardson extrapolation": `A1(h) = (t^k A0(h/t) - A0(h)) / (t^k - 1)`,
///   which is the formula above for `t = 2`, `k = 2`.
///
/// The allowance beyond `tol * s` has two parts for each element:
/// - `|D(h) - D(h/2)| / 3`, the correction that Richardson applies, which estimates the
///   truncation error of `D(h/2)`. It is a conservative bound for the error of `R`, which
///   is smaller by about `h^2`. It lets a model with strong curvature pass, and a wrong
///   derivative smaller than this term goes unseen.
/// - `ERR_FACTOR * eps * E_R`, the rounding of `R` in the form `close_err` uses
///   (`8 eps E = 16 u E`). `4 D(h/2)` is exact, the subtraction and the division by 3 round
///   once each, so to first order
///   `E_R = (4 E_D(h/2) + E_D(h) + |4 D(h/2) - D(h)|) / 3 + |R|`, with `E_D` from [`central`].
fn richardson(at: impl Fn(f64) -> (Vec<f64>, Vec<f64>), h: f64) -> Fd {
    let (gp1, ep1) = at(h);
    let (gm1, em1) = at(-h);
    let (gp2, ep2) = at(0.5 * h);
    let (gm2, em2) = at(-0.5 * h);
    let (value, extra) = (0..gp1.len())
        .map(|i| {
            let (d1, e1) = central(gp1[i], gm1[i], ep1[i], em1[i], h);
            let (d2, e2) = central(gp2[i], gm2[i], ep2[i], em2[i], 0.5 * h);
            let t = 4.0 * d2 - d1;
            let r = t / 3.0;
            let e_r = (4.0 * e2 + e1 + t.abs()) / 3.0 + r.abs();
            let trunc = (d1 - d2).abs() / 3.0;
            (
                r,
                finite_or_zero(trunc) + ERR_FACTOR * f64::EPSILON * finite_or_zero(e_r),
            )
        })
        .unzip();
    Fd { value, extra }
}

/// The result of [`richardson`].
struct Fd {
    value: Vec<f64>,
    /// Allowance for each element on top of `tol * s`.
    extra: Vec<f64>,
}

/// One sample point of a finite-difference row.
struct FdPoint {
    x: Vec<f64>,
    p: Vec<f64>,
    /// The values of the DAG entry under test.
    exact: Vec<f64>,
    /// The Richardson values and their allowances.
    fd: Fd,
}

/// One finite-difference row. `skipped` is the number of drawn sample points that were left
/// out because the model value there is not finite; a row that compared nothing fails.
fn fd_report(label: &str, tol: f64, skipped: usize, pts: Vec<FdPoint>) -> EntryReport {
    let mut rep = EntryReport::new(label.to_string());
    rep.skipped_points = skipped;
    for pt in pts {
        let s = scale(&pt.exact).max(f64::MIN_POSITIVE);
        let it = pt
            .exact
            .iter()
            .zip(&pt.fd.value)
            .zip(&pt.fd.extra)
            .map(|((&a, &b), &extra)| element(a, b, tol * s + extra));
        rep.record(it, s, &pt.x, &pt.p);
    }
    rep.finish()
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

    entries.extend(fd_checks(ir, fb, opt, &mut rng));
    Report {
        entries,
        points: opt.points,
        seed: opt.seed,
    }
}

fn fd_checks(ir: &mut IrModel, fb: &Fallback, opt: &Options, rng: &mut Rng) -> Vec<EntryReport> {
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
            return vec![EntryReport::unbuilt("FD".into(), &e)];
        }
    }
    let ir: &IrModel = ir;
    let shift = |x: &[f64], d: &[f64], s: f64| -> Vec<f64> {
        x.iter().zip(d).map(|(a, b)| a + s * b).collect()
    };
    let h_dir = f64::EPSILON.cbrt();
    let mut a_pts = Vec::new();
    let mut fa_pts = Vec::new();
    let mut b_pts = Vec::new();
    let mut c_pts = Vec::new();
    let (samples, drawn) = usable_points(ir, fb, rng, opt, &[n, n, n]);
    let skipped = drawn - samples.len();
    for smp in &samples {
        let (x, p) = (&smp.x, &smp.p);
        let (v, w, z) = (&smp.dirs[0], &smp.dirs[1], &smp.dirs[2]);
        // A: column j by central differences of f.
        let exact = dag(ir, &EntryKind::JacU, x, p, &[]);
        let mut fd = Fd {
            value: vec![0.0; n * n],
            extra: vec![0.0; n * n],
        };
        for j in 0..n {
            let at = |s: f64| {
                let mut xs = x.clone();
                xs[j] += s;
                val_err(ir, &EntryKind::F, &xs, p, &[])
            };
            let col = richardson(at, step(x[j]));
            for (i, (d, e)) in col.value.iter().zip(&col.extra).enumerate() {
                fd.value[i + n * j] = *d;
                fd.extra[i + n * j] = *e;
            }
        }
        a_pts.push(FdPoint {
            x: x.clone(),
            p: p.clone(),
            exact,
            fd,
        });
        // f_alpha.
        if n_p > 0 {
            let exact = dag(ir, &EntryKind::JacP, x, p, &[]);
            let mut fd = Fd {
                value: vec![0.0; n * n_p],
                extra: vec![0.0; n * n_p],
            };
            for j in 0..n_p {
                let at = |s: f64| {
                    let mut ps = p.clone();
                    ps[j] += s;
                    val_err(ir, &EntryKind::F, x, &ps, &[])
                };
                let col = richardson(at, step(p[j]));
                for (i, (d, e)) in col.value.iter().zip(&col.extra).enumerate() {
                    fd.value[i + n * j] = *d;
                    fd.extra[i + n * j] = *e;
                }
            }
            fa_pts.push(FdPoint {
                x: x.clone(),
                p: p.clone(),
                exact,
                fd,
            });
        }
        // B(v, w) by differencing A v = Dir[U](v) along w.
        let vw: Vec<f64> = v.iter().chain(w).copied().collect();
        let exact_b = dag(ir, &u2, x, p, &vw);
        let fd_b = richardson(|s| val_err(ir, &u1, &shift(x, w, s), p, v), h_dir);
        b_pts.push(FdPoint {
            x: x.clone(),
            p: p.clone(),
            exact: exact_b,
            fd: fd_b,
        });
        // C(v, w, z) by differencing B(v, w) along z.
        let vwz: Vec<f64> = v.iter().chain(w).chain(z).copied().collect();
        let exact_c = dag(ir, &u3, x, p, &vwz);
        let fd_c = richardson(|s| val_err(ir, &u2, &shift(x, z, s), p, &vw), h_dir);
        c_pts.push(FdPoint {
            x: x.clone(),
            p: p.clone(),
            exact: exact_c,
            fd: fd_c,
        });
    }
    let mut out = vec![fd_report("A (FD)", FD_TOL_FIRST, skipped, a_pts)];
    if n_p > 0 {
        out.push(fd_report("f_alpha (FD)", FD_TOL_FIRST, skipped, fa_pts));
    }
    out.push(fd_report("B (FD)", FD_TOL_HIGHER, skipped, b_pts));
    out.push(fd_report("C (FD)", FD_TOL_HIGHER, skipped, c_pts));
    out
}

#[cfg(test)]
#[path = "check_tests.rs"]
mod tests;
