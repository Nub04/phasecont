//! Generic tree-walking interpreter, dual numbers and the f64 fallback evaluator.
//!
//! [`Interp`] walks the expanded AST of a checked model without going through the DAG, so
//! it is an independent check on `lower` and `eval`. It is generic over [`Scalar`]: `f64`
//! gives values, nested [`Dual`] gives derivatives of any order. `impl Scalar for f64`
//! calls the shared `math` functions and the `^` rule of the lowering (a constant integer
//! exponent with `|n| <= 16` becomes `powi` in the order documented on `Op::PowI`), so
//! values match `eval.rs` bit for bit on ordinary points. [`Fallback`] serves every
//! [`EntryKind`] in `f64`; it is the evaluator used where no JIT is available.
//!
//! Where the DAG folds `x + 0`, `x * 0` or `x * 1`, `eval.rs` can differ from this literal
//! walk on signed zeros and on dead inf or NaN terms. Finite nonzero points agree.

use crate::delays::{self, DelaySlot, Names, SlotKey};
use crate::entry::{EntryKind, Slot};
use crate::lower::IrModel;
use crate::math;
use pc_ode::{BinaryOp, Expr, Ident, Model, ModelInfo, Stmt, StmtKind, UnaryOp, Value};
use std::collections::{HashMap, HashSet};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Deepest expression or function nesting the walker follows, for a small scalar.
const MAX_DEPTH: usize = 1000;

/// Stack the recursive walk may use, in bytes. The Windows main thread has 1 MiB.
const STACK_BUDGET: usize = 768 * 1024;

/// Bytes of stack one nesting level costs: a fixed part plus a multiple of the scalar size
/// (each level holds several scalars in `ev`, `ev_inner`, `binary` and `call`). Measured
/// from stack addresses on a user-function chain: release 256 bytes for `f64` and 2344 for
/// `Dual^5`; debug 1110 and 7400. The constants round up.
#[cfg(debug_assertions)]
const LEVEL_FIXED: usize = 900;
#[cfg(debug_assertions)]
const LEVEL_PER_SCALAR: usize = 29;
#[cfg(not(debug_assertions))]
const LEVEL_FIXED: usize = 190;
#[cfg(not(debug_assertions))]
const LEVEL_PER_SCALAR: usize = 9;

/// Nesting limit for scalar type `S`: 1000 for `f64`, smaller for nested duals.
pub(crate) fn depth_limit<S>() -> usize {
    let per_level = LEVEL_FIXED + LEVEL_PER_SCALAR * std::mem::size_of::<S>();
    (STACK_BUDGET / per_level).clamp(1, MAX_DEPTH)
}

/// A number type the interpreter can compute with.
///
/// Every builtin is a method. Comparisons, logic and `if` use [`Scalar::value`], so a
/// dual number takes the branch its primal value takes.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn from_f64(v: f64) -> Self;
    /// NaN in the value and in every tangent.
    fn nan() -> Self;
    /// The primal value.
    fn value(self) -> f64;
    /// True when the value and every tangent are exactly zero.
    fn is_zero(self) -> bool;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log10(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn erf(self) -> Self;
    fn erfc(self) -> Self;
    fn lgamma(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn heav(self) -> Self;
    fn sign(self) -> Self;
    /// `atan2(self, x)`, with `self` the y argument.
    fn atan2(self, x: Self) -> Self;
    fn pow(self, y: Self) -> Self;
    /// XPPAUT `mod(self, y)`.
    fn modulo(self, y: Self) -> Self;
    /// `J_n(x)` with `self` the order `n`.
    fn besselj(self, x: Self) -> Self;
    /// `Y_n(x)` with `self` the order `n`.
    fn bessely(self, x: Self) -> Self;
    fn min(self, y: Self) -> Self;
    fn max(self, y: Self) -> Self;
    /// `psi^(m)(self)`.
    fn polygamma(self, m: u32) -> Self;
    /// Integer power in the multiplication order of `Op::PowI`.
    fn powi(self, n: i32) -> Self;
    /// Table lookup in a block laid out as in `math::table`.
    fn table(self, data: &[f64]) -> Self;
    /// Slope of the segment `table` uses.
    fn table_slope(self, data: &[f64]) -> Self;
}

impl Scalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn value(self) -> f64 {
        self
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn sin(self) -> Self {
        math::sin(self)
    }
    fn cos(self) -> Self {
        math::cos(self)
    }
    fn tan(self) -> Self {
        math::tan(self)
    }
    fn asin(self) -> Self {
        math::asin(self)
    }
    fn acos(self) -> Self {
        math::acos(self)
    }
    fn atan(self) -> Self {
        math::atan(self)
    }
    fn sinh(self) -> Self {
        math::sinh(self)
    }
    fn cosh(self) -> Self {
        math::cosh(self)
    }
    fn tanh(self) -> Self {
        math::tanh(self)
    }
    fn exp(self) -> Self {
        math::exp(self)
    }
    fn ln(self) -> Self {
        math::ln(self)
    }
    fn log10(self) -> Self {
        math::log10(self)
    }
    fn sqrt(self) -> Self {
        math::sqrt(self)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn erf(self) -> Self {
        math::erf(self)
    }
    fn erfc(self) -> Self {
        math::erfc(self)
    }
    fn lgamma(self) -> Self {
        math::lgamma(self)
    }
    fn floor(self) -> Self {
        math::floor(self)
    }
    fn ceil(self) -> Self {
        math::ceil(self)
    }
    fn heav(self) -> Self {
        math::heav(self)
    }
    fn sign(self) -> Self {
        math::sign(self)
    }
    fn atan2(self, x: Self) -> Self {
        math::atan2(self, x)
    }
    fn pow(self, y: Self) -> Self {
        math::pow(self, y)
    }
    fn modulo(self, y: Self) -> Self {
        math::pc_mod(self, y)
    }
    fn besselj(self, x: Self) -> Self {
        math::besselj(self, x)
    }
    fn bessely(self, x: Self) -> Self {
        math::bessely(self, x)
    }
    fn min(self, y: Self) -> Self {
        if self < y { self } else { y }
    }
    fn max(self, y: Self) -> Self {
        if self > y { self } else { y }
    }
    fn polygamma(self, m: u32) -> Self {
        math::polygamma(f64::from(m), self)
    }
    fn powi(self, n: i32) -> Self {
        let mut m = n.unsigned_abs();
        if m == 0 {
            return 1.0;
        }
        let mut base = self;
        let mut acc: Option<f64> = None;
        loop {
            if m & 1 == 1 {
                acc = Some(match acc {
                    None => base,
                    Some(a) => a * base,
                });
            }
            m >>= 1;
            if m == 0 {
                break;
            }
            base *= base;
        }
        let r = acc.unwrap_or(1.0);
        if n < 0 { 1.0 / r } else { r }
    }
    fn table(self, data: &[f64]) -> Self {
        math::table(data, self)
    }
    fn table_slope(self, data: &[f64]) -> Self {
        math::table_slope(data, self)
    }
}

/// First-order dual number `re + eps * e` over any scalar. Nesting `Dual<Dual<f64>>` and
/// so on gives higher derivatives, because every rule is written in `T`'s own operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<T> {
    pub re: T,
    pub eps: T,
}

impl<T: Scalar> Dual<T> {
    fn zero() -> T {
        T::from_f64(0.0)
    }

    fn one() -> T {
        T::from_f64(1.0)
    }

    /// `tangent * d()`, or `None` when the tangent is zero (so a zero tangent never meets
    /// an infinite or NaN partial).
    fn term(tangent: T, d: impl FnOnce() -> T) -> Option<T> {
        if tangent.is_zero() {
            None
        } else {
            Some(tangent * d())
        }
    }

    fn sum(a: Option<T>, b: Option<T>) -> T {
        match (a, b) {
            (Some(a), Some(b)) => a + b,
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => Self::zero(),
        }
    }

    /// Result with value `re` and tangent `self.eps * d()`.
    fn chain(self, re: T, d: impl FnOnce() -> T) -> Self {
        Dual {
            re,
            eps: Self::sum(Self::term(self.eps, d), None),
        }
    }

    /// Result with value `re` and tangent zero.
    fn flat(re: T) -> Self {
        Dual {
            re,
            eps: Self::zero(),
        }
    }
}

impl<T: Scalar> Add for Dual<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Dual {
            re: self.re + o.re,
            eps: self.eps + o.eps,
        }
    }
}

impl<T: Scalar> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Dual {
            re: self.re - o.re,
            eps: self.eps - o.eps,
        }
    }
}

impl<T: Scalar> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Dual {
            re: self.re * o.re,
            eps: Self::sum(Self::term(self.eps, || o.re), Self::term(o.eps, || self.re)),
        }
    }
}

impl<T: Scalar> Div for Dual<T> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let q = self.re / o.re;
        let ta = if self.eps.is_zero() {
            None
        } else {
            Some(self.eps / o.re)
        };
        Dual {
            re: q,
            eps: Self::sum(ta, Self::term(o.eps, || -(q / o.re))),
        }
    }
}

impl<T: Scalar> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Dual {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

impl<T: Scalar> Scalar for Dual<T> {
    fn from_f64(v: f64) -> Self {
        Self::flat(T::from_f64(v))
    }
    fn nan() -> Self {
        Dual {
            re: T::nan(),
            eps: T::nan(),
        }
    }
    fn value(self) -> f64 {
        self.re.value()
    }
    fn is_zero(self) -> bool {
        self.re.is_zero() && self.eps.is_zero()
    }
    fn sin(self) -> Self {
        self.chain(self.re.sin(), || self.re.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.re.cos(), || -self.re.sin())
    }
    fn tan(self) -> Self {
        let t = self.re.tan();
        self.chain(t, || Self::one() + t * t)
    }
    fn asin(self) -> Self {
        self.chain(self.re.asin(), || {
            Self::one() / (Self::one() - self.re * self.re).sqrt()
        })
    }
    fn acos(self) -> Self {
        self.chain(self.re.acos(), || {
            -(Self::one() / (Self::one() - self.re * self.re).sqrt())
        })
    }
    fn atan(self) -> Self {
        self.chain(self.re.atan(), || {
            Self::one() / (Self::one() + self.re * self.re)
        })
    }
    fn sinh(self) -> Self {
        self.chain(self.re.sinh(), || self.re.cosh())
    }
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), || self.re.sinh())
    }
    fn tanh(self) -> Self {
        let t = self.re.tanh();
        self.chain(t, || Self::one() - t * t)
    }
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, || self.re.exp())
    }
    fn ln(self) -> Self {
        self.chain(self.re.ln(), || Self::one() / self.re)
    }
    fn log10(self) -> Self {
        self.chain(self.re.log10(), || {
            Self::one() / (self.re * T::from_f64(std::f64::consts::LN_10))
        })
    }
    fn sqrt(self) -> Self {
        self.chain(self.re.sqrt(), || {
            Self::one() / (T::from_f64(2.0) * self.re.sqrt())
        })
    }
    fn abs(self) -> Self {
        self.chain(self.re.abs(), || self.re.sign())
    }
    fn erf(self) -> Self {
        self.chain(self.re.erf(), || erf_slope(self.re))
    }
    fn erfc(self) -> Self {
        self.chain(self.re.erfc(), || -erf_slope(self.re))
    }
    fn lgamma(self) -> Self {
        self.chain(self.re.lgamma(), || self.re.polygamma(0))
    }
    fn floor(self) -> Self {
        Self::flat(self.re.floor())
    }
    fn ceil(self) -> Self {
        Self::flat(self.re.ceil())
    }
    fn heav(self) -> Self {
        Self::flat(self.re.heav())
    }
    fn sign(self) -> Self {
        Self::flat(self.re.sign())
    }
    fn atan2(self, x: Self) -> Self {
        let (y, x) = (self, x);
        let d = || y.re * y.re + x.re * x.re;
        Dual {
            re: y.re.atan2(x.re),
            eps: Self::sum(
                Self::term(y.eps, || x.re / d()),
                Self::term(x.eps, || -(y.re / d())),
            ),
        }
    }
    fn pow(self, y: Self) -> Self {
        let x = self;
        let r = x.re.pow(y.re);
        Dual {
            re: r,
            eps: Self::sum(
                Self::term(x.eps, || y.re * x.re.pow(y.re - Self::one())),
                if r.value() == 0.0 {
                    None
                } else {
                    Self::term(y.eps, || r * x.re.ln())
                },
            ),
        }
    }
    fn modulo(self, y: Self) -> Self {
        let m = self.re.modulo(y.re);
        let tx = if self.eps.is_zero() {
            None
        } else {
            Some(self.eps)
        };
        Dual {
            re: m,
            eps: Self::sum(tx, Self::term(y.eps, || -((self.re - m) / y.re))),
        }
    }
    fn besselj(self, x: Self) -> Self {
        let n = self.re;
        Dual {
            re: n.besselj(x.re),
            eps: Self::sum(
                Self::term(x.eps, || {
                    (n_minus(n).besselj(x.re) - n_plus(n).besselj(x.re)) * T::from_f64(0.5)
                }),
                None,
            ),
        }
    }
    fn bessely(self, x: Self) -> Self {
        let n = self.re;
        Dual {
            re: n.bessely(x.re),
            eps: Self::sum(
                Self::term(x.eps, || {
                    (n_minus(n).bessely(x.re) - n_plus(n).bessely(x.re)) * T::from_f64(0.5)
                }),
                None,
            ),
        }
    }
    fn min(self, y: Self) -> Self {
        if self.value() < y.value() { self } else { y }
    }
    fn max(self, y: Self) -> Self {
        if self.value() > y.value() { self } else { y }
    }
    fn polygamma(self, m: u32) -> Self {
        self.chain(self.re.polygamma(m), || self.re.polygamma(m + 1))
    }
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::from_f64(1.0);
        }
        self.chain(self.re.powi(n), || {
            T::from_f64(f64::from(n)) * self.re.powi(n - 1)
        })
    }
    fn table(self, data: &[f64]) -> Self {
        self.chain(self.re.table(data), || self.re.table_slope(data))
    }
    fn table_slope(self, data: &[f64]) -> Self {
        Self::flat(self.re.table_slope(data))
    }
}

/// `2/sqrt(pi) * exp(-x^2)`, the derivative of `erf` (DLMF 7.10).
fn erf_slope<T: Scalar>(x: T) -> T {
    T::from_f64(std::f64::consts::FRAC_2_SQRT_PI) * (-(x * x)).exp()
}

/// `trunc(n) - 1` as a constant: DLMF 10.6.1 shifts the order `k = trunc(n)` that
/// `besselj` and `bessely` use. The order has no tangent.
fn n_minus<T: Scalar>(n: T) -> T {
    T::from_f64(n.value().trunc() - 1.0)
}

/// `trunc(n) + 1` as a constant, see [`n_minus`].
fn n_plus<T: Scalar>(n: T) -> T {
    T::from_f64(n.value().trunc() + 1.0)
}

/// A scalar built from nested duals by seeding directions, for directional derivatives.
trait Nest: Scalar {
    /// Value `u` whose tangent at nesting level `s` is `dirs[s]` (missing entries are 0).
    fn seed(u: f64, dirs: &[f64]) -> Self;
    /// The coefficient of `eps_0 eps_1 ... eps_{k-1}`.
    fn top(self) -> f64;
}

impl Nest for f64 {
    fn seed(u: f64, _dirs: &[f64]) -> Self {
        u
    }
    fn top(self) -> f64 {
        self
    }
}

impl<T: Nest> Nest for Dual<T> {
    fn seed(u: f64, dirs: &[f64]) -> Self {
        Dual {
            re: T::seed(u, dirs.get(1..).unwrap_or(&[])),
            eps: T::seed(dirs.first().copied().unwrap_or(0.0), &[]),
        }
    }
    fn top(self) -> f64 {
        self.eps.top()
    }
}

// ---------------------------------------------------------------------------

/// Owns copies of everything the walker reads, so it outlives the parsed model.
pub struct Interp {
    n: usize,
    n_p: usize,
    states: HashMap<String, usize>,
    params: HashMap<String, usize>,
    numbers: HashMap<String, f64>,
    /// Derived parameters and fixed quantities.
    named: HashMap<String, Expr>,
    funcs: HashMap<String, (Vec<String>, Expr)>,
    table_ids: HashMap<String, usize>,
    tables: Vec<Vec<f64>>,
    f: Vec<Expr>,
    aux: Vec<Expr>,
    /// Parameter index of the hidden slot of each delayed state and delay time. Empty
    /// unless the model was lowered with `lower_dde`.
    delay_slots: HashMap<SlotKey, usize>,
}

impl Names for Interp {
    fn state(&self, key: &str) -> Option<usize> {
        self.states.get(key).copied()
    }

    fn param(&self, key: &str) -> Option<usize> {
        self.params.get(key).copied()
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.numbers.get(key).copied()
    }
}

fn flatten<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a Stmt>) {
    for s in stmts {
        match &s.kind {
            StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => flatten(body, out),
            _ => out.push(s),
        }
    }
}

fn num(v: &Option<Value>) -> f64 {
    match v {
        Some(Value::Num { value, .. }) => *value,
        _ => 0.0,
    }
}

impl Interp {
    /// `model` and `info` are `Loaded::expanded` and `Loaded::info`; `ir` is the lowering
    /// of the same model. The tables come from `ir.dag.tables` (they were baked from the
    /// parameter defaults and initial values at lowering), and `ir` also fixes `n` and
    /// `n_p`.
    pub fn new(model: &Model, _info: &ModelInfo, ir: &IrModel) -> Self {
        let mut stmts = Vec::new();
        flatten(&model.statements, &mut stmts);
        let mut it = Interp {
            n: ir.n(),
            n_p: ir.n_p(),
            states: HashMap::new(),
            params: HashMap::new(),
            numbers: HashMap::new(),
            named: HashMap::new(),
            funcs: HashMap::new(),
            table_ids: HashMap::new(),
            tables: ir.dag.tables.clone(),
            f: Vec::new(),
            aux: Vec::new(),
            delay_slots: ir
                .delays
                .iter()
                .enumerate()
                .map(|(j, d)| (d.key(), ir.n_user_params + j))
                .collect(),
        };
        let mut n_tables = 0;
        for s in stmts {
            match &s.kind {
                StmtKind::Ode { name, rhs, .. } | StmtKind::Map { name, rhs } => {
                    it.states.insert(name.key.clone(), it.f.len());
                    it.f.push(rhs.clone());
                }
                StmtKind::Par { entries } => {
                    for e in entries {
                        let j = it.params.len();
                        it.params.insert(e.name.key.clone(), j);
                    }
                }
                StmtKind::Number { entries } => {
                    for e in entries {
                        it.numbers.insert(e.name.key.clone(), num(&e.value));
                    }
                }
                StmtKind::Derived { name, rhs } | StmtKind::Fixed { name, rhs } => {
                    it.named.insert(name.key.clone(), rhs.clone());
                }
                StmtKind::Aux { rhs, .. } => it.aux.push(rhs.clone()),
                StmtKind::Function { name, params, rhs } => {
                    let keys = params.iter().map(|p| p.key.clone()).collect();
                    it.funcs.insert(name.key.clone(), (keys, rhs.clone()));
                }
                StmtKind::Table { name, .. } => {
                    it.table_ids.insert(name.key.clone(), n_tables);
                    n_tables += 1;
                }
                _ => {}
            }
        }
        it
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn n_p(&self) -> usize {
        self.n_p
    }

    pub fn n_aux(&self) -> usize {
        self.aux.len()
    }

    /// Evaluates the state right sides into `out_f` and the `aux` quantities into
    /// `out_aux`. `x` has length n and `p` has length n_p + 1 (the last slot is `t`). A
    /// short input, a model nested past the depth limit or an unknown name gives NaN in
    /// the affected outputs; extra output slots are NaN.
    pub fn eval<S: Scalar>(&self, x: &[S], p: &[S], out_f: &mut [S], out_aux: &mut [S]) {
        let nan = S::nan();
        if x.len() < self.n || p.len() <= self.n_p {
            out_f.fill(nan);
            out_aux.fill(nan);
            return;
        }
        let mut cx = Ctx {
            it: self,
            x,
            p,
            frames: Vec::new(),
            cache: HashMap::new(),
            visiting: HashSet::new(),
            depth: 0,
            max_depth: depth_limit::<S>(),
            failed: false,
        };
        for (i, o) in out_f.iter_mut().enumerate() {
            *o = match self.f.get(i) {
                Some(e) => cx.ev(e).0,
                None => nan,
            };
        }
        for (i, o) in out_aux.iter_mut().enumerate() {
            *o = match self.aux.get(i) {
                Some(e) => cx.ev(e).0,
                None => nan,
            };
        }
        if cx.failed {
            out_f.fill(nan);
            out_aux.fill(nan);
        }
    }
}

/// A value and whether it is a compile-time constant. The lowering folds constant
/// subtrees, and its `^` rule looks at the folded exponent, so the walker tracks the same.
type V<S> = (S, bool);

struct Frame<'a, S> {
    params: &'a [String],
    args: Vec<V<S>>,
}

struct Ctx<'a, S: Scalar> {
    it: &'a Interp,
    x: &'a [S],
    p: &'a [S],
    frames: Vec<Frame<'a, S>>,
    /// Fixed and derived quantities, evaluated once per call in the global scope.
    cache: HashMap<&'a str, V<S>>,
    visiting: HashSet<&'a str>,
    depth: usize,
    max_depth: usize,
    failed: bool,
}

fn b2f(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

fn small_int(v: f64) -> Option<i32> {
    (v.fract() == 0.0 && v.abs() <= 16.0).then_some(v as i32)
}

impl<'a, S: Scalar> Ctx<'a, S> {
    fn nan() -> V<S> {
        (S::nan(), true)
    }

    fn ev(&mut self, e: &'a Expr) -> V<S> {
        if self.depth >= self.max_depth {
            self.failed = true;
            return Self::nan();
        }
        self.depth += 1;
        let r = self.ev_inner(e);
        self.depth -= 1;
        r
    }

    fn ev_inner(&mut self, e: &'a Expr) -> V<S> {
        match e {
            Expr::Num { value, .. } => (S::from_f64(*value), true),
            Expr::Name(id) => self.name(id),
            Expr::Index(_) | Expr::Error { .. } => Self::nan(),
            Expr::Unary { op, operand, .. } => self.unary(*op, operand),
            Expr::Binary { op, lhs, rhs, .. } => self.binary(*op, lhs, rhs),
            Expr::If {
                cond, then, else_, ..
            } => {
                let (c, cc) = self.ev(cond);
                let (v, vc) = if c.value() != 0.0 {
                    self.ev(then)
                } else {
                    self.ev(else_)
                };
                (v, cc && vc)
            }
            Expr::Call { name, args, .. } => self.call(name, args),
        }
    }

    #[inline(never)]
    fn unary(&mut self, op: UnaryOp, operand: &'a Expr) -> V<S> {
        let (a, c) = self.ev(operand);
        match op {
            UnaryOp::Neg => (-a, c),
            UnaryOp::Not => (S::from_f64(b2f(a.value() == 0.0)), c),
        }
    }

    #[inline(never)]
    fn binary(&mut self, op: BinaryOp, l: &'a Expr, r: &'a Expr) -> V<S> {
        let (a, ca) = self.ev(l);
        let (b, cb) = self.ev(r);
        let (x, y) = (a.value(), b.value());
        let flag = |t: bool| S::from_f64(b2f(t));
        let v = match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Pow | BinaryOp::StarStar => match small_int(y) {
                Some(n) if cb => a.powi(n),
                _ => a.pow(b),
            },
            BinaryOp::Lt => flag(x < y),
            BinaryOp::Gt => flag(x > y),
            BinaryOp::Le => flag(x <= y),
            BinaryOp::Ge => flag(x >= y),
            BinaryOp::Eq => flag(x == y),
            BinaryOp::And => flag(x != 0.0 && y != 0.0),
            BinaryOp::Or => flag(x != 0.0 || y != 0.0),
        };
        (v, ca && cb)
    }

    #[inline(never)]
    fn name(&mut self, id: &'a Ident) -> V<S> {
        let key = id.key.as_str();
        let it = self.it;
        if let Some(fr) = self.frames.last() {
            if let Some(pos) = fr.params.iter().position(|p| p == key) {
                return fr.args.get(pos).copied().unwrap_or_else(Self::nan);
            }
            if let Some(k) = key.strip_prefix("arg")
                && let Ok(k @ 1..=9) = k.parse::<usize>()
                && let Some(&v) = fr.args.get(k - 1)
            {
                return v;
            }
        }
        if key == "t" {
            return (self.p.get(it.n_p).copied().unwrap_or(S::nan()), false);
        }
        if key == "pi" {
            return (S::from_f64(std::f64::consts::PI), true);
        }
        if let Some(&i) = it.states.get(key) {
            return (self.x.get(i).copied().unwrap_or(S::nan()), false);
        }
        if let Some(&j) = it.params.get(key) {
            return (self.p.get(j).copied().unwrap_or(S::nan()), false);
        }
        if let Some(&v) = it.numbers.get(key) {
            return (S::from_f64(v), true);
        }
        if let Some((k, rhs)) = it.named.get_key_value(key) {
            return self.named(k.as_str(), rhs);
        }
        Self::nan()
    }

    /// A fixed quantity or derived parameter, evaluated in the global scope.
    fn named(&mut self, key: &'a str, rhs: &'a Expr) -> V<S> {
        if let Some(&v) = self.cache.get(key) {
            return v;
        }
        if !self.visiting.insert(key) {
            return Self::nan();
        }
        let saved = std::mem::take(&mut self.frames);
        let v = self.ev(rhs);
        self.frames = saved;
        self.visiting.remove(key);
        self.cache.insert(key, v);
        v
    }

    /// A `delay` call is a read of its hidden parameter slot, by the rule `lower_dde` uses.
    fn delay(&mut self, args: &'a [Expr]) -> V<S> {
        let it = self.it;
        let formals = self.frames.last().map_or(&[][..], |f| f.params);
        let idx = delays::classify(args, formals, it)
            .ok()
            .and_then(|(state, time)| it.delay_slots.get(&DelaySlot { state, time }.key()));
        match idx {
            Some(&j) => (self.p.get(j).copied().unwrap_or(S::nan()), false),
            None => Self::nan(),
        }
    }

    #[inline(never)]
    fn call(&mut self, name: &'a Ident, args: &'a [Expr]) -> V<S> {
        let key = name.key.as_str();
        let it = self.it;
        if let Some((params, body)) = it.funcs.get(key) {
            if params.len() != args.len() {
                return Self::nan();
            }
            let vals: Vec<V<S>> = args.iter().map(|a| self.ev(a)).collect();
            if self.frames.len() >= self.max_depth {
                self.failed = true;
                return Self::nan();
            }
            self.frames.push(Frame { params, args: vals });
            let v = self.ev(body);
            self.frames.pop();
            return v;
        }
        if let Some(&id) = it.table_ids.get(key) {
            let [a] = args else {
                return Self::nan();
            };
            let (x, c) = self.ev(a);
            let data = it.tables.get(id).map(Vec::as_slice).unwrap_or(&[]);
            return (x.table(data), c);
        }
        if key == "delay" {
            return self.delay(args);
        }
        match args {
            [a] => {
                let (x, c) = self.ev(a);
                match call1(key, x) {
                    Some(r) => (r, c),
                    None => Self::nan(),
                }
            }
            [a, b] => {
                let (x, ca) = self.ev(a);
                let (y, cb) = self.ev(b);
                match call2(key, x, y) {
                    Some(r) => (r, ca && cb),
                    None => Self::nan(),
                }
            }
            _ => Self::nan(),
        }
    }
}

fn call1<S: Scalar>(key: &str, x: S) -> Option<S> {
    Some(match key {
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "exp" => x.exp(),
        "ln" | "log" => x.ln(),
        "log10" => x.log10(),
        "sqrt" => x.sqrt(),
        "abs" => x.abs(),
        "erf" => x.erf(),
        "erfc" => x.erfc(),
        "lgamma" => x.lgamma(),
        "flr" => x.floor(),
        "ceil" => x.ceil(),
        "heav" => x.heav(),
        "sign" => x.sign(),
        _ => return None,
    })
}

fn call2<S: Scalar>(key: &str, x: S, y: S) -> Option<S> {
    Some(match key {
        "atan2" => x.atan2(y),
        "mod" => x.modulo(y),
        "min" => x.min(y),
        "max" => x.max(y),
        "besselj" => x.besselj(y),
        "bessely" => x.bessely(y),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------

/// The f64 evaluator for every [`EntryKind`]: values from `Interp` over `f64`, Jacobian
/// columns from `Dual<f64>` and directional derivatives from nested duals (up to five
/// slots). It is the evaluator for hosts without a JIT.
pub struct Fallback {
    interp: Interp,
    n: usize,
    n_p: usize,
    n_aux: usize,
}

impl Fallback {
    pub fn new(model: &Model, info: &ModelInfo, ir: &IrModel) -> Self {
        Self::from_interp(Interp::new(model, info, ir))
    }

    pub fn from_interp(interp: Interp) -> Self {
        Fallback {
            n: interp.n(),
            n_p: interp.n_p(),
            n_aux: interp.n_aux(),
            interp,
        }
    }

    /// Evaluates one entry. Inputs and outputs follow the contract in [`crate::entry`].
    /// A wrong input length or a bad slot count fills the outputs with NaN.
    pub fn eval(&self, kind: &EntryKind, x: &[f64], p: &[f64], v: &[f64], out: &mut [f64]) {
        if x.len() < self.n || p.len() <= self.n_p {
            out.fill(f64::NAN);
            return;
        }
        match kind {
            EntryKind::F => {
                let mut aux = vec![0.0; self.n_aux];
                let mut f = vec![0.0; self.n];
                self.interp.eval(x, p, &mut f, &mut aux);
                copy_out(out, &f);
            }
            EntryKind::Aux => {
                let mut aux = vec![0.0; self.n_aux];
                let mut f = vec![0.0; self.n];
                self.interp.eval(x, p, &mut f, &mut aux);
                copy_out(out, &aux);
            }
            EntryKind::JacU => self.jac(x, p, out, self.n, true),
            EntryKind::JacP => self.jac(x, p, out, self.n_p, false),
            EntryKind::Dir(slots) => match slots.len() {
                1 => self.dir::<Dual<f64>>(slots, x, p, v, out),
                2 => self.dir::<Dual<Dual<f64>>>(slots, x, p, v, out),
                3 => self.dir::<Dual<Dual<Dual<f64>>>>(slots, x, p, v, out),
                4 => self.dir::<Dual<Dual<Dual<Dual<f64>>>>>(slots, x, p, v, out),
                5 => self.dir::<Dual<Dual<Dual<Dual<Dual<f64>>>>>>(slots, x, p, v, out),
                _ => out.fill(f64::NAN),
            },
        }
    }

    /// Columns of the Jacobian with respect to the states (`wrt_u`) or the parameters.
    fn jac(&self, x: &[f64], p: &[f64], out: &mut [f64], cols: usize, wrt_u: bool) {
        out.fill(0.0);
        let c = |v: f64| Dual { re: v, eps: 0.0 };
        let mut xs: Vec<Dual<f64>> = x.iter().take(self.n).map(|&v| c(v)).collect();
        let mut ps: Vec<Dual<f64>> = p.iter().take(self.n_p + 1).map(|&v| c(v)).collect();
        let mut f = vec![c(0.0); self.n];
        let mut aux = vec![c(0.0); self.n_aux];
        for j in 0..cols {
            let slot = if wrt_u { xs.get_mut(j) } else { ps.get_mut(j) };
            if let Some(s) = slot {
                s.eps = 1.0;
            }
            self.interp.eval(&xs, &ps, &mut f, &mut aux);
            let slot = if wrt_u { xs.get_mut(j) } else { ps.get_mut(j) };
            if let Some(s) = slot {
                s.eps = 0.0;
            }
            for (i, fi) in f.iter().enumerate() {
                if let Some(o) = out.get_mut(i + self.n * j) {
                    *o = fi.eps;
                }
            }
        }
    }

    fn dir<S: Nest>(&self, slots: &[Slot], x: &[f64], p: &[f64], v: &[f64], out: &mut [f64]) {
        let (n, n_p) = (self.n, self.n_p);
        let total: usize = slots.iter().map(|s| s.len(n, n_p)).sum();
        if v.len() < total {
            out.fill(f64::NAN);
            return;
        }
        // dirs[i][s]: component of slot s's direction along variable i (u, then p).
        let mut dirs = vec![[0.0f64; 5]; n + n_p];
        let mut off = 0;
        for (s, slot) in slots.iter().enumerate() {
            let (u_len, p_len) = match slot {
                Slot::U => (n, 0),
                Slot::P => (0, n_p),
                Slot::Full => (n, n_p),
            };
            for (i, d) in dirs.iter_mut().enumerate() {
                let pos = if i < n {
                    (u_len > 0).then_some(off + i)
                } else {
                    (p_len > 0).then_some(off + u_len + (i - n))
                };
                if let Some(q) = pos {
                    d[s] = v.get(q).copied().unwrap_or(f64::NAN);
                }
            }
            off += u_len + p_len;
        }
        let k = slots.len();
        let seed = |u: f64, i: usize| S::seed(u, dirs.get(i).map_or(&[][..], |d| &d[..k]));
        let xs: Vec<S> = x
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &u)| seed(u, i))
            .collect();
        let mut ps: Vec<S> = p
            .iter()
            .take(n_p)
            .enumerate()
            .map(|(j, &u)| seed(u, n + j))
            .collect();
        ps.push(S::seed(p.get(n_p).copied().unwrap_or(f64::NAN), &[]));
        let mut f = vec![S::from_f64(0.0); n];
        let mut aux = vec![S::from_f64(0.0); self.n_aux];
        self.interp.eval(&xs, &ps, &mut f, &mut aux);
        let vals: Vec<f64> = f.iter().map(|fi| fi.top()).collect();
        copy_out(out, &vals);
    }
}

/// Copies `src` into `out`; extra `out` slots become NaN.
fn copy_out(out: &mut [f64], src: &[f64]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = src.get(i).copied().unwrap_or(f64::NAN);
    }
}

#[cfg(test)]
#[path = "interp_tests.rs"]
mod tests;
