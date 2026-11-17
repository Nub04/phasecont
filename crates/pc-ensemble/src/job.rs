//! The job a run takes, the result it gives back and the errors in between.
//!
//! Nothing here has a default. A caller states the precision, both ends of the time span,
//! the tolerances, the step limit and the bound, so a number that changes a result never
//! comes from this crate.

use core::fmt;

use pc_ir::IrModel;

use crate::device::Backend;
use crate::real::Real;

/// The scalar type the integration runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Single,
    Double,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::Single => "single",
            Precision::Double => "double",
        }
    }
}

/// How the device compiler may round the arithmetic of a GPU kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Math {
    /// Every operation rounds once, as written. Results are bit for bit the same on every
    /// backend and in every run.
    Exact,
    /// The compiler may fuse a multiply and an add into one fused multiply-add. Only a CUDA
    /// double precision run takes it ([`check_math`]). The bits differ from an exact run.
    Fma,
}

impl Math {
    pub fn name(self) -> &'static str {
        match self {
            Math::Exact => "exact",
            Math::Fma => "fma",
        }
    }

    pub fn from_name(name: &str) -> Option<Math> {
        [Math::Exact, Math::Fma]
            .into_iter()
            .find(|m| m.name() == name)
    }
}

/// Refuses a math tier the backend and precision do not take. Only CUDA in double precision
/// takes [`Math::Fma`]: it is the one case where the fused tier measured faster, and every
/// other backend refuses it instead of running exact under an fma label.
pub fn check_math(backend: Backend, precision: Precision, math: Math) -> Result<(), EnsembleError> {
    match (math, backend, precision) {
        (Math::Exact, _, _) | (Math::Fma, Backend::Cuda, Precision::Double) => Ok(()),
        (Math::Fma, _, _) => Err(EnsembleError::BadJob(format!(
            "math fma runs only on cuda in double precision, not on {} in {} precision; use math exact",
            backend.name(),
            precision.name()
        ))),
    }
}

/// Parameter values of the members.
#[derive(Clone, Copy, Debug)]
pub enum Params<'a> {
    /// One set of `n_p` values for every member.
    Shared(&'a [f64]),
    /// `members * n_p` values, member-major.
    PerMember(&'a [f64]),
}

/// What a run records.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    /// The state at `t1` only.
    Final,
    /// The state at each of these times. The times are strictly increasing, lie in
    /// `(t0, t1]` and the last one is `t1`.
    Samples(Vec<f64>),
}

/// The integration method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Classical fourth order Runge-Kutta. `dt` is the largest step: the span between two
    /// output times is cut into the fewest equal steps that are not longer than `dt`.
    Rk4 { dt: f64 },
    /// Dormand-Prince 5(4) with the controller of `pc_engine::integrate`, one step size
    /// per member. `hmax` may be infinite.
    Dp5 { rtol: f64, atol: f64, hmax: f64 },
    /// Iteration of a map, x(k+1) = F(x(k), p, t = k), one call of F per iterate (SPEC
    /// section 8aa). The times are whole numbers.
    Iter,
}

/// A native right-hand side of a model, compiled by the caller (pc-jit), for CPU runs. A run
/// of the same precision calls it in place of the [`crate::tape::CpuTape`] interpreter; a run
/// of the other precision ignores it.
///
/// The function reads `x` (`n` values) and `p` (`n_p` parameters, then `t`) and writes `n`
/// values to `out`, which it must not alias. It gives the bits of the tape of its precision,
/// with no fused multiply-add.
#[derive(Clone, Copy, Debug)]
pub struct NativeRhs(pub(crate) NativeFn);

/// The function behind a [`NativeRhs`], by precision.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeFn {
    Double(unsafe extern "C" fn(*const f64, *const f64, *mut f64)),
    Single(unsafe extern "C" fn(*const f32, *const f32, *mut f32)),
}

/// The largest stack frame, in bytes, that a caller may hand to [`NativeRhs::new`]. A rayon
/// worker has a 2 MiB stack and the integrator's own frames sit below the call, so the limit
/// leaves most of it free. A caller whose compiled `F` needs more passes no native function.
pub const NATIVE_FRAME_LIMIT: u32 = 256 * 1024;

impl NativeRhs {
    /// Wraps `f`, a double precision `F`.
    ///
    /// # Safety
    /// `f` must be the `F` entry of this job's model, with exactly `n` outputs. It must give
    /// the same bits as the `f64` tape for every input, and be callable from any thread at the
    /// same time. Its stack frame must fit a rayon worker's 2 MiB stack. The code behind it
    /// must stay valid until every run that holds this value has returned.
    pub unsafe fn new(f: unsafe extern "C" fn(*const f64, *const f64, *mut f64)) -> Self {
        NativeRhs(NativeFn::Double(f))
    }

    /// Wraps `f`, a single precision `F`.
    ///
    /// # Safety
    /// The contract of [`NativeRhs::new`], with the `f32` tape in place of the `f64` one.
    pub unsafe fn new_f32(f: unsafe extern "C" fn(*const f32, *const f32, *mut f32)) -> Self {
        NativeRhs(NativeFn::Single(f))
    }

    /// The precision the function was compiled for.
    pub fn precision(self) -> Precision {
        match self.0 {
            NativeFn::Double(_) => Precision::Double,
            NativeFn::Single(_) => Precision::Single,
        }
    }

    /// The double precision function, or `None` for a single precision one.
    pub(crate) fn double(self) -> Option<unsafe extern "C" fn(*const f64, *const f64, *mut f64)> {
        match self.0 {
            NativeFn::Double(f) => Some(f),
            NativeFn::Single(_) => None,
        }
    }

    /// The single precision function, or `None` for a double precision one.
    pub(crate) fn single(self) -> Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32)> {
        match self.0 {
            NativeFn::Single(f) => Some(f),
            NativeFn::Double(_) => None,
        }
    }
}

#[derive(Clone)]
pub struct Job<'a> {
    pub model: &'a IrModel,
    pub members: usize,
    /// `members * n` values, member-major.
    pub x0: &'a [f64],
    pub params: Params<'a>,
    pub t0: f64,
    pub t1: f64,
    pub output: Output,
    pub method: Method,
    /// A member stops with [`Status::Bounds`] when a state component exceeds this in
    /// magnitude. Infinity turns the test off.
    pub bounds: f64,
    /// Accepted steps (RK4: steps) per member before [`Status::MaxSteps`].
    pub max_steps: u32,
    pub precision: Precision,
    /// The rounding the device compiler may use. [`check_math`] says which backends take
    /// [`Math::Fma`]; [`run_on`](crate::run::run_on) refuses the rest.
    pub math: Math,
    /// A compiled right-hand side for a CPU run of its precision. A run of the other precision
    /// and GPU runs ignore it. `None` runs the tape interpreter, which gives the same bits.
    pub native: Option<NativeRhs>,
}

/// How a member ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Reached `t1`.
    Ok = 0,
    /// A state component exceeded `bounds`.
    Bounds = 1,
    /// The adaptive step fell below the roundoff limit.
    StepTooSmall = 2,
    /// `max_steps` steps were taken before `t1`.
    MaxSteps = 3,
    /// A state component or the error estimate was NaN or infinite.
    NonFinite = 4,
    /// Not finished yet. Never appears in a finished result.
    Running = 5,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Bounds => "bounds",
            Status::StepTooSmall => "step_too_small",
            Status::MaxSteps => "max_steps",
            Status::NonFinite => "nonfinite",
            Status::Running => "running",
        }
    }

    pub fn code(self) -> u8 {
        self as u8
    }

    /// The status of a code written by a kernel, `None` for a code no status has.
    pub fn from_code(code: u32) -> Option<Status> {
        match code {
            0 => Some(Status::Ok),
            1 => Some(Status::Bounds),
            2 => Some(Status::StepTooSmall),
            3 => Some(Status::MaxSteps),
            4 => Some(Status::NonFinite),
            5 => Some(Status::Running),
            _ => None,
        }
    }
}

/// Where and how a result was computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    /// `"cpu"`, `"cuda"`, `"opencl"` or `"metal"`.
    pub backend: String,
    pub device: String,
    pub driver: String,
    pub precision: Precision,
    pub math: Math,
}

impl Label {
    /// True when the numbers carry single precision rounding, so they may not feed
    /// continuation, normal forms or Floquet multipliers before a double precision rerun.
    pub fn single_precision(&self) -> bool {
        self.precision == Precision::Single
    }

    /// True when the device compiler fused multiplies and adds, so the bits differ from an
    /// exact run and the numbers may not feed continuation, normal forms or Floquet
    /// multipliers before an exact rerun.
    pub fn fused(&self) -> bool {
        self.math == Math::Fma
    }
}

#[derive(Clone, Debug)]
pub struct EnsembleResult {
    /// States per member.
    pub n: usize,
    pub members: usize,
    /// Output times.
    pub times: Vec<f64>,
    /// `members * times.len() * n` values. A sample a member did not reach is NaN.
    pub states: Vec<f64>,
    pub status: Vec<Status>,
    /// The time each member reached.
    pub t_reached: Vec<f64>,
    pub accepted: Vec<u32>,
    pub rejected: Vec<u32>,
    pub label: Label,
}

impl EnsembleResult {
    /// The state of `member` at output time `sample`.
    pub fn state(&self, member: usize, sample: usize) -> &[f64] {
        let at = (member * self.times.len() + sample) * self.n;
        &self.states[at..at + self.n]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EnsembleError {
    /// An input is wrong. The text names it.
    BadJob(String),
    /// The relative tolerance is below what the precision resolves.
    ToleranceTooTight(String),
    /// The model uses a construct the ensemble kernels cannot run. The text names it.
    Unsupported(String),
    /// The backend cannot start: a library is missing, the driver is too old or there is no
    /// device. The text says which.
    GpuUnavailable(String),
    /// The device compiler rejected the kernel, or the compiled kernel did not load. The text
    /// carries the compiler log when there is one.
    GpuCompile(String),
    /// A copy, a launch or a synchronize failed on the device. The text names the step.
    GpuLaunch(String),
    /// The driver stopped a launch that ran too long (the Windows display watchdog).
    GpuTimeout(String),
    Cancelled,
}

impl fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnsembleError::BadJob(m) => write!(f, "bad ensemble job: {m}"),
            EnsembleError::ToleranceTooTight(m) => write!(f, "tolerance too tight: {m}"),
            EnsembleError::Unsupported(m) => write!(f, "unsupported model: {m}"),
            EnsembleError::GpuUnavailable(m) => write!(f, "GPU unavailable: {m}"),
            EnsembleError::GpuCompile(m) => write!(f, "GPU compile failed: {m}"),
            EnsembleError::GpuLaunch(m) => write!(f, "GPU launch failed: {m}"),
            EnsembleError::GpuTimeout(m) => write!(f, "GPU launch timed out: {m}"),
            EnsembleError::Cancelled => write!(f, "ensemble run cancelled"),
        }
    }
}

impl std::error::Error for EnsembleError {}

/// The smallest relative tolerance `dopri5` accepts is `13 * uround`.
pub const TOLERANCE_FACTOR: f64 = 13.0;

/// A job that passed [`validate`]: the sample times are spelled out and the sizes agree.
#[derive(Clone, Debug)]
pub struct Checked {
    pub n: usize,
    pub n_p: usize,
    pub times: Vec<f64>,
}

fn bad<T>(m: impl Into<String>) -> Result<T, EnsembleError> {
    Err(EnsembleError::BadJob(m.into()))
}

/// Checks every input of `job` and returns the output times.
pub fn validate(job: &Job) -> Result<Checked, EnsembleError> {
    let n = job.model.n();
    let n_p = job.model.n_p();
    match (job.model.kind, job.method) {
        (pc_ode::ModelKind::None, _) => {
            return Err(EnsembleError::Unsupported(
                "the model has no ODE and no map".into(),
            ));
        }
        (pc_ode::ModelKind::Map, Method::Rk4 { .. } | Method::Dp5 { .. }) => {
            return Err(EnsembleError::Unsupported(
                "the model is a map; rk4 and dp5 integrate an ODE, so run a map with method iter"
                    .into(),
            ));
        }
        (pc_ode::ModelKind::Ode, Method::Iter) => {
            return bad("method iter iterates a map, and the model is an ODE; use rk4 or dp5");
        }
        (pc_ode::ModelKind::Ode, _) | (pc_ode::ModelKind::Map, Method::Iter) => {}
    }
    if n == 0 {
        return bad("the model has no states");
    }
    if job.members == 0 {
        return bad("members must be at least 1");
    }
    if job.x0.len() != job.members * n {
        return bad(format!(
            "x0 has {} values, expected {} members x {n} states",
            job.x0.len(),
            job.members
        ));
    }
    let want_p = match job.params {
        Params::Shared(p) => (p.len(), n_p),
        Params::PerMember(p) => (p.len(), job.members * n_p),
    };
    if want_p.0 != want_p.1 {
        return bad(format!(
            "params has {} values, expected {}",
            want_p.0, want_p.1
        ));
    }
    if !(job.t0.is_finite() && job.t1.is_finite()) {
        return bad("t0 and t1 must be finite");
    }
    if job.t1 <= job.t0 {
        return bad("t1 must be greater than t0; the ensemble engine only integrates forward");
    }
    if job.bounds.is_nan() || job.bounds <= 0.0 {
        return bad("bounds must be positive (infinity turns the test off)");
    }
    if job.max_steps == 0 {
        return bad("max_steps must be at least 1");
    }
    match job.precision {
        Precision::Single => check_inputs::<f32>(job, n, n_p)?,
        Precision::Double => check_inputs::<f64>(job, n, n_p)?,
    }
    let times = match &job.output {
        Output::Final => vec![job.t1],
        Output::Samples(t) => {
            if t.is_empty() {
                return bad("the sample list is empty");
            }
            let mut prev = job.t0;
            for (i, &s) in t.iter().enumerate() {
                if !s.is_finite() || s <= prev || s > job.t1 {
                    return bad(format!(
                        "sample time {i} ({s}) must be finite, greater than the one before it and at most t1"
                    ));
                }
                prev = s;
            }
            if prev != job.t1 {
                return bad("the last sample time must be t1");
            }
            t.clone()
        }
    };
    match job.precision {
        Precision::Single => check_times::<f32>(job.t0, &times)?,
        Precision::Double => check_times::<f64>(job.t0, &times)?,
    }
    match job.method {
        Method::Rk4 { dt } => {
            if !(dt.is_finite() && dt > 0.0) {
                return bad(format!("dt must be positive and finite, got {dt}"));
            }
            // The step counters of a kernel are 32 bit, and in single precision the step
            // index is exact only up to 2^24.
            let cap = match job.precision {
                Precision::Single => 16_777_216.0,
                Precision::Double => f64::from(u32::MAX),
            };
            let mut a = job.t0;
            for &b in &times {
                let m = ((b - a) / dt).ceil().max(1.0);
                if m > cap || m.is_nan() {
                    return bad(format!(
                        "dt {dt:e} needs {m:e} steps between {a} and {b}, over the step limit of {cap:e} per sample interval in {} precision",
                        job.precision.name()
                    ));
                }
                a = b;
            }
        }
        Method::Dp5 { rtol, atol, hmax } => {
            if !(rtol.is_finite() && rtol > 0.0) {
                return bad(format!("rtol must be positive and finite, got {rtol}"));
            }
            if !(atol.is_finite() && atol > 0.0) {
                return bad(format!("atol must be positive and finite, got {atol}"));
            }
            if hmax.is_nan() || hmax <= 0.0 {
                return bad(format!("hmax must be positive, got {hmax}"));
            }
            let uround = match job.precision {
                Precision::Single => <f32 as Real>::UROUND_F64,
                Precision::Double => <f64 as Real>::UROUND_F64,
            };
            let least = TOLERANCE_FACTOR * uround;
            if rtol < least {
                return Err(EnsembleError::ToleranceTooTight(format!(
                    "rtol {rtol:e} is below {least:e}, 13 times the unit roundoff of {} precision",
                    job.precision.name()
                )));
            }
        }
        Method::Iter => {
            // The iterate index of a segment is a 32-bit counter, and `a + (real)i` must be
            // exact in the job precision: times up to 2^24 and 2^24 iterates per segment in
            // single precision, times up to 2^53 and u32::MAX iterates in double precision.
            let (limit, cap) = match job.precision {
                Precision::Single => (16_777_216.0, 16_777_216.0),
                Precision::Double => (9_007_199_254_740_992.0, f64::from(u32::MAX)),
            };
            for &v in std::iter::once(&job.t0).chain(&times) {
                if v.fract() != 0.0 {
                    return bad(format!("method iter needs whole-number times, got {v}"));
                }
            }
            for &v in std::iter::once(&job.t0).chain(&times) {
                if v.abs() > limit {
                    return bad(format!(
                        "method iter in {} precision needs times of magnitude at most {limit}, got {v}",
                        job.precision.name()
                    ));
                }
            }
            let mut a = job.t0;
            for &b in &times {
                let m = b - a;
                if m > cap {
                    return bad(format!(
                        "method iter makes {m} iterates between {a} and {b}, over the limit of {cap} per sample interval in {} precision",
                        job.precision.name()
                    ));
                }
                a = b;
            }
        }
    }
    Ok(Checked { n, n_p, times })
}

/// Every input must be finite and must stay finite after rounding to `T`.
fn check_inputs<T: Real>(job: &Job, n: usize, n_p: usize) -> Result<(), EnsembleError> {
    for (i, &v) in job.x0.iter().enumerate() {
        if !T::from_f64(v).is_finite() {
            return bad(format!(
                "x0[{}] of member {} ({v}) is not finite in {} precision",
                i % n,
                i / n,
                T::NAME
            ));
        }
    }
    let p = match job.params {
        Params::Shared(p) | Params::PerMember(p) => p,
    };
    for (i, &v) in p.iter().enumerate() {
        if !T::from_f64(v).is_finite() {
            return bad(format!(
                "parameter {} ({v}) is not finite in {} precision",
                i % n_p.max(1),
                T::NAME
            ));
        }
    }
    for (name, v) in [("t0", job.t0), ("t1", job.t1)] {
        if !T::from_f64(v).is_finite() {
            return bad(format!("{name} is not finite in {} precision", T::NAME));
        }
    }
    Ok(())
}

/// The output times must stay strictly increasing after rounding to `T`.
fn check_times<T: Real>(t0: f64, times: &[f64]) -> Result<(), EnsembleError> {
    let mut prev = T::from_f64(t0);
    for (i, &s) in times.iter().enumerate() {
        let r = T::from_f64(s);
        // A NaN sample time is refused here too, since it compares as neither greater nor less.
        if r.partial_cmp(&prev) != Some(std::cmp::Ordering::Greater) {
            return bad(format!(
                "sample time {i} ({s}) is not later than the one before it in {} precision",
                T::NAME
            ));
        }
        prev = r;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_cuda_double_takes_fma() {
        for backend in Backend::ALL {
            for precision in [Precision::Single, Precision::Double] {
                assert_eq!(check_math(backend, precision, Math::Exact), Ok(()));
                let fma = check_math(backend, precision, Math::Fma);
                if (backend, precision) == (Backend::Cuda, Precision::Double) {
                    assert_eq!(fma, Ok(()));
                } else {
                    assert!(matches!(fma, Err(EnsembleError::BadJob(_))), "{fma:?}");
                }
            }
        }
    }

    #[test]
    fn math_names_round_trip() {
        for m in [Math::Exact, Math::Fma] {
            assert_eq!(Math::from_name(m.name()), Some(m));
        }
        assert_eq!(Math::from_name("fast"), None);
    }
}
