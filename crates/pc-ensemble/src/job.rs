//! The job a run takes, the result it gives back and the errors in between.
//!
//! Nothing here has a default. A caller states the precision, both ends of the time span,
//! the tolerances, the step limit and the bound, so a number that changes a result never
//! comes from this crate.

use core::fmt;

use pc_ir::IrModel;

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
}

/// Where and how a result was computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    /// `"cpu"`, `"cuda"`, `"opencl"` or `"metal"`.
    pub backend: String,
    pub device: String,
    pub driver: String,
    pub precision: Precision,
}

impl Label {
    /// True when the numbers carry single precision rounding, so they may not feed
    /// continuation, normal forms or Floquet multipliers before a double precision rerun.
    pub fn single_precision(&self) -> bool {
        self.precision == Precision::Single
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
    Cancelled,
}

impl fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnsembleError::BadJob(m) => write!(f, "bad ensemble job: {m}"),
            EnsembleError::ToleranceTooTight(m) => write!(f, "tolerance too tight: {m}"),
            EnsembleError::Unsupported(m) => write!(f, "unsupported model: {m}"),
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
    if job.model.kind != pc_ode::ModelKind::Ode {
        return Err(EnsembleError::Unsupported(
            "the model is not an ODE system".into(),
        ));
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
