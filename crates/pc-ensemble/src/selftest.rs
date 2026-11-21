//! The self-test behind `pc-cli gpu selftest` and, later, `phasecont.gpuSelfTest`.
//!
//! It runs a few small ensembles through a backend and checks them against closed forms
//! and against themselves. The same checks run on every backend: a backend is a function
//! from a [`Job`] to an [`EnsembleResult`].
//!
//! * `analytic-rk4` and `analytic-dp5`: decay, a harmonic oscillator and logistic growth
//!   against their exact solutions, error scaled by `1 + |exact|`.
//! * `iterate`: the map `x(t+1) = a*x + t` with the method `iter` against a loop in `f64`,
//!   error scaled by `1 + |exact|`. It checks the time a backend passes to the map.
//! * `determinism`: the same job twice, one member alone, and the members in reverse order
//!   must give the same bits. The number of values that differ is the error, the limit is 0.
//!   The jobs are RK4 and Dormand-Prince on a model of `+ - * /` and the Henon map with `iter`.
//! * `recursion`: `x' = -s(k)*x/k` with the recursive `s(n) = n` under RK4 against
//!   `x0*exp(-t)`, at depths up to 4,096 frames, error scaled by `1 + |exact|`. One more
//!   member asks for exactly the frames of the backend's cap and must end with
//!   `recursion_depth` while the others finish; if it does not, the error is infinite. The
//!   limit is that of `analytic-rk4`.
//! * Bit-exactness: for a backend that is not the CPU reference, fixed-step RK4 on a model
//!   of `+ - * /` only and the Henon map with `iter` must match the CPU backend in the same
//!   precision bit for bit. The CPU backend is the reference, so its report says `None`.

use std::sync::atomic::AtomicBool;

use pc_ir::IrModel;

use crate::cpu::run_cpu;
use crate::device::{Backend, DeviceInfo, find_device};
use crate::job::{
    EnsembleError, EnsembleResult, Job, Label, Math, Method, Output, Params, Precision, Status,
    check_math,
};
use crate::run::run_on;

/// A backend as the checks see it.
pub type Runner<'a> = &'a dyn Fn(&Job) -> Result<EnsembleResult, EnsembleError>;

/// The device a self-test runs on.
#[derive(Clone, Copy, Debug)]
pub struct Selected {
    pub backend: Backend,
    pub device: usize,
    pub precision: Precision,
}

#[derive(Debug)]
pub enum SelfTestError {
    /// The backend is not built in, or has no device with this index. Not a pass.
    NoDevice(String),
    /// The backend refused or failed a job.
    Run(EnsembleError),
    /// The backend refused or failed a job of the fma tier, on a device that takes it.
    Fma(EnsembleError),
}

impl std::fmt::Display for SelfTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelfTestError::NoDevice(m) => f.write_str(m),
            SelfTestError::Run(e) => write!(f, "the backend failed a self-test job: {e}"),
            SelfTestError::Fma(e) => {
                write!(f, "the backend failed a self-test job of the fma tier: {e}")
            }
        }
    }
}

impl std::error::Error for SelfTestError {}

impl From<EnsembleError> for SelfTestError {
    fn from(e: EnsembleError) -> Self {
        SelfTestError::Run(e)
    }
}

/// One group of checks and its worst error.
#[derive(Clone, Debug, PartialEq)]
pub struct Tier {
    pub name: &'static str,
    pub max_error: f64,
    pub limit: f64,
}

impl Tier {
    pub fn pass(&self) -> bool {
        self.max_error.is_finite() && self.max_error <= self.limit
    }
}

#[derive(Clone, Debug)]
pub struct Report {
    pub label: Label,
    pub device: DeviceInfo,
    pub tiers: Vec<Tier>,
    /// `None` when the backend is the reference the comparison would use.
    pub bit_exact: Option<bool>,
    /// How far the fma tier moves the bit-exact jobs, on a device that takes that tier
    /// ([`check_math`]), else `None`. It is a report only and never counts toward `pass`.
    pub fma: Option<FmaDifference>,
    pub pass: bool,
}

/// The fma tier against the exact tier on the same device, over the jobs of the bit-exact
/// comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct FmaDifference {
    /// Worst `|fma - exact| / (1 + |exact|)` over the members whose status is the same in
    /// both runs.
    pub max_difference: f64,
    /// Members whose status differs between the two runs.
    pub status_differ: usize,
}

const DECAY: &str = "par a=1.3\nx'=-a*x\ndone\n";
const OSC: &str = "x'=v\nv'=-x\ndone\n";
const LOGISTIC: &str = "x'=x*(1-x)\ndone\n";
/// Only `+ - * /`, so a device that rounds those correctly gives the CPU bits.
const VDP: &str = "par mu=1.5\nx'=v\nv'=mu*(1-x*x)*v-x/(2+v*v)\ndone\n";
/// A map whose value depends on the time `t` the backend passes to it.
const AFFINE: &str = "par a=0.5\nx(t+1)=a*x+t\ndone\n";
/// The Henon map, only `+ - *`. From the start states of `x0_for(2)` it stays bounded over
/// [0, 50] and approaches the fixed point (1.0895, 0.3268), multipliers about 0.3716 and
/// -0.8074.
const HENON: &str = "par a=0.2,b=0.3\nx(t+1)=1-a*x^2+y\ny(t+1)=b*x\ndone\n";
const HENON_PAR: [f64; 2] = [0.2, 0.3];
/// `s(k)` recurses `k + 1` frames deep and returns `k`, so `x' = -x` at every depth.
const RECURSIVE: &str = "par k=1\ns(n)=if(n<1)then(0)else(1+s(n-1))\nx'=-s(k)*x/k\ndone\n";
/// The depths of the `recursion` tier that finish on every backend; 4,095 takes the 4,096
/// frames of a GPU stack.
const RECURSION_K: [f64; 6] = [1.0, 2.0, 7.0, 64.0, 1000.0, 4095.0];
const HENON_T1: f64 = 50.0;

fn model(src: &str) -> IrModel {
    use pc_ode::{MapLoader, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("selftest.ode", src);
    let loaded = load(top, &mut map, &mut MapLoader::new());
    pc_ir::lower(&loaded.expanded, &loaded.info).expect("a self-test model lowers")
}

const MEMBERS: usize = 8;

/// `members * n` initial values that differ per member and keep logistic growth in (0, 1).
fn x0_for(n: usize) -> Vec<f64> {
    (0..MEMBERS * n)
        .map(|k| {
            let (m, i) = (k / n, k % n);
            0.1 + 0.1 * m as f64 + 0.05 * i as f64
        })
        .collect()
}

struct Case {
    src: &'static str,
    params: &'static [f64],
    t1: f64,
    /// The exact state at `t1` from the initial state of one member.
    exact: fn(&[f64], f64) -> Vec<f64>,
}

fn cases() -> [Case; 3] {
    [
        Case {
            src: DECAY,
            params: &[1.3],
            t1: 1.0,
            exact: |x, t| vec![x[0] * (-1.3 * t).exp()],
        },
        Case {
            src: OSC,
            params: &[],
            t1: std::f64::consts::TAU,
            exact: |x, t| {
                vec![
                    x[0] * t.cos() + x[1] * t.sin(),
                    x[1] * t.cos() - x[0] * t.sin(),
                ]
            },
        },
        Case {
            src: LOGISTIC,
            params: &[],
            t1: 2.0,
            exact: |x, t| {
                let e = t.exp();
                vec![x[0] * e / (1.0 - x[0] + x[0] * e)]
            },
        },
    ]
}

fn job<'a>(
    m: &'a IrModel,
    x0: &'a [f64],
    params: &'a [f64],
    t1: f64,
    method: Method,
    precision: Precision,
) -> Job<'a> {
    job_in(m, x0, params, t1, method, precision, Math::Exact)
}

fn job_in<'a>(
    m: &'a IrModel,
    x0: &'a [f64],
    params: &'a [f64],
    t1: f64,
    method: Method,
    precision: Precision,
    math: Math,
) -> Job<'a> {
    Job {
        model: m,
        members: x0.len() / m.n(),
        x0,
        params: Params::Shared(params),
        t0: 0.0,
        t1,
        output: Output::Final,
        method,
        bounds: f64::INFINITY,
        max_steps: 10_000_000,
        precision,
        math,
        native: None,
    }
}

/// Worst `|got - exact| / (1 + |exact|)` over the cases for one method.
fn analytic(run: Runner, method: Method, precision: Precision) -> Result<f64, EnsembleError> {
    let mut worst = 0.0f64;
    for c in cases() {
        let m = model(c.src);
        let n = m.n();
        let x0 = x0_for(n);
        let r = run(&job(&m, &x0, c.params, c.t1, method, precision))?;
        for i in 0..MEMBERS {
            if r.status[i] != Status::Ok {
                return Ok(f64::INFINITY);
            }
            let exact = (c.exact)(&x0[i * n..(i + 1) * n], c.t1);
            for (g, e) in r.state(i, 0).iter().zip(&exact) {
                worst = worst.max((g - e).abs() / (1.0 + e.abs()));
            }
        }
    }
    Ok(worst)
}

fn rk4(precision: Precision) -> (Method, f64) {
    // Measured on the CPU backend: 7.2e-7 single, 2.5e-10 double. The double error is
    // truncation (dt = 0.01); the single error is roundoff over about 600 steps. The limits
    // leave 14 and 4 times that for the contraction and libm differences of a GPU.
    let limit = match precision {
        Precision::Single => 1e-5,
        Precision::Double => 1e-9,
    };
    (Method::Rk4 { dt: 0.01 }, limit)
}

fn dp5(precision: Precision) -> (Method, f64) {
    // Measured on the CPU backend: 4.3e-5 single (rtol 1e-4), 2.7e-10 double (rtol 1e-9).
    // The limits leave 23 and 37 times that.
    match precision {
        Precision::Single => (
            Method::Dp5 {
                rtol: 1e-4,
                atol: 1e-6,
                hmax: f64::INFINITY,
            },
            1e-3,
        ),
        Precision::Double => (
            Method::Dp5 {
                rtol: 1e-9,
                atol: 1e-11,
                hmax: f64::INFINITY,
            },
            1e-8,
        ),
    }
}

/// Worst `|got - exact| / (1 + |exact|)` of `RECURSIVE` at the depths of `RECURSION_K`,
/// or infinity when a member does not finish or the member at the cap of `frames` frames
/// finishes.
fn recursion(run: Runner, precision: Precision, frames: usize) -> Result<f64, EnsembleError> {
    let (method, _) = rk4(precision);
    let m = model(RECURSIVE);
    let mut k = RECURSION_K.to_vec();
    // `s(frames)` needs one frame more than the cap.
    k.push(frames as f64);
    let x0: Vec<f64> = (0..k.len()).map(|i| 0.2 + 0.1 * i as f64).collect();
    let mut j = job(&m, &x0, &[], 1.0, method, precision);
    j.params = Params::PerMember(&k);
    let r = run(&j)?;
    let last = k.len() - 1;
    if r.status[last] != Status::RecursionDepth {
        return Ok(f64::INFINITY);
    }
    let mut worst = 0.0f64;
    for (i, &start) in x0.iter().enumerate().take(last) {
        if r.status[i] != Status::Ok {
            return Ok(f64::INFINITY);
        }
        let e = start * (-1.0f64).exp();
        worst = worst.max((r.state(i, 0)[0] - e).abs() / (1.0 + e.abs()));
    }
    Ok(worst)
}

/// Worst `|got - exact| / (1 + |exact|)` for `AFFINE` over [0, 10], where the exact value is
/// a loop in `f64` that computes `(a*x) + t`.
fn iterate(run: Runner, precision: Precision) -> Result<f64, EnsembleError> {
    const A: f64 = 0.5;
    const T1: f64 = 10.0;
    let m = model(AFFINE);
    let x0 = x0_for(1);
    let r = run(&job(&m, &x0, &[A], T1, Method::Iter, precision))?;
    let mut worst = 0.0f64;
    for (i, &start) in x0.iter().enumerate() {
        if r.status[i] != Status::Ok {
            return Ok(f64::INFINITY);
        }
        let mut e = start;
        for k in 0..T1 as u32 {
            e = A * e + f64::from(k);
        }
        let g = r.state(i, 0)[0];
        worst = worst.max((g - e).abs() / (1.0 + e.abs()));
    }
    Ok(worst)
}

/// The limit of the `iterate` tier. Measured on the CPU backend: 4.5e-8 single, which is
/// the rounding of the start states and of the sums to `f32`, and 0 double, where the
/// backend does the operations of the loop. The final values are about 16, and a time off
/// by one at iterate k of the 10 moves them by 0.5^(9 - k), so the smallest such error is
/// 0.5^9 / 17, about 1.1e-4, above both limits.
fn iterate_limit(precision: Precision) -> f64 {
    match precision {
        Precision::Single => 1e-5,
        Precision::Double => 1e-12,
    }
}

fn same_bits(a: &EnsembleResult, b: &EnsembleResult) -> usize {
    let value = a
        .states
        .iter()
        .zip(&b.states)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let status = a
        .status
        .iter()
        .zip(&b.status)
        .filter(|(x, y)| x != y)
        .count();
    let steps = a
        .accepted
        .iter()
        .zip(&b.accepted)
        .filter(|(x, y)| x != y)
        .count();
    value + status + steps
}

/// Number of values that change when the members of one job run twice, alone, or reversed.
fn differences(
    run: Runner,
    m: &IrModel,
    params: &[f64],
    t1: f64,
    method: Method,
    precision: Precision,
) -> Result<usize, EnsembleError> {
    let n = m.n();
    let x0 = x0_for(n);
    let full = run(&job(m, &x0, params, t1, method, precision))?;
    let mut differ = same_bits(&full, &run(&job(m, &x0, params, t1, method, precision))?);

    let member = 3;
    let alone = run(&job(
        m,
        &x0[member * n..(member + 1) * n],
        params,
        t1,
        method,
        precision,
    ))?;
    differ += full
        .state(member, 0)
        .iter()
        .zip(alone.state(0, 0))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    let rev: Vec<f64> = (0..MEMBERS)
        .rev()
        .flat_map(|i| x0[i * n..(i + 1) * n].iter().copied())
        .collect();
    let back = run(&job(m, &rev, params, t1, method, precision))?;
    for i in 0..MEMBERS {
        differ += full
            .state(i, 0)
            .iter()
            .zip(back.state(MEMBERS - 1 - i, 0))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    Ok(differ)
}

/// Number of values that change when the same members run twice, alone, or reversed.
fn determinism(run: Runner, precision: Precision) -> Result<f64, EnsembleError> {
    let vdp = model(VDP);
    let mut differ = 0;
    for method in [rk4(precision).0, dp5(precision).0] {
        differ += differences(run, &vdp, &[1.5], 3.0, method, precision)?;
    }
    let henon = model(HENON);
    differ += differences(run, &henon, &HENON_PAR, HENON_T1, Method::Iter, precision)?;
    Ok(differ as f64)
}

/// The two models of the bit-exact comparison: fixed-step RK4 of `VDP` and the Henon map
/// with `iter`.
fn bit_models() -> (IrModel, IrModel) {
    (model(VDP), model(HENON))
}

fn bit_jobs<'a>(
    vdp: &'a IrModel,
    henon: &'a IrModel,
    x0_vdp: &'a [f64],
    x0_henon: &'a [f64],
    precision: Precision,
    math: Math,
) -> [Job<'a>; 2] {
    [
        job_in(
            vdp,
            x0_vdp,
            &[1.5],
            3.0,
            Method::Rk4 { dt: 0.01 },
            precision,
            math,
        ),
        job_in(
            henon,
            x0_henon,
            &HENON_PAR,
            HENON_T1,
            Method::Iter,
            precision,
            math,
        ),
    ]
}

/// Whether `run` matches the CPU backend bit for bit on fixed-step RK4 of `VDP` and on the
/// Henon map with `iter`.
fn bit_exact(run: Runner, precision: Precision) -> Result<bool, EnsembleError> {
    let (vdp, henon) = bit_models();
    let (x0_vdp, x0_henon) = (x0_for(vdp.n()), x0_for(henon.n()));
    for j in &bit_jobs(&vdp, &henon, &x0_vdp, &x0_henon, precision, Math::Exact) {
        let want = run_cpu(j, u64::MAX, &AtomicBool::new(false))?;
        let got = run(j)?;
        if same_bits(&want, &got) != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Runs the jobs of [`bit_exact`] on `run` in both math tiers and compares the two.
fn fma_difference(run: Runner, precision: Precision) -> Result<FmaDifference, SelfTestError> {
    let (vdp, henon) = bit_models();
    let (x0_vdp, x0_henon) = (x0_for(vdp.n()), x0_for(henon.n()));
    let exact = bit_jobs(&vdp, &henon, &x0_vdp, &x0_henon, precision, Math::Exact);
    let fused = bit_jobs(&vdp, &henon, &x0_vdp, &x0_henon, precision, Math::Fma);
    let mut out = FmaDifference {
        max_difference: 0.0,
        status_differ: 0,
    };
    for (e, f) in exact.iter().zip(&fused) {
        let want = run(e)?;
        let got = run(f).map_err(SelfTestError::Fma)?;
        for i in 0..MEMBERS {
            if want.status[i] != got.status[i] {
                out.status_differ += 1;
                continue;
            }
            for (g, w) in got.state(i, 0).iter().zip(want.state(i, 0)) {
                out.max_difference = out.max_difference.max((g - w).abs() / (1.0 + w.abs()));
            }
        }
    }
    Ok(out)
}

/// Runs the checks on `run`. `compare_bits` adds the bit-exact comparison with the CPU
/// backend; pass `false` for the CPU backend itself. `compare_fma` adds the report of
/// [`FmaDifference`]; pass `true` only for a device that takes the fma tier.
pub fn run_with(
    run: Runner,
    device: DeviceInfo,
    label: Label,
    compare_bits: bool,
    compare_fma: bool,
) -> Result<Report, SelfTestError> {
    let precision = label.precision;
    let (m4, l4) = rk4(precision);
    let (m5, l5) = dp5(precision);
    let tiers = vec![
        Tier {
            name: "analytic-rk4",
            max_error: analytic(run, m4, precision)?,
            limit: l4,
        },
        Tier {
            name: "analytic-dp5",
            max_error: analytic(run, m5, precision)?,
            limit: l5,
        },
        Tier {
            name: "iterate",
            max_error: iterate(run, precision)?,
            limit: iterate_limit(precision),
        },
        Tier {
            name: "determinism",
            max_error: determinism(run, precision)?,
            limit: 0.0,
        },
        Tier {
            name: "recursion",
            max_error: recursion(run, precision, device.backend.call_frames())?,
            limit: l4,
        },
    ];
    let bit_exact = if compare_bits {
        Some(bit_exact(run, precision)?)
    } else {
        None
    };
    // The fma report runs after `pass` is fixed and never changes it.
    let pass = tiers.iter().all(Tier::pass) && bit_exact != Some(false);
    let fma = if compare_fma {
        Some(fma_difference(run, precision)?)
    } else {
        None
    };
    Ok(Report {
        label,
        device,
        tiers,
        bit_exact,
        fma,
        pass,
    })
}

/// Runs the self-test on the chosen device.
pub fn run_selftest(sel: Selected) -> Result<Report, SelfTestError> {
    let device = find_device(sel.backend, sel.device).map_err(SelfTestError::NoDevice)?;
    let label = Label {
        backend: device.backend.name().to_string(),
        device: device.name.clone(),
        driver: device.driver.clone(),
        precision: sel.precision,
        math: Math::Exact,
    };
    let cancel = AtomicBool::new(false);
    let on = device.clone();
    let run = |j: &Job| run_on(&on, j, &cancel);
    // Comparing the CPU backend with the CPU loop proves nothing, so only a device of
    // another backend gets the bit comparison.
    let compare_bits = sel.backend != Backend::Cpu;
    let compare_fma = check_math(device.backend, sel.precision, Math::Fma).is_ok();
    run_with(&run, device, label, compare_bits, compare_fma)
}
