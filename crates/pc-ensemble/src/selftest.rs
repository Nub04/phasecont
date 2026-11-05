//! The self-test behind `pc-cli gpu selftest` and, later, `phasecont.gpuSelfTest`.
//!
//! It runs a few small ensembles through a backend and checks them against closed forms
//! and against themselves. The same checks run on every backend: a backend is a function
//! from a [`Job`] to an [`EnsembleResult`].
//!
//! * `analytic-rk4` and `analytic-dp5`: decay, a harmonic oscillator and logistic growth
//!   against their exact solutions, error scaled by `1 + |exact|`.
//! * `determinism`: the same job twice, one member alone, and the members in reverse order
//!   must give the same bits. The number of values that differ is the error, the limit is 0.
//! * Bit-exactness: for a backend that is not the CPU reference, fixed-step RK4 on a model
//!   of `+ - * /` only must match the CPU backend in the same precision bit for bit.
//!   The CPU backend is the reference, so its report says `None`.

use std::sync::atomic::AtomicBool;

use pc_ir::IrModel;

use crate::cpu::run_cpu;
use crate::device::{Backend, DeviceInfo, find_device};
use crate::job::{
    EnsembleError, EnsembleResult, Job, Label, Method, Output, Params, Precision, Status,
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
}

impl std::fmt::Display for SelfTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelfTestError::NoDevice(m) => f.write_str(m),
            SelfTestError::Run(e) => write!(f, "the backend failed a self-test job: {e}"),
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
    pub pass: bool,
}

const DECAY: &str = "par a=1.3\nx'=-a*x\ndone\n";
const OSC: &str = "x'=v\nv'=-x\ndone\n";
const LOGISTIC: &str = "x'=x*(1-x)\ndone\n";
/// Only `+ - * /`, so a device that rounds those correctly gives the CPU bits.
const VDP: &str = "par mu=1.5\nx'=v\nv'=mu*(1-x*x)*v-x/(2+v*v)\ndone\n";

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

/// Number of values that change when the same members run twice, alone, or reversed.
fn determinism(run: Runner, precision: Precision) -> Result<f64, EnsembleError> {
    let m = model(VDP);
    let n = m.n();
    let x0 = x0_for(n);
    let mut differ = 0;
    for method in [rk4(precision).0, dp5(precision).0] {
        let full = run(&job(&m, &x0, &[1.5], 3.0, method, precision))?;
        differ += same_bits(&full, &run(&job(&m, &x0, &[1.5], 3.0, method, precision))?);

        let member = 3;
        let alone = run(&job(
            &m,
            &x0[member * n..(member + 1) * n],
            &[1.5],
            3.0,
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
        let back = run(&job(&m, &rev, &[1.5], 3.0, method, precision))?;
        for i in 0..MEMBERS {
            differ += full
                .state(i, 0)
                .iter()
                .zip(back.state(MEMBERS - 1 - i, 0))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
    }
    Ok(differ as f64)
}

/// Whether `run` matches the CPU backend bit for bit on fixed-step RK4 of `VDP`.
fn bit_exact(run: Runner, precision: Precision) -> Result<bool, EnsembleError> {
    let m = model(VDP);
    let x0 = x0_for(m.n());
    let j = job(&m, &x0, &[1.5], 3.0, Method::Rk4 { dt: 0.01 }, precision);
    let want = run_cpu(&j, u64::MAX, &AtomicBool::new(false))?;
    let got = run(&j)?;
    Ok(same_bits(&want, &got) == 0)
}

/// Runs the checks on `run`. `compare_bits` adds the bit-exact comparison with the CPU
/// backend; pass `false` for the CPU backend itself.
pub fn run_with(
    run: Runner,
    device: DeviceInfo,
    label: Label,
    compare_bits: bool,
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
            name: "determinism",
            max_error: determinism(run, precision)?,
            limit: 0.0,
        },
    ];
    let bit_exact = if compare_bits {
        Some(bit_exact(run, precision)?)
    } else {
        None
    };
    let pass = tiers.iter().all(Tier::pass) && bit_exact != Some(false);
    Ok(Report {
        label,
        device,
        tiers,
        bit_exact,
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
    };
    let cancel = AtomicBool::new(false);
    let on = device.clone();
    let run = |j: &Job| run_on(&on, j, &cancel);
    // Comparing the CPU backend with the CPU loop proves nothing, so only a device of
    // another backend gets the bit comparison.
    let compare_bits = sel.backend != Backend::Cpu;
    run_with(&run, device, label, compare_bits)
}
