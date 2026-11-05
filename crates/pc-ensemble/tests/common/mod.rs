//! Shared by the ensemble tests: builds a model from `.ode` text and fills in a job.

#![allow(dead_code)]

use std::sync::atomic::AtomicBool;

use pc_engine::{Evaluator, ModelEval};
use pc_ensemble::{EnsembleError, EnsembleResult, Job, Method, Output, Params, Precision, run_cpu};
use pc_ir::IrModel;

pub fn build_model(src: &str) -> IrModel {
    use pc_ode::{MapLoader, Severity, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    match pc_ir::lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    }
}

/// The same model as an f64 evaluator of `pc-engine`, to compare against `integrate`.
pub fn engine_model(src: &str) -> Box<dyn ModelEval> {
    use pc_ode::{MapLoader, Severity, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(l.diagnostics.iter().all(|d| d.severity != Severity::Error));
    pc_engine::build_allowing_t(&l.expanded, &l.info, Evaluator::Fallback).expect("engine model")
}

/// Decay `x' = -a x`, `a` a parameter.
pub const DECAY: &str = "par a=1\nx'=-a*x\ndone\n";
/// Harmonic oscillator, period 2 pi.
pub const OSC: &str = "x'=v\nv'=-x\ndone\n";
/// Logistic growth with rate 1 and capacity 1.
pub const LOGISTIC: &str = "x'=x*(1-x)\ndone\n";
/// Forced by time: x(t) = sin(t) from x(0) = 0.
pub const FORCED: &str = "x'=cos(t)\ndone\n";
/// Van der Pol, a nonlinear model with only + - *.
pub const VDP: &str = "par mu=1.5\nx'=v\nv'=mu*(1-x*x)*v-x\ndone\n";
/// A model with several transcendental functions.
pub const MIXED: &str = "par a=0.7\nx'=sin(x)*exp(-0.1*x*x)+a*cos(t)\ny'=tanh(x-y)-0.2*y\ndone\n";

pub const NO_CANCEL: fn() -> AtomicBool = || AtomicBool::new(false);

/// A job with no bound, a generous step limit and the final state as output.
pub fn job<'a>(
    model: &'a IrModel,
    members: usize,
    x0: &'a [f64],
    params: &'a [f64],
    t1: f64,
    method: Method,
    precision: Precision,
) -> Job<'a> {
    Job {
        model,
        members,
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

pub const DP5_TIGHT: Method = Method::Dp5 {
    rtol: 1e-10,
    atol: 1e-12,
    hmax: f64::INFINITY,
};
pub const DP5_STD: Method = Method::Dp5 {
    rtol: 1e-8,
    atol: 1e-10,
    hmax: f64::INFINITY,
};

/// A tolerance single precision can resolve.
pub const DP5_SINGLE: Method = Method::Dp5 {
    rtol: 1e-4,
    atol: 1e-6,
    hmax: f64::INFINITY,
};

/// Runs on the CPU in one launch.
pub fn run(job: &Job) -> EnsembleResult {
    run_cpu(job, u64::MAX, &NO_CANCEL()).expect("run")
}

pub fn run_err(job: &Job) -> EnsembleError {
    run_cpu(job, u64::MAX, &NO_CANCEL()).expect_err("expected an error")
}

pub fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
