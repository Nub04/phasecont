//! Equilibrium branches: the EP problem over a model.

use crate::branch::{Branch, Options, Point};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::palc::run_branch;
use crate::problem::Problem;
use crate::refine::refine_specials;
use pc_ir::EntryKind;
use pc_linalg::Lu;
use std::sync::atomic::{AtomicBool, Ordering};

/// Where and along which parameter an equilibrium branch starts.
#[derive(Clone, Debug)]
pub struct EpSetup {
    /// Index of the continuation parameter among the model parameters.
    pub free: usize,
    /// All `n_p` parameter values; the engine appends t = 0.
    pub params: Vec<f64>,
    /// Guess for the starting equilibrium.
    pub u0: Vec<f64>,
    /// Set when the start is a special point; no special point is looked for on the first
    /// step. Default false.
    pub start_is_special: bool,
    /// First tangent (u, alpha), `n + 1` values, set by `switch_bp`. When present the run
    /// starts at `u0` without the equilibrium solve and leaves along this tangent
    /// (normalized; `Options::direction` -1 reverses it). Default `None`.
    pub tangent0: Option<Vec<f64>>,
}

impl EpSetup {
    /// Setup with the defaults of every later field.
    pub fn new(free: usize, params: Vec<f64>, u0: Vec<f64>) -> EpSetup {
        EpSetup {
            free,
            params,
            u0,
            start_is_special: false,
            tangent0: None,
        }
    }
}

/// F(u, alpha) = 0 with x = (u, alpha), alpha being parameter `free`.
pub struct EpProblem<'a> {
    model: &'a mut dyn ModelEval,
    free: usize,
    /// Parameters with t = 0 appended; entry `free` is overwritten by alpha.
    p: Vec<f64>,
    jac_u: Vec<f64>,
    jac_p: Vec<f64>,
}

impl<'a> EpProblem<'a> {
    /// Checks the setup against the model.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &EpSetup,
    ) -> Result<EpProblem<'a>, EngineError> {
        check_setup(model, setup)?;
        let mut p = setup.params.clone();
        p.push(0.0);
        let (n, n_p) = (model.n(), model.n_p());
        Ok(EpProblem {
            model,
            free: setup.free,
            p,
            jac_u: vec![0.0; n * n],
            jac_p: vec![0.0; n * n_p],
        })
    }
}

fn check_setup(model: &dyn ModelEval, setup: &EpSetup) -> Result<(), EngineError> {
    if model.n() == 0 {
        return Err(EngineError::BadProblem("the model has no variables".into()));
    }
    if setup.params.len() != model.n_p() {
        return Err(EngineError::BadProblem(format!(
            "expected {} parameters, got {}",
            model.n_p(),
            setup.params.len()
        )));
    }
    if setup.free >= model.n_p() {
        return Err(EngineError::BadProblem(format!(
            "parameter index {} is out of range",
            setup.free
        )));
    }
    if setup.u0.len() != model.n() {
        return Err(EngineError::BadProblem(format!(
            "expected {} state values, got {}",
            model.n(),
            setup.u0.len()
        )));
    }
    Ok(())
}

impl Problem for EpProblem<'_> {
    fn dim(&self) -> usize {
        self.model.n() + 1
    }

    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.model.n();
        self.p[self.free] = x[n];
        self.model.eval(&EntryKind::F, &x[..n], &self.p, &[], out)
    }

    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        let n = self.model.n();
        self.p[self.free] = x[n];
        self.model
            .eval(&EntryKind::JacU, &x[..n], &self.p, &[], &mut self.jac_u)?;
        self.model
            .eval(&EntryKind::JacP, &x[..n], &self.p, &[], &mut self.jac_p)?;
        out[..n * n].copy_from_slice(&self.jac_u);
        out[n * n..].copy_from_slice(&self.jac_p[self.free * n..(self.free + 1) * n]);
        Ok(())
    }

    fn state_jacobian(&mut self, x: &[f64]) -> Result<Option<Vec<f64>>, EngineError> {
        let n = self.model.n();
        self.p[self.free] = x[n];
        self.model
            .eval(&EntryKind::JacU, &x[..n], &self.p, &[], &mut self.jac_u)?;
        Ok(Some(self.jac_u.clone()))
    }
}

/// Newton's method on F(u) = 0 at the parameters of `setup`, from `setup.u0`. Uses the
/// tolerances and the iteration cap of the corrector; gives `NewtonFailed` otherwise.
pub fn equilibrium(
    model: &mut dyn ModelEval,
    setup: &EpSetup,
    opts: &Options,
) -> Result<Vec<f64>, EngineError> {
    check_setup(model, setup)?;
    let n = model.n();
    let mut p = setup.params.clone();
    p.push(0.0);
    let mut u = setup.u0.clone();
    let mut f = vec![0.0; n];
    let mut jac = vec![0.0; n * n];
    let mut res = f64::INFINITY;
    for _ in 0..opts.max_newton {
        model.eval(&EntryKind::F, &u, &p, &[], &mut f)?;
        model.eval(&EntryKind::JacU, &u, &p, &[], &mut jac)?;
        let lu = Lu::new(&jac, n)?;
        for fi in f.iter_mut() {
            *fi = -*fi;
        }
        lu.solve(&mut f)?;
        let mut dxn = 0.0_f64;
        for (ui, d) in u.iter_mut().zip(&f) {
            *ui += d;
            dxn = dxn.max(d.abs());
        }
        model.eval(&EntryKind::F, &u, &p, &[], &mut f)?;
        res = f.iter().fold(0.0, |m, v| m.max(v.abs()));
        if dxn <= opts.tol_x && res <= opts.tol_f {
            return Ok(u);
        }
    }
    Err(EngineError::NewtonFailed {
        iters: opts.max_newton,
        residual: res,
    })
}

/// Continues the equilibrium found from `setup.u0` along parameter `setup.free`. Each LP,
/// BP and H found on the way is refined with its defining system afterwards (see
/// `refine_specials`).
pub fn continue_ep(
    model: &mut dyn ModelEval,
    setup: &EpSetup,
    opts: &Options,
    cancel: &AtomicBool,
    on_point: &mut dyn FnMut(&Point),
) -> Result<Branch, EngineError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(EngineError::Cancelled);
    }
    // With a first tangent the start is a branch point, where A is singular, so Newton on
    // F(u) = 0 is skipped and the start must already be an equilibrium.
    let u = if setup.tangent0.is_some() {
        setup.u0.clone()
    } else {
        equilibrium(model, setup, opts)?
    };
    let mut start = setup.clone();
    start.u0 = u.clone();
    let mut x = u;
    x.push(setup.params[setup.free]);
    let mut problem = EpProblem::new(model, &start)?;
    if setup.tangent0.is_some() {
        let mut f = vec![0.0; x.len() - 1];
        problem.residual(&x, &mut f)?;
        let res = f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if res.is_nan() || res > opts.tol_f {
            return Err(EngineError::BadProblem(
                "the start of the switched branch is not an equilibrium".into(),
            ));
        }
    }
    let mut branch = run_branch(
        &mut problem,
        &x,
        setup.tangent0.as_deref(),
        opts,
        setup.start_is_special,
        cancel,
        on_point,
    )?;
    drop(problem);
    refine_specials(model, &setup.params, setup.free, opts, cancel, &mut branch);
    Ok(branch)
}
