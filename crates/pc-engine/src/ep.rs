//! Equilibrium branches: the EP problem over a model.

use crate::branch::{Branch, Options, Point, StopReason};
use crate::cycle::{nearest_pair_is_complex, pair_product, shifted_product};
use crate::error::EngineError;
use crate::model::ModelEval;
use crate::normal::attach_map_coefficients;
use crate::palc::run_branch;
use crate::problem::Problem;
use crate::refine::refine_specials;
use pc_ir::EntryKind;
use pc_linalg::{Lu, null_right, singular_values};
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
///
/// On the residual `G = F - x` of a map (`ModelEval::map_residual`, SPEC section 8v) the
/// state matrix is `G_u + I = F_u`, so the stored eigenvalues are the multipliers, and the
/// two codim-2 slots hold `psi_PD = prod (mu_i + 1)` and `psi_NS = prod_{i<j} (mu_i mu_j - 1)`.
pub struct EpProblem<'a> {
    model: &'a mut dyn ModelEval,
    free: usize,
    /// Parameters with t = 0 appended; entry `free` is overwritten by alpha.
    p: Vec<f64>,
    jac_u: Vec<f64>,
    jac_p: Vec<f64>,
    map: bool,
    detect_pd: bool,
    detect_ns: bool,
}

/// Names of the codim-2 tests of a fixed point branch of a map.
const MAP_LABELS: [&str; 2] = ["PD", "NS"];

impl<'a> EpProblem<'a> {
    /// Checks the setup against the model.
    pub fn new(
        model: &'a mut dyn ModelEval,
        setup: &EpSetup,
    ) -> Result<EpProblem<'a>, EngineError> {
        check_setup(model, setup, true)?;
        let mut p = setup.params.clone();
        p.push(0.0);
        let (n, n_p) = (model.n(), model.n_p());
        let map = model.map_residual();
        Ok(EpProblem {
            model,
            free: setup.free,
            p,
            jac_u: vec![0.0; n * n],
            jac_p: vec![0.0; n * n_p],
            map,
            detect_pd: true,
            detect_ns: true,
        })
    }
}

/// `need_free` is false for a plain Newton solve, which never varies a parameter, so a model
/// without parameters can run it.
fn check_setup(model: &dyn ModelEval, setup: &EpSetup, need_free: bool) -> Result<(), EngineError> {
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
    if need_free && setup.free >= model.n_p() {
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
        let mut a = self.jac_u.clone();
        if self.map {
            for i in 0..n {
                a[i + n * i] += 1.0;
            }
        }
        Ok(Some(a))
    }

    fn codim2_labels(&self) -> &'static [&'static str] {
        if self.map { &MAP_LABELS } else { &[] }
    }

    fn codim2_tests(
        &mut self,
        _x: &[f64],
        _tangent: &[f64],
        eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        if !self.map {
            return Ok(Vec::new());
        }
        let have = !eigs.is_empty();
        let pd = if self.detect_pd && have {
            shifted_product(eigs, 1.0)
        } else {
            f64::NAN
        };
        let ns = if self.detect_ns && have {
            pair_product(eigs)
        } else {
            f64::NAN
        };
        Ok(vec![pd, ns])
    }

    /// A zero of `psi_NS` is an NS point only when the pair with product closest to 1 is
    /// complex; a real pair with product 1 is a neutral saddle and gets no label.
    fn codim2_label(&self, i: usize, eigs: &[(f64, f64)]) -> Option<&'static str> {
        match i {
            0 => Some("PD"),
            1 => nearest_pair_is_complex(eigs).then_some("NS"),
            _ => None,
        }
    }
}

/// Newton's method on F(u) = 0 at the parameters of `setup`, from `setup.u0`. Uses the
/// tolerances and the iteration cap of the corrector; gives `NewtonFailed` otherwise.
pub fn equilibrium(
    model: &mut dyn ModelEval,
    setup: &EpSetup,
    opts: &Options,
) -> Result<Vec<f64>, EngineError> {
    check_setup(model, setup, false)?;
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
/// `refine_specials`). On the residual of a map (SPEC section 8v) the H test is off, PD and
/// NS follow `Options::detect_pd` and `Options::detect_ns`, and the PD and NS points get
/// their normal form coefficients without refinement.
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
    // F(u) = 0 is skipped and the start must already be an equilibrium. A special start
    // without a tangent (LP, H or BP) skips it too, since A is singular at an LP and a
    // fixed-alpha solve fails there.
    let special = setup.start_is_special && setup.tangent0.is_none();
    let u = if setup.tangent0.is_some() || special {
        check_setup(model, setup, true)?;
        setup.u0.clone()
    } else {
        equilibrium(model, setup, opts)?
    };
    let mut start = setup.clone();
    start.u0 = u.clone();
    let mut x = u;
    x.push(setup.params[setup.free]);
    let map = model.map_residual();
    let map_opts = Options {
        detect_h: false,
        ..opts.clone()
    };
    let opts = if map { &map_opts } else { opts };
    let mut problem = EpProblem::new(model, &start)?;
    problem.detect_pd = opts.detect_pd;
    problem.detect_ns = opts.detect_ns;
    let mut tangent0 = setup.tangent0.clone();
    if tangent0.is_some() || special {
        let mut f = vec![0.0; x.len() - 1];
        problem.residual(&x, &mut f)?;
        let res = f.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if res.is_nan() || res > opts.tol_f {
            let what = if special {
                "the special start point is not an equilibrium"
            } else {
                "the start of the switched branch is not an equilibrium"
            };
            return Err(EngineError::BadProblem(what.into()));
        }
    }
    if special {
        tangent0 = Some(special_tangent(&mut problem, &x)?);
    }
    let mut branch = run_branch(
        &mut problem,
        &x,
        tangent0.as_deref(),
        opts,
        setup.start_is_special,
        cancel,
        on_point,
    )?;
    drop(problem);
    if branch.stop != StopReason::Cancelled {
        refine_specials(model, &setup.params, setup.free, opts, cancel, &mut branch);
        // A recursion cap error in the coefficients ends the run with that error (SPEC 8af-5).
        if map
            && let Err(e) = attach_map_coefficients(model, &setup.params, setup.free, &mut branch)
        {
            branch.end_with_error(e);
        }
    }
    Ok(branch)
}

/// Relative size of the smallest singular value of `[F_u F_alpha]` below which the matrix
/// counts as rank deficient. This is the constant `palc::kernel_tangent` uses to tell a
/// two-dimensional null space from a one-dimensional one.
const RANK_TOL: f64 = 1e-8;

/// First tangent at a special start. The `n` by `n + 1` matrix `[F_u F_alpha]` has a
/// one-dimensional null space at an LP and at an H, so the tangent is that null vector.
/// At a BP the rank is `n - 1`, no single tangent exists, and the start is refused.
///
/// The sign is fixed so that the alpha component is positive; at an LP that component is
/// zero up to rounding, and then the first state component of size above `RANK_TOL` is made
/// positive instead. `Options::direction` -1 reverses the result in `run_branch`.
fn special_tangent(problem: &mut EpProblem<'_>, x: &[f64]) -> Result<Vec<f64>, EngineError> {
    let n = x.len() - 1;
    let mut jac = vec![0.0; n * (n + 1)];
    problem.jacobian(x, &mut jac)?;
    let s = singular_values(&jac, n, n + 1)?;
    if s.iter().any(|v| !v.is_finite()) {
        return Err(EngineError::NotFinite("Jacobian at the start".into()));
    }
    if s[n - 1] <= RANK_TOL * s[0] {
        return Err(EngineError::BadProblem(
            "the start is a branch point, where the tangent is not unique; \
             use the Switch option to leave on the other branch"
                .into(),
        ));
    }
    let mut t = null_right(&jac, n, n + 1, 1)?.swap_remove(0);
    let lead = if t[n].abs() > RANK_TOL {
        t[n]
    } else {
        t.iter()
            .copied()
            .find(|c| c.abs() > RANK_TOL)
            .unwrap_or(1.0)
    };
    if lead < 0.0 {
        for c in t.iter_mut() {
            *c = -*c;
        }
    }
    Ok(t)
}
