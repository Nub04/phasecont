//! Points, special points and the result of a continuation run.

use crate::error::EngineError;

/// One point of a branch.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    /// State part of the solution.
    pub u: Vec<f64>,
    /// The continuation parameter.
    pub alpha: f64,
    /// Unit tangent, `u.len() + 1` values.
    pub tangent: Vec<f64>,
    /// Eigenvalues (or multipliers) of A = F_u as (re, im); empty for problems without a
    /// state block.
    pub eigenvalues: Vec<(f64, f64)>,
    /// Test function values in the order `[psi_LP, psi_BP, psi_H]`. A test that is off
    /// (`Options::detect_*`) or not defined at the point is NaN.
    pub tests: [f64; 3],
    /// Step length that reached this point (0 for the first point).
    pub h: f64,
    /// Newton iterations of the corrector that reached this point.
    pub newton_iters: usize,
}

/// Extra data stored with a special point.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Extras {
    /// Normal form coefficient (a for LP, l1 for H).
    pub coefficient: Option<f64>,
    /// Frequency of a Hopf point.
    pub omega: Option<f64>,
    /// Which normalization the coefficient used.
    pub normalization: Option<String>,
    /// Whether Newton on the defining system succeeded.
    pub refined: bool,
}

/// A detected bifurcation point.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecialPoint {
    pub label: String,
    /// Index of the first branch point after the special point (the stored k + 1 of the
    /// step k to k + 1 that contained it).
    pub index: usize,
    pub point: Point,
    pub extras: Extras,
}

/// Why a run stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum StopReason {
    /// `max_steps` accepted steps.
    MaxSteps,
    /// The last step landed on a bound of alpha.
    Bounds,
    /// The step length fell below `hmin`.
    StepTooSmall,
    /// A bordered matrix was singular.
    Singular,
    /// The cancel flag was set.
    Cancelled,
    /// Any other error, for example a model entry that gave NaN.
    Error(EngineError),
}

/// The result of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    pub points: Vec<Point>,
    pub specials: Vec<SpecialPoint>,
    pub stop: StopReason,
    /// Messages about the run, for example that Hopf detection is off.
    pub warnings: Vec<String>,
}

/// Settings of a continuation run.
#[derive(Clone, Debug)]
pub struct Options {
    /// First step length.
    pub h0: f64,
    pub hmin: f64,
    pub hmax: f64,
    /// Accepted steps before the run stops with `MaxSteps`.
    pub max_steps: usize,
    /// Tolerance on the Newton update, inf-norm.
    pub tol_x: f64,
    /// Tolerance on the residual, inf-norm.
    pub tol_f: f64,
    /// Newton iterations per corrector call.
    pub max_newton: usize,
    /// Sets the sign of the alpha component of the first tangent from the bordered solve.
    /// With a supplied `tangent0` it multiplies that whole vector by the sign instead.
    pub direction: f64,
    /// Bounds on alpha; a run stops when it reaches one.
    pub bounds: (f64, f64),
    pub detect_lp: bool,
    pub detect_bp: bool,
    pub detect_h: bool,
    /// Iterations of the root search that locates a special point or a bound.
    pub max_locate: usize,
    /// Tolerance of that search and of the refinement Newton.
    pub tol_refine: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            h0: 0.01,
            hmin: 1e-8,
            hmax: 0.1,
            max_steps: 1000,
            tol_x: 1e-10,
            tol_f: 1e-10,
            max_newton: 6,
            direction: 1.0,
            bounds: (f64::NEG_INFINITY, f64::INFINITY),
            detect_lp: true,
            detect_bp: true,
            detect_h: true,
            max_locate: 30,
            tol_refine: 1e-12,
        }
    }
}
