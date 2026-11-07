//! Continuation core and problem types (EP, LP, H, LC, LPC, PD, NS).
//!
//! The model layer comes first: [`ModelEval`] evaluates the entries of a model in f64,
//! [`build`] makes one from a loaded `.ode` model, and [`EngineError`] is the error type
//! of the whole crate. On top of it, [`continue_ep`] follows an equilibrium branch with
//! pseudo-arclength continuation. [`Run`] does the same on a worker thread with poll and
//! cancel, and [`install_hook`] makes worker panics quiet and recoverable.

mod aot;
mod branch;
mod codim2;
mod curve2;
mod cycle;
mod dde;
mod ep;
mod error;
mod hopfcurve;
mod ivp;
mod locate;
mod lpcurve;
mod model;
mod normal;
mod palc;
mod panic;
mod problem;
mod refine;
mod run;
mod switch;
mod tests_fn;

pub use aot::{AotEval, engine_kinds, entry_symbol};
pub use branch::{Branch, Extras, Options, Point, SpecialPoint, StopReason};
pub use curve2::CurveSetup;
pub use cycle::{
    AdjointResidual, AdjointSetup, CYCLE_LABELS, CollocationBasis, CycleBifurcation,
    CycleCurveStart, CycleNormalForm, CycleProblem, CycleSetup, CycleStart, HOM_CURVE_LABELS,
    HomProblem, HomStart, HomStartKind, HopfStart, InclinationFlips, LPC_CURVE_LABELS, LpcProblem,
    NCOL_RANGE, NS_CURVE_LABELS, NsProblem, PD_CURVE_LABELS, PdProblem, PdStart, PeriodicOrbit,
    adjoint_residual, continue_cycle, continue_hom_curve, continue_lpc_curve, continue_ns_curve,
    continue_pd_curve, curve_point_normal_form, inclination_flips, lc_point_normal_form,
    periodic_orbit,
};
pub use dde::{
    CharRoot, DdeBranch, DdeCycle, DdeCycleSetup, DdeCycleStart, DdeHopfBranch, DdeHopfCycle,
    DdeHopfPoint, DdeHopfSpecial, DdePoint, DdeSpecial, HopfExtras as DdeHopfExtras,
    HopfHopf as DdeHopfHopf, HopfPairs as DdeHopfPairs, HopfSetup as DdeHopfSetup, LinearDde,
    Lyapunov as DdeLyapunov, Lyapunov2 as DdeLyapunov2, RootOptions, RootSet,
    ZeroHopf as DdeZeroHopf, characteristic_roots,
    continue_equilibrium as dde_continue_equilibrium, continue_hopf as dde_continue_hopf,
    continue_hopf_with as dde_continue_hopf_with,
    continue_hopf_with_lyapunov as dde_continue_hopf_with_lyapunov,
    correct_cycle as dde_correct_cycle, cycle_jacobian as dde_cycle_jacobian,
    cycle_residual as dde_cycle_residual, equilibrium as dde_equilibrium,
    hopf_cycle as dde_hopf_cycle, hopf_hopf as dde_hopf_hopf, hopf_pairs as dde_hopf_pairs,
    linearize as dde_linearize, lyapunov as dde_lyapunov, lyapunov2 as dde_lyapunov2,
    residual as dde_residual, stability as dde_stability, zero_hopf as dde_zero_hopf,
};
pub use ep::{EpProblem, EpSetup, continue_ep, equilibrium};
pub use error::EngineError;
pub use hopfcurve::{HOPF_CURVE_LABELS, HopfCurveProblem, continue_hopf_curve};
pub use ivp::{IvpOptions, IvpSetup, integrate};
pub use lpcurve::{LP_CURVE_LABELS, LpCurveProblem, continue_lp_curve};
pub use model::{DdeModel, Evaluator, ModelEval, build, build_allowing_t, build_dde};
pub use normal::{
    GH_NORMALIZATION, H_NORMALIZATION, LP_NORMALIZATION, first_lyapunov, fold_coefficient,
    hopf_eigenvectors, special_coefficient,
};
pub use palc::continue_branch;
pub use panic::{install_hook, payload_message, recording, take_last_panic};
pub use problem::Problem;
pub use refine::{
    BpSystem, DefiningSystem, HopfSystem, LpSystem, MAX_REFINE_ITERS, NewtonResult, NewtonStep,
    Refined, newton, refine_special, refine_specials,
};
pub use run::{Run, RunKind, RunSpec, RunState, Status};
pub use switch::{special_index, switch_bp};
pub use tests_fn::{HopfZero, classify_h_zero};
