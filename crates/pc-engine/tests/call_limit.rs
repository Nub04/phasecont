//! A recursion cap hit is never hidden (SPEC 8af-5, Errors): `EngineError::CallLimit` ends a
//! continuation run with `StopReason::Error(CallLimit)` and the points computed so far, and
//! no step, locate pass or post-run pass turns it into a NaN, a missing coefficient, a failed
//! trial or an unrefined point. Any other error at the same sites keeps its old handling.
//!
//! The cap is the constant `MAX_ACTIVE_CALLS` with no knob, so a real recursive model cannot
//! make one chosen call fail. Each test injects the error at one pass instead: `Line` is a
//! toy problem with fault hooks for the PALC and locate plumbing, and
//! `common::inject::Inject` wraps a real model for the post-run passes and the leaf sites.

mod common;

use common::inject::{Inject, cap_error, is_form, is_mixed_pair};
use common::load;
use pc_engine::{
    Branch, CurveSetup, CycleCurveStart, CycleSetup, CycleStart, DdeHopfBranch, DdeHopfExtras,
    DdeHopfSetup, DdeModel, EngineError, EpSetup, Evaluator, MapAsRoot, ModelEval, Options,
    Problem, RootOptions, StopReason, build, build_dde, continue_branch, continue_cycle,
    continue_ep, continue_hopf_curve, continue_lp_curve, continue_lpc_curve, continue_map_ns_curve,
    continue_map_pd_curve, continue_ns_curve, continue_pd_curve, dde_continue_equilibrium,
    dde_continue_hopf_with, dde_continue_hopf_with_lyapunov,
};
use pc_ir::{EntryKind, Slot};
use pc_ode::{MapLoader, SourceMap, load as load_source};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;

fn is_cap(stop: &StopReason) -> bool {
    matches!(stop, StopReason::Error(EngineError::CallLimit(_)))
}

/// The one observable of the rule: the branch is returned `Ok` and its stop reason is the cap.
#[track_caller]
fn assert_cap_stop(br: &Branch, what: &str) {
    assert!(
        is_cap(&br.stop),
        "{what}: expected stop Error(CallLimit), got {:?} (warnings {:?})",
        br.stop,
        br.warnings
    );
}

fn model_of(name: &str) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, Evaluator::Jit).unwrap()
}

// ---- PALC and locate plumbing, on a toy problem ----

/// Which entry point of `Line` is being called.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Site {
    Residual,
    Jacobian,
    Codim2,
}

type LineRule = Box<dyn FnMut(Site, &[f64]) -> bool>;

/// Two unknowns `(x0, x1)`, one equation `x0 = 0`, and one codim-2 test `ZZ` that crosses
/// zero at `root()`. `rule` picks the calls that fail with `error`.
struct Line {
    rule: LineRule,
    error: EngineError,
    hits: Rc<Cell<usize>>,
}

impl Line {
    fn new(error: EngineError, rule: impl FnMut(Site, &[f64]) -> bool + 'static) -> Line {
        Line {
            rule: Box::new(rule),
            error,
            hits: Rc::new(Cell::new(0)),
        }
    }

    fn check(&mut self, site: Site, x: &[f64]) -> Result<(), EngineError> {
        if (self.rule)(site, x) {
            self.hits.set(self.hits.get() + 1);
            return Err(self.error.clone());
        }
        Ok(())
    }
}

impl Problem for Line {
    fn dim(&self) -> usize {
        2
    }
    fn residual(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.check(Site::Residual, x)?;
        out[0] = x[0];
        Ok(())
    }
    fn jacobian(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), EngineError> {
        self.check(Site::Jacobian, x)?;
        out.copy_from_slice(&[1.0, 0.0]);
        Ok(())
    }
    fn ep_tests(&self) -> bool {
        false
    }
    fn codim2_labels(&self) -> &'static [&'static str] {
        &["ZZ"]
    }
    fn codim2_tests(
        &mut self,
        x: &[f64],
        _tangent: &[f64],
        _eigs: &[(f64, f64)],
    ) -> Result<Vec<f64>, EngineError> {
        self.check(Site::Codim2, x)?;
        // Not linear, so the locate search takes several trials and the polish has a slope.
        // A value of exactly 0.0 is replaced by 1e-18, so that the located point is not an
        // exact zero and the polish has a step to take.
        let v = x[1] * (1.0 + x[1]) - 0.05;
        Ok(vec![if v == 0.0 { 1e-18 } else { v }])
    }
}

/// The zero of `ZZ`: the positive root of `x (1 + x) = 0.05`, which is not a number that the
/// search lands on exactly.
fn root() -> f64 {
    (-1.0 + 1.2_f64.sqrt()) / 2.0
}

fn line_opts() -> Options {
    Options {
        h0: 0.1,
        hmax: 0.1,
        max_steps: 6,
        bounds: (-1.0, 1.0),
        ..Options::default()
    }
}

fn run_line(line: &mut Line, opts: &Options) -> Branch {
    continue_branch(
        line,
        &[0.0, 0.0],
        opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

/// What every `Line` test checks besides the stop reason: the run kept the points before the
/// hit, and no point has a NaN codim-2 value that the cap could have written.
#[track_caller]
fn assert_line_points_kept(br: &Branch) {
    assert!(br.points.len() >= 2, "points kept: {}", br.points.len());
    for p in &br.points {
        assert!(
            p.codim2.iter().all(|v| v.is_finite()),
            "a NaN codim-2 value at alpha {}: {:?}",
            p.alpha,
            p.codim2
        );
    }
}

/// A clean run of `Line` finds ZZ between the first two points and refines it.
#[test]
fn line_without_a_fault_locates_and_refines_zz() {
    let mut line = Line::new(cap_error(), |_, _| false);
    let br = run_line(&mut line, &line_opts());
    let zz: Vec<_> = br.specials.iter().filter(|s| s.label == "ZZ").collect();
    assert_eq!(zz.len(), 1, "{:?} {:?}", br.stop, br.warnings);
    assert!(zz[0].extras.refined);
    assert!((zz[0].point.alpha - root()).abs() < 1e-9);
    assert!(!is_cap(&br.stop));
}

/// locate.rs `trial`, corrector arm: a trial point whose corrector hits the cap is
/// `Corrected::Stop(Error(CallLimit))`, which `trial` turned into `Trial::Failed`, and the
/// search then ended without a zero ("could not locate").
#[test]
fn locate_trial_corrector_cap_stops_the_run() {
    let mut line = Line::new(cap_error(), |site, x| {
        site == Site::Residual && (x[1] - root()).abs() < 1e-3
    });
    let hits = line.hits.clone();
    let br = run_line(&mut line, &line_opts());
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "trial corrector");
    assert_line_points_kept(&br);
    assert!(br.specials.is_empty(), "{:?}", br.specials);
}

/// locate.rs `trial`, tangent and tests arm: the corrector of the trial succeeds and the test
/// functions there hit the cap, which `trial` turned into `Trial::Failed` through its `_`
/// arm.
#[test]
fn locate_trial_tests_cap_stops_the_run() {
    let mut line = Line::new(cap_error(), |site, x| {
        site == Site::Codim2 && (x[1] - root()).abs() < 1e-3
    });
    let hits = line.hits.clone();
    let br = run_line(&mut line, &line_opts());
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "trial tests");
    assert_line_points_kept(&br);
    assert!(br.specials.is_empty(), "{:?}", br.specials);
}

/// locate.rs `trial`, tangent arm: the Jacobian of the tangent computation at the trial point
/// hits the cap. On this problem a converged corrector calls residual, Jacobian, residual
/// (the check after a zero step), so the tangent is the Jacobian call that follows that
/// triple. The corrector's own Jacobian call follows a lone residual call and does not match.
#[test]
fn locate_trial_tangent_cap_stops_the_run() {
    let before = Rc::new(Cell::new((Site::Codim2, Site::Codim2)));
    let before_in = before.clone();
    let mut line = Line::new(cap_error(), move |site, x| {
        let (older, prev) = before_in.replace((before_in.get().1, site));
        site == Site::Jacobian
            && prev == Site::Residual
            && older == Site::Jacobian
            && (x[1] - root()).abs() < 1e-3
    });
    let hits = line.hits.clone();
    let br = run_line(&mut line, &line_opts());
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "trial tangent");
    assert_line_points_kept(&br);
    assert!(br.specials.is_empty(), "{:?}", br.specials);
}

/// palc.rs, bounds landing: the landed point is built and its test functions hit the cap. The
/// driver kept a bare point, warned "tests at the bound point are missing" and stopped with
/// `Bounds`.
#[test]
fn bounds_landing_tests_cap_stops_the_run() {
    let opts = Options {
        bounds: (-1.0, 0.35),
        ..line_opts()
    };
    let mut line = Line::new(cap_error(), |site, x| {
        site == Site::Codim2 && (x[1] - 0.35).abs() < 1e-3
    });
    let hits = line.hits.clone();
    let br = run_line(&mut line, &opts);
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "bounds landing");
    assert_line_points_kept(&br);
    // The points before the landing have their ZZ value and the zero near 0.05 is kept.
    assert!(br.points.len() >= 4, "points {}", br.points.len());
    assert_eq!(
        br.specials.iter().filter(|s| s.label == "ZZ").count(),
        1,
        "{:?}",
        br.specials
    );
}

/// locate.rs `refine_zero`: the corrector of the polish step hits the cap, which `refine_zero`
/// turned into `Ok(None)`, so the zero stayed unrefined. The fault arms when the polish
/// evaluates the tests at the forward-difference point `x + sqrt(eps) (1 + |x|) tau`, and
/// fails the residual call that follows.
#[test]
fn refine_zero_corrector_cap_stops_the_run() {
    let seen: Rc<std::cell::RefCell<Vec<f64>>> = Rc::default();
    let armed = Rc::new(Cell::new(false));
    let (seen_in, armed_in) = (seen.clone(), armed.clone());
    let mut line = Line::new(cap_error(), move |site, x| match site {
        Site::Codim2 => {
            let step = |a: f64| f64::EPSILON.sqrt() * (1.0 + a.abs());
            if seen_in
                .borrow()
                .iter()
                .any(|a| (x[1] - a - step(*a)).abs() < 1e-13)
            {
                armed_in.set(true);
            }
            seen_in.borrow_mut().push(x[1]);
            false
        }
        // One shot: if refine_zero hides this failure, the run goes on and the test fails.
        Site::Residual => armed_in.replace(false),
        Site::Jacobian => false,
    });
    let hits = line.hits.clone();
    let br = run_line(&mut line, &line_opts());
    assert!(hits.get() > 0, "the polish never reached its corrector");
    assert_cap_stop(&br, "refine_zero corrector");
    assert_line_points_kept(&br);
}

/// Other errors at the same sites keep their old handling: a failed trial ends the search
/// ("could not locate"), the zero is not reported and the run goes on to its end.
#[test]
fn locate_trial_other_error_keeps_the_old_result() {
    for site in [Site::Residual, Site::Codim2] {
        let mut line = Line::new(EngineError::NotFinite("test".into()), move |s, x| {
            s == site && (x[1] - root()).abs() < 1e-3
        });
        let hits = line.hits.clone();
        let br = run_line(&mut line, &line_opts());
        assert!(hits.get() > 0, "{site:?}: the fault never fired");
        assert!(!is_cap(&br.stop), "{site:?}: {:?}", br.stop);
        assert!(
            br.warnings
                .iter()
                .any(|w| w.contains("could not locate ZZ")),
            "{site:?}: {:?}",
            br.warnings
        );
        assert!(br.specials.is_empty(), "{site:?}: {:?}", br.specials);
    }
}

// ---- Post-run passes of the equilibrium driver ----

fn fold_setup() -> (EpSetup, Options) {
    (
        EpSetup::new(0, vec![-1.0], vec![-1.0]),
        Options {
            max_steps: 150,
            ..Options::default()
        },
    )
}

fn run_fold(m: &mut dyn ModelEval) -> Branch {
    let (setup, opts) = fold_setup();
    continue_ep(m, &setup, &opts, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

/// Calls with a `Full` slot are the defining systems of `refine_special`; the run itself
/// uses only F, JacU and JacP.
fn has_full(kind: &EntryKind) -> bool {
    matches!(kind, EntryKind::Dir(s) if s.contains(&Slot::Full))
}

/// refine.rs `refine_specials`, refine arm: the Newton polish of the LP point hits the cap.
/// The `Err(_)` arm skipped the point with a warning "could not refine LP".
#[test]
fn refine_specials_cap_stops_the_run() {
    let mut m = model_of("fold");
    let clean = run_fold(&mut *m);
    assert_eq!(clean.specials.len(), 1, "{:?}", clean.specials);
    let mut inj = Inject::call_limit(&mut *m, |c| has_full(c.kind));
    let hits = inj.hits.clone();
    let br = run_fold(&mut inj);
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "refine_specials");
    assert_eq!(br.points.len(), clean.points.len(), "points kept");
    assert!(
        br.warnings.iter().all(|w| !w.contains("could not refine")),
        "the cap was reported as a warning: {:?}",
        br.warnings
    );
}

/// refine.rs `refine_specials`, coefficient arm: the fold coefficient of the refined LP hits
/// the cap. The `Err(_)` arm pushed "could not compute the <a|l1> coefficient" and went on.
#[test]
fn lp_coefficient_cap_stops_the_run() {
    let mut m = model_of("fold");
    let clean = run_fold(&mut *m);
    assert!(clean.specials[0].extras.coefficient.is_some());
    let mut inj = Inject::call_limit(&mut *m, |c| is_form(c.kind, 2));
    let hits = inj.hits.clone();
    let br = run_fold(&mut inj);
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "LP coefficient");
    assert_eq!(br.points.len(), clean.points.len(), "points kept");
    for sp in &br.specials {
        assert!(
            sp.extras.coefficient.is_none_or(|c| !c.is_nan()),
            "a NaN coefficient: {:?}",
            sp.extras
        );
    }
    assert!(
        br.warnings.iter().all(|w| !w.contains("coefficient")),
        "the cap was reported as a warning: {:?}",
        br.warnings
    );
}

/// Another error at the coefficient site keeps its old handling: a warning, no coefficient
/// and the run ends the way it did before.
#[test]
fn lp_coefficient_other_error_keeps_the_old_result() {
    let mut m = model_of("fold");
    let clean = run_fold(&mut *m);
    let mut inj = Inject::with_error(&mut *m, EngineError::NotFinite("test".into()), |c| {
        is_form(c.kind, 2)
    });
    let hits = inj.hits.clone();
    let br = run_fold(&mut inj);
    assert!(hits.get() > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    assert_eq!(br.stop, clean.stop);
    assert_eq!(br.specials.len(), 1);
    assert!(br.specials[0].extras.coefficient.is_none());
    assert!(
        br.warnings
            .iter()
            .any(|w| w.contains("could not compute the <a|l1> coefficient")),
        "{:?}",
        br.warnings
    );
}

/// refine.rs `refine_specials`, other error at the refine arm: a warning and the point stays
/// unrefined.
#[test]
fn refine_specials_other_error_keeps_the_old_result() {
    let mut m = model_of("fold");
    let mut inj = Inject::with_error(&mut *m, EngineError::NotFinite("test".into()), |c| {
        has_full(c.kind)
    });
    let hits = inj.hits.clone();
    let br = run_fold(&mut inj);
    assert!(hits.get() > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    assert!(
        br.warnings
            .iter()
            .any(|w| w.contains("could not refine LP")),
        "{:?}",
        br.warnings
    );
    assert!(!br.specials[0].extras.refined);
}

// ---- Normal form coefficients of a map ----

/// normal.rs `attach_map_coefficients`: the flip coefficient of the PD point hits the cap. The
/// `Err(e)` arm pushed "no normal form coefficient" and went on.
#[test]
fn map_flip_coefficient_cap_stops_the_run() {
    let run = |inner: &mut dyn ModelEval| {
        let mut g = MapAsRoot::new(inner);
        let setup = EpSetup::new(0, vec![-0.1, 0.7, 0.5, 0.3], vec![0.0, 0.0]);
        let opts = Options {
            bounds: (f64::NEG_INFINITY, 0.1),
            ..Options::default()
        };
        continue_ep(&mut g, &setup, &opts, &AtomicBool::new(false), &mut |_| {}).unwrap()
    };
    let mut m = model_of("map_flip");
    let clean = run(&mut *m);
    assert!(clean.specials.iter().any(|s| s.label == "PD"));
    let mut inj = Inject::call_limit(&mut *m, |c| is_form(c.kind, 2));
    let hits = inj.hits.clone();
    let br = run(&mut inj);
    assert!(hits.get() > 0, "the fault never fired");
    assert_cap_stop(&br, "map flip coefficient");
    assert_eq!(br.points.len(), clean.points.len(), "points kept");
    assert!(
        br.warnings.iter().all(|w| !w.contains("coefficient")),
        "the cap was reported as a warning: {:?}",
        br.warnings
    );
}

/// The same site with another error keeps the warning and the empty coefficient.
#[test]
fn map_flip_coefficient_other_error_keeps_the_old_result() {
    let mut m = model_of("map_flip");
    let mut g_inner = Inject::with_error(&mut *m, EngineError::NotFinite("test".into()), |c| {
        is_form(c.kind, 2)
    });
    let hits = g_inner.hits.clone();
    let mut g = MapAsRoot::new(&mut g_inner);
    let setup = EpSetup::new(0, vec![-0.1, 0.7, 0.5, 0.3], vec![0.0, 0.0]);
    let opts = Options {
        bounds: (f64::NEG_INFINITY, 0.1),
        ..Options::default()
    };
    let br = continue_ep(&mut g, &setup, &opts, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert!(hits.get() > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    let pd = br.specials.iter().find(|s| s.label == "PD").unwrap();
    assert!(pd.extras.coefficient.is_none());
    assert!(
        br.warnings
            .iter()
            .any(|w| w.contains("no normal form coefficient")),
        "{:?}",
        br.warnings
    );
}

// ---- Codim-2 curves ----

/// The Hopf curve of the Bautin model from b2 = 0.5 down through the GH at b2 = 0, as in
/// `codim2.rs`. The start is built with the plain model and `run` follows the curve with the
/// model it is given.
fn bautin_curve(run: impl FnOnce(&mut dyn ModelEval, &CurveSetup, &Options) -> Branch) -> Branch {
    let mut m = model_of("bautin");
    let cancel = AtomicBool::new(false);
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, vec![-0.5, 0.5, 1.0], vec![0.0, 0.0]),
        &Options {
            bounds: (-1.0, 1.0),
            ..Options::default()
        },
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let h: Vec<_> = ep.specials.iter().filter(|s| s.label == "H").collect();
    assert_eq!(h.len(), 1, "{:?}", ep.specials);
    let setup = CurveSetup::new(
        [0, 1],
        vec![h[0].point.alpha, 0.5, 1.0],
        h[0].point.u.clone(),
    );
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 0.8),
        ..Options::default()
    };
    run(&mut *m, &setup, &opts)
}

fn hopf_run(m: &mut dyn ModelEval, setup: &CurveSetup, opts: &Options) -> Branch {
    continue_hopf_curve(m, setup, opts, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

/// Index of the GH test in `Point.codim2` on a Hopf curve.
const GH: usize = 0;

/// hopfcurve.rs `codim2_tests`: the first Lyapunov coefficient at a point of the curve hits the
/// cap and was stored as NaN (`unwrap_or(NAN)`), which the run read as "GH undefined here".
/// The fault starts after the first points, so the start itself is a valid point.
#[test]
fn hopf_curve_first_lyapunov_cap_stops_the_run() {
    let br = bautin_curve(|m, setup, opts| {
        let mut inj = Inject::call_limit(m, |c| is_form(c.kind, 2) && c.p[1] < 0.3);
        let hits = inj.hits.clone();
        let br = hopf_run(&mut inj, setup, opts);
        assert!(hits.get() > 0, "the fault never fired");
        br
    });
    assert_cap_stop(&br, "first Lyapunov");
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
    for p in &br.points {
        assert!(
            p.alpha >= 0.3 || !p.codim2[GH].is_nan(),
            "a NaN GH test written at b2 = {}",
            p.alpha
        );
    }
}

/// Another error at the same site keeps the NaN: the GH test is undefined at those points and
/// the run goes on to its bound.
#[test]
fn hopf_curve_first_lyapunov_other_error_keeps_the_nan() {
    let br = bautin_curve(|m, setup, opts| {
        let mut inj = Inject::with_error(m, EngineError::NotFinite("test".into()), |c| {
            is_form(c.kind, 2) && c.p[1] < 0.3
        });
        let hits = inj.hits.clone();
        let br = hopf_run(&mut inj, setup, opts);
        assert!(hits.get() > 0, "the fault never fired");
        br
    });
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    let late: Vec<_> = br.points.iter().filter(|p| p.alpha < 0.3).collect();
    assert!(!late.is_empty());
    assert!(late.iter().all(|p| p.codim2[GH].is_nan()));
}

/// hopfcurve.rs `annotate`, GH arm: the second Lyapunov coefficient at the GH point hits the
/// cap. The `if let Ok` left the coefficient empty and the branch stopped as before. Forms of
/// order 4 and above are used only here, since the curve tests stop at order 3.
#[test]
fn hopf_curve_gh_annotation_cap_stops_the_run() {
    let mut clean_points = 0;
    bautin_curve(|m, setup, opts| {
        let br = hopf_run(m, setup, opts);
        clean_points = br.points.len();
        br
    });
    let br = bautin_curve(|m, setup, opts| {
        let mut inj = Inject::call_limit(m, |c| is_form(c.kind, 4));
        let hits = inj.hits.clone();
        let br = hopf_run(&mut inj, setup, opts);
        assert!(hits.get() > 0, "the fault never fired");
        br
    });
    assert_cap_stop(&br, "GH annotation");
    assert_eq!(br.points.len(), clean_points, "points kept");
    for sp in &br.specials {
        assert!(
            sp.extras.coefficient.is_none_or(|c| !c.is_nan()),
            "a NaN coefficient: {:?}",
            sp.extras
        );
    }
}

/// Another error in the GH annotation keeps the old result: no coefficient, no warning, the
/// stop reason of the run.
#[test]
fn hopf_curve_gh_annotation_other_error_keeps_the_old_result() {
    let br = bautin_curve(|m, setup, opts| {
        let mut inj = Inject::with_error(m, EngineError::NotFinite("test".into()), |c| {
            is_form(c.kind, 4)
        });
        let hits = inj.hits.clone();
        let br = hopf_run(&mut inj, setup, opts);
        assert!(hits.get() > 0, "the fault never fired");
        br
    });
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    let gh = br.specials.iter().find(|s| s.label == "GH").unwrap();
    assert!(gh.extras.coefficient.is_none());
}

/// The LP curve of the Bogdanov-Takens model through the BT, as `codim2.rs` `bt_lp_branch`.
fn bt_lp_curve(run: impl FnOnce(&mut dyn ModelEval, &CurveSetup, &Options) -> Branch) -> Branch {
    let s = 1.0;
    let mut m = model_of("bt");
    let cancel = AtomicBool::new(false);
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, vec![-1.0, 0.5, s], vec![-1.0, 0.0]),
        &Options {
            bounds: (-1.0, 1.0),
            ..Options::default()
        },
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let lp: Vec<_> = ep.specials.iter().filter(|s| s.label == "LP").collect();
    assert_eq!(lp.len(), 1, "{:?}", ep.specials);
    let setup = CurveSetup::new(
        [0, 1],
        vec![lp[0].point.alpha, 0.5, s],
        lp[0].point.u.clone(),
    );
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 1.0),
        ..Options::default()
    };
    run(&mut *m, &setup, &opts)
}

fn lp_run(m: &mut dyn ModelEval, setup: &CurveSetup, opts: &Options) -> Branch {
    continue_lp_curve(m, setup, opts, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

/// lpcurve.rs `annotate`, BT arm: the BT coefficients hit the cap. The `if let Ok` left them
/// empty. The curve tests call `B(q, q)` and the BT coefficients also call `B(q0, q1)`, so the
/// rule picks a bilinear form of two different vectors.
#[test]
fn lp_curve_bt_annotation_cap_stops_the_run() {
    let mut clean_points = 0;
    bt_lp_curve(|m, setup, opts| {
        let br = lp_run(m, setup, opts);
        clean_points = br.points.len();
        assert!(br.specials.iter().any(|s| s.label == "BT"));
        br
    });
    let br = bt_lp_curve(|m, setup, opts| {
        let mut inj = Inject::call_limit(m, is_mixed_pair);
        let hits = inj.hits.clone();
        let br = lp_run(&mut inj, setup, opts);
        assert!(hits.get() > 0, "the fault never fired");
        br
    });
    assert_cap_stop(&br, "BT annotation");
    assert_eq!(br.points.len(), clean_points, "points kept");
    for sp in &br.specials {
        assert!(
            sp.extras.coefficient.is_none_or(|c| !c.is_nan())
                && sp.extras.coefficient2.is_none_or(|c| !c.is_nan()),
            "a NaN coefficient: {:?}",
            sp.extras
        );
    }
}

/// Another error in the BT annotation keeps the old result: empty coefficients, no warning.
#[test]
fn lp_curve_bt_annotation_other_error_keeps_the_old_result() {
    let br = bt_lp_curve(|m, setup, opts| {
        let mut inj = Inject::with_error(m, EngineError::NotFinite("test".into()), |c| {
            is_mixed_pair(c)
        });
        let hits = inj.hits.clone();
        let br = lp_run(&mut inj, setup, opts);
        assert!(hits.get() > 0, "the fault never fired");
        br
    });
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    let bt = br.specials.iter().find(|s| s.label == "BT").unwrap();
    assert!(bt.extras.coefficient.is_none() && bt.extras.coefficient2.is_none());
}

// ---- Cycle runs and cycle curves ----

use std::f64::consts::PI;

const NTST: usize = 40;
const NCOL: usize = 4;
const CURVE_NTST: usize = 20;

fn ring_orbit(beta: f64, nm: usize) -> (Vec<f64>, f64) {
    let r = beta.sqrt();
    let orbit = (0..=nm)
        .flat_map(|k| {
            let a = 2.0 * PI * k as f64 / nm as f64;
            [r * a.cos(), r * a.sin(), 0.0, 0.0]
        })
        .collect();
    (orbit, PI)
}

fn polar_orbit(s: f64, nm: usize) -> (Vec<f64>, f64) {
    let (omega, r) = (1.0 + s, s.sqrt());
    let period = 2.0 * PI / omega;
    let orbit = (0..=nm)
        .flat_map(|k| {
            let a = omega * period * k as f64 / nm as f64;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    (orbit, period)
}

/// Any entry of the normal form code: the cycle runs and curves use `F`, `JacU`, `JacP` only.
fn is_dir(c: &common::inject::Call) -> bool {
    matches!(c.kind, EntryKind::Dir(_))
}

/// The bilinear form of one vector with itself, `B(q, q)`, which every coefficient of a cycle
/// curve calls. The curve systems call `Dir` too, with two different vectors or a parameter slot.
fn is_square_form_below(c: &common::inject::Call, below: f64) -> bool {
    let h = c.v.len() / 2;
    matches!(c.kind, EntryKind::Dir(s) if s.as_slice() == [Slot::U, Slot::U])
        && c.v[..h] == c.v[h..]
        && c.p[1] < below
}

/// The LC run of `cyc_pd` from the ring at beta = 0.2 up through the flip at beta = 1/4, as in
/// `cycle_normal_forms.rs`.
fn pd_cycle_run(error: EngineError) -> (Branch, usize) {
    let mut m = model_of("cyc_pd");
    let mut inj = Inject::with_error(&mut *m, error, is_dir);
    let hits = inj.hits.clone();
    let (orbit, period) = ring_orbit(0.2, NTST * NCOL);
    let setup = CycleSetup::new(0, vec![0.2], NTST, NCOL, orbit, period);
    let opts = Options {
        direction: 1.0,
        max_newton: 15,
        hmax: 0.03,
        bounds: (0.05, 0.4),
        ..Options::default()
    };
    let br = continue_cycle(
        &mut inj,
        &CycleStart::Orbit(setup),
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap();
    (br, hits.get())
}

/// cycle/normal.rs `attach_coefficients` (called from `continue_cycle`): the coefficient at the
/// PD point hits the cap, which became the warning "no normal form coefficient".
#[test]
fn cycle_coefficient_cap_stops_the_run() {
    let (br, hits) = pd_cycle_run(cap_error());
    assert!(hits > 0, "the fault never fired");
    assert_cap_stop(&br, "cycle attach_coefficients");
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
}

#[test]
fn cycle_coefficient_other_error_keeps_the_old_result() {
    let (br, hits) = pd_cycle_run(EngineError::NotFinite("test".into()));
    assert!(hits > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    let pd = br.specials.iter().find(|s| s.label == "PD").unwrap();
    assert!(pd.extras.coefficient.is_none());
    assert!(
        br.warnings
            .iter()
            .any(|w| w.contains("no normal form coefficient")),
        "{:?}",
        br.warnings
    );
}

fn curve_start(params: Vec<f64>, (orbit, period): (Vec<f64>, f64)) -> CycleCurveStart {
    CycleCurveStart {
        free: [0, 1],
        params,
        orbit,
        period,
        mesh: (0..=CURVE_NTST)
            .map(|j| j as f64 / CURVE_NTST as f64)
            .collect(),
        ntst: CURVE_NTST,
        ncol: NCOL,
    }
}

type CurveRun = fn(
    &mut dyn ModelEval,
    &CycleCurveStart,
    &Options,
    &AtomicBool,
    &mut dyn FnMut(&pc_engine::Point),
) -> Result<Branch, EngineError>;

/// One curve of `cycle_normal_forms.rs` with the injected model; returns the branch and the
/// number of injected failures.
fn cycle_curve(
    name: &str,
    run: CurveRun,
    start: &CycleCurveStart,
    bounds: (f64, f64),
    below: f64,
    error: EngineError,
) -> (Branch, usize) {
    let mut m = model_of(name);
    let mut inj = Inject::with_error(&mut *m, error, move |c| is_square_form_below(c, below));
    let hits = inj.hits.clone();
    let opts = Options {
        direction: -1.0,
        max_newton: 15,
        hmax: 0.1,
        bounds,
        ..Options::default()
    };
    let br = run(&mut inj, start, &opts, &AtomicBool::new(false), &mut |_| {}).unwrap();
    (br, hits.get())
}

/// The three cycle curves with the test that holds the coefficient: LPC (CPC, index 2), PD (GPD,
/// index 2) and NS (CH, index 6).
/// A curve case: model, follower, start, bounds of alpha, the value of `p[1]` below which the
/// fault arms, and the index of the coefficient test.
type CurveCase = (
    &'static str,
    CurveRun,
    CycleCurveStart,
    (f64, f64),
    f64,
    usize,
);

fn cycle_curves() -> Vec<CurveCase> {
    let ring = || curve_start(vec![0.25, 0.5], ring_orbit(0.25, CURVE_NTST * NCOL));
    let q0 = 0.3_f64;
    let cusp = curve_start(
        vec![-2.0 * q0.powi(3), 3.0 * q0 * q0],
        polar_orbit(1.0 + q0, CURVE_NTST * NCOL),
    );
    vec![
        (
            "cyc_cpc",
            |m, s, o, c, f| continue_lpc_curve(m, s, o, c, f),
            cusp,
            (-0.5, 0.3),
            0.15,
            2,
        ),
        (
            "cyc_gpd",
            |m, s, o, c, f| continue_pd_curve(m, s, o, c, f),
            ring(),
            (-0.5, 0.6),
            0.3,
            2,
        ),
        (
            "cyc_ch",
            |m, s, o, c, f| continue_ns_curve(m, s, o, c, f),
            ring(),
            (-0.5, 0.6),
            0.3,
            6,
        ),
    ]
}

/// cycle/lpc.rs, cycle/pd.rs and cycle/ns.rs `codim2_tests`: the coefficient at a curve point
/// hits the cap, which became a NaN CPC, GPD or CH test (`unwrap_or(NAN)`).
#[test]
fn cycle_curve_coefficient_cap_stops_the_run() {
    let mut bad = Vec::new();
    for (name, run, start, bounds, below, idx) in cycle_curves() {
        let (br, hits) = cycle_curve(name, run, &start, bounds, below, cap_error());
        assert!(hits > 0, "{name}: the fault never fired");
        if !is_cap(&br.stop) {
            bad.push(format!(
                "{name}: expected stop Error(CallLimit), got {:?} (warnings {:?})",
                br.stop, br.warnings
            ));
        }
        if br.points.iter().any(|p| p.codim2[idx].is_nan()) {
            bad.push(format!("{name}: a NaN coefficient test was written"));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn cycle_curve_coefficient_other_error_keeps_the_nan() {
    for (name, run, start, bounds, below, idx) in cycle_curves() {
        let (br, hits) = cycle_curve(
            name,
            run,
            &start,
            bounds,
            below,
            EngineError::NotFinite("t".into()),
        );
        assert!(hits > 0, "{name}: the fault never fired");
        assert!(!is_cap(&br.stop), "{name}: {:?}", br.stop);
        assert!(
            br.points.iter().any(|p| p.codim2[idx].is_nan()),
            "{name}: no NaN coefficient test"
        );
    }
}

// ---- Map curves ----

/// The cubic form `C(q, q, q)`, which the GPD and CH coefficients call once the parameter
/// `p[1]` of the curve is past `after`, so that the start is a valid point. The curve systems
/// use `Dir` with a `Full` slot, or `Dir` with two vectors.
fn is_cubic_after(c: &common::inject::Call, after: f64) -> bool {
    matches!(c.kind, EntryKind::Dir(s) if s.as_slice() == [Slot::U, Slot::U, Slot::U])
        && c.p[1] > after
}

/// The map PD curve of the generalized Henon map (`map_curve_codim2.rs`) with a failing cubic
/// form; returns the branch and the number of injected failures.
fn map_pd_curve(error: EngineError) -> (Branch, usize) {
    let mut m = model_of("map_henon_gen");
    let mut inj = Inject::with_error(&mut *m, error, |c| is_cubic_after(c, 0.5));
    let hits = inj.hits.clone();
    let setup = CurveSetup::new([0, 1], vec![0.625, 0.0], vec![0.5, 0.5]);
    let opts = Options {
        direction: 1.0,
        bounds: (-2.0, 2.5),
        ..Options::default()
    };
    let br = continue_map_pd_curve(
        &mut inj,
        &setup,
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap();
    (br, hits.get())
}

/// The map NS curve of the same model, with the CH coefficient at index 7.
fn map_ns_curve(error: EngineError) -> (Branch, usize) {
    let mut m = model_of("map_henon_gen");
    let mut inj = Inject::with_error(&mut *m, error, |c| is_cubic_after(c, 1.4));
    let hits = inj.hits.clone();
    let setup = CurveSetup::new([0, 1], vec![0.69, 1.15], vec![0.3, 0.3]);
    let opts = Options {
        direction: 1.0,
        bounds: (0.0, 2.0),
        ..Options::default()
    };
    let br = continue_map_ns_curve(
        &mut inj,
        &setup,
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap();
    (br, hits.get())
}

/// mapcurve.rs `gpd` (index 2 of a PD curve): the cubic form hits the cap, which became a NaN
/// GPD test.
#[test]
fn map_pd_curve_gpd_cap_stops_the_run() {
    let (br, hits) = map_pd_curve(cap_error());
    assert!(hits > 0, "the fault never fired");
    assert_cap_stop(&br, "map PD curve GPD");
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
    assert!(
        br.points.iter().all(|p| !p.codim2[2].is_nan()),
        "a NaN GPD test was written"
    );
}

#[test]
fn map_pd_curve_gpd_other_error_keeps_the_nan() {
    let (br, hits) = map_pd_curve(EngineError::NotFinite("t".into()));
    assert!(hits > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    assert!(
        br.points.iter().any(|p| p.codim2[2].is_nan()),
        "no NaN GPD test"
    );
}

/// mapcurve.rs NS `codim2_tests` (CH, index 7): `map_ns_core` hits the cap, which `.ok()` and
/// `map_or(NAN)` turned into a NaN CH test.
#[test]
fn map_ns_curve_ch_cap_stops_the_run() {
    let (br, hits) = map_ns_curve(cap_error());
    assert!(hits > 0, "the fault never fired");
    assert_cap_stop(&br, "map NS curve CH");
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
}

#[test]
fn map_ns_curve_ch_other_error_keeps_the_nan() {
    let clean = map_ns_curve_clean();
    let (br, hits) = map_ns_curve(EngineError::NotFinite("t".into()));
    assert!(hits > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    let nans = |b: &Branch| b.points.iter().filter(|p| p.codim2[7].is_nan()).count();
    assert!(
        nans(&br) > nans(&clean),
        "{} against {}",
        nans(&br),
        nans(&clean)
    );
}

fn map_ns_curve_clean() -> Branch {
    let mut m = model_of("map_henon_gen");
    let mut inj = Inject::with_error(&mut *m, cap_error(), |_| false);
    let setup = CurveSetup::new([0, 1], vec![0.69, 1.15], vec![0.3, 0.3]);
    let opts = Options {
        direction: 1.0,
        bounds: (0.0, 2.0),
        ..Options::default()
    };
    continue_map_ns_curve(
        &mut inj,
        &setup,
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

// ---- Delay Hopf curves ----

const WRIGHT3: &str = "par a=1.5, b=0\nu'=-a*delay(u,1)*(1+u)+b*u^3\ndone\n";

/// The Hopf curve of Wright's cubic equation (`dde_hopf_lyap.rs`) with every `Dir` entry whose
/// slot count is at least `k` failing once `p[1]` is above `after`. The Hopf curve system uses
/// `F`, `JacU` and `JacP` only, so `Dir` is the first Lyapunov coefficient (3 slots at most) or
/// `l2` (4 and 5 slots).
fn wright_run(error: EngineError, k: usize, after: f64) -> (DdeHopfBranch, usize) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", WRIGHT3);
    let l = load_source(top, &mut map, &mut MapLoader::new());
    let m = build_dde(&l.expanded, &l.info, Evaluator::Jit).unwrap();
    // The leaked evaluator lives to the end of the test process.
    let inner: &'static mut dyn ModelEval = Box::leak(m.eval);
    let inj = Inject::with_error(inner, error, move |c| {
        matches!(c.kind, EntryKind::Dir(s) if s.len() >= k) && c.p[1] > after
    });
    let hits = inj.hits.clone();
    let mut m = DdeModel {
        eval: Box::new(inj),
        delays: m.delays,
        n_user_params: m.n_user_params,
    };
    let setup = DdeHopfSetup {
        free: [0, 1],
        params: vec![1.57, 0.0],
        x0: vec![0.001],
        omega: 1.57,
    };
    let opts = Options {
        hmax: 0.05,
        bounds: (-1.0, 1.0),
        direction: 1.0,
        tol_x: 1e-12,
        tol_f: 1e-12,
        max_newton: 20,
        ..Options::default()
    };
    let br =
        dde_continue_hopf_with_lyapunov(&mut m, &setup, &opts, &AtomicBool::new(false)).unwrap();
    (br, hits.get())
}

/// dde/hopf.rs `l1_at`: the first Lyapunov coefficient hits the cap, which `Err(_) => NAN`
/// stored as a NaN `l1` (and so as an undefined GH test).
#[test]
fn dde_hopf_l1_cap_stops_the_run() {
    let (br, hits) = wright_run(cap_error(), 1, 0.1);
    assert!(hits > 0, "the fault never fired");
    assert!(
        matches!(br.stop, StopReason::Error(EngineError::CallLimit(_))),
        "dde l1: expected stop Error(CallLimit), got {:?}",
        br.stop
    );
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
    assert!(
        br.points.iter().all(|p| p.l1.is_none_or(|v| !v.is_nan())),
        "a NaN l1 was written"
    );
}

#[test]
fn dde_hopf_l1_other_error_keeps_the_nan() {
    let (br, hits) = wright_run(EngineError::NotFinite("t".into()), 1, 0.1);
    assert!(hits > 0, "the fault never fired");
    assert!(
        !matches!(br.stop, StopReason::Error(EngineError::CallLimit(_))),
        "{:?}",
        br.stop
    );
    assert!(
        br.points.iter().any(|p| p.l1.is_none_or(|v| v.is_nan())),
        "no undefined l1"
    );
}

/// dde/hopf.rs `l2_at` in the end-of-run pass over the GH points: `l2` needs the 4 and 5 slot
/// forms, which the cap fails after the run reached its bound.
#[test]
fn dde_hopf_l2_cap_stops_the_run() {
    let (br, hits) = wright_run(cap_error(), 4, f64::NEG_INFINITY);
    assert!(hits > 0, "the fault never fired");
    assert!(
        matches!(br.stop, StopReason::Error(EngineError::CallLimit(_))),
        "dde l2: expected stop Error(CallLimit), got {:?} (specials {:?})",
        br.stop,
        br.specials
            .iter()
            .map(|s| (&s.label, s.l2))
            .collect::<Vec<_>>()
    );
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
}

#[test]
fn dde_hopf_l2_other_error_keeps_the_old_result() {
    let (br, hits) = wright_run(EngineError::NotFinite("t".into()), 4, f64::NEG_INFINITY);
    assert!(hits > 0, "the fault never fired");
    assert!(
        !matches!(br.stop, StopReason::Error(EngineError::CallLimit(_))),
        "{:?}",
        br.stop
    );
    let gh = br.specials.iter().find(|s| s.label == "GH").unwrap();
    assert!(gh.l2.is_none());
}

// ---- Homoclinic curves ----

/// The HOM curve of `sandstede` with the inclination flip tests on (`hom_inclination.rs`).
/// The adjoint solve of the flip tests evaluates `F` then `JacU` at every collocation node in
/// one unbroken sequence of 600 or more calls, and the curve system does not. So a `JacU` that
/// ends 300 such pairs belongs to the adjoint solve; it fails once the free parameter `p[5]` is
/// above 0.2, so that the start is a valid point. IFS and IFU are at
/// indices 12 and 13 of `Point::codim2`.
fn hom_flip_run(error: EngineError) -> (Branch, usize) {
    use pc_engine::{HomStart, HomStartKind, continue_hom_curve};
    let period = 20.0;
    let (ntst, ncol) = ((8.0 * period) as usize, 4);
    let mut m = model_of("sandstede");
    let mut pairs = 0usize;
    let mut inj = Inject::with_error(&mut *m, error, move |c| {
        let after_f = matches!(c.prev, Some(EntryKind::F));
        match c.kind {
            EntryKind::JacU if after_f => pairs += 1,
            EntryKind::F if matches!(c.prev, Some(EntryKind::JacU)) => {}
            _ => pairs = 0,
        }
        matches!(c.kind, EntryKind::JacU) && pairs >= 300 && c.p[5] > 0.2
    });
    let hits = inj.hits.clone();
    let start = HomStart {
        kind: HomStartKind::Orbit,
        free: [4, 5],
        params: vec![-0.25, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        orbit: common::plane_orbit(-0.25, 1.0, period, ntst, ncol, 1),
        period,
        mesh: None,
        ntst,
        ncol,
        inclination_flips: true,
    };
    let opts = Options {
        max_newton: 15,
        hmax: 0.05,
        bounds: (-0.1, 0.5),
        ..Options::default()
    };
    let br = continue_hom_curve(
        &mut inj,
        &start,
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap();
    (br, hits.get())
}

/// cycle/hom.rs `flip_tests`: the IFS and IFU solve hits the cap, which `.ok()` turned into
/// NaN tests.
#[test]
fn hom_flip_tests_cap_stops_the_run() {
    let (br, hits) = hom_flip_run(cap_error());
    assert!(hits > 0, "the fault never fired");
    assert_cap_stop(&br, "hom flip tests");
    assert!(br.points.len() >= 3, "points kept: {}", br.points.len());
    assert!(
        br.points
            .iter()
            .all(|p| p.codim2[12].is_finite() && p.codim2[13].is_finite()),
        "a NaN IFS or IFU test was written"
    );
}

#[test]
fn hom_flip_tests_other_error_keeps_the_nan() {
    let (br, hits) = hom_flip_run(EngineError::NotFinite("t".into()));
    assert!(hits > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    assert!(
        br.points
            .iter()
            .any(|p| p.codim2[12].is_nan() && p.codim2[13].is_nan()),
        "no NaN IFS and IFU"
    );
}

// ---- One-shot sweeps ----
//
// A sweep fails one call of a run, once, for each call in a range, and checks that the run
// ends in the cap error whenever the failure fired. It reaches the passes that no predicate on
// the call can single out, such as the delay Hopf-Hopf test, the second critical frequency and
// the classification floor of the equilibrium branch.

/// What one run with a one-shot fault gives: the stop reason or the error, the number of
/// injected failures and the number of model calls made.
struct Outcome {
    result: Result<StopReason, EngineError>,
    hits: usize,
    calls: usize,
}

impl Outcome {
    /// The failure ended the run as the rule asks: a kept branch that stops with the cap error
    /// or, when `err_ok` is set (a failure before the run has a point to keep), the cap error
    /// out of the run.
    fn ends_in_the_cap(&self, err_ok: bool) -> bool {
        match &self.result {
            Ok(stop) => is_cap(stop),
            Err(e) => err_ok && matches!(e, EngineError::CallLimit(_)),
        }
    }
}

/// Runs `run(Some(k))` for each `k` in `ks(total)`, where `total` is the number of calls of the
/// clean run `run(None)`, and returns the calls whose failure did not end in the cap error.
/// The cap error out of the run counts only for `k < err_before`.
fn hidden_failures(
    ks: impl Fn(usize) -> Vec<usize>,
    err_before: usize,
    run: impl Fn(Option<usize>) -> Outcome,
) -> Vec<String> {
    let clean = run(None);
    assert_eq!(clean.hits, 0);
    assert!(
        matches!(&clean.result, Ok(stop) if !is_cap(stop)),
        "the clean run: {:?}",
        clean.result
    );
    let mut bad = Vec::new();
    for k in ks(clean.calls) {
        let o = run(Some(k));
        assert_eq!(o.hits, 1, "call {k} of {}", clean.calls);
        if !o.ends_in_the_cap(k < err_before) {
            bad.push(format!("call {k} of {}: {:?}", clean.calls, o.result));
        }
    }
    bad
}

/// The delay model `src` behind an `Inject` that fails the call number `k` once, and the
/// counter of calls the run made.
fn one_shot_dde(src: &str, k: Option<usize>) -> (DdeModel, Rc<Cell<usize>>, Rc<Cell<usize>>) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load_source(top, &mut map, &mut MapLoader::new());
    let m = build_dde(&l.expanded, &l.info, Evaluator::Jit).unwrap();
    // The leaked evaluator lives to the end of the test process.
    let inner: &'static mut dyn ModelEval = Box::leak(m.eval);
    let calls = Rc::new(Cell::new(0));
    let seen = calls.clone();
    let inj = Inject::call_limit(inner, move |c| {
        seen.set(c.index + 1);
        Some(c.index) == k
    });
    let hits = inj.hits.clone();
    let m = DdeModel {
        eval: Box::new(inj),
        delays: m.delays,
        n_user_params: m.n_user_params,
    };
    (m, hits, calls)
}

const COUPLED: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)\n\
    y'=-d*x-b*delay(y,1.5)\n\
    done\n";

/// The Hopf curve of the coupled delay model up through its Hopf-Hopf point (`dde_hopf_hh.rs`,
/// test 3), with one call failing once. dde/hopf.rs `hh_at` met the failure as a lost pair and a
/// NaN test, and `omega2_at` as a missing second frequency.
fn coupled_hh_at(k: Option<usize>) -> Outcome {
    coupled_hh_steps(k, Options::default().max_steps)
}

/// `coupled_hh_at` with at most `max_steps` steps.
fn coupled_hh_steps(k: Option<usize>, max_steps: usize) -> Outcome {
    let (mut m, hits, calls) = one_shot_dde(COUPLED, k);
    let setup = DdeHopfSetup {
        free: [0, 1],
        params: vec![1.46, 0.8, 1.0, 0.1],
        x0: vec![0.001, 0.001],
        omega: 1.54,
    };
    let opts = Options {
        hmax: 0.05,
        bounds: (0.5, 1.4),
        direction: 1.0,
        tol_x: 1e-12,
        tol_f: 1e-12,
        max_newton: 20,
        max_steps,
        ..Options::default()
    };
    let extras = DdeHopfExtras {
        lyapunov: false,
        hh: Some(RootOptions {
            n: 30,
            re_min: -3.0,
        }),
        coefficients: false,
    };
    let r = dde_continue_hopf_with(&mut m, &setup, &opts, &AtomicBool::new(false), &extras)
        .map(|b| b.stop);
    Outcome {
        result: r,
        hits: hits.get(),
        calls: calls.get(),
    }
}

/// Every `stride`-th call, and each of the last `tail` calls.
fn every(stride: usize, tail: usize) -> impl Fn(usize) -> Vec<usize> {
    move |total| {
        let mut ks: Vec<usize> = (0..total).step_by(stride).collect();
        ks.extend(total.saturating_sub(tail)..total);
        ks.sort_unstable();
        ks.dedup();
        ks
    }
}

#[test]
fn delay_hopf_hopf_test_and_second_frequency_cap_stop_the_run() {
    // A run with no step makes the calls of the start and its first point; a failure after
    // those must keep the branch.
    let first = coupled_hh_steps(None, 0).calls;
    let bad = hidden_failures(every(29, 80), first, coupled_hh_at);
    assert!(bad.is_empty(), "{bad:#?}");
}

const HUTCH: &str = "par r=1.5, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n";

/// The equilibrium branch of Hutchinson's equation through its Hopf point (`dde_branch.rs`),
/// with one call failing once. dde/branch.rs `omega_floor` met the failure as the constant
/// floor.
fn hutchinson_at(k: Option<usize>) -> Outcome {
    let (mut m, hits, calls) = one_shot_dde(HUTCH, k);
    let setup = EpSetup::new(0, vec![1.3, 1.0], vec![1.0]);
    let ro = RootOptions {
        n: 30,
        re_min: -0.5,
    };
    let opts = Options {
        hmax: 0.05,
        bounds: (1.2, 2.0),
        direction: 1.0,
        ..Options::default()
    };
    let r = dde_continue_equilibrium(&mut m, &setup, &ro, &opts, &AtomicBool::new(false))
        .map(|b| b.stop);
    Outcome {
        result: r,
        hits: hits.get(),
        calls: calls.get(),
    }
}

#[test]
fn delay_branch_classification_floor_cap_stops_the_run() {
    // `continue_equilibrium` gives the error out of the run (STATUS, P44 item 5, limits).
    let bad = hidden_failures(every(1, 0), usize::MAX, hutchinson_at);
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The known limit, pinned: a cap hit late in the delay equilibrium branch is an error out of
/// the run and the points before it are lost. A change that keeps them should change this test.
#[test]
fn delay_equilibrium_branch_cap_is_an_error_out_of_the_run() {
    let clean = hutchinson_at(None);
    let o = hutchinson_at(Some(clean.calls - 1));
    assert_eq!(o.hits, 1);
    assert!(
        matches!(&o.result, Err(EngineError::CallLimit(_))),
        "{:?}",
        o.result
    );
}

// ---- Homoclinic accept ----

/// The HOM curve of `sandstede` as in `hom_flip_run` but without the flip tests, with the call
/// number `k` failing once. cycle/hom.rs `accept` evaluates `JacU` at the saddle twice (the
/// eigenvalue check and the bases) and `let Ok(..) else { return }` dropped a failure there.
/// Returns the outcome and the kinds and states of all calls, for the sweep to find those two.
fn hom_accept_at(k: Option<usize>) -> (Outcome, Vec<(bool, Vec<f64>)>) {
    hom_accept_steps(k, 3)
}

/// `hom_accept_at` with at most `max_steps` steps.
fn hom_accept_steps(k: Option<usize>, max_steps: usize) -> (Outcome, Vec<(bool, Vec<f64>)>) {
    use pc_engine::{HomStart, HomStartKind, continue_hom_curve};
    let period = 20.0;
    let (ntst, ncol) = ((8.0 * period) as usize, 4);
    let mut m = model_of("sandstede");
    let calls = Rc::new(Cell::new(0));
    let seen = calls.clone();
    let trace = Rc::new(std::cell::RefCell::new(Vec::new()));
    let log = trace.clone();
    let mut inj = Inject::call_limit(&mut *m, move |c| {
        seen.set(c.index + 1);
        log.borrow_mut()
            .push((matches!(c.kind, EntryKind::JacU), c.x.to_vec()));
        Some(c.index) == k
    });
    let hits = inj.hits.clone();
    let start = HomStart {
        kind: HomStartKind::Orbit,
        free: [4, 5],
        params: vec![-0.25, 1.0, -3.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        orbit: common::plane_orbit(-0.25, 1.0, period, ntst, ncol, 1),
        period,
        mesh: None,
        ntst,
        ncol,
        inclination_flips: false,
    };
    let opts = Options {
        max_newton: 15,
        hmax: 0.05,
        max_steps,
        bounds: (-0.1, 0.5),
        ..Options::default()
    };
    let r = continue_hom_curve(
        &mut inj,
        &start,
        &opts,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .map(|b| b.stop);
    let out = Outcome {
        result: r,
        hits: hits.get(),
        calls: calls.get(),
    };
    let t = trace.borrow().clone();
    (out, t)
}

#[test]
fn hom_accept_cap_stops_the_run() {
    let (clean, trace) = hom_accept_at(None);
    assert!(
        matches!(&clean.result, Ok(s) if !is_cap(s)),
        "{:?}",
        clean.result
    );
    // The two `JacU` calls of an accept are neighbours at the same state; the curve system's
    // own `JacU` calls come between `F` calls.
    let accept_calls: Vec<usize> = (1..trace.len())
        .filter(|&i| trace[i].0 && trace[i - 1].0 && trace[i].1 == trace[i - 1].1)
        .flat_map(|i| [i - 1, i])
        .collect();
    // The start's Newton solve also evaluates `JacU` twice at the saddle, before there is a
    // point to keep. A run with no step makes those calls and the first point's; a failure
    // after them must keep the branch.
    let first = hom_accept_steps(None, 0).0.calls;
    assert!(
        accept_calls.iter().any(|&k| k >= first),
        "no accept call after the first point in the trace"
    );
    let mut bad = Vec::new();
    for k in accept_calls {
        let (o, _) = hom_accept_at(Some(k));
        assert_eq!(o.hits, 1, "call {k}");
        if !o.ends_in_the_cap(k < first) {
            bad.push(format!("call {k} of {}: {:?}", clean.calls, o.result));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

// ---- Annotation arms of the LP and Hopf curves: CP, ZH and HH ----
//
// The post-run `annotate` pass of both curves is the last thing the run does with the model,
// and the curve itself makes no call of the kinds below that the pass makes at the end. So a
// first run through a counting double gives the number of calls, and the injected run fails
// the last call (inside the coefficient code) or, for the HH arm, the call before the last
// `JacU` (the arm's own state Jacobian, which comes before the one in `double_hopf_at`).

type Runner<'a> = &'a mut dyn FnMut(&mut dyn ModelEval, &CurveSetup, &Options) -> Branch;
type Builder = fn(Runner) -> Branch;
type RunFn = fn(&mut dyn ModelEval, &CurveSetup, &Options) -> Branch;

/// Which call of the annotation pass fails.
#[derive(Clone, Copy, Debug)]
enum Tail {
    /// The last call of the run: inside the coefficient code.
    Last,
    /// The `JacU` before the last one: the state Jacobian of the HH arm.
    ArmJacobian,
}

/// Runs the curve `build` with `go` once through a counting double, then again with the call
/// that `tail` names failing with `error`. Returns the failing run and the number of failures.
fn tail_run(build: Builder, go: RunFn, error: EngineError, tail: Tail) -> (Branch, usize) {
    let total = Rc::new(Cell::new(0));
    let last_jacu = Rc::new(Cell::new(0));
    let (t, j) = (total.clone(), last_jacu.clone());
    build(&mut |m, s, o| {
        let (t, j) = (t.clone(), j.clone());
        let mut inj = Inject::call_limit(m, move |c| {
            t.set(c.index + 1);
            if matches!(c.kind, EntryKind::JacU) {
                j.set(c.index);
            }
            false
        });
        go(&mut inj, s, o)
    });
    let target = match tail {
        Tail::Last => total.get() - 1,
        Tail::ArmJacobian => last_jacu.get() - 1,
    };
    let hits = Rc::new(Cell::new(0));
    let h = hits.clone();
    let br = build(&mut |m, s, o| {
        let mut inj = Inject::with_error(m, error.clone(), move |c| c.index == target);
        let r = go(&mut inj, s, o);
        h.set(inj.hits.get());
        r
    });
    (br, hits.get())
}

/// The run without a fault.
fn clean_run(build: Builder, go: RunFn) -> Branch {
    build(&mut |m, s, o| go(m, s, o))
}

/// The LP curve of the cusp model through the CP at the origin, as `codim2.rs` `test7_cusp_cp`.
fn cusp_lp_curve(run: Runner) -> Branch {
    let mut m = model_of("cusp");
    let cancel = AtomicBool::new(false);
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, vec![0.0, 1.0], vec![-1.0]),
        &Options {
            bounds: (-1.0, 1.0),
            ..Options::default()
        },
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let lp: Vec<_> = ep.specials.iter().filter(|s| s.label == "LP").collect();
    assert!(!lp.is_empty(), "no LP on the equilibrium branch");
    let setup = CurveSetup::new([0, 1], vec![lp[0].point.alpha, 1.0], lp[0].point.u.clone());
    let opts = Options {
        bounds: (-0.5, 1.5),
        ..Options::default()
    };
    run(&mut *m, &setup, &opts)
}

const ZH_P: [f64; 11] = [1.5, 0.8, 0.3, 0.6, 2.0, 1.0, 0.4, 0.5, -0.3, -0.4, 0.7];

/// The parameters of `zh.ode` with b1 chosen for the equilibrium x = -1 on y = z = 0.
fn zh_params(b2: f64) -> Vec<f64> {
    let x0: f64 = -1.0;
    let mut p = vec![-(ZH_P[0] * x0 * x0 + ZH_P[2] * x0 * x0 * x0), b2];
    p.extend_from_slice(&ZH_P);
    p
}

/// The LP curve of `zh.ode` through its ZH point (`codim2.rs` `zh_on_lp_curve`).
fn zh_lp_curve(run: Runner) -> Branch {
    let mut m = model_of("zh");
    let cancel = AtomicBool::new(false);
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, zh_params(0.5), vec![-1.0, 0.0, 0.0]),
        &Options {
            bounds: (-2.0, 1.0),
            ..Options::default()
        },
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let lp: Vec<_> = ep.specials.iter().filter(|s| s.label == "LP").collect();
    assert_eq!(lp.len(), 1, "{:?}", ep.specials);
    let mut p = zh_params(0.5);
    p[0] = lp[0].point.alpha;
    let setup = CurveSetup::new([0, 1], p, lp[0].point.u.clone());
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 1.0),
        ..Options::default()
    };
    run(&mut *m, &setup, &opts)
}

/// The Hopf curve of model `name` started at the single H of its equilibrium branch in b1.
fn hopf_curve_from_h(
    name: &str,
    params: Vec<f64>,
    u0: Vec<f64>,
    ep_bounds: (f64, f64),
    opts: &Options,
    run: Runner,
) -> Branch {
    let mut m = model_of(name);
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(0, params.clone(), u0),
        &Options {
            bounds: ep_bounds,
            ..Options::default()
        },
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap();
    let h: Vec<_> = ep.specials.iter().filter(|s| s.label == "H").collect();
    assert_eq!(h.len(), 1, "{:?}", ep.specials);
    let mut q = params;
    q[0] = h[0].point.alpha;
    let setup = CurveSetup::new([0, 1], q, h[0].point.u.clone());
    run(&mut *m, &setup, opts)
}

/// The Hopf curve of `zh.ode` through its ZH point (`codim2.rs` `zh_on_hopf_curve`).
fn zh_hopf_curve(run: Runner) -> Branch {
    let opts = Options {
        bounds: (-0.5, 1.0),
        ..Options::default()
    };
    hopf_curve_from_h(
        "zh",
        zh_params(0.3),
        vec![-1.0, 0.0, 0.0],
        (-2.0, 1.0),
        &opts,
        run,
    )
}

/// The Hopf curve of `hh.ode` through its HH point (`codim2.rs` `hh_on_hopf_curve`).
fn hh_hopf_curve(run: Runner) -> Branch {
    let p = vec![
        -0.3, 0.4, 2.0, 0.9, -0.8, 0.3, 1.7, -0.4, 0.6, 0.5, -1.1, 0.2,
    ];
    let opts = Options {
        bounds: (-0.5, 1.0),
        direction: -1.0,
        ..Options::default()
    };
    hopf_curve_from_h("hh", p, vec![0.0; 4], (-0.5, 0.5), &opts, run)
}

fn special<'a>(br: &'a Branch, label: &str) -> &'a pc_engine::SpecialPoint {
    br.specials
        .iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("no {label} in {:?}", br.specials))
}

fn no_nan(sp: &pc_engine::SpecialPoint) {
    let e = &sp.extras;
    assert!(
        e.coefficient.is_none_or(|c| !c.is_nan())
            && e.coefficient2.is_none_or(|c| !c.is_nan())
            && e.values.iter().all(|(_, v)| !v.is_nan()),
        "a NaN coefficient: {e:?}"
    );
}

/// The cap test of one annotation arm: the stop is the cap, the points are those of the clean
/// run, the special is the one of the clean run without coefficients, and nothing is NaN.
fn check_cap_arm(build: Builder, go: RunFn, label: &str, tail: Tail) {
    let clean = clean_run(build, go);
    let have = |sp: &pc_engine::SpecialPoint| {
        sp.extras.coefficient.is_some() || !sp.extras.values.is_empty()
    };
    assert!(
        have(special(&clean, label)),
        "the clean {label} has no coefficient"
    );
    let (br, hits) = tail_run(build, go, cap_error(), tail);
    assert!(hits > 0, "the fault never fired");
    assert_cap_stop(&br, &format!("{label} annotation, {tail:?}"));
    assert_eq!(br.points.len(), clean.points.len(), "points kept");
    let sp = special(&br, label);
    assert!(!have(sp), "{label} kept a value: {:?}", sp.extras);
    no_nan(sp);
}

/// The paired test: another error keeps the old result, which is no coefficient, no warning
/// and the stop reason of the clean run.
fn check_other_arm(build: Builder, go: RunFn, label: &str, tail: Tail) {
    let clean = clean_run(build, go);
    let (br, hits) = tail_run(build, go, EngineError::NotFinite("test".into()), tail);
    assert!(hits > 0, "the fault never fired");
    assert!(!is_cap(&br.stop), "{:?}", br.stop);
    assert_eq!(br.stop, clean.stop);
    assert_eq!(br.warnings, clean.warnings, "a warning was added");
    assert_eq!(br.points.len(), clean.points.len());
    let sp = special(&br, label);
    assert!(
        sp.extras.coefficient.is_none() && sp.extras.values.is_empty(),
        "{label} kept a value: {:?}",
        sp.extras
    );
}

/// lpcurve.rs `annotate`, CP arm: the cusp coefficient hits the cap. The `?` was missing, so
/// the coefficient stayed empty and the run went on.
#[test]
fn lp_curve_cp_annotation_cap_stops_the_run() {
    check_cap_arm(cusp_lp_curve, lp_run, "CP", Tail::Last);
}

#[test]
fn lp_curve_cp_annotation_other_error_keeps_the_old_result() {
    check_other_arm(cusp_lp_curve, lp_run, "CP", Tail::Last);
}

/// lpcurve.rs `annotate`, ZH arm: the zero-Hopf coefficients hit the cap.
#[test]
fn lp_curve_zh_annotation_cap_stops_the_run() {
    check_cap_arm(zh_lp_curve, lp_run, "ZH", Tail::Last);
}

#[test]
fn lp_curve_zh_annotation_other_error_keeps_the_old_result() {
    check_other_arm(zh_lp_curve, lp_run, "ZH", Tail::Last);
}

/// hopfcurve.rs `annotate`, ZH arm.
#[test]
fn hopf_curve_zh_annotation_cap_stops_the_run() {
    check_cap_arm(zh_hopf_curve, hopf_run, "ZH", Tail::Last);
}

#[test]
fn hopf_curve_zh_annotation_other_error_keeps_the_old_result() {
    check_other_arm(zh_hopf_curve, hopf_run, "ZH", Tail::Last);
}

/// hopfcurve.rs `annotate`, HH arm, in `double_hopf_at`.
#[test]
fn hopf_curve_hh_annotation_cap_stops_the_run() {
    check_cap_arm(hh_hopf_curve, hopf_run, "HH", Tail::Last);
}

#[test]
fn hopf_curve_hh_annotation_other_error_keeps_the_old_result() {
    check_other_arm(hh_hopf_curve, hopf_run, "HH", Tail::Last);
}

/// The Hopf curve of `bt.ode` through its BT point (`codim2.rs` `test8_bt_h`, s = 1).
fn bt_hopf_curve(run: Runner) -> Branch {
    let opts = Options {
        bounds: (-0.3, 1.0),
        ..Options::default()
    };
    hopf_curve_from_h(
        "bt",
        vec![-1.0, 0.5, 1.0],
        vec![-1.0, 0.0],
        (-1.0, 1.0),
        &opts,
        run,
    )
}

/// hopfcurve.rs `annotate`, BT arm: the BT coefficients hit the cap.
#[test]
fn hopf_curve_bt_annotation_cap_stops_the_run() {
    check_cap_arm(bt_hopf_curve, hopf_run, "BT", Tail::Last);
}

#[test]
fn hopf_curve_bt_annotation_other_error_keeps_the_old_result() {
    check_other_arm(bt_hopf_curve, hopf_run, "BT", Tail::Last);
}

/// hopfcurve.rs `annotate`, HH arm, at the state Jacobian that finds the second frequency. A
/// cap error there was dropped by `.ok()` and `and_then`, and the special had no coefficients.
#[test]
fn hopf_curve_hh_jacobian_cap_stops_the_run() {
    check_cap_arm(hh_hopf_curve, hopf_run, "HH", Tail::ArmJacobian);
}

#[test]
fn hopf_curve_hh_jacobian_other_error_keeps_the_old_result() {
    check_other_arm(hh_hopf_curve, hopf_run, "HH", Tail::ArmJacobian);
}

// ---- Delay Hopf curve: coefficients at ZH and HH (dde/hopf.rs end-of-run pass) ----

const ZH_DELAY: &str = "par a=0, tau=1, b=1\nx'=-(a+y)*x-b*delay(x,tau)\ny'=a-y*y\ndone\n";
const COUPLED_NL: &str = "par a=1.4, b=1.2, c=1, d=0.1\n\
    x'=c*y-a*delay(x,1)+x*x+x*y-x*x*x\n\
    y'=-d*x-b*delay(y,1.5)+y*y*y+x*y*y\n\
    done\n";

/// A delay Hopf run with `coefficients` on (`dde_hopf_coef.rs`), where the last call of the run
/// fails with `error` when `fail` is set. The end-of-run pass makes the last call of the run,
/// inside `zero_hopf` or `hopf_hopf`. Returns the branch, the failures and the call count.
fn coefficient_run(
    src: &str,
    setup: &DdeHopfSetup,
    opts: &Options,
    hh: bool,
    fail: Option<(EngineError, usize)>,
) -> (DdeHopfBranch, usize, usize) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load_source(top, &mut map, &mut MapLoader::new());
    let m = build_dde(&l.expanded, &l.info, Evaluator::Jit).unwrap();
    // The leaked evaluator lives to the end of the test process.
    let inner: &'static mut dyn ModelEval = Box::leak(m.eval);
    let calls = Rc::new(Cell::new(0));
    let seen = calls.clone();
    let (error, target) = fail.unzip();
    let inj = Inject::with_error(inner, error.unwrap_or_else(cap_error), move |c| {
        seen.set(c.index + 1);
        Some(c.index) == target
    });
    let hits = inj.hits.clone();
    let mut m = DdeModel {
        eval: Box::new(inj),
        delays: m.delays,
        n_user_params: m.n_user_params,
    };
    let extras = DdeHopfExtras {
        lyapunov: false,
        hh: hh.then_some(RootOptions {
            n: 30,
            re_min: -3.0,
        }),
        coefficients: true,
    };
    let br = dde_continue_hopf_with(&mut m, setup, opts, &AtomicBool::new(false), &extras).unwrap();
    (br, hits.get(), calls.get())
}

type DdeCase = (&'static str, DdeHopfSetup, Options, bool, &'static str);

/// The ZH and HH runs of `dde_hopf_coef.rs`.
fn dde_cases() -> Vec<DdeCase> {
    let o = |hmax, bounds| Options {
        hmax,
        bounds,
        direction: 1.0,
        tol_x: 1e-12,
        tol_f: 1e-12,
        max_newton: 20,
        ..Options::default()
    };
    vec![
        (
            ZH_DELAY,
            DdeHopfSetup {
                free: [0, 1],
                params: vec![0.07, 1.4071360574511174, 1.0],
                x0: vec![0.001, -0.249],
                omega: 0.98,
            },
            o(0.05, (1.3, 2.0)),
            false,
            "ZH",
        ),
        (
            COUPLED_NL,
            DdeHopfSetup {
                free: [0, 1],
                params: vec![1.46, 0.8, 1.0, 0.1],
                x0: vec![0.001, 0.001],
                omega: 1.54,
            },
            o(0.05, (0.5, 1.4)),
            true,
            "HH",
        ),
    ]
}

fn dde_special<'a>(br: &'a DdeHopfBranch, label: &str) -> &'a pc_engine::DdeHopfSpecial {
    br.specials
        .iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("no {label} in {:?}", br.specials))
}

/// dde/hopf.rs end-of-run pass, `special_coefficients` at a ZH and at an HH special: the call
/// fails with the cap, the stop is the cap, the points and the earlier specials are kept and
/// the special has no coefficients, no NaN and no warning for it.
#[test]
fn dde_hopf_special_coefficients_cap_stops_the_run() {
    for (src, setup, opts, hh, label) in dde_cases() {
        let (clean, _, total) = coefficient_run(src, &setup, &opts, hh, None);
        assert!(
            !dde_special(&clean, label).coefficients.is_empty(),
            "the clean {label} has no coefficients"
        );
        let (br, hits, _) = coefficient_run(src, &setup, &opts, hh, Some((cap_error(), total - 1)));
        assert_eq!(hits, 1, "{label}: the fault never fired");
        assert!(
            matches!(br.stop, StopReason::Error(EngineError::CallLimit(_))),
            "{label}: expected stop Error(CallLimit), got {:?} (warnings {:?})",
            br.stop,
            br.warnings
        );
        assert_eq!(br.points.len(), clean.points.len(), "{label}: points kept");
        assert_eq!(br.specials.len(), clean.specials.len());
        let sp = dde_special(&br, label);
        assert!(sp.coefficients.is_empty(), "{label}: {:?}", sp.coefficients);
        assert!(
            br.warnings.iter().all(|w| !w.starts_with("coefficients")),
            "{label}: {:?}",
            br.warnings
        );
    }
}

/// Another error at the same call keeps the old result: no coefficients and the warning that
/// `coefficient_warning` words, and the stop reason of the clean run.
#[test]
fn dde_hopf_special_coefficients_other_error_keeps_the_old_result() {
    for (src, setup, opts, hh, label) in dde_cases() {
        let (clean, _, total) = coefficient_run(src, &setup, &opts, hh, None);
        let err = EngineError::NotFinite("test".into());
        let (br, hits, _) = coefficient_run(src, &setup, &opts, hh, Some((err, total - 1)));
        assert_eq!(hits, 1, "{label}: the fault never fired");
        assert_eq!(br.stop, clean.stop, "{label}");
        assert!(dde_special(&br, label).coefficients.is_empty());
        let w: Vec<_> = br
            .warnings
            .iter()
            .filter(|w| w.starts_with("coefficients"))
            .collect();
        assert_eq!(w.len(), 1, "{label}: {:?}", br.warnings);
        assert!(w[0].contains(label), "{label}: {w:?}");
    }
}
