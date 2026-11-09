//! The problem file and its options, read by hand from a `serde_json::Value` so that
//! every error can name the key that is wrong. Unknown keys are errors.
//!
//! `pc-cli` reads a whole problem file with [`Mode::Cli`]. The C ABI uses [`Mode::Abi`],
//! where the state and the parameters come as buffers and the keys `u0` and `params`
//! are errors.

use pc_engine::{
    CYCLE_LABELS, CurveSetup, CycleCurveStart, CycleSetup, CycleStart, EpSetup, Evaluator,
    HOM_CURVE_LABELS, HOPF_CURVE_LABELS, HomStart, HomStartKind, HopfStart, IvpOptions,
    LP_CURVE_LABELS, LPC_CURVE_LABELS, NCOL_RANGE, NS_CURVE_LABELS, Options, PD_CURVE_LABELS,
    PdStart, RootOptions, RunKind, RunSpec,
};
use pc_engine::{DDE_HOPF_OMEGA_MIN, DdeFloquetOptions, DdeHopfExtras};
use pc_ir::IrModel;
use pc_ode::{Model, ModelInfo};
use serde_json::{Map, Value};

/// What a run follows: an equilibrium branch, the curve of fold (LP) or Hopf points of two
/// parameters, a branch of periodic orbits (LC), or the curve of LPC, PD or NS points of
/// cycles in two parameters, or the curve of orbits homoclinic to a saddle (HOM) in two
/// parameters. Cycles and homoclinic orbits live on N mesh intervals of m collocation
/// points. An initial value run (`Ivp`) integrates the model over a time span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ep,
    /// The equilibrium branch of a delay equation (SPEC 8s). It reads as `"EP"` in the JSON.
    DdeEp,
    /// The Hopf curve of a delay equilibrium (SPEC 8s, slice 2). It reads as `"H"` in the JSON.
    DdeHopf,
    /// The branch of periodic orbits of a delay equation from a Hopf point (SPEC 8s, slice 4).
    /// It reads as `"LC"` in the JSON. `count` is the number of multipliers kept per point, 0
    /// without `floquet`.
    DdeLc {
        ntst: usize,
        ncol: usize,
        count: usize,
    },
    Lp,
    Hopf,
    Lc {
        ntst: usize,
        ncol: usize,
    },
    Lpc {
        ntst: usize,
        ncol: usize,
    },
    Pd {
        ntst: usize,
        ncol: usize,
    },
    Ns {
        ntst: usize,
        ncol: usize,
    },
    Hom {
        ntst: usize,
        ncol: usize,
    },
    Ivp,
}

/// The two test columns of a delay Hopf run: the first Lyapunov coefficient and the Hopf-Hopf
/// test (SPEC 8s, slice 2). They are also the labels that the run can add to `BT` and `ZH`.
const DDE_HOPF_LABELS: [&str; 2] = ["GH", "HH"];

impl Kind {
    /// The `type` string of the problem JSON.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Ep | Kind::DdeEp => "EP",
            Kind::Lp => "LP",
            Kind::Hopf | Kind::DdeHopf => "H",
            Kind::Lc { .. } | Kind::DdeLc { .. } => "LC",
            Kind::Lpc { .. } => "LPC",
            Kind::Pd { .. } => "PD",
            Kind::Ns { .. } => "NS",
            Kind::Hom { .. } => "HOM",
            Kind::Ivp => "IVP",
        }
    }

    /// Labels of the test values in `Point::codim2`, in order: the codim-2 points of a
    /// curve, the LPC, PD and NS points of a cycle branch, or the codim-2 points of a HOM
    /// curve. Empty for an equilibrium branch.
    pub fn labels(self) -> &'static [&'static str] {
        match self {
            Kind::Ep | Kind::DdeEp | Kind::Ivp => &[],
            Kind::Lp => &LP_CURVE_LABELS,
            Kind::Hopf => &HOPF_CURVE_LABELS,
            Kind::DdeHopf => &DDE_HOPF_LABELS,
            Kind::Lc { .. } | Kind::DdeLc { .. } => &CYCLE_LABELS,
            Kind::Lpc { .. } => &LPC_CURVE_LABELS,
            Kind::Pd { .. } => &PD_CURVE_LABELS,
            Kind::Ns { .. } => &NS_CURVE_LABELS,
            Kind::Hom { .. } => &HOM_CURVE_LABELS,
        }
    }

    /// (N, m) of a run of cycles or of a HOM curve, `None` for the other kinds.
    pub fn mesh_size(self) -> Option<(usize, usize)> {
        match self {
            Kind::Lc { ntst, ncol }
            | Kind::Lpc { ntst, ncol }
            | Kind::Pd { ntst, ncol }
            | Kind::Ns { ntst, ncol }
            | Kind::Hom { ntst, ncol } => Some((ntst, ncol)),
            _ => None,
        }
    }

    /// Values of `Point::u` after the profile in a run of cycles with `n` states: T on an
    /// LC branch, (T, a1) on an LPC or PD curve, (T, a1, κ) on an NS curve and (T, x0, a1)
    /// on a HOM curve, with x0 the saddle. 0 for the other kinds, and `None` when `n + 2`
    /// overflows.
    pub fn cycle_tail(self, n: usize) -> Option<usize> {
        match self {
            Kind::Lc { .. } => Some(1),
            Kind::Lpc { .. } | Kind::Pd { .. } => Some(2),
            Kind::Ns { .. } => Some(3),
            Kind::Hom { .. } => n.checked_add(2),
            _ => Some(0),
        }
    }
}

/// Where a run of cycles starts, and what its start state holds.
#[derive(Clone, Debug, PartialEq)]
pub enum CycleFrom {
    /// A Hopf point: the equilibrium there, `n` values.
    Hopf,
    /// A guessed orbit on the uniform mesh: the orbit at the N m + 1 base points,
    /// point-major, then the period. On a HOM curve the orbit is the homoclinic orbit and
    /// the period is the truncation time T.
    Orbit,
    /// A PD point of an earlier LC run: its orbit at the base points of `mesh`, then its
    /// period. The new run doubles the orbit onto its own N and m.
    Pd { mesh: Vec<f64> },
    /// The start of an LPC, PD or NS curve, a special point of an LC run: its orbit at the
    /// base points of `mesh`, then its period. The curve moves the orbit onto its own N
    /// and m. The start of a HOM curve from a long-period point of an LC run has the same
    /// layout.
    Point { mesh: Vec<f64> },
}

/// Keys that only a run of cycles takes.
const CYCLE_KEYS: [&str; 3] = ["ntst", "ncol", "mesh"];

/// Values in one row of a run of cycles or a HOM curve of `kind` with `n` states, or
/// `None` when that overflows or `kind` has no mesh: `n (N m + 1) + N + 2 n` plus 8 for
/// LC, LPC and PD, 13 for NS and `n + 18` for HOM. `pc-ffi`'s row layout documents the
/// parts. A delay LC row has no mesh and no eigenvalues of a Jacobian: it is
/// `n (N m + 1) + 5 + 2 c` with `c` multipliers (SPEC 8s, slice 4).
pub fn cycle_stride(kind: Kind, n: usize) -> Option<usize> {
    if let Kind::DdeLc { ntst, ncol, count } = kind {
        // alpha, the period, h, iters and trivial_error, then the real and imaginary parts
        return ntst
            .checked_mul(ncol)?
            .checked_add(1)?
            .checked_mul(n)?
            .checked_add(5)?
            .checked_add(count.checked_mul(2)?);
    }
    let (ntst, ncol) = kind.mesh_size()?;
    // alpha, the tail of u, the last mesh point, the tests, h and iters
    let rest = kind
        .cycle_tail(n)?
        .checked_add(1 + 1 + kind.labels().len() + 2)?;
    ntst.checked_mul(ncol)?
        .checked_add(1)?
        .checked_mul(n)?
        .checked_add(ntst)?
        .checked_add(n.checked_mul(2)?)?
        .checked_add(rest)
}

/// A problem file after validation.
#[derive(Debug)]
pub struct Problem {
    /// The start of an `Ep` run. For a curve, `setup.u0` and `setup.params` are the start of
    /// the curve and `setup.free` is the first of its two parameters.
    pub setup: EpSetup,
    pub opts: Options,
    pub switch: Option<String>,
    pub kind: Kind,
    /// The two parameters of a curve run, `[a1, a2]`, as positions among the parameters.
    pub curve_free: Option<[usize; 2]>,
    /// The start of an LC run.
    pub cycle: Option<CycleFrom>,
    /// The time span and the solver options of an IVP run. The start state is `setup.u0` and
    /// the parameters are `setup.params`.
    pub ivp: Option<([f64; 2], IvpOptions)>,
    /// Whether a HOM run computes the inclination flip tests IFS and IFU (SPEC §8b-3). The key
    /// `inclination_flips` sets it and it is off when the key is missing.
    pub inclination_flips: bool,
    /// The characteristic-root count of a delay-equation run (SPEC 8s), `Some` exactly when
    /// `kind` is `DdeEp`. It sits here and not in `Kind`, which stays `Copy + Eq`.
    pub roots: Option<RootOptions>,
    /// The frequency of the start of a delay Hopf run, `Some` exactly when `kind` is `DdeHopf`.
    pub omega: Option<f64>,
    /// Whether a delay Hopf run computes `l1` and labels `GH`. Off when the key is missing.
    pub lyapunov: bool,
    /// Whether a delay Hopf run computes the coefficients at its `ZH` and `HH` specials
    /// (SPEC 8s, slice 3). Off when the key is missing.
    pub coefficients: bool,
    /// The size of the first guess of a delay cycle run, `Some` exactly when `kind` is
    /// `DdeLc`. The key `eps` has no default (SPEC 8s, slice 4).
    pub eps: Option<f64>,
    /// The Floquet options of a delay cycle run, `Some` when the key `floquet` is there. The
    /// count is also in `Kind::DdeLc`.
    pub floquet: Option<DdeFloquetOptions>,
}

impl Problem {
    /// The engine's run kind, with the start taken from `setup` as it is now.
    pub fn run_kind(&self) -> RunKind {
        let curve = |free: [usize; 2]| {
            CurveSetup::new(free, self.setup.params.clone(), self.setup.u0.clone())
        };
        let point = |free: [usize; 2], mesh: &Vec<f64>| {
            let (orbit, period) = self.orbit_period();
            let (ntst, ncol) = self.kind.mesh_size().unwrap_or((0, 0));
            CycleCurveStart {
                free,
                params: self.setup.params.clone(),
                orbit,
                period,
                mesh: mesh.clone(),
                ntst,
                ncol,
            }
        };
        match (self.kind, self.curve_free, &self.cycle) {
            (Kind::Ivp, ..) if let Some((tspan, opts)) = &self.ivp => RunKind::Ivp {
                tspan: *tspan,
                opts: opts.clone(),
            },
            (Kind::Lp, Some(free), _) => RunKind::LpCurve(curve(free)),
            (Kind::Hopf, Some(free), _) => RunKind::HopfCurve(curve(free)),
            (Kind::Lc { ntst, ncol }, _, Some(from)) => {
                RunKind::Cycle(self.cycle_start(ntst, ncol, from))
            }
            (Kind::Lpc { .. }, Some(free), Some(CycleFrom::Point { mesh })) => {
                RunKind::LpcCurve(point(free, mesh))
            }
            (Kind::Pd { .. }, Some(free), Some(CycleFrom::Point { mesh })) => {
                RunKind::PdCurve(point(free, mesh))
            }
            (Kind::Ns { .. }, Some(free), Some(CycleFrom::Point { mesh })) => {
                RunKind::NsCurve(point(free, mesh))
            }
            (Kind::Hom { ntst, ncol }, Some(free), Some(from)) => {
                let (orbit, period) = self.orbit_period();
                let (kind, mesh) = match from {
                    CycleFrom::Point { mesh } => (HomStartKind::Cycle, Some(mesh.clone())),
                    // The parser gives a HOM run only `Orbit` or `Point` starts.
                    _ => (HomStartKind::Orbit, None),
                };
                RunKind::HomCurve(HomStart {
                    kind,
                    free,
                    params: self.setup.params.clone(),
                    orbit,
                    period,
                    mesh,
                    ntst,
                    ncol,
                    inclination_flips: self.inclination_flips,
                })
            }
            (Kind::DdeEp, ..) if let Some(roots) = self.roots => RunKind::DdeEq(roots),
            (Kind::DdeLc { ntst, ncol, .. }, ..)
                if let (Some(omega), Some(eps)) = (self.omega, self.eps) =>
            {
                RunKind::DdeCycle {
                    free: self.setup.free,
                    omega,
                    eps,
                    ntst,
                    ncol,
                    floquet: self.floquet.clone(),
                }
            }
            (Kind::DdeHopf, Some(free), _) if let Some(omega) = self.omega => RunKind::DdeHopf {
                free,
                omega,
                extras: DdeHopfExtras {
                    lyapunov: self.lyapunov,
                    hh: self.roots,
                    coefficients: self.coefficients,
                },
            },
            _ => RunKind::Ep,
        }
    }

    /// `setup.u0` as an orbit followed by its period. An empty state gives a NaN period,
    /// which the engine rejects.
    fn orbit_period(&self) -> (Vec<f64>, f64) {
        match self.setup.u0.split_last() {
            Some((t, orbit)) => (orbit.to_vec(), *t),
            None => (Vec::new(), f64::NAN),
        }
    }

    fn cycle_start(&self, ntst: usize, ncol: usize, from: &CycleFrom) -> CycleStart {
        let free = self.setup.free;
        let params = self.setup.params.clone();
        let (orbit, period) = self.orbit_period();
        match from {
            CycleFrom::Hopf => CycleStart::Hopf(HopfStart {
                free,
                params,
                u0: self.setup.u0.clone(),
                ntst,
                ncol,
            }),
            CycleFrom::Orbit => {
                CycleStart::Orbit(CycleSetup::new(free, params, ntst, ncol, orbit, period))
            }
            // The parser gives an LC run no `Point` start; it reads as a PD start.
            CycleFrom::Pd { mesh } | CycleFrom::Point { mesh } => CycleStart::Pd(PdStart {
                free,
                params,
                orbit,
                period,
                mesh: mesh.clone(),
                ntst,
                ncol,
            }),
        }
    }

    /// Checks that a start state of `got` values fits the run: `n` values, except for an
    /// LC or HOM run from an orbit, an LC run from a PD point, a curve of cycles and a HOM
    /// curve from a cycle, where it is the orbit at the base points of its mesh followed by
    /// the period.
    pub fn check_u0_len(&self, n: usize, got: usize) -> Result<(), String> {
        let want = match (self.kind, &self.cycle) {
            (Kind::Lc { ntst, ncol } | Kind::Hom { ntst, ncol }, Some(CycleFrom::Orbit)) => {
                n * (ntst * ncol + 1) + 1
            }
            (_, Some(CycleFrom::Pd { mesh } | CycleFrom::Point { mesh })) => {
                // got = n (N m + 1) + 1, with N from the mesh and m in NCOL_RANGE.
                let old = mesh.len() - 1;
                let fits = n > 0 && got > 0 && (got - 1).is_multiple_of(n) && {
                    let k = (got - 1) / n;
                    k > 1 && (k - 1).is_multiple_of(old) && NCOL_RANGE.contains(&((k - 1) / old))
                };
                return if fits {
                    Ok(())
                } else {
                    Err(format!(
                        "u0: expected n (N m + 1) + 1 values with n = {n}, N = {old} from mesh \
                         and m from {} to {}, got {got}",
                        NCOL_RANGE.start(),
                        NCOL_RANGE.end()
                    ))
                };
            }
            _ => n,
        };
        if got == want {
            Ok(())
        } else {
            Err(format!("u0: expected {want} values, got {got}"))
        }
    }

    /// The run to start, with the start taken from `setup` as it is now.
    pub fn into_spec(self, model: Model, info: ModelInfo, evaluator: Evaluator) -> RunSpec {
        let kind = self.run_kind();
        match kind {
            RunKind::Ep => {
                let mut spec = RunSpec::new(model, info, evaluator, self.setup, self.opts);
                spec.switch_at = self.switch;
                spec
            }
            RunKind::Ivp { .. }
            | RunKind::DdeEq(_)
            | RunKind::DdeHopf { .. }
            | RunKind::DdeCycle { .. } => {
                let mut spec = RunSpec::new(model, info, evaluator, self.setup, self.opts);
                spec.kind = kind;
                spec
            }
            curve => RunSpec::curve(model, info, evaluator, curve, self.opts),
        }
    }
}

/// Who is reading the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The command line: `u0` and `params` may appear in the JSON.
    Cli,
    /// The C ABI: `u0` and `params` are passed as buffers, so the JSON must not have them.
    /// `setup.u0` and `setup.params` hold the model defaults until the caller overwrites them.
    Abi,
}

/// A number that is finite.
fn number(v: &Value, path: &str) -> Result<f64, String> {
    match v.as_f64() {
        Some(x) if x.is_finite() => Ok(x),
        _ => Err(format!("{path}: expected a number")),
    }
}

fn positive(v: &Value, path: &str) -> Result<f64, String> {
    let x = number(v, path)?;
    if x > 0.0 {
        Ok(x)
    } else {
        Err(format!("{path}: must be greater than 0"))
    }
}

fn count(v: &Value, path: &str) -> Result<usize, String> {
    match v.as_u64().and_then(|n| usize::try_from(n).ok()) {
        Some(n) if n >= 1 => Ok(n),
        _ => Err(format!("{path}: expected an integer of at least 1")),
    }
}

fn object<'a>(v: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    v.as_object()
        .ok_or_else(|| format!("{path}: expected an object"))
}

/// Errors on the first key of `map` that is not in `allowed`.
fn only_keys(map: &Map<String, Value>, allowed: &[&str], path: &str) -> Result<(), String> {
    for k in map.keys() {
        if !allowed.contains(&k.as_str()) {
            let at = if path.is_empty() {
                k.clone()
            } else {
                format!("{path}.{k}")
            };
            return Err(format!("{at}: unknown key"));
        }
    }
    Ok(())
}

/// Position of the parameter `name` (names match without regard to case).
fn param_index(ir: &IrModel, name: &str) -> Option<usize> {
    let key = name.to_lowercase();
    // The hidden delay slots of a delay model come after the user parameters and are not names.
    ir.param_names[..ir.n_user_params]
        .iter()
        .position(|p| p.to_lowercase() == key)
}

/// A label followed by a 1-based ordinal, for example `BP1`: the form that
/// `pc_engine::special_index` reads.
pub fn check_switch(name: &str) -> bool {
    match name.find(|c: char| c.is_ascii_digit()) {
        Some(cut) if cut > 0 => {
            let digits = &name[cut..];
            digits.bytes().all(|b| b.is_ascii_digit())
                && digits.parse::<usize>().is_ok_and(|n| n >= 1)
        }
        _ => false,
    }
}

/// Reads and checks a problem object against the model.
pub fn parse_problem(v: &Value, ir: &IrModel, mode: Mode) -> Result<Problem, String> {
    let top = v
        .as_object()
        .ok_or_else(|| "$: expected an object".to_string())?;
    only_keys(
        top,
        &[
            "type",
            "free",
            "u0",
            "params",
            "options",
            "switch",
            "start_is_special",
            "ntst",
            "ncol",
            "start",
            "mesh",
            "tspan",
            "inclination_flips",
            "roots",
            "omega",
            "lyapunov",
            "coefficients",
            "eps",
            "floquet",
        ],
        "",
    )?;
    // A delay model has hidden delay slots after its user parameters (SPEC 8s).
    let delay = !ir.delays.is_empty();
    let kind = match top.get("type") {
        None => return Err("type: missing".into()),
        // A delay equation takes no other run, whatever its other keys say (SPEC 8s).
        Some(Value::String(s))
            if delay && matches!(s.as_str(), "LP" | "LPC" | "PD" | "NS" | "HOM") =>
        {
            return Err(DELAY_TYPE_REFUSAL.into());
        }
        Some(Value::String(s)) if s == "EP" && delay => Kind::DdeEp,
        Some(Value::String(s)) if s == "EP" => Kind::Ep,
        Some(Value::String(s)) if s == "LP" => Kind::Lp,
        Some(Value::String(s)) if s == "H" && delay => Kind::DdeHopf,
        Some(Value::String(s)) if s == "H" => Kind::Hopf,
        Some(Value::String(s)) if s == "IVP" => return parse_ivp(top, ir, mode),
        Some(Value::String(s)) if s == "LC" && delay => {
            cycle_size(top, ir.n(), |ntst, ncol| Kind::DdeLc {
                ntst,
                ncol,
                count: 0,
            })?
        }
        Some(Value::String(s)) if s == "LC" => {
            cycle_size(top, ir.n(), |ntst, ncol| Kind::Lc { ntst, ncol })?
        }
        Some(Value::String(s)) if s == "LPC" => {
            cycle_size(top, ir.n(), |ntst, ncol| Kind::Lpc { ntst, ncol })?
        }
        Some(Value::String(s)) if s == "PD" => {
            cycle_size(top, ir.n(), |ntst, ncol| Kind::Pd { ntst, ncol })?
        }
        Some(Value::String(s)) if s == "NS" => {
            cycle_size(top, ir.n(), |ntst, ncol| Kind::Ns { ntst, ncol })?
        }
        Some(Value::String(s)) if s == "HOM" => {
            cycle_size(top, ir.n(), |ntst, ncol| Kind::Hom { ntst, ncol })?
        }
        Some(_) => {
            return Err(
                "type: expected \"EP\", \"LP\", \"H\", \"LC\", \"LPC\", \"PD\", \
                        \"NS\", \"HOM\" or \"IVP\""
                    .into(),
            );
        }
    };
    if top.contains_key("tspan") {
        return Err("tspan: only an IVP run takes this key".into());
    }
    let roots = parse_roots(top, kind)?;
    let (omega, lyapunov, coefficients) = parse_hopf_keys(top, kind)?;
    let (eps, floquet) = parse_cycle_keys(top, kind)?;
    // The count of multipliers is part of the row layout, so the kind carries it.
    let kind = match (kind, &floquet) {
        (Kind::DdeLc { ntst, ncol, .. }, Some(f)) => {
            let kind = Kind::DdeLc {
                ntst,
                ncol,
                count: f.count,
            };
            if cycle_stride(kind, ir.n()).is_none() {
                return Err("floquet.count: too large".into());
            }
            kind
        }
        _ => kind,
    };
    let curve_of_cycles = matches!(kind, Kind::Lpc { .. } | Kind::Pd { .. } | Kind::Ns { .. });
    let one_free = matches!(
        kind,
        Kind::Ep | Kind::DdeEp | Kind::Lc { .. } | Kind::DdeLc { .. }
    );
    let (free, curve_free) = match (one_free, top.get("free")) {
        (_, None) => return Err("free: missing".into()),
        (true, Some(Value::String(name))) => (
            param_index(ir, name).ok_or_else(|| format!("free: unknown parameter \"{name}\""))?,
            None,
        ),
        (true, Some(_)) => return Err("free: expected a string".into()),
        (_, Some(Value::Array(names))) if names.len() == 2 => {
            let mut at = [0; 2];
            for (i, name) in names.iter().enumerate() {
                let name = name
                    .as_str()
                    .ok_or_else(|| format!("free[{i}]: expected a string"))?;
                at[i] = param_index(ir, name)
                    .ok_or_else(|| format!("free[{i}]: unknown parameter \"{name}\""))?;
            }
            if at[0] == at[1] {
                return Err("free: the two parameters must differ".into());
            }
            (at[0], Some(at))
        }
        (_, Some(_)) => return Err("free: expected two parameter names".into()),
    };

    if mode == Mode::Abi {
        if top.contains_key("u0") {
            return Err("u0: not allowed here; pass the state as a buffer".into());
        }
        if top.contains_key("params") {
            return Err("params: not allowed here; pass the parameters as a buffer".into());
        }
    }

    let params = parse_params(top, ir)?;

    let cycle = match kind {
        Kind::Lc { .. } => Some(cycle_from(top)?),
        Kind::Hom { .. } => Some(hom_from(top)?),
        _ if curve_of_cycles => Some(point_from(top)?),
        _ => {
            if matches!(kind, Kind::DdeLc { .. }) {
                // A delay cycle run starts at a Hopf point on a uniform mesh (SPEC 8s, slice 4).
                for key in ["start", "mesh"] {
                    if top.contains_key(key) {
                        return Err(format!(
                            "{key}: a delay cycle run starts at a Hopf point on a uniform \
                             mesh and takes no {key}"
                        ));
                    }
                }
            } else {
                if let Some(key) = CYCLE_KEYS.iter().find(|k| top.contains_key(**k)) {
                    return Err(format!(
                        "{key}: only an LC, LPC, PD, NS or HOM run takes this key"
                    ));
                }
                if top.contains_key("start") {
                    return Err("start: only an LC or HOM run takes this key".into());
                }
            }
            None
        }
    };

    let u0 = match top.get("u0") {
        // A start from an orbit has no default state.
        None if matches!(kind, Kind::Hom { .. }) && mode == Mode::Cli => {
            return Err(
                "u0: missing; a HOM curve needs the orbit and the period (or T) of its start"
                    .into(),
            );
        }
        None if matches!(cycle, Some(CycleFrom::Orbit | CycleFrom::Pd { .. }))
            && mode == Mode::Cli =>
        {
            return Err(
                "u0: missing; an LC run from an orbit or a PD point needs the orbit and the \
                 period"
                    .into(),
            );
        }
        None if curve_of_cycles && mode == Mode::Cli => {
            return Err(
                "u0: missing; a curve of cycles needs the orbit and the period of its start".into(),
            );
        }
        None if matches!(kind, Kind::DdeLc { .. }) && mode == Mode::Cli => {
            return Err(
                "u0: missing; a delay cycle run needs the equilibrium at its Hopf point".into(),
            );
        }
        None => ir.init.clone(),
        Some(v) => u0_values(v)?,
    };

    let opts = match top.get("options") {
        None => Options::default(),
        Some(o) => parse_options(o)?,
    };
    let has_option = |key: &str| top.get("options").is_some_and(|o| o.get(key).is_some());
    if kind == Kind::DdeEp
        && top
            .get("options")
            .and_then(|o| o.get("detect"))
            .and_then(Value::as_object)
            .is_some_and(|d| d.contains_key("h"))
    {
        return Err(
            "options.detect.h: Hopf points come from the root count, not from detect".into(),
        );
    }
    match kind {
        Kind::Lc { .. } if has_option("detect") => {
            return Err("options.detect: an LC run always detects LPC, PD and NS".into());
        }
        Kind::Lc { .. } => {}
        _ if curve_of_cycles && has_option("detect") => {
            return Err("options.detect: a curve of cycles always runs its codim-2 tests".into());
        }
        _ if curve_of_cycles => {}
        Kind::Hom { .. } if has_option("detect") => {
            return Err("options.detect: a HOM curve always runs its codim-2 tests".into());
        }
        Kind::Hom { .. } => {}
        Kind::DdeHopf if has_option("detect") => {
            return Err(
                "options.detect: a delay Hopf run always looks for its codim-2 points \
                        and takes no detect"
                    .into(),
            );
        }
        Kind::DdeLc { .. } if has_option("detect") => {
            return Err(
                "options.detect: a delay cycle run labels LPC, PD and NS itself and takes no \
                 detect"
                    .into(),
            );
        }
        Kind::DdeLc { .. } if has_option("mesh_adapt") => {
            return Err(
                "options.mesh_adapt: a delay cycle run keeps its uniform mesh and takes no \
                 mesh_adapt"
                    .into(),
            );
        }
        _ if has_option("mesh_adapt") => {
            return Err(
                "options.mesh_adapt: only an LC, LPC, PD, NS or HOM run adapts its mesh".into(),
            );
        }
        _ => {}
    }

    let switch = match top.get("switch") {
        None => None,
        Some(Value::String(s)) if check_switch(s) => Some(s.clone()),
        Some(Value::String(s)) => {
            return Err(format!(
                "switch: \"{s}\" is not a label followed by a number of at least 1, like \"BP1\""
            ));
        }
        Some(_) => return Err("switch: expected a string".into()),
    };

    let start_is_special = match top.get("start_is_special") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("start_is_special: expected true or false".into()),
    };

    let inclination_flips = match top.get("inclination_flips") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("inclination_flips: expected true or false".into()),
    };

    if top.contains_key("inclination_flips") && !matches!(kind, Kind::Hom { .. }) {
        return Err("inclination_flips: only a HOM run takes this key".into());
    }

    if !matches!(kind, Kind::Ep | Kind::DdeEp) {
        if switch.is_some() {
            return Err("switch: only an EP run can switch branches".into());
        }
        if top.contains_key("start_is_special") {
            return Err("start_is_special: only an EP run starts at a special point".into());
        }
    }

    if matches!(kind, Kind::DdeEp | Kind::DdeHopf | Kind::DdeLc { .. }) {
        if switch.is_some() {
            return Err("switch: a delay-equation run cannot switch branches".into());
        }
        if top.contains_key("start_is_special") {
            return Err("start_is_special: a delay-equation run starts by Newton from u0".into());
        }
    }

    let mut setup = EpSetup::new(free, params, u0);
    setup.start_is_special = start_is_special;
    let problem = Problem {
        setup,
        opts,
        switch,
        kind,
        curve_free,
        cycle,
        ivp: None,
        inclination_flips,
        roots,
        omega,
        lyapunov,
        coefficients,
        eps,
        floquet,
    };
    if mode == Mode::Cli {
        problem.check_u0_len(ir.n(), problem.setup.u0.len())?;
    }
    Ok(problem)
}

/// The `roots` key of a delay-equation run (SPEC 8s), with `n` and `re_min` both given. It is
/// required for a delay `"EP"` run, optional for a delay `"H"` run (it turns the `HH` label on)
/// and an error for any other kind. `Some` exactly when the key is there.
fn parse_roots(top: &Map<String, Value>, kind: Kind) -> Result<Option<RootOptions>, String> {
    let Some(v) = top.get("roots") else {
        return if kind == Kind::DdeEp {
            Err("roots: missing; a delay-equation run needs n and re_min".into())
        } else {
            Ok(None)
        };
    };
    if !matches!(kind, Kind::DdeEp | Kind::DdeHopf) {
        return Err("roots: only a delay-equation run takes this key".into());
    }
    let map = object(v, "roots")?;
    only_keys(map, &["n", "re_min"], "roots")?;
    let n = count(map.get("n").ok_or("roots.n: missing")?, "roots.n")?;
    let re_min = number(
        map.get("re_min").ok_or("roots.re_min: missing")?,
        "roots.re_min",
    )?;
    if re_min >= 0.0 {
        return Err("roots.re_min: must be less than 0".into());
    }
    Ok(Some(RootOptions { n, re_min }))
}

/// The refusal of a type that a delay equation does not take (SPEC 8s). At the C ABI the types
/// `LP`, `LPC`, `PD`, `NS`, `HOM` and `IVP` are stopped by the delay gate before the parser
/// sees them, so this text is what `pc-cli` and a direct `parse_problem` call report.
const DELAY_TYPE_REFUSAL: &str = "type: a delay equation takes only \"EP\", \"H\" or \"LC\"";

/// The `omega`, `lyapunov` and `coefficients` keys of a delay Hopf run, and the `omega` key of a
/// delay cycle run (SPEC 8s, slices 2 to 4). `omega` is required there and has no default, and
/// it must be above the engine's floor, so a bad value is refused at the start. `lyapunov` and
/// `coefficients` are false when the key is left out and must be booleans otherwise, and only a
/// Hopf run takes them. Any other kind refuses all three.
fn parse_hopf_keys(
    top: &Map<String, Value>,
    kind: Kind,
) -> Result<(Option<f64>, bool, bool), String> {
    let cycle = matches!(kind, Kind::DdeLc { .. });
    if kind != Kind::DdeHopf {
        let keys: &[&str] = if cycle {
            &["lyapunov", "coefficients"]
        } else {
            &["omega", "lyapunov", "coefficients"]
        };
        for key in keys {
            if top.contains_key(*key) {
                return Err(format!(
                    "{key}: only a delay-equation Hopf run takes this key"
                ));
            }
        }
        if !cycle {
            return Ok((None, false, false));
        }
    }
    let what = if cycle { "cycle" } else { "Hopf" };
    let omega = number(
        top.get("omega").ok_or_else(|| {
            format!("omega: missing; a delay {what} run needs the frequency of its start")
        })?,
        "omega",
    )?;
    if omega <= DDE_HOPF_OMEGA_MIN {
        return Err(format!("omega: must be above {DDE_HOPF_OMEGA_MIN:e}"));
    }
    if cycle {
        return Ok((Some(omega), false, false));
    }
    let flag = |key: &str| match top.get(key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(format!("{key}: expected true or false")),
    };
    Ok((Some(omega), flag("lyapunov")?, flag("coefficients")?))
}

/// The `eps` and `floquet` keys of a delay cycle run (SPEC 8s, slice 4). `eps` is required and
/// has no default. `floquet` is an object with the integers `nhist` and `count`, both of at
/// least 1 and both required, and no other member. The engine checks the values against the
/// model. Any other kind refuses both keys.
fn parse_cycle_keys(
    top: &Map<String, Value>,
    kind: Kind,
) -> Result<(Option<f64>, Option<DdeFloquetOptions>), String> {
    if !matches!(kind, Kind::DdeLc { .. }) {
        for key in ["eps", "floquet"] {
            if top.contains_key(key) {
                return Err(format!(
                    "{key}: only a delay-equation cycle run takes this key"
                ));
            }
        }
        return Ok((None, None));
    }
    let eps = match top.get("eps").and_then(Value::as_f64) {
        Some(x) if x.is_finite() && x > 0.0 => x,
        _ => return Err("eps: must be a number above 0".into()),
    };
    let floquet = match top.get("floquet") {
        None => None,
        Some(v) => {
            let map = object(v, "floquet")?;
            only_keys(map, &["nhist", "count"], "floquet")?;
            Some(DdeFloquetOptions {
                nhist: count(
                    map.get("nhist").ok_or("floquet.nhist: missing")?,
                    "floquet.nhist",
                )?,
                count: count(
                    map.get("count").ok_or("floquet.count: missing")?,
                    "floquet.count",
                )?,
            })
        }
    };
    Ok((Some(eps), floquet))
}

/// The model's parameter values with the ones that `params` sets.
fn parse_params(top: &Map<String, Value>, ir: &IrModel) -> Result<Vec<f64>, String> {
    // A delay model lists its user parameters only; the hidden slots are set by the engine.
    let mut params = ir.par_values[..ir.n_user_params].to_vec();
    if let Some(p) = top.get("params") {
        for (name, val) in object(p, "params")? {
            let path = format!("params.{name}");
            let i = param_index(ir, name)
                .ok_or_else(|| format!("{path}: unknown parameter \"{name}\""))?;
            params[i] = number(val, &path)?;
        }
    }
    Ok(params)
}

/// The numbers of a `u0` array.
fn u0_values(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(a) => a
            .iter()
            .enumerate()
            .map(|(i, x)| number(x, &format!("u0[{i}]")))
            .collect(),
        _ => Err("u0: expected an array of numbers".into()),
    }
}

/// An `IVP` problem: `tspan`, the solver options, and the start state and parameters.
fn parse_ivp(top: &Map<String, Value>, ir: &IrModel, mode: Mode) -> Result<Problem, String> {
    if !ir.delays.is_empty() {
        return Err(DELAY_TYPE_REFUSAL.into());
    }
    if top.contains_key("free") {
        return Err("free: an IVP run has no free parameter".into());
    }
    if let Some(key) = CYCLE_KEYS.iter().find(|k| top.contains_key(**k)) {
        return Err(format!(
            "{key}: only an LC, LPC, PD, NS or HOM run takes this key"
        ));
    }
    if top.contains_key("start") {
        return Err("start: only an LC or HOM run takes this key".into());
    }
    if top.contains_key("switch") {
        return Err("switch: only an EP run can switch branches".into());
    }
    if top.contains_key("start_is_special") {
        return Err("start_is_special: only an EP run starts at a special point".into());
    }
    if top.contains_key("inclination_flips") {
        return Err("inclination_flips: only a HOM run takes this key".into());
    }
    if top.contains_key("roots") {
        return Err("roots: only a delay-equation run takes this key".into());
    }
    for key in ["omega", "lyapunov", "coefficients", "eps", "floquet"] {
        if top.contains_key(key) {
            return Err(format!("{key}: only a delay-equation run takes this key"));
        }
    }
    let tspan = match top.get("tspan") {
        None => return Err("tspan: missing".into()),
        Some(Value::Array(a)) if a.len() == 2 => {
            [number(&a[0], "tspan[0]")?, number(&a[1], "tspan[1]")?]
        }
        Some(_) => return Err("tspan: expected two numbers".into()),
    };
    if tspan[0] == tspan[1] {
        return Err("tspan: the two times must differ".into());
    }
    if mode == Mode::Abi {
        if top.contains_key("u0") {
            return Err("u0: not allowed here; pass the state as a buffer".into());
        }
        if top.contains_key("params") {
            return Err("params: not allowed here; pass the parameters as a buffer".into());
        }
    }
    let params = parse_params(top, ir)?;
    let u0 = match top.get("u0") {
        None => ir.init.clone(),
        Some(v) => u0_values(v)?,
    };
    let opts = match top.get("options") {
        None => IvpOptions::default(),
        Some(o) => parse_ivp_options(o)?,
    };
    let problem = Problem {
        setup: EpSetup::new(0, params, u0),
        opts: Options::default(),
        switch: None,
        kind: Kind::Ivp,
        curve_free: None,
        cycle: None,
        ivp: Some((tspan, opts)),
        inclination_flips: false,
        roots: None,
        omega: None,
        lyapunov: false,
        coefficients: false,
        eps: None,
        floquet: None,
    };
    if mode == Mode::Cli {
        problem.check_u0_len(ir.n(), problem.setup.u0.len())?;
    }
    Ok(problem)
}

/// The options of an IVP run: `rtol`, `atol`, `max_steps` and `hmax`. Missing ones keep
/// the defaults of `IvpOptions`.
fn parse_ivp_options(v: &Value) -> Result<IvpOptions, String> {
    let map = object(v, "options")?;
    only_keys(map, &["rtol", "atol", "max_steps", "hmax"], "options")?;
    let mut opts = IvpOptions::default();
    if let Some(x) = map.get("rtol") {
        opts.rtol = positive(x, "options.rtol")?;
    }
    if let Some(x) = map.get("atol") {
        opts.atol = positive(x, "options.atol")?;
    }
    if let Some(x) = map.get("max_steps") {
        opts.max_steps = count(x, "options.max_steps")?;
    }
    if let Some(x) = map.get("hmax") {
        opts.hmax = positive(x, "options.hmax")?;
    }
    Ok(opts)
}

/// Reads `ntst` and `ncol` of a problem of cycles and makes its kind with `make`.
fn cycle_size(
    top: &Map<String, Value>,
    n: usize,
    make: fn(usize, usize) -> Kind,
) -> Result<Kind, String> {
    let ntst = count(top.get("ntst").ok_or("ntst: missing")?, "ntst")?;
    let ncol = match top.get("ncol").ok_or("ncol: missing")?.as_u64() {
        Some(m) if usize::try_from(m).is_ok_and(|m| NCOL_RANGE.contains(&m)) => m as usize,
        _ => {
            return Err(format!(
                "ncol: expected an integer from {} to {}",
                NCOL_RANGE.start(),
                NCOL_RANGE.end()
            ));
        }
    };
    if n == 0 {
        return Err("type: a run of cycles needs at least one state variable".into());
    }
    let kind = make(ntst, ncol);
    if cycle_stride(kind, n).is_none() {
        return Err("ntst: ntst times ncol is too large".into());
    }
    Ok(kind)
}

/// Reads `start` and `mesh` of an LC problem.
fn cycle_from(top: &Map<String, Value>) -> Result<CycleFrom, String> {
    let from = match top.get("start") {
        None => return Err("start: missing".into()),
        Some(Value::String(s)) if s == "hopf" => CycleFrom::Hopf,
        Some(Value::String(s)) if s == "orbit" => CycleFrom::Orbit,
        Some(Value::String(s)) if s == "pd" => {
            let v = top
                .get("mesh")
                .ok_or("mesh: missing; a start from a PD point needs the mesh of its orbit")?;
            CycleFrom::Pd {
                mesh: parse_mesh(v)?,
            }
        }
        Some(_) => return Err("start: expected \"hopf\", \"orbit\" or \"pd\"".into()),
    };
    if !matches!(from, CycleFrom::Pd { .. }) && top.contains_key("mesh") {
        return Err("mesh: only a start from a PD point takes a mesh".into());
    }
    Ok(from)
}

/// Reads `mesh` of an LPC, PD or NS problem, whose start is a special point of an LC run.
fn point_from(top: &Map<String, Value>) -> Result<CycleFrom, String> {
    if top.contains_key("start") {
        return Err("start: only an LC or HOM run takes this key".into());
    }
    let v = top
        .get("mesh")
        .ok_or("mesh: missing; a curve of cycles needs the mesh of the orbit at its start")?;
    Ok(CycleFrom::Point {
        mesh: parse_mesh(v)?,
    })
}

/// Reads `start` and `mesh` of a HOM problem. The start has no default: `"orbit"` is an
/// orbit on the uniform mesh and takes no `mesh`, `"cycle"` is a long-period point of an
/// LC run and needs the mesh of its orbit.
fn hom_from(top: &Map<String, Value>) -> Result<CycleFrom, String> {
    match top.get("start") {
        None => Err("start: missing; a HOM run starts from \"orbit\" or \"cycle\"".into()),
        Some(Value::String(s)) if s == "orbit" => {
            if top.contains_key("mesh") {
                return Err("mesh: a HOM start from an orbit uses the uniform mesh".into());
            }
            Ok(CycleFrom::Orbit)
        }
        Some(Value::String(s)) if s == "cycle" => {
            let v = top
                .get("mesh")
                .ok_or("mesh: missing; a HOM start from a cycle needs the mesh of its orbit")?;
            Ok(CycleFrom::Point {
                mesh: parse_mesh(v)?,
            })
        }
        Some(_) => Err("start: expected \"orbit\" or \"cycle\"".into()),
    }
}

/// A mesh: at least two numbers that rise strictly from exactly 0 to exactly 1.
fn parse_mesh(v: &Value) -> Result<Vec<f64>, String> {
    let a = v
        .as_array()
        .filter(|a| a.len() >= 2)
        .ok_or("mesh: expected an array of at least 2 numbers")?;
    let mesh = a
        .iter()
        .enumerate()
        .map(|(i, x)| number(x, &format!("mesh[{i}]")))
        .collect::<Result<Vec<f64>, String>>()?;
    if mesh[0] != 0.0 || mesh[mesh.len() - 1] != 1.0 {
        return Err("mesh: must run from 0 to 1".into());
    }
    if mesh.windows(2).any(|w| w[0] >= w[1]) {
        return Err("mesh: must rise strictly".into());
    }
    Ok(mesh)
}

/// Reads the `options` object.
pub fn parse_options(v: &Value) -> Result<Options, String> {
    let map = object(v, "options")?;
    only_keys(
        map,
        &[
            "h0",
            "hmin",
            "hmax",
            "max_steps",
            "tol_x",
            "tol_f",
            "max_newton",
            "direction",
            "bounds",
            "detect",
            "mesh_adapt",
        ],
        "options",
    )?;
    let mut o = Options::default();
    for (key, val) in map {
        let path = format!("options.{key}");
        match key.as_str() {
            "h0" => o.h0 = positive(val, &path)?,
            "hmin" => o.hmin = positive(val, &path)?,
            "hmax" => o.hmax = positive(val, &path)?,
            "tol_x" => o.tol_x = positive(val, &path)?,
            "tol_f" => o.tol_f = positive(val, &path)?,
            "max_steps" => o.max_steps = count(val, &path)?,
            "max_newton" => o.max_newton = count(val, &path)?,
            "mesh_adapt" => {
                o.mesh_adapt = val
                    .as_u64()
                    .and_then(|k| usize::try_from(k).ok())
                    .ok_or_else(|| format!("{path}: expected an integer of at least 0"))?
            }
            "direction" => {
                let d = number(val, &path)?;
                if d == 0.0 {
                    return Err(format!("{path}: must not be 0"));
                }
                o.direction = d.signum();
            }
            "bounds" => {
                let a = val
                    .as_array()
                    .filter(|a| a.len() == 2)
                    .ok_or_else(|| format!("{path}: expected [lo, hi]"))?;
                let lo = number(&a[0], &format!("{path}[0]"))?;
                let hi = number(&a[1], &format!("{path}[1]"))?;
                if lo >= hi {
                    return Err(format!("{path}: lo must be less than hi"));
                }
                o.bounds = (lo, hi);
            }
            "detect" => {
                let d = object(val, &path)?;
                only_keys(d, &["lp", "bp", "h"], &path)?;
                for (k, flag) in d {
                    let p = format!("{path}.{k}");
                    let b = flag
                        .as_bool()
                        .ok_or_else(|| format!("{p}: expected true or false"))?;
                    match k.as_str() {
                        "lp" => o.detect_lp = b,
                        "bp" => o.detect_bp = b,
                        _ => o.detect_h = b,
                    }
                }
            }
            // `only_keys` has rejected every other key.
            _ => {}
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_ode::{MapLoader, SourceMap, load};
    use serde_json::json;

    /// Two states, one parameter, nonzero initial values.
    fn model() -> IrModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", "x'=a-x\ny'=x-y\npar a=2\ninit x=0.5,y=1.5\ndone\n");
        let loaded = load(top, &mut map, &mut MapLoader::new());
        pc_ir::lower(&loaded.expanded, &loaded.info).expect("the test model lowers")
    }

    fn base() -> Value {
        json!({"type": "EP", "free": "a"})
    }

    fn with(key: &str, v: Value) -> Value {
        let mut b = base();
        b[key] = v;
        b
    }

    /// Two parameters, for curves.
    fn model2() -> IrModel {
        let mut map = SourceMap::new();
        let top = map.add(
            "t.ode",
            "x'=a-x\ny'=b*x-y\npar a=2,b=3\ninit x=0.5,y=1.5\ndone\n",
        );
        let loaded = load(top, &mut map, &mut MapLoader::new());
        pc_ir::lower(&loaded.expanded, &loaded.info).expect("the test model lowers")
    }

    #[test]
    fn curve_types_take_two_names_and_build_a_curve_run() {
        let ir = model2();
        for (ty, kind) in [("LP", Kind::Lp), ("H", Kind::Hopf)] {
            let v = json!({"type": ty, "free": ["B", "a"]});
            let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
            assert_eq!(p.kind, kind);
            assert_eq!(p.curve_free, Some([1, 0]));
            let run = p.run_kind();
            let (RunKind::LpCurve(cs) | RunKind::HopfCurve(cs)) = run else {
                panic!("expected a curve kind");
            };
            assert_eq!(cs.free, [1, 0]);
            assert_eq!(cs.u0, ir.init);
            assert_eq!(cs.params, ir.par_values);
        }
        let p = parse_problem(&json!({"type": "EP", "free": "a"}), &ir, Mode::Abi).unwrap();
        assert_eq!(p.kind, Kind::Ep);
        assert_eq!(p.curve_free, None);
        assert!(matches!(p.run_kind(), RunKind::Ep));
    }

    #[test]
    fn curve_types_check_free_switch_and_start_is_special() {
        let ir = model2();
        let cases = [
            (
                json!({"type": "LP", "free": "a"}),
                "free: expected two parameter names",
            ),
            (
                json!({"type": "LP", "free": ["a"]}),
                "free: expected two parameter names",
            ),
            (
                json!({"type": "LP", "free": ["a", 1]}),
                "free[1]: expected a string",
            ),
            (
                json!({"type": "LP", "free": ["a", "A"]}),
                "free: the two parameters must differ",
            ),
            (
                json!({"type": "H", "free": ["a", "zz"]}),
                "free[1]: unknown parameter \"zz\"",
            ),
            (
                json!({"type": "EP", "free": ["a", "b"]}),
                "free: expected a string",
            ),
            (
                json!({"type": "LP", "free": ["a", "b"], "switch": "BP1"}),
                "switch: only an EP run can switch branches",
            ),
            (
                json!({"type": "H", "free": ["a", "b"], "start_is_special": true}),
                "start_is_special: only an EP run starts at a special point",
            ),
            (
                json!({"type": "BP", "free": "a"}),
                "type: expected \"EP\", \"LP\", \"H\", \"LC\", \"LPC\", \"PD\", \"NS\", \"HOM\" or \"IVP\"",
            ),
        ];
        for (v, msg) in cases {
            assert_eq!(parse_problem(&v, &ir, Mode::Abi).unwrap_err(), msg, "{v}");
        }
    }

    #[test]
    fn switch_names() {
        for ok in ["BP1", "H12", "LP3"] {
            assert!(check_switch(ok), "{ok}");
        }
        for bad in ["BP", "1", "BP0", "BP1x", "", "B1P2"] {
            assert!(!check_switch(bad), "{bad}");
        }
    }

    #[test]
    fn abi_rejects_u0_and_params() {
        let ir = model();
        let e = parse_problem(&with("u0", json!([1.0, 2.0])), &ir, Mode::Abi).unwrap_err();
        assert_eq!(e, "u0: not allowed here; pass the state as a buffer");
        let e = parse_problem(&with("params", json!({"a": 1.0})), &ir, Mode::Abi).unwrap_err();
        assert_eq!(
            e,
            "params: not allowed here; pass the parameters as a buffer"
        );
    }

    #[test]
    fn abi_defaults_match_cli_defaults() {
        let ir = model();
        let cli = parse_problem(&base(), &ir, Mode::Cli).unwrap();
        let abi = parse_problem(&base(), &ir, Mode::Abi).unwrap();
        assert_eq!(abi.setup.u0, cli.setup.u0);
        assert_eq!(abi.setup.params, cli.setup.params);
        assert_eq!(abi.setup.u0, ir.init);
        assert_eq!(abi.setup.params, ir.par_values);
    }

    #[test]
    fn cli_accepts_u0_and_params() {
        let ir = model();
        let mut v = with("u0", json!([3.0, 4.0]));
        v["params"] = json!({"a": 7.0});
        let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
        assert_eq!(p.setup.u0, vec![3.0, 4.0]);
        assert_eq!(p.setup.params, vec![7.0]);
    }

    #[test]
    fn start_is_special_is_a_bool() {
        let ir = model();
        for mode in [Mode::Cli, Mode::Abi] {
            let p = parse_problem(&base(), &ir, mode).unwrap();
            assert!(!p.setup.start_is_special);
            let p = parse_problem(&with("start_is_special", json!(true)), &ir, mode).unwrap();
            assert!(p.setup.start_is_special);
            let p = parse_problem(&with("start_is_special", json!(false)), &ir, mode).unwrap();
            assert!(!p.setup.start_is_special);
            let e = parse_problem(&with("start_is_special", json!("yes")), &ir, mode).unwrap_err();
            assert_eq!(e, "start_is_special: expected true or false");
        }
    }

    /// Doubles that MATLAB writes with `%.17g` must read back bit for bit.
    #[test]
    fn options_numbers_round_trip_bitwise() {
        let mut values: Vec<f64> = vec![
            1e-10,
            8.0 / 3.0,
            470.0 / 19.0,
            0.1,
            1.0 / 3.0,
            5e-324,
            f64::MAX,
            2.6666666666666665,
        ];
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        while values.len() < 1008 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f64::from_bits(state);
            if x.is_finite() {
                values.push(x);
            }
        }
        for &v in &values {
            let text = format!("{v:.17e}");
            // bounds [v, MAX] or [-MAX, v]
            let b = if v < f64::MAX {
                format!("[{text}, {:.17e}]", f64::MAX)
            } else {
                format!("[{:.17e}, {text}]", -f64::MAX)
            };
            let json = format!("{{\"bounds\": {b}}}");
            let o = parse_options(&serde_json::from_str(&json).unwrap()).unwrap();
            let got = if v < f64::MAX { o.bounds.0 } else { o.bounds.1 };
            assert_eq!(got.to_bits(), v.to_bits(), "bounds {text}");
            if v != 5e-324 && v != 0.0 {
                let a = v.abs();
                let json = format!("{{\"h0\": {a:.17e}}}");
                let o = parse_options(&serde_json::from_str(&json).unwrap()).unwrap();
                assert_eq!(o.h0.to_bits(), a.to_bits(), "h0 {a:.17e}");
            }
        }
    }

    fn lc(start: &str) -> Value {
        json!({"type": "LC", "free": "a", "ntst": 2, "ncol": 3, "start": start})
    }

    #[test]
    fn an_lc_problem_builds_a_cycle_run_for_each_start() {
        let ir = model();
        let p = parse_problem(&lc("hopf"), &ir, Mode::Abi).unwrap();
        assert_eq!(p.kind, Kind::Lc { ntst: 2, ncol: 3 });
        assert_eq!(p.cycle, Some(CycleFrom::Hopf));
        let RunKind::Cycle(CycleStart::Hopf(h)) = p.run_kind() else {
            panic!("expected a Hopf start");
        };
        assert_eq!((h.free, h.ntst, h.ncol), (0, 2, 3));
        assert_eq!(h.u0, ir.init);
        assert_eq!(h.params, ir.par_values);

        // An orbit on the uniform mesh: 2 states at N m + 1 = 7 points, then T.
        let mut orbit: Vec<f64> = (0..14).map(f64::from).collect();
        orbit.push(6.5);
        let mut v = lc("orbit");
        v["u0"] = json!(orbit);
        let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
        let RunKind::Cycle(CycleStart::Orbit(c)) = p.run_kind() else {
            panic!("expected an orbit start");
        };
        assert_eq!(c.profile, orbit[..14]);
        assert_eq!(c.period, 6.5);
        assert_eq!((c.free, c.ntst, c.ncol), (0, 2, 3));

        // A PD point of a run on 1 interval of 2 points: 2 states at 3 points, then T.
        let mut v = lc("pd");
        v["mesh"] = json!([0.0, 1.0]);
        v["u0"] = json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5]);
        let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
        let RunKind::Cycle(CycleStart::Pd(d)) = p.run_kind() else {
            panic!("expected a PD start");
        };
        assert_eq!(d.orbit, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(d.period, 7.5);
        assert_eq!(d.mesh, [0.0, 1.0]);
        assert_eq!((d.ntst, d.ncol), (2, 3));
    }

    #[test]
    fn lc_keys_are_checked() {
        let ir = model();
        let put = |mut v: Value, key: &str, x: Value| {
            v[key] = x;
            v
        };
        let drop = |mut v: Value, key: &str| {
            v.as_object_mut().unwrap().remove(key);
            v
        };
        let cases = [
            (drop(lc("hopf"), "ntst"), "ntst: missing".to_string()),
            (
                put(lc("hopf"), "ntst", json!(0)),
                "ntst: expected an integer of at least 1".into(),
            ),
            (drop(lc("hopf"), "ncol"), "ncol: missing".into()),
            (
                put(lc("hopf"), "ncol", json!(8)),
                "ncol: expected an integer from 2 to 7".into(),
            ),
            (
                put(lc("hopf"), "ncol", json!(1)),
                "ncol: expected an integer from 2 to 7".into(),
            ),
            (
                put(lc("hopf"), "ntst", json!(u64::MAX / 2)),
                "ntst: ntst times ncol is too large".into(),
            ),
            (drop(lc("hopf"), "start"), "start: missing".into()),
            (
                lc("bp"),
                "start: expected \"hopf\", \"orbit\" or \"pd\"".into(),
            ),
            (
                put(lc("hopf"), "free", json!(["a", "a"])),
                "free: expected a string".into(),
            ),
            (
                put(lc("hopf"), "mesh", json!([0.0, 1.0])),
                "mesh: only a start from a PD point takes a mesh".into(),
            ),
            (
                lc("pd"),
                "mesh: missing; a start from a PD point needs the mesh of its orbit".into(),
            ),
            (
                put(lc("pd"), "mesh", json!([0.0])),
                "mesh: expected an array of at least 2 numbers".into(),
            ),
            (
                put(lc("pd"), "mesh", json!([0.0, "x", 1.0])),
                "mesh[1]: expected a number".into(),
            ),
            (
                put(lc("pd"), "mesh", json!([0.1, 1.0])),
                "mesh: must run from 0 to 1".into(),
            ),
            (
                put(lc("pd"), "mesh", json!([0.0, 0.5, 0.5, 1.0])),
                "mesh: must rise strictly".into(),
            ),
            (
                put(lc("hopf"), "switch", json!("BP1")),
                "switch: only an EP run can switch branches".into(),
            ),
            (
                put(lc("hopf"), "start_is_special", json!(true)),
                "start_is_special: only an EP run starts at a special point".into(),
            ),
            (
                put(lc("hopf"), "options", json!({"detect": {"lp": true}})),
                "options.detect: an LC run always detects LPC, PD and NS".into(),
            ),
            (
                put(lc("hopf"), "options", json!({"mesh_adapt": -1})),
                "options.mesh_adapt: expected an integer of at least 0".into(),
            ),
            (
                with("options", json!({"mesh_adapt": 3})),
                "options.mesh_adapt: only an LC, LPC, PD, NS or HOM run adapts its mesh".into(),
            ),
        ];
        for (v, msg) in cases {
            assert_eq!(parse_problem(&v, &ir, Mode::Abi).unwrap_err(), msg, "{v}");
        }
        for key in ["ntst", "ncol", "mesh"] {
            let e = parse_problem(&with(key, json!(2)), &ir, Mode::Abi).unwrap_err();
            assert_eq!(
                e,
                format!("{key}: only an LC, LPC, PD, NS or HOM run takes this key")
            );
        }
        let e = parse_problem(&with("start", json!(2)), &ir, Mode::Abi).unwrap_err();
        assert_eq!(e, "start: only an LC or HOM run takes this key");
        let p = parse_problem(
            &put(lc("hopf"), "options", json!({"mesh_adapt": 0})),
            &ir,
            Mode::Abi,
        )
        .unwrap();
        assert_eq!(p.opts.mesh_adapt, 0);
    }

    #[test]
    fn the_start_state_length_follows_the_start() {
        let ir = model();
        // A Hopf start takes the equilibrium, n values, with ir.init as the CLI default.
        let p = parse_problem(&lc("hopf"), &ir, Mode::Cli).unwrap();
        assert_eq!(p.setup.u0, ir.init);
        assert_eq!(p.check_u0_len(2, 2), Ok(()));
        assert_eq!(
            p.check_u0_len(2, 15),
            Err("u0: expected 2 values, got 15".into())
        );
        // An orbit start takes n (N m + 1) + 1 = 15 values and has no default.
        let p = parse_problem(&lc("orbit"), &ir, Mode::Abi).unwrap();
        assert_eq!(p.check_u0_len(2, 15), Ok(()));
        assert_eq!(
            p.check_u0_len(2, 14),
            Err("u0: expected 15 values, got 14".into())
        );
        let missing = "u0: missing; an LC run from an orbit or a PD point needs the orbit and \
                       the period";
        assert_eq!(
            parse_problem(&lc("orbit"), &ir, Mode::Cli).unwrap_err(),
            missing
        );
        let mut v = lc("orbit");
        v["u0"] = json!([1.0, 2.0]);
        assert_eq!(
            parse_problem(&v, &ir, Mode::Cli).unwrap_err(),
            "u0: expected 15 values, got 2"
        );
        // A PD start on a mesh of 2 intervals takes n (2 m + 1) + 1 values, m from 2 to 7.
        let mut v = lc("pd");
        v["mesh"] = json!([0.0, 0.25, 1.0]);
        let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
        for m in 2..=7 {
            assert_eq!(p.check_u0_len(2, 2 * (2 * m + 1) + 1), Ok(()), "m = {m}");
        }
        let bad = "u0: expected n (N m + 1) + 1 values with n = 2, N = 2 from mesh and m from 2 \
                   to 7, got";
        // 7 is m = 1, 9 has an odd point count, 35 is m = 8.
        for got in [0, 1, 2, 3, 6, 7, 9, 35] {
            assert_eq!(
                p.check_u0_len(2, got),
                Err(format!("{bad} {got}")),
                "got = {got}"
            );
        }
        assert_eq!(parse_problem(&v, &ir, Mode::Cli).unwrap_err(), missing);
    }

    #[test]
    fn cycle_stride_checks_overflow() {
        let lc = Kind::Lc { ntst: 2, ncol: 3 };
        // [alpha, u(2 * 7), T, mesh(3), mult(4), tests(3), h, iters]
        assert_eq!(cycle_stride(lc, 2), Some(1 + 14 + 1 + 3 + 4 + 3 + 2));
        // [a2, u(2 * 7), T, a1, mesh(3), mult(4), LPPD, LPNS or PDNS, CPC or GPD, R1 or R2,
        // h, iters]
        for kind in [
            Kind::Lpc { ntst: 2, ncol: 3 },
            Kind::Pd { ntst: 2, ncol: 3 },
        ] {
            assert_eq!(cycle_stride(kind, 2), Some(1 + 14 + 2 + 3 + 4 + 4 + 2));
        }
        // [a2, u(2 * 7), T, a1, kappa, mesh(3), mult(4), R1 to R4, LPNS, PDNS, CH, h, iters]
        let ns = Kind::Ns { ntst: 2, ncol: 3 };
        assert_eq!(cycle_stride(ns, 2), Some(1 + 14 + 3 + 3 + 4 + 7 + 2));
        let huge = Kind::Lc {
            ntst: usize::MAX / 2,
            ncol: 3,
        };
        assert_eq!(cycle_stride(huge, 2), None);
        assert_eq!(
            cycle_stride(Kind::Lc { ntst: 1, ncol: 2 }, usize::MAX / 2),
            None
        );
        assert_eq!(cycle_stride(Kind::Ep, 2), None);
        // [a2, u(2 * 7), T, x0(2), a1, mesh(3), eig(4), codim2(14), h, iters]
        let hom = Kind::Hom { ntst: 2, ncol: 3 };
        assert_eq!(cycle_stride(hom, 2), Some(1 + 14 + 4 + 3 + 4 + 14 + 2));
    }

    /// A HOM problem on N = 2, m = 3, from an orbit on that mesh.
    fn hom_orbit() -> Value {
        let mut u0: Vec<f64> = (0..14).map(f64::from).collect();
        u0.push(20.0);
        json!({
            "type": "HOM",
            "free": ["b", "a"],
            "start": "orbit",
            "ntst": 2,
            "ncol": 3,
            "u0": u0,
        })
    }

    #[test]
    fn a_hom_problem_from_an_orbit_builds_a_hom_run() {
        let ir = model2();
        let p = parse_problem(&hom_orbit(), &ir, Mode::Cli).unwrap();
        assert_eq!(p.kind, Kind::Hom { ntst: 2, ncol: 3 });
        assert_eq!(p.curve_free, Some([1, 0]));
        assert_eq!(p.cycle, Some(CycleFrom::Orbit));
        let RunKind::HomCurve(h) = p.run_kind() else {
            panic!("expected a HOM run");
        };
        assert_eq!(h.kind, HomStartKind::Orbit);
        assert_eq!(h.free, [1, 0]);
        assert_eq!(h.params, [2.0, 3.0]);
        assert_eq!(h.orbit, (0..14).map(f64::from).collect::<Vec<_>>());
        assert_eq!(h.period, 20.0);
        assert_eq!(h.mesh, None);
        assert_eq!((h.ntst, h.ncol), (2, 3));
        assert!(!h.inclination_flips);
        // The orbit sits on the uniform mesh of the run, so u0 has n (N m + 1) + 1 values.
        let mut short = hom_orbit();
        short["u0"] = json!([1.0, 2.0]);
        assert_eq!(
            parse_problem(&short, &ir, Mode::Cli).unwrap_err(),
            "u0: expected 15 values, got 2"
        );
    }

    /// SPEC §8b-3: the key is a bool, off when missing, and only a HOM run takes it.
    #[test]
    fn inclination_flips_is_a_bool_that_only_a_hom_run_takes() {
        let ir = model2();
        let flips = |v: Value| {
            let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
            let RunKind::HomCurve(h) = p.run_kind() else {
                panic!("expected a HOM run");
            };
            h.inclination_flips
        };
        let mut v = hom_orbit();
        assert!(!flips(v.clone()));
        v["inclination_flips"] = json!(true);
        assert!(flips(v.clone()));
        v["inclination_flips"] = json!(false);
        assert!(!flips(v.clone()));
        v["inclination_flips"] = json!(1);
        assert_eq!(
            parse_problem(&v, &ir, Mode::Cli).unwrap_err(),
            "inclination_flips: expected true or false"
        );
        for other in [json!({"type": "EP", "free": "a"}), curve_of_cycles("LPC")] {
            let mut w = other;
            w["inclination_flips"] = json!(true);
            assert_eq!(
                parse_problem(&w, &ir, Mode::Cli).unwrap_err(),
                "inclination_flips: only a HOM run takes this key"
            );
        }
        let ivp = json!({"type": "IVP", "tspan": [0, 1], "inclination_flips": true});
        assert_eq!(
            parse_problem(&ivp, &ir, Mode::Cli).unwrap_err(),
            "inclination_flips: only a HOM run takes this key"
        );
    }

    #[test]
    fn a_hom_problem_from_a_cycle_takes_the_mesh_of_the_cycle() {
        let ir = model2();
        let mut v = curve_of_cycles("HOM");
        v["start"] = json!("cycle");
        let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
        assert_eq!(
            p.cycle,
            Some(CycleFrom::Point {
                mesh: vec![0.0, 1.0]
            })
        );
        let RunKind::HomCurve(h) = p.run_kind() else {
            panic!("expected a HOM run");
        };
        assert_eq!(h.kind, HomStartKind::Cycle);
        assert_eq!(h.orbit, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(h.period, 7.5);
        assert_eq!(h.mesh, Some(vec![0.0, 1.0]));
        assert_eq!((h.ntst, h.ncol), (2, 3));
    }

    #[test]
    fn hom_problems_reject_bad_starts_and_keys() {
        let ir = model2();
        let err = |edit: &dyn Fn(&mut Value)| {
            let mut v = hom_orbit();
            edit(&mut v);
            parse_problem(&v, &ir, Mode::Cli).unwrap_err()
        };
        // No default start.
        assert_eq!(
            err(&|v| {
                v.as_object_mut().unwrap().remove("start");
            }),
            "start: missing; a HOM run starts from \"orbit\" or \"cycle\""
        );
        assert_eq!(
            err(&|v| v["start"] = json!("hopf")),
            "start: expected \"orbit\" or \"cycle\""
        );
        assert_eq!(
            err(&|v| v["mesh"] = json!([0.0, 1.0])),
            "mesh: a HOM start from an orbit uses the uniform mesh"
        );
        assert_eq!(
            err(&|v| v["start"] = json!("cycle")),
            "mesh: missing; a HOM start from a cycle needs the mesh of its orbit"
        );
        assert_eq!(
            err(&|v| v["free"] = json!("a")),
            "free: expected two parameter names"
        );
        assert_eq!(
            err(&|v| v["options"] = json!({"detect": {"lp": false}})),
            "options.detect: a HOM curve always runs its codim-2 tests"
        );
        assert_eq!(
            err(&|v| {
                v.as_object_mut().unwrap().remove("u0");
            }),
            "u0: missing; a HOM curve needs the orbit and the period (or T) of its start"
        );
        assert_eq!(
            err(&|v| {
                v.as_object_mut().unwrap().remove("ntst");
            }),
            "ntst: missing"
        );
        // The C ABI passes the start as a buffer and adapts the mesh when asked.
        let mut v = hom_orbit();
        v.as_object_mut().unwrap().remove("u0");
        v["options"] = json!({"mesh_adapt": 3});
        let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
        assert_eq!(p.opts.mesh_adapt, 3);
        assert!(p.check_u0_len(2, 15).is_ok());
        assert!(p.check_u0_len(2, 14).is_err());
    }

    /// An LPC, PD or NS problem on N = 2, m = 3 from a point of a run on 1 interval of 2
    /// points.
    fn curve_of_cycles(ty: &str) -> Value {
        json!({
            "type": ty,
            "free": ["b", "a"],
            "ntst": 2,
            "ncol": 3,
            "mesh": [0.0, 1.0],
            "u0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5],
        })
    }

    #[test]
    fn curves_of_cycles_build_a_cycle_curve_run() {
        let ir = model2();
        for (ty, kind, name) in [
            ("LPC", Kind::Lpc { ntst: 2, ncol: 3 }, "LPC"),
            ("PD", Kind::Pd { ntst: 2, ncol: 3 }, "PD"),
            ("NS", Kind::Ns { ntst: 2, ncol: 3 }, "NS"),
        ] {
            let p = parse_problem(&curve_of_cycles(ty), &ir, Mode::Cli).unwrap();
            assert_eq!(p.kind, kind);
            assert_eq!(p.kind.name(), name);
            assert_eq!(p.kind.mesh_size(), Some((2, 3)));
            assert_eq!(p.curve_free, Some([1, 0]));
            assert_eq!(
                p.cycle,
                Some(CycleFrom::Point {
                    mesh: vec![0.0, 1.0]
                })
            );
            let s = match (kind, p.run_kind()) {
                (Kind::Lpc { .. }, RunKind::LpcCurve(s))
                | (Kind::Pd { .. }, RunKind::PdCurve(s))
                | (Kind::Ns { .. }, RunKind::NsCurve(s)) => s,
                (_, other) => panic!("{ty}: the wrong run kind {other:?}"),
            };
            assert_eq!(s.free, [1, 0]);
            assert_eq!(s.params, ir.par_values);
            assert_eq!(s.orbit, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            assert_eq!(s.period, 7.5);
            assert_eq!(s.mesh, [0.0, 1.0]);
            assert_eq!((s.ntst, s.ncol), (2, 3));
        }
        assert_eq!(Kind::Lpc { ntst: 2, ncol: 3 }.labels(), LPC_CURVE_LABELS);
        assert_eq!(Kind::Pd { ntst: 2, ncol: 3 }.labels(), PD_CURVE_LABELS);
        assert_eq!(Kind::Ns { ntst: 2, ncol: 3 }.labels(), NS_CURVE_LABELS);
        // Through the C ABI the start comes as a buffer, so u0 is not in the JSON.
        let mut v = curve_of_cycles("NS");
        v.as_object_mut().unwrap().remove("u0");
        let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
        assert_eq!(p.check_u0_len(2, 7), Ok(()));
        assert!(p.check_u0_len(2, 6).is_err());
        let p = parse_problem(
            &put(curve_of_cycles("PD"), "options", json!({"mesh_adapt": 2})),
            &ir,
            Mode::Cli,
        )
        .unwrap();
        assert_eq!(p.opts.mesh_adapt, 2);
    }

    fn put(mut v: Value, key: &str, x: Value) -> Value {
        v[key] = x;
        v
    }

    #[test]
    fn curve_of_cycles_keys_are_checked() {
        let ir = model2();
        let drop = |mut v: Value, key: &str| {
            v.as_object_mut().unwrap().remove(key);
            v
        };
        let bad_u0 = "u0: expected n (N m + 1) + 1 values with n = 2, N = 1 from mesh and m \
                      from 2 to 7, got 6";
        for ty in ["LPC", "PD", "NS"] {
            let c = || curve_of_cycles(ty);
            let cases = [
                (drop(c(), "ntst"), "ntst: missing"),
                (
                    put(c(), "ncol", json!(1)),
                    "ncol: expected an integer from 2 to 7",
                ),
                (
                    put(c(), "free", json!("a")),
                    "free: expected two parameter names",
                ),
                (
                    drop(c(), "mesh"),
                    "mesh: missing; a curve of cycles needs the mesh of the orbit at its start",
                ),
                (
                    put(c(), "mesh", json!([0.0, 0.5])),
                    "mesh: must run from 0 to 1",
                ),
                (
                    put(c(), "start", json!("pd")),
                    "start: only an LC or HOM run takes this key",
                ),
                (
                    drop(c(), "u0"),
                    "u0: missing; a curve of cycles needs the orbit and the period of its start",
                ),
                (
                    put(c(), "u0", json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
                    bad_u0,
                ),
                (
                    put(c(), "options", json!({"detect": {"lp": true}})),
                    "options.detect: a curve of cycles always runs its codim-2 tests",
                ),
                (
                    put(c(), "switch", json!("BP1")),
                    "switch: only an EP run can switch branches",
                ),
                (
                    put(c(), "start_is_special", json!(true)),
                    "start_is_special: only an EP run starts at a special point",
                ),
            ];
            for (v, msg) in cases {
                assert_eq!(parse_problem(&v, &ir, Mode::Cli).unwrap_err(), msg, "{v}");
            }
        }
    }

    fn ivp() -> Value {
        json!({"type": "IVP", "tspan": [0.0, 5.0]})
    }

    #[test]
    fn an_ivp_problem_takes_tspan_and_its_own_options() {
        let ir = model();
        let p = parse_problem(&ivp(), &ir, Mode::Abi).unwrap();
        assert_eq!(p.kind, Kind::Ivp);
        assert_eq!(Kind::Ivp.name(), "IVP");
        assert!(Kind::Ivp.labels().is_empty());
        let (tspan, opts) = p.ivp.clone().unwrap();
        assert_eq!(tspan, [0.0, 5.0]);
        assert_eq!(opts, pc_engine::IvpOptions::default());
        match p.run_kind() {
            RunKind::Ivp { tspan, opts } => {
                assert_eq!(tspan, [0.0, 5.0]);
                assert_eq!(opts, pc_engine::IvpOptions::default());
            }
            other => panic!("{other:?}"),
        }
        let mut v = ivp();
        v["tspan"] = json!([3.0, -1.0]);
        v["options"] = json!({"rtol": 1e-6, "atol": 1e-9, "max_steps": 50, "hmax": 0.25});
        let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
        let (tspan, opts) = p.ivp.unwrap();
        assert_eq!(tspan, [3.0, -1.0]);
        assert_eq!(opts.rtol, 1e-6);
        assert_eq!(opts.atol, 1e-9);
        assert_eq!(opts.max_steps, 50);
        assert_eq!(opts.hmax, 0.25);
    }

    #[test]
    fn an_ivp_problem_is_refused_when_it_is_malformed() {
        let ir = model();
        let put = |k: &str, x: Value| {
            let mut v = ivp();
            v[k] = x;
            v
        };
        let opt = |o: Value| put("options", o);
        let mut no_tspan = ivp();
        no_tspan.as_object_mut().unwrap().remove("tspan");
        let cases = [
            (no_tspan, "tspan: missing"),
            (put("tspan", json!([0.0])), "tspan: expected two numbers"),
            (
                put("tspan", json!([0.0, 1.0, 2.0])),
                "tspan: expected two numbers",
            ),
            (put("tspan", json!("0 5")), "tspan: expected two numbers"),
            (
                put("tspan", json!([0.0, "x"])),
                "tspan[1]: expected a number",
            ),
            (
                put("tspan", json!([2.0, 2.0])),
                "tspan: the two times must differ",
            ),
            (
                put("free", json!("a")),
                "free: an IVP run has no free parameter",
            ),
            (
                put("switch", json!("BP1")),
                "switch: only an EP run can switch branches",
            ),
            (
                put("ntst", json!(4)),
                "ntst: only an LC, LPC, PD, NS or HOM run takes this key",
            ),
            (
                opt(json!({"rtol": 0})),
                "options.rtol: must be greater than 0",
            ),
            (
                opt(json!({"rtol": -1e-3})),
                "options.rtol: must be greater than 0",
            ),
            (
                opt(json!({"atol": 0})),
                "options.atol: must be greater than 0",
            ),
            (
                opt(json!({"hmax": 0})),
                "options.hmax: must be greater than 0",
            ),
            (
                opt(json!({"max_steps": 0})),
                "options.max_steps: expected an integer of at least 1",
            ),
            (opt(json!({"hmin": 1e-9})), "options.hmin: unknown key"),
            (opt(json!({"h0": 0.1})), "options.h0: unknown key"),
            (
                opt(json!({"direction": 1})),
                "options.direction: unknown key",
            ),
            (opt(json!("fast")), "options: expected an object"),
        ];
        for (v, msg) in cases {
            assert_eq!(parse_problem(&v, &ir, Mode::Abi).unwrap_err(), msg, "{v}");
        }
        // The ABI passes the state and the parameters as buffers.
        let v = put("u0", json!([1.0, 2.0]));
        assert!(
            parse_problem(&v, &ir, Mode::Abi)
                .unwrap_err()
                .starts_with("u0:")
        );
    }

    #[test]
    fn the_command_line_reads_u0_and_params_of_an_ivp_problem() {
        let ir = model();
        let mut v = ivp();
        v["u0"] = json!([1.0, 2.0]);
        v["params"] = json!({"a": 7.0});
        let p = parse_problem(&v, &ir, Mode::Cli).unwrap();
        assert_eq!(p.setup.u0, vec![1.0, 2.0]);
        assert_eq!(p.setup.params, vec![7.0]);
        // Without u0 the run starts at the model's initial values.
        let p = parse_problem(&ivp(), &ir, Mode::Cli).unwrap();
        assert_eq!(p.setup.u0, vec![0.5, 1.5]);
        let mut v = ivp();
        v["u0"] = json!([1.0]);
        assert_eq!(
            parse_problem(&v, &ir, Mode::Cli).unwrap_err(),
            "u0: expected 2 values, got 1"
        );
    }

    #[test]
    fn the_type_error_lists_ivp() {
        let ir = model();
        let msg = parse_problem(&json!({"type": "XX"}), &ir, Mode::Abi).unwrap_err();
        assert!(msg.contains("\"IVP\""), "{msg}");
    }

    /// A scalar delay model.
    fn delay_model() -> IrModel {
        let mut map = SourceMap::new();
        let top = map.add("t.ode", "par r=1.3, tau=1\nx'=r*x*(1-delay(x,tau))\ndone\n");
        let loaded = load(top, &mut map, &mut MapLoader::new());
        pc_ir::lower_dde(&loaded.expanded, &loaded.info).expect("the delay model lowers")
    }

    #[test]
    fn a_delay_equation_takes_only_ep_h_and_lc() {
        let ir = delay_model();
        let text = "type: a delay equation takes only \"EP\", \"H\" or \"LC\"";
        for ty in ["LP", "LPC", "PD", "NS", "HOM", "IVP"] {
            let v = json!({"type": ty, "free": "tau"});
            let msg = parse_problem(&v, &ir, Mode::Abi).unwrap_err();
            assert_eq!(msg, text, "type {ty}");
        }
        // A type that no run has gets the generic text.
        let msg = parse_problem(&json!({"type": "FOO"}), &ir, Mode::Abi).unwrap_err();
        assert!(msg.starts_with("type: expected"), "{msg}");
    }

    #[test]
    fn a_delay_cycle_run_reads_its_keys() {
        let ir = delay_model();
        let v = json!({
            "type": "LC", "free": "tau", "ntst": 8, "ncol": 3,
            "omega": 1.5, "eps": 0.05, "floquet": {"nhist": 6, "count": 2},
        });
        let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
        assert_eq!(
            p.kind,
            Kind::DdeLc {
                ntst: 8,
                ncol: 3,
                count: 2
            }
        );
        assert_eq!(p.omega, Some(1.5));
        assert_eq!(p.eps, Some(0.05));
        assert_eq!(p.floquet, Some(DdeFloquetOptions { nhist: 6, count: 2 }));
        assert_eq!(p.kind.name(), "LC");
        assert_eq!(p.kind.labels(), ["LPC", "PD", "NS"]);
        let RunKind::DdeCycle {
            free,
            omega,
            eps,
            ntst,
            ncol,
            floquet,
        } = p.run_kind()
        else {
            panic!("expected a delay cycle run");
        };
        assert_eq!((free, omega, eps, ntst, ncol), (1, 1.5, 0.05, 8, 3));
        assert_eq!(floquet, Some(DdeFloquetOptions { nhist: 6, count: 2 }));
        // Without `floquet` the count is 0.
        let v = json!({
            "type": "LC", "free": "tau", "ntst": 8, "ncol": 3, "omega": 1.5, "eps": 0.05,
        });
        let p = parse_problem(&v, &ir, Mode::Abi).unwrap();
        assert_eq!(
            p.kind,
            Kind::DdeLc {
                ntst: 8,
                ncol: 3,
                count: 0
            }
        );
        assert_eq!(p.floquet, None);
    }

    #[test]
    fn a_delay_cycle_run_names_the_key_it_refuses() {
        let ir = delay_model();
        let ok = || {
            json!({
                "type": "LC", "free": "tau", "ntst": 8, "ncol": 3,
                "omega": 1.5, "eps": 0.05,
            })
        };
        let cases = [
            ("eps", json!(0.0), "eps: must be a number above 0"),
            ("eps", json!(-1.0), "eps: must be a number above 0"),
            ("eps", json!("x"), "eps: must be a number above 0"),
            ("eps", Value::Null, "eps: must be a number above 0"),
            ("omega", json!(1e-9), "omega: must be above 1e-8"),
            ("floquet", json!({"nhist": 6}), "floquet.count"),
            ("floquet", json!({"count": 2}), "floquet.nhist"),
            ("floquet", json!({"nhist": 0, "count": 2}), "floquet.nhist"),
            (
                "floquet",
                json!({"nhist": 6, "count": 2, "x": 1}),
                "floquet",
            ),
            ("start", json!("orbit"), "start"),
            ("mesh", json!([0.0, 1.0]), "mesh"),
            ("roots", json!(1), "roots"),
            ("lyapunov", json!(true), "lyapunov"),
            ("coefficients", json!(true), "coefficients"),
            ("switch", json!(true), "switch"),
            ("tspan", json!([0.0, 1.0]), "tspan"),
        ];
        for (key, value, want) in cases {
            let mut v = ok();
            v[key] = value;
            let msg = parse_problem(&v, &ir, Mode::Abi).unwrap_err();
            assert!(msg.contains(want), "{key}: {msg}");
        }
        for key in ["eps", "omega"] {
            let mut v = ok();
            v.as_object_mut().unwrap().remove(key);
            let msg = parse_problem(&v, &ir, Mode::Abi).unwrap_err();
            assert!(msg.starts_with(key), "{key}: {msg}");
        }
        for key in ["detect", "mesh_adapt"] {
            let mut v = ok();
            v["options"] = json!({ key: 1 });
            let msg = parse_problem(&v, &ir, Mode::Abi).unwrap_err();
            assert!(msg.contains(key), "{key}: {msg}");
        }
        let mut v = ok();
        v["free"] = json!(["tau", "r"]);
        assert!(parse_problem(&v, &ir, Mode::Abi).is_err());
    }

    #[test]
    fn the_delay_cycle_keys_are_errors_on_an_ode_run() {
        let ir = model2();
        for (key, value) in [
            ("eps", json!(0.05)),
            ("floquet", json!({"nhist": 6, "count": 2})),
        ] {
            let mut v = json!({"type": "LC", "free": "a", "ntst": 8, "ncol": 3});
            v[key] = value;
            let msg = parse_problem(&v, &ir, Mode::Abi).unwrap_err();
            assert!(msg.starts_with(key), "{key}: {msg}");
        }
    }
}
