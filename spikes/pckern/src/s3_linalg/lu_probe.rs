//! Sparse LU threshold probe for spikecli (not in the contract). Times faer's
//! symbolic LU, numeric LU and solve on the full S3 matrix and on the reduced
//! system of method (b), once per supernodal threshold setting.

use std::fmt::{self, Write as _};
use std::time::Instant;

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::sparse::SparseColMat;
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::lu::{LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu};
use faer::{Conj, MatMut, Par};

use super::matrix::{Params, Problem, ranged};
use super::run::{max_pairwise_rel_diff, since};
use super::{condense, sparse};
use crate::ffi::{Args, Json, Result, bad_arg};
use crate::s2_jit::median;

/// Keys that `run` accepts.
const KEYS: [&str; 9] = [
    "n",
    "m",
    "N",
    "T",
    "seed",
    "reps",
    "budget_ms",
    "settings",
    "on",
];

/// Threshold settings with their names, in the order they run.
const SETTINGS: [(&str, SupernodalThreshold); 3] = [
    ("auto", SupernodalThreshold::AUTO),
    ("simplicial", SupernodalThreshold::FORCE_SIMPLICIAL),
    ("supernodal", SupernodalThreshold::FORCE_SUPERNODAL),
];

/// Matrices in the order they run.
const MATRICES: [&str; 2] = ["full", "reduced"];

/// Probe arguments after the checks. `settings` and `on` hold indices into
/// SETTINGS and MATRICES, ascending.
pub struct Opts {
    pub params: Params,
    pub reps: usize,
    pub budget_ms: usize,
    pub settings: Vec<usize>,
    pub on: Vec<usize>,
}

/// Reads a list of names joined by '+' (commas separate pairs in `Args`).
/// Returns the indices of the named items in the order of `names`.
fn subset(a: &Args, key: &str, names: &[&str]) -> Result<Vec<usize>> {
    let all = names.join("+");
    let mut seen = vec![false; names.len()];
    for item in a.str_or(key, &all).split('+') {
        let k = names.iter().position(|name| *name == item).ok_or_else(|| {
            bad_arg(format!(
                "{key}: unknown item {item:?}; use any of {}",
                names.join(", ")
            ))
        })?;
        if seen[k] {
            return Err(bad_arg(format!("{key}: item {item:?} given twice")));
        }
        seen[k] = true;
    }
    Ok((0..names.len()).filter(|&k| seen[k]).collect())
}

impl Opts {
    pub fn parse(text: &str) -> Result<Opts> {
        let a = Args::parse(text, &KEYS)?;
        let setting_names = SETTINGS.map(|(name, _)| name);
        Ok(Opts {
            params: Params::from_args(&a)?,
            reps: ranged(&a, "reps", 5, 1, 1000)?,
            budget_ms: ranged(&a, "budget_ms", 30000, 1, 3_600_000)?,
            settings: subset(&a, "settings", &setting_names)?,
            on: subset(&a, "on", &MATRICES)?,
        })
    }
}

/// Timed runs of one threshold setting on one matrix.
pub struct SettingResult {
    pub threshold: &'static str,
    pub variant: &'static str,
    pub reps_done: usize,
    pub symbolic_ns: Vec<f64>,
    pub numeric_ns: Vec<f64>,
    pub solve_ns: Vec<f64>,
    pub backward_error: f64,
    pub error: String,
    pub x: Vec<f64>,
}

/// Every setting on one matrix.
pub struct MatrixResult {
    pub name: &'static str,
    pub size: usize,
    pub nnz: usize,
    pub settings: Vec<SettingResult>,
    pub max_rel_diff: f64,
}

/// ‖Ax - b‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞) from the CSC arrays, or NaN when x is not
/// finite.
fn csc_backward_error(a: &SparseColMat<usize, f64>, b: &[f64], x: &[f64]) -> f64 {
    if x.len() != a.ncols() || x.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let mut r: Vec<f64> = b.iter().map(|v| -v).collect();
    let mut row_abs = vec![0.0; a.nrows()];
    let (s, val) = (a.symbolic(), a.val());
    for (j, &xj) in x.iter().enumerate() {
        for k in s.col_range(j) {
            let i = s.row_idx()[k];
            r[i] += val[k] * xj;
            row_abs[i] += val[k].abs();
        }
    }
    let inf = |v: &[f64]| v.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    inf(&r) / (inf(&row_abs) * inf(x) + inf(b))
}

/// Number of bytes of Debug output that `variant` reads.
const PREFIX_CAP: usize = 40;

/// Keeps the first PREFIX_CAP bytes of formatted text, then fails, so that a
/// large structure is not formatted in full.
struct Prefix {
    text: String,
    full: bool,
}

impl fmt::Write for Prefix {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.full {
            return Err(fmt::Error);
        }
        let room = PREFIX_CAP - self.text.len();
        if s.len() <= room {
            self.text.push_str(s);
            return Ok(());
        }
        let mut end = room;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        self.text.push_str(&s[..end]);
        self.full = true;
        Err(fmt::Error)
    }
}

/// The LU that faer picked. faer 0.24.4 keeps `SymbolicLu::raw` private
/// (lu.rs:1810-1816), so this reads the start of the derived Debug output.
fn variant(sym: &SymbolicLu<usize>) -> &'static str {
    let mut p = Prefix {
        text: String::new(),
        full: false,
    };
    // The sink fails after PREFIX_CAP bytes, so an error here is expected.
    let _ = write!(p, "{sym:?}");
    if p.text.starts_with("SymbolicLu { raw: Simplicial") {
        "simplicial"
    } else if p.text.starts_with("SymbolicLu { raw: Supernodal") {
        "supernodal"
    } else {
        ""
    }
}

/// Workspace for the numeric LU and the solve that `sym` describes.
fn scratch(sym: &SymbolicLu<usize>) -> StackReq {
    StackReq::any_of(&[
        sym.factorize_numeric_lu_scratch::<f64>(Par::Seq, Default::default()),
        sym.solve_in_place_scratch::<f64>(1, Par::Seq),
    ])
}

/// Run 0 is the warm-up and is not recorded. Each later run times the
/// symbolic LU, the numeric LU and the solve on their own. The runs stop
/// after `reps` timed runs, or once the timed runs exceed `budget_ns`.
fn timed_runs(
    a: &SparseColMat<usize, f64>,
    b: &[f64],
    t: SupernodalThreshold,
    reps: usize,
    budget_ns: f64,
    r: &mut SettingResult,
) -> std::result::Result<(), String> {
    let params = LuSymbolicParams {
        supernodal_flop_ratio_threshold: t,
        ..Default::default()
    };
    let mut req = StackReq::EMPTY;
    let mut buf = MemBuffer::new(req);
    let mut numeric = NumericLu::<usize, f64>::new();
    let mut x = b.to_vec();
    let mut total = 0.0;
    for run in 0..=reps {
        if run > 1 && total > budget_ns {
            break;
        }
        let t0 = Instant::now();
        let sym = factorize_symbolic_lu(a.symbolic(), params)
            .map_err(|e| format!("symbolic LU failed: {e:?}"))?;
        let ts = since(t0);
        // Untimed: the variant, and the workspace if this run needs more.
        if run == 0 {
            r.variant = variant(&sym);
        }
        if scratch(&sym) != req {
            req = scratch(&sym);
            buf = MemBuffer::try_new(req).map_err(|_| {
                format!("cannot allocate {} bytes of LU workspace", req.size_bytes())
            })?;
        }
        let t1 = Instant::now();
        let lu = sym
            .factorize_numeric_lu(
                &mut numeric,
                a.as_ref(),
                Par::Seq,
                MemStack::new(&mut buf),
                Default::default(),
            )
            .map_err(|e| format!("numeric LU failed: {e:?}"))?;
        let tn = since(t1);
        x.copy_from_slice(b);
        let rhs = MatMut::from_column_major_slice_mut(&mut x, b.len(), 1);
        let t2 = Instant::now();
        lu.solve_in_place_with_conj(Conj::No, rhs, Par::Seq, MemStack::new(&mut buf));
        let tv = since(t2);
        if run > 0 {
            r.symbolic_ns.push(ts);
            r.numeric_ns.push(tn);
            r.solve_ns.push(tv);
            total += ts + tn + tv;
            r.reps_done = run;
        }
    }
    r.x = x;
    Ok(())
}

/// Runs setting `k` on A x = b. A failed faer call leaves its text in
/// `error`, no timings, no x and a NaN backward error.
fn measure(
    a: &SparseColMat<usize, f64>,
    b: &[f64],
    k: usize,
    reps: usize,
    budget_ns: f64,
    bwe: &dyn Fn(&[f64]) -> f64,
) -> SettingResult {
    let mut r = SettingResult {
        threshold: SETTINGS[k].0,
        variant: "",
        reps_done: 0,
        symbolic_ns: Vec::new(),
        numeric_ns: Vec::new(),
        solve_ns: Vec::new(),
        backward_error: f64::NAN,
        error: String::new(),
        x: Vec::new(),
    };
    match timed_runs(a, b, SETTINGS[k].1, reps, budget_ns, &mut r) {
        Ok(()) => r.backward_error = bwe(&r.x),
        Err(e) => {
            r.error = e;
            r.reps_done = 0;
            r.symbolic_ns.clear();
            r.numeric_ns.clear();
            r.solve_ns.clear();
            r.x.clear();
        }
    }
    r
}

/// Runs every setting in `o.settings` on every matrix in `o.on`. Building
/// the matrices is not timed.
pub fn probe(o: &Opts) -> Result<Vec<MatrixResult>> {
    faer::set_global_parallelism(Par::Seq);
    let p = Problem::build(&o.params)?;
    let budget_ns = o.budget_ms as f64 * 1e6;
    let mut out = Vec::with_capacity(o.on.len());
    for &mi in &o.on {
        let full = MATRICES[mi] == "full";
        let (a, b) = if full {
            (sparse::csc(&p)?, p.b.clone())
        } else {
            let cond = condense::condense_all(&p, false);
            let bd = condense::border_rows(&p, &cond);
            condense::reduced_system(&p, &cond, &bd)?
        };
        let bwe = |x: &[f64]| {
            if full {
                p.backward_error(x)
            } else {
                csc_backward_error(&a, &b, x)
            }
        };
        let settings: Vec<SettingResult> = o
            .settings
            .iter()
            .map(|&k| measure(&a, &b, k, o.reps, budget_ns, &bwe))
            .collect();
        let xs: Vec<&[f64]> = settings
            .iter()
            .filter(|s| s.error.is_empty())
            .map(|s| s.x.as_slice())
            .collect();
        let max_rel_diff = if xs.len() < 2 {
            f64::NAN
        } else {
            max_pairwise_rel_diff(&xs)
        };
        out.push(MatrixResult {
            name: MATRICES[mi],
            size: a.nrows(),
            nnz: a.symbolic().compute_nnz(),
            settings,
            max_rel_diff,
        });
    }
    Ok(out)
}

/// Median, min and max of one step. NaN loses to any number in f64::min and
/// f64::max, so all three are NaN (null) only when there are no runs.
fn stats(ns: &[f64]) -> Json {
    let mut j = Json::default();
    j.num("median", median(ns))
        .num("min", ns.iter().copied().fold(f64::NAN, f64::min))
        .num("max", ns.iter().copied().fold(f64::NAN, f64::max));
    j
}

/// Parses the arguments, runs the probe and returns the JSON report.
pub fn run(args: &str) -> Result<String> {
    let o = Opts::parse(args)?;
    let results = probe(&o)?;
    let mut j = Json::default();
    j.str("spike", "s3lu")
        .str("target", env!("PCKERN_TARGET"))
        .str("profile", env!("PCKERN_PROFILE"))
        .int("n", o.params.n as i64)
        .int("m", o.params.m as i64)
        .int("N", o.params.big_n as i64);
    match i64::try_from(o.params.seed) {
        Ok(seed) => j.int("seed", seed),
        // A seed above i64::MAX prints as an f64 and may lose digits.
        Err(_) => j.num("seed", o.params.seed as f64),
    };
    j.int("reps", o.reps as i64)
        .int("budget_ms", o.budget_ms as i64);
    for m in results {
        let settings = m
            .settings
            .iter()
            .map(|s| {
                let mut e = Json::default();
                e.str("threshold", s.threshold)
                    .str("variant", s.variant)
                    .int("reps_done", s.reps_done as i64)
                    .obj("symbolic_ns", stats(&s.symbolic_ns))
                    .obj("numeric_ns", stats(&s.numeric_ns))
                    .obj("solve_ns", stats(&s.solve_ns))
                    .num("backward_error", s.backward_error)
                    .str("error", &s.error);
                e
            })
            .collect();
        let mut mj = Json::default();
        mj.int("size", m.size as i64)
            .int("nnz", m.nnz as i64)
            .objs("settings", settings)
            .num("max_rel_diff", m.max_rel_diff);
        j.obj(m.name, mj);
    }
    Ok(j.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::Error;
    use crate::s2_jit::median;
    use crate::s3_linalg::matrix::nnz_formula;
    use faer::sparse::Triplet;

    fn small(extra: &str) -> Vec<MatrixResult> {
        let o = Opts::parse(&format!("n=2 m=2 N=3 reps=1 {extra}")).unwrap();
        probe(&o).unwrap()
    }

    #[test]
    fn every_setting_solves_both_matrices() {
        let out = small("");
        let names: Vec<&str> = out.iter().map(|m| m.name).collect();
        assert_eq!(names, ["full", "reduced"]);
        // nr = n(Nm + 1) + 2 = 16 and nred = n(N + 1) + 2 = 10.
        assert_eq!((out[0].size, out[1].size), (16, 10));
        assert_eq!(out[0].nnz, nnz_formula(2, 2, 3));
        for m in &out {
            let names: Vec<&str> = m.settings.iter().map(|s| s.threshold).collect();
            assert_eq!(names, ["auto", "simplicial", "supernodal"]);
            for s in &m.settings {
                let what = format!("{} {}", m.name, s.threshold);
                assert_eq!(s.error, "", "{what}");
                assert!(s.backward_error <= 1e-13, "{what}: {:e}", s.backward_error);
                assert_eq!(s.reps_done, 1, "{what}");
                for ns in [&s.symbolic_ns, &s.numeric_ns, &s.solve_ns] {
                    assert_eq!(ns.len(), 1, "{what}");
                    assert!(ns[0] > 0.0, "{what}");
                }
            }
            assert!(m.max_rel_diff <= 1e-12, "{}: {:e}", m.name, m.max_rel_diff);
        }
    }

    #[test]
    fn forced_thresholds_report_their_variant() {
        for m in &small("") {
            assert_eq!(m.settings[1].variant, "simplicial", "{}", m.name);
            assert_eq!(m.settings[2].variant, "supernodal", "{}", m.name);
            let auto = m.settings[0].variant;
            assert!(auto == "simplicial" || auto == "supernodal", "{auto:?}");
        }
    }

    #[test]
    fn lists_run_in_the_fixed_order() {
        let o = Opts::parse("settings=supernodal+auto on=reduced").unwrap();
        assert_eq!((o.settings, o.on), (vec![0, 2], vec![1]));
        let out = small("settings=simplicial on=reduced");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].name, "reduced");
        assert_eq!(out[0].settings.len(), 1);
        // One setting leaves no pair to compare.
        assert!(out[0].max_rel_diff.is_nan());
    }

    #[test]
    fn bad_arguments_are_errors() {
        for (args, needle) in [
            ("settings=bogus", "bogus"),
            ("on=x", "\"x\""),
            ("settings=auto+auto", "auto"),
            ("on=full+reduced+full", "full"),
            ("settings=", "\"\""),
            ("on=reduced+", "\"\""),
            ("reps=0", "reps"),
            ("reps=1001", "reps"),
            ("budget_ms=0", "budget_ms"),
            ("budget_ms=3600001", "budget_ms"),
            ("n=0", "n=0"),
            ("threads=2", "threads"),
        ] {
            match Opts::parse(args) {
                Err(Error::BadArg(msg)) => assert!(msg.contains(needle), "{args}: {msg}"),
                Err(e) => panic!("{args}: expected BadArg, got {e:?}"),
                Ok(_) => panic!("{args}: expected BadArg, got Ok"),
            }
        }
    }

    #[test]
    fn the_budget_stops_the_runs_early() {
        // On the default shape (n=4 m=4 N=40, a 646 x 646 matrix) one run
        // takes about 0.4 ms even in a release build, so 1000 runs need
        // about 0.4 s against the 1 ms budget.
        let text = "reps=1000 budget_ms=1 settings=simplicial on=full";
        let out = probe(&Opts::parse(text).unwrap()).unwrap();
        let s = &out[0].settings[0];
        assert!(s.reps_done >= 1 && s.reps_done < 1000, "{}", s.reps_done);
        assert_eq!(s.symbolic_ns.len(), s.reps_done);
        let total: f64 = [&s.symbolic_ns, &s.numeric_ns, &s.solve_ns]
            .iter()
            .flat_map(|v| v.iter())
            .sum();
        assert!(total > 1e6, "stopped after {total} ns");
    }

    #[test]
    fn a_failing_factorization_fills_error_and_leaves_nan() {
        // Column 1 is empty, so the simplicial LU finds no pivot there.
        let t = [
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(2, 2, 1.0),
        ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &t).unwrap();
        let b = [1.0, 2.0, 3.0];
        let r = measure(&a, &b, 1, 3, 1e9, &|x| csc_backward_error(&a, &b, x));
        assert!(r.error.contains("numeric"), "{:?}", r.error);
        assert_eq!((r.variant, r.reps_done), ("simplicial", 0));
        assert!(median(&r.numeric_ns).is_nan() && r.backward_error.is_nan());
    }

    #[test]
    fn run_prints_every_top_level_key() {
        let json = run("n=2 m=2 N=3 reps=1").unwrap();
        assert!(json.starts_with("{\"spike\":\"s3lu\""), "{json}");
        for key in [
            "target",
            "profile",
            "n",
            "m",
            "N",
            "seed",
            "reps",
            "budget_ms",
            "full",
            "reduced",
        ] {
            assert!(
                json.contains(&format!("\"{key}\":")),
                "{key} missing: {json}"
            );
        }
    }
}
