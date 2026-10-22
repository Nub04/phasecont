//! Timing of the three methods and the JSON reports of contract sections 3
//! and 4.

use std::hint::black_box;
use std::time::Instant;

use faer::Par;

use super::condense;
use super::matrix::{Params, Problem, ranged};
use super::sparse;
use crate::ffi::{Args, Json, Result, bad_arg, internal};
use crate::s2_jit::median;

/// Keys that `s3_solve` accepts.
pub const SOLVE_KEYS: [&str; 3] = ["method", "threads", "reps"];
const RUN_KEYS: [&str; 7] = ["n", "m", "N", "T", "seed", "threads", "reps"];
/// Largest reduced order for the dense reduced solve.
pub const DENSE_MAX: usize = 5000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    SparseLu,
    CondensedSparse,
    CondensedDense,
}

impl Method {
    pub fn parse(s: &str) -> Result<Method> {
        match s {
            "sparse_lu" => Ok(Method::SparseLu),
            "condensed_sparse" => Ok(Method::CondensedSparse),
            "condensed_dense" => Ok(Method::CondensedDense),
            _ => Err(bad_arg(format!(
                "method={s} is unknown; use sparse_lu, condensed_sparse or condensed_dense"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Method::SparseLu => "sparse_lu",
            Method::CondensedSparse => "condensed_sparse",
            Method::CondensedDense => "condensed_dense",
        }
    }
}

/// Timed runs of one named step, in nanoseconds.
struct Timing {
    name: &'static str,
    ns: Vec<f64>,
}

impl Timing {
    fn json(&self) -> Json {
        let mut j = Json::default();
        j.str("name", self.name)
            .num("median_ns", median(&self.ns))
            .num(
                "min_ns",
                self.ns.iter().copied().fold(f64::INFINITY, f64::min),
            )
            .num("max_ns", self.ns.iter().copied().fold(0.0, f64::max));
        j
    }
}

/// x from the last timed run and the timings of one method.
pub struct Outcome {
    pub x: Vec<f64>,
    timings: Vec<Timing>,
}

fn named(steps: Vec<(&'static str, Vec<f64>)>) -> Vec<Timing> {
    steps
        .into_iter()
        .map(|(name, ns)| Timing { name, ns })
        .collect()
}

pub(super) fn since(t0: Instant) -> f64 {
    t0.elapsed().as_nanos() as f64
}

/// Method (a). "first" covers symbolic LU, numeric LU and solve; "repeat"
/// reuses the symbolic part. Building the CSC matrix is not timed.
fn sparse_lu(p: &Problem, reps: usize) -> Result<Outcome> {
    let a = sparse::csc(p)?;
    let finish = |lu: &faer::sparse::linalg::solvers::Lu<usize, f64>| {
        let mut x = p.b.clone();
        sparse::solve_in_place(lu, &mut x, false);
        x
    };
    let first_run = || -> Result<Vec<f64>> {
        let sym = sparse::symbolic(&a)?;
        Ok(finish(&sparse::numeric(&sym, &a)?))
    };
    black_box(first_run()?);
    let mut first = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        black_box(first_run()?);
        first.push(since(t0));
    }
    let sym = sparse::symbolic(&a)?;
    let mut x = black_box(finish(&sparse::numeric(&sym, &a)?));
    let mut repeat = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        x = black_box(finish(&sparse::numeric(&sym, &a)?));
        repeat.push(since(t0));
    }
    let timings = named(vec![("first", first), ("repeat", repeat)]);
    Ok(Outcome { x, timings })
}

/// Method (b) with a sparse or dense reduced solve. "first" builds the
/// reduced pattern and its symbolic LU in every run; "repeat" reuses them
/// and also records the condense, reduced and backsub steps.
fn condensed(p: &Problem, reps: usize, dense: bool, parallel: bool) -> Result<Outcome> {
    let par = if parallel { Par::rayon(0) } else { Par::Seq };
    let one = |pat: Option<&condense::Pattern>| -> Result<(Vec<f64>, [f64; 3])> {
        let t0 = Instant::now();
        let cond = condense::condense_all(p, parallel);
        let t1 = Instant::now();
        let bd = condense::border_rows(p, &cond);
        let xr = match (dense, pat) {
            (true, _) => condense::solve_dense(p, &cond, &bd, par),
            (false, Some(pat)) => condense::solve_sparse(p, &cond, &bd, pat)?,
            (false, None) => {
                let pat = condense::pattern(p, &cond, &bd)?;
                condense::solve_sparse(p, &cond, &bd, &pat)?
            }
        };
        let t2 = Instant::now();
        let x = condense::backsub_all(p, &cond, &xr, parallel);
        let d = |a: Instant, b: Instant| (b - a).as_nanos() as f64;
        Ok((x, [d(t0, t1), d(t1, t2), since(t2)]))
    };
    black_box(one(None)?);
    let mut first = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        black_box(one(None)?);
        first.push(since(t0));
    }
    let pat = if dense {
        None
    } else {
        let cond = condense::condense_all(p, parallel);
        let bd = condense::border_rows(p, &cond);
        Some(condense::pattern(p, &cond, &bd)?)
    };
    let (mut x, _) = black_box(one(pat.as_ref())?);
    let mut steps: [Vec<f64>; 4] = Default::default();
    for _ in 0..reps {
        let t0 = Instant::now();
        let (xi, parts) = black_box(one(pat.as_ref())?);
        steps[0].push(since(t0));
        for (s, v) in steps[1..].iter_mut().zip(parts) {
            s.push(v);
        }
        x = xi;
    }
    let [repeat, cond, reduced, backsub] = steps;
    let timings = named(vec![
        ("first", first),
        ("repeat", repeat),
        ("condense", cond),
        ("reduced", reduced),
        ("backsub", backsub),
    ]);
    Ok(Outcome { x, timings })
}

/// Settings of one `s3_solve` call.
#[derive(Clone, Copy, Debug)]
pub struct SolveArgs {
    pub method: Method,
    pub threads: usize,
    pub reps: usize,
}

impl SolveArgs {
    pub fn parse(text: &str) -> Result<SolveArgs> {
        let a = Args::parse(text, &SOLVE_KEYS)?;
        Ok(SolveArgs {
            method: Method::parse(a.str_or("method", "sparse_lu"))?,
            threads: ranged(&a, "threads", 1, 1, 256)?,
            reps: ranged(&a, "reps", 20, 1, 10000)?,
        })
    }
}

/// Rejects the dense reduced solve above order DENSE_MAX.
pub fn check(p: &Problem, method: Method) -> Result<()> {
    let nred = p.nred();
    if method == Method::CondensedDense && nred > DENSE_MAX {
        return Err(bad_arg(format!(
            "condensed_dense needs nred <= {DENSE_MAX}, but nred={nred}"
        )));
    }
    Ok(())
}

/// Runs one method. More than one thread runs in a local rayon pool, and
/// faer's global parallelism stays sequential, so only the code that asks
/// for the pool uses it. sparse_lu stays sequential at any thread count.
pub fn solve(p: &Problem, method: Method, threads: usize, reps: usize) -> Result<Outcome> {
    check(p, method)?;
    faer::set_global_parallelism(Par::Seq);
    let parallel = threads > 1;
    let go = || match method {
        Method::SparseLu => sparse_lu(p, reps),
        Method::CondensedSparse => condensed(p, reps, false, parallel),
        Method::CondensedDense => condensed(p, reps, true, parallel),
    };
    if !parallel {
        return go();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("pckern-s3-{i}"))
        .build()
        .map_err(|e| internal(format!("s3: cannot start the thread pool: {e}")))?;
    pool.install(go)
}

fn timings_json(out: &Outcome) -> Vec<Json> {
    out.timings.iter().map(Timing::json).collect()
}

/// The JSON of one `s3_solve` call (contract section 3).
pub fn solve_json(p: &Problem, s: &SolveArgs, out: &Outcome) -> String {
    let mut j = Json::default();
    j.str("spike", "s3")
        .str("method", s.method.name())
        .int("threads", s.threads as i64)
        .int("reps", s.reps as i64)
        .int("n", p.n as i64)
        .int("m", p.m as i64)
        .int("N", p.big_n as i64)
        .int("nr", p.nr as i64)
        .int("nnz", p.nnz as i64)
        .int("nred", p.nred() as i64)
        .num("backward_error", p.backward_error(&out.x))
        .objs("timings", timings_json(out));
    j.finish()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Largest ‖x_i - x_j‖₂ / ‖x_0‖₂ over all pairs. NaN wins over any number,
/// so a broken solution cannot pass.
pub(super) fn max_pairwise_rel_diff(xs: &[&[f64]]) -> f64 {
    let scale = norm2(xs[0]);
    let mut worst = 0.0f64;
    for (i, a) in xs.iter().enumerate() {
        for b in &xs[i + 1..] {
            let d: Vec<f64> = a.iter().zip(b.iter()).map(|(u, v)| u - v).collect();
            let r = norm2(&d) / scale;
            worst = if r.is_nan() || worst.is_nan() {
                f64::NAN
            } else {
                worst.max(r)
            };
        }
    }
    worst
}

fn bits_equal(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(u, v)| u.to_bits() == v.to_bits())
}

/// ‖A‖₁‖A⁻¹‖₁ estimated from a sparse LU. Not timed.
fn cond1_est(p: &Problem) -> Result<f64> {
    let a = sparse::csc(p)?;
    let lu = sparse::numeric(&sparse::symbolic(&a)?, &a)?;
    Ok(p.norm1() * sparse::inv_norm1_est(&lu, p.nr))
}

/// Builds the problem once and runs every method (contract section 4).
pub fn run(args: &str) -> Result<String> {
    let a = Args::parse(args, &RUN_KEYS)?;
    let params = Params::from_args(&a)?;
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let threads = ranged(&a, "threads", cores.clamp(1, 256), 1, 256)?;
    let reps = ranged(&a, "reps", 20, 1, 10000)?;
    let p = Problem::build(&params)?;
    let mut plan = vec![(Method::SparseLu, 1), (Method::CondensedSparse, 1)];
    if threads > 1 {
        plan.push((Method::CondensedSparse, threads));
    }
    if p.nred() <= DENSE_MAX {
        plan.push((Method::CondensedDense, 1));
        if threads > 1 {
            plan.push((Method::CondensedDense, threads));
        }
    }
    let mut outs = Vec::with_capacity(plan.len());
    for &(method, t) in &plan {
        outs.push(solve(&p, method, t, reps)?);
    }
    let errors: Vec<f64> = outs.iter().map(|o| p.backward_error(&o.x)).collect();
    let xs: Vec<&[f64]> = outs.iter().map(|o| o.x.as_slice()).collect();
    let diff = max_pairwise_rel_diff(&xs);
    let bitwise = threads == 1 || bits_equal(&outs[1].x, &outs[2].x);
    let pass = errors.iter().all(|&e| e <= 1e-13) && diff <= 1e-10 && bitwise;
    let methods = plan
        .iter()
        .zip(&outs)
        .zip(&errors)
        .map(|((&(method, t), out), &e)| {
            let mut j = Json::default();
            j.str("method", method.name())
                .int("threads", t as i64)
                .num("backward_error", e)
                .objs("timings", timings_json(out));
            j
        })
        .collect();
    let mut j = Json::default();
    j.str("spike", "s3")
        .str("target", env!("PCKERN_TARGET"))
        .str("profile", env!("PCKERN_PROFILE"))
        .int("n", p.n as i64)
        .int("m", p.m as i64)
        .int("N", p.big_n as i64)
        .int("nr", p.nr as i64)
        .int("nnz", p.nnz as i64)
        .int("nred", p.nred() as i64)
        .int("threads", threads as i64)
        .int("reps", reps as i64)
        .num("cond1_est", cond1_est(&p)?)
        .objs("methods", methods)
        .num("max_pairwise_rel_diff", diff)
        .boolean("sparse_threads_bitwise_equal", bitwise)
        .boolean("pass", pass);
    Ok(j.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_method_solves_edge_shapes() {
        for (n, m, big_n) in [(1, 2, 1), (2, 10, 3), (5, 3, 7), (1, 10, 2)] {
            let params = Params {
                n,
                m,
                big_n,
                t: 0.7,
                seed: 11,
            };
            let p = Problem::build(&params).unwrap();
            let one = solve(&p, Method::CondensedSparse, 1, 1).unwrap().x;
            let three = solve(&p, Method::CondensedSparse, 3, 1).unwrap().x;
            assert!(
                bits_equal(&one, &three),
                "{params:?}: threads change the bits"
            );
            for method in [
                Method::SparseLu,
                Method::CondensedSparse,
                Method::CondensedDense,
            ] {
                for threads in [1, 3] {
                    let x = solve(&p, method, threads, 1).unwrap().x;
                    let e = p.backward_error(&x);
                    assert!(e <= 1e-13, "{params:?} {method:?}@{threads}: error {e:e}");
                }
            }
        }
    }

    #[test]
    fn max_pairwise_rel_diff_lets_nan_win() {
        let a = [1.0, 0.0];
        let b = [1.0, 1e-12];
        let c = [f64::NAN, 0.0];
        assert!((max_pairwise_rel_diff(&[&a, &b]) - 1e-12).abs() < 1e-24);
        assert!(max_pairwise_rel_diff(&[&a, &c, &b]).is_nan());
        assert!(max_pairwise_rel_diff(&[&c, &a, &b]).is_nan());
    }
}
