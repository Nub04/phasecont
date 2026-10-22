//! S2: Cranelift JIT of FitzHugh-Nagumo against AOT Rust and AOT C.
//!
//! x = [v, w] and p = [eps, a, b, I]. The right-hand side is
//! [v - v^3/3 - w + I, eps (v + a - b w)] and the Jacobian is stored column
//! major as [1 - v^2, eps, -1, -eps b]. The JIT code, the Rust references and
//! `spikes/mex/fhn_aot.c` use the same operation order and no fused
//! multiply-add, so their results should agree bit for bit.

use std::ffi::c_char;
use std::hint::black_box;
use std::time::Instant;

use cranelift_codegen::ir::{self, AbiParam, InstBuilder, MemFlagsData, UserFuncName, types};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};

use crate::RhsFn;
use crate::ffi::{Args, Json, Result, bad_arg, barrier, internal, into_c_string, out_arg, str_arg};
use crate::rng::SplitMix64;

/// The parameter values from `docs/KICKOFF.md`.
pub const P_NOMINAL: [f64; 4] = [0.08, 0.7, 0.8, 0.5];

/// Called from JIT code. Shims must not panic: JIT frames have no unwind
/// information, and the panic barrier sits outside the JIT call.
extern "C" fn pc_shim_exp(x: f64) -> f64 {
    x.exp()
}

/// Owns the JIT code. Dropping it frees the code pages, so the function
/// pointers must not outlive it.
pub struct Jit {
    module: Option<JITModule>,
    pub rhs: RhsFn,
    pub jac: RhsFn,
    /// out[0] = exp(x[0]) through `pc_shim_exp`, to check the shim path.
    pub exp: RhsFn,
    pub code_bytes: usize,
    pub compile_ms: f64,
}

impl Drop for Jit {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: the function pointers die with self.
            unsafe { m.free_memory() };
        }
    }
}

fn cl<T, E: std::fmt::Display>(what: &str, r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| internal(format!("cranelift: {what}: {e}")))
}

#[derive(Clone, Copy, PartialEq)]
enum Body {
    Rhs,
    Jac,
    Exp,
}

impl Jit {
    pub fn new() -> Result<Jit> {
        let t0 = Instant::now();
        let mut fb = settings::builder();
        for (k, v) in [
            ("opt_level", "speed"),
            ("is_pic", "true"),
            ("use_colocated_libcalls", "false"),
            // Windows grows the stack through a guard page, so frames over
            // 4 KiB need probes. Nothing provides __cranelift_probestack.
            ("enable_probestack", "true"),
            ("probestack_strategy", "inline"),
        ] {
            cl(k, fb.set(k, v))?;
        }
        let isa = cl("host ISA", cranelift_native::builder())?;
        let isa = cl("ISA flags", isa.finish(settings::Flags::new(fb)))?;
        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
        jb.symbol("pc_shim_exp", pc_shim_exp as *const u8);
        let mut module = JITModule::new(jb);

        let target = module.target_config();
        let ptr = target.pointer_type();
        let mut sig = module.make_signature();
        sig.params.extend([AbiParam::new(ptr); 3]);
        let mut shim_sig = module.make_signature();
        shim_sig.params.push(AbiParam::new(types::F64));
        shim_sig.returns.push(AbiParam::new(types::F64));
        let shim = cl(
            "declare pc_shim_exp",
            module.declare_function("pc_shim_exp", Linkage::Import, &shim_sig),
        )?;

        let mut ctx = module.make_context();
        let mut fbc = FunctionBuilderContext::new();
        let mut code_bytes = 0;
        let mut ids = Vec::new();
        for (name, body) in [
            ("fhn_rhs", Body::Rhs),
            ("fhn_jac", Body::Jac),
            ("shim_exp", Body::Exp),
        ] {
            let id = cl(name, module.declare_function(name, Linkage::Local, &sig))?;
            ctx.func.signature = sig.clone();
            ctx.func.name = UserFuncName::user(0, id.as_u32());
            let shim_ref =
                (body == Body::Exp).then(|| module.declare_func_in_func(shim, &mut ctx.func));
            build_body(&mut ctx.func, &mut fbc, body, shim_ref, target);
            cl(name, module.define_function(id, &mut ctx))?;
            code_bytes += ctx
                .compiled_code()
                .map_or(0, |c| c.code_info().total_size as usize);
            module.clear_context(&mut ctx);
            ids.push(id);
        }
        // Panics on an unresolved symbol; the caller's barrier catches that.
        cl("finalize", module.finalize_definitions())?;
        let get = |id: FuncId| -> RhsFn {
            let code = module.get_finalized_function(id);
            // SAFETY: the function was built with the RhsFn parameters and
            // the host's default calling convention.
            unsafe { std::mem::transmute::<*const u8, RhsFn>(code) }
        };
        let (rhs, jac, exp) = (get(ids[0]), get(ids[1]), get(ids[2]));
        Ok(Jit {
            module: Some(module),
            rhs,
            jac,
            exp,
            code_bytes,
            compile_ms: t0.elapsed().as_secs_f64() * 1e3,
        })
    }
}

// x, p and out are aligned f64 arrays that the caller guarantees, so the
// accesses cannot trap.
fn load(b: &mut FunctionBuilder, base: ir::Value, k: i32) -> ir::Value {
    b.ins()
        .load(types::F64, MemFlagsData::trusted(), base, 8 * k)
}

fn store(b: &mut FunctionBuilder, v: ir::Value, base: ir::Value, k: i32) {
    b.ins().store(MemFlagsData::trusted(), v, base, 8 * k);
}

fn build_body(
    func: &mut ir::Function,
    fbc: &mut FunctionBuilderContext,
    body: Body,
    shim: Option<ir::FuncRef>,
    target: TargetFrontendConfig,
) {
    let mut b = FunctionBuilder::new(func, fbc);
    let block = b.create_block();
    b.append_block_params_for_function_params(block);
    b.switch_to_block(block);
    b.seal_block(block);
    let (x, p, out) = {
        let v = b.block_params(block);
        (v[0], v[1], v[2])
    };
    match body {
        Body::Rhs => {
            let (v, w) = (load(&mut b, x, 0), load(&mut b, x, 1));
            let eps = load(&mut b, p, 0);
            let a = load(&mut b, p, 1);
            let bp = load(&mut b, p, 2);
            let i = load(&mut b, p, 3);
            // out[0] = ((v - ((v*v)*v)/3) - w) + I
            let vv = b.ins().fmul(v, v);
            let vvv = b.ins().fmul(vv, v);
            let three = b.ins().f64const(3.0);
            let cube = b.ins().fdiv(vvv, three);
            let r0 = b.ins().fsub(v, cube);
            let r0 = b.ins().fsub(r0, w);
            let r0 = b.ins().fadd(r0, i);
            // out[1] = eps * ((v + a) - b*w)
            let s = b.ins().fadd(v, a);
            let bw = b.ins().fmul(bp, w);
            let s = b.ins().fsub(s, bw);
            let r1 = b.ins().fmul(eps, s);
            store(&mut b, r0, out, 0);
            store(&mut b, r1, out, 1);
        }
        Body::Jac => {
            let v = load(&mut b, x, 0);
            let eps = load(&mut b, p, 0);
            let bp = load(&mut b, p, 2);
            let one = b.ins().f64const(1.0);
            let vv = b.ins().fmul(v, v);
            let j00 = b.ins().fsub(one, vv);
            let j01 = b.ins().f64const(-1.0);
            let eb = b.ins().fmul(eps, bp);
            let j11 = b.ins().fneg(eb);
            store(&mut b, j00, out, 0);
            store(&mut b, eps, out, 1);
            store(&mut b, j01, out, 2);
            store(&mut b, j11, out, 3);
        }
        Body::Exp => {
            let v = load(&mut b, x, 0);
            if let Some(f) = shim {
                let call = b.ins().call(f, &[v]);
                let r = b.inst_results(call)[0];
                store(&mut b, r, out, 0);
            }
        }
    }
    b.ins().return_(&[]);
    b.finalize(target);
}

#[inline(always)]
fn rhs_ref(x: &[f64; 2], p: &[f64; 4], out: &mut [f64; 2]) {
    let (v, w) = (x[0], x[1]);
    out[0] = v - v * v * v / 3.0 - w + p[3];
    out[1] = p[0] * (v + p[1] - p[2] * w);
}

#[inline(always)]
fn jac_ref(x: &[f64; 2], p: &[f64; 4], out: &mut [f64; 4]) {
    let v = x[0];
    out[0] = 1.0 - v * v;
    out[1] = p[0];
    out[2] = -1.0;
    out[3] = -(p[0] * p[2]);
}

// The Rust references behind a function pointer, so a call costs the same as
// a call into JIT code.

#[inline(never)]
unsafe extern "C" fn rhs_extern(x: *const f64, p: *const f64, out: *mut f64) {
    // SAFETY: the RhsFn contract gives x 2, p 4 and out 2 elements.
    unsafe { rhs_ref(&*x.cast(), &*p.cast(), &mut *out.cast()) }
}

#[inline(never)]
unsafe extern "C" fn jac_extern(x: *const f64, p: *const f64, out: *mut f64) {
    // SAFETY: the RhsFn contract gives x 2, p 4 and out 4 elements.
    unsafe { jac_ref(&*x.cast(), &*p.cast(), &mut *out.cast()) }
}

unsafe extern "C" fn exp_extern(x: *const f64, _p: *const f64, out: *mut f64) {
    // SAFETY: x and out have at least one element.
    unsafe { *out = (*x).exp() }
}

#[cfg(fhn_c)]
unsafe extern "C" {
    fn fhn_aot_rhs(x: *const f64, p: *const f64, out: *mut f64);
    fn fhn_aot_jac(x: *const f64, p: *const f64, out: *mut f64);
}

/// `spikes/mex/fhn_aot.c`, when build.rs compiled it (host builds only).
fn native_c() -> Option<(RhsFn, RhsFn, &'static str)> {
    #[cfg(fhn_c)]
    let c: Option<(RhsFn, RhsFn, &'static str)> =
        Some((fhn_aot_rhs, fhn_aot_jac, env!("PCKERN_CC")));
    #[cfg(not(fhn_c))]
    let c = None;
    c
}

/// Folds every output bit into the checksum. Each call is a bijection of the
/// running value, so a call whose `h` differs always changes the checksum.
/// Several differing calls can still cancel. A left rotation doubles the value
/// modulo 2^64 - 1, so two runs that each keep `h` constant for a multiple of
/// 64 calls often get the same checksum. S2 moves its inputs on every call, so
/// its `h` is not constant.
#[inline(always)]
fn fold<const N: usize>(acc: u64, out: &[f64; N]) -> u64 {
    let mut h = 0u64;
    for (k, v) in out.iter().enumerate() {
        h ^= v.to_bits().rotate_left(17 * k as u32);
    }
    acc.rotate_left(1).wrapping_add(h)
}

/// v sweeps [-2, 2) and w sweeps (-1, 1], so no term can move out of the loop.
#[inline(always)]
fn input(k: usize, step: f64) -> [f64; 2] {
    let t = k as f64 * step;
    [-2.0 + 4.0 * t, 1.0 - 2.0 * t]
}

/// `f` must write N values.
fn loop_ptr<const N: usize>(f: RhsFn, calls: usize) -> u64 {
    let f = black_box(f);
    let p = black_box(P_NOMINAL);
    let step = 1.0 / calls as f64;
    let mut out = [0.0; N];
    let mut acc = 0;
    for k in 0..calls {
        let x = input(k, step);
        // SAFETY: x has 2, p 4 and out N elements, and f writes N.
        unsafe { f(x.as_ptr(), p.as_ptr(), out.as_mut_ptr()) };
        acc = fold(acc, &out);
    }
    acc
}

fn loop_inline<const N: usize, F: Fn(&[f64; 2], &[f64; 4], &mut [f64; N])>(
    f: F,
    calls: usize,
) -> u64 {
    let p = P_NOMINAL;
    let step = 1.0 / calls as f64;
    let mut out = [0.0; N];
    let mut acc = 0;
    for k in 0..calls {
        let x = input(k, step);
        // black_box keeps the terms that depend only on p inside the loop.
        f(&x, black_box(&p), &mut out);
        acc = fold(acc, &out);
    }
    acc
}

struct Timing {
    name: &'static str,
    /// Nanoseconds per call, one entry per repeat.
    ns: Vec<f64>,
    checksum: u64,
}

/// One variant of a timing group: its name, and a run of n calls that
/// returns the checksum. Only the run is an indirect call; the loop of calls
/// inside it stays as `loop_ptr` or `loop_inline` compiled it.
type Variant<'a> = (&'static str, &'a dyn Fn(usize) -> u64);

/// The order of the variants in rep `r` of a group of `v`: (r + k) % v for
/// k = 0..v. Over v consecutive reps every variant runs once in every place.
fn rep_order(r: usize, v: usize) -> impl Iterator<Item = usize> {
    (0..v).map(move |k| (r + k) % v)
}

/// One untimed warm-up run per variant, which sets its checksum, then `reps`
/// rounds that time every variant once, in the order of `rep_order`. Every
/// timed run must give the same checksum. The rows keep the variants' order.
fn time_group(variants: &[Variant], reps: usize, calls: usize) -> Result<Vec<Timing>> {
    let mut t: Vec<Timing> = variants
        .iter()
        .map(|&(name, run)| Timing {
            name,
            ns: Vec::with_capacity(reps),
            checksum: run(calls),
        })
        .collect();
    for r in 0..reps {
        for i in rep_order(r, variants.len()) {
            let (name, run) = variants[i];
            let t0 = Instant::now();
            let c = black_box(run(calls));
            let dt = t0.elapsed();
            t[i].ns.push(dt.as_secs_f64() * 1e9 / calls as f64);
            if c != t[i].checksum {
                return Err(internal(format!(
                    "s2: {name}: checksum changed between runs"
                )));
            }
        }
    }
    Ok(t)
}

/// The jit time over another variant's time, rep by rep, with its median,
/// min and max.
#[derive(Debug, PartialEq)]
struct Ratio {
    name: &'static str,
    median: f64,
    min: f64,
    max: f64,
    per_rep: Vec<f64>,
}

fn ratio(name: &'static str, jit: &Timing, other: &Timing) -> Ratio {
    let per_rep: Vec<f64> = jit.ns.iter().zip(&other.ns).map(|(a, b)| a / b).collect();
    Ratio {
        name,
        median: median(&per_rep),
        min: per_rep.iter().copied().fold(f64::INFINITY, f64::min),
        max: per_rep.iter().copied().fold(0.0, f64::max),
        per_rep,
    }
}

/// Each group holds jit, rust via pointer, rust inlined and, when there is a
/// C column, c. Rows: rhs jit/pointer, rhs jit/c, jac jit/pointer, jac jit/c,
/// where the jit/c rows exist only with a C column.
fn ratios(rhs: &[Timing], jac: &[Timing]) -> Vec<Ratio> {
    let mut rows = Vec::new();
    for (group, to_ptr, to_c) in [
        (rhs, "rhs jit/pointer", "rhs jit/c"),
        (jac, "jac jit/pointer", "jac jit/c"),
    ] {
        rows.push(ratio(to_ptr, &group[0], &group[1]));
        if let Some(c) = group.get(3) {
            rows.push(ratio(to_c, &group[0], c));
        }
    }
    rows
}

pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Distance in units in the last place. 0 means equal (+0 and -0 count as
/// equal); a NaN against a number gives u64::MAX.
fn ulp(a: f64, b: f64) -> u64 {
    fn ord(x: f64) -> i64 {
        let i = x.to_bits() as i64;
        if i < 0 { i64::MIN - i } else { i }
    }
    match (a.is_nan(), b.is_nan()) {
        (true, true) => 0,
        (false, false) => ord(a).abs_diff(ord(b)),
        _ => u64::MAX,
    }
}

/// Largest ulp difference between `f` and `g` over random x in [-3, 3]^2
/// and p within 50% of the nominal values. Both must write N values.
fn max_ulp<const N: usize>(f: RhsFn, g: RhsFn, samples: usize, seed: u64) -> f64 {
    let mut rng = SplitMix64::new(seed);
    let mut worst = 0;
    for _ in 0..samples {
        let x = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
        let p = P_NOMINAL.map(|q| q * rng.range(0.5, 1.5));
        let (mut a, mut b) = ([0.0; N], [0.0; N]);
        // SAFETY: x has 2, p 4, a and b N elements, and f and g write N.
        unsafe {
            f(x.as_ptr(), p.as_ptr(), a.as_mut_ptr());
            g(x.as_ptr(), p.as_ptr(), b.as_mut_ptr());
        }
        for k in 0..N {
            worst = worst.max(ulp(a[k], b[k]));
        }
    }
    worst as f64
}

fn suite(args: &str, c: Option<(RhsFn, RhsFn, &str)>) -> Result<String> {
    let a = Args::parse(args, &["calls", "reps", "samples", "seed"])?;
    let calls = a.usize_or("calls", 10_000_000)?;
    let reps = a.usize_or("reps", 5)?;
    let samples = a.usize_or("samples", 100_000)?;
    let seed = a.u64_or("seed", 2026)?;
    if calls == 0 || reps == 0 || samples == 0 {
        return Err(bad_arg("s2: calls, reps and samples must be positive"));
    }
    let jit = Jit::new()?;

    let mut ulps = Json::default();
    ulps.num("jit_rhs", max_ulp::<2>(jit.rhs, rhs_extern, samples, seed))
        .num("jit_jac", max_ulp::<4>(jit.jac, jac_extern, samples, seed))
        .num(
            "jit_shim_exp",
            max_ulp::<1>(jit.exp, exp_extern, samples, seed),
        );
    if let Some((cr, cj, _)) = c {
        ulps.num("c_rhs", max_ulp::<2>(cr, rhs_extern, samples, seed))
            .num("c_jac", max_ulp::<4>(cj, jac_extern, samples, seed));
    }

    let jit_rhs = |n| loop_ptr::<2>(jit.rhs, n);
    let ptr_rhs = |n| loop_ptr::<2>(rhs_extern, n);
    let inline_rhs = |n| loop_inline(rhs_ref, n);
    let c_rhs = c.map(|(cr, _, _)| move |n| loop_ptr::<2>(cr, n));
    let mut rhs: Vec<Variant> = vec![
        ("jit rhs", &jit_rhs),
        ("rust rhs via pointer", &ptr_rhs),
        ("rust rhs inlined", &inline_rhs),
    ];
    if let Some(f) = &c_rhs {
        rhs.push(("c rhs", f));
    }
    let jit_jac = |n| loop_ptr::<4>(jit.jac, n);
    let ptr_jac = |n| loop_ptr::<4>(jac_extern, n);
    let inline_jac = |n| loop_inline(jac_ref, n);
    let c_jac = c.map(|(_, cj, _)| move |n| loop_ptr::<4>(cj, n));
    let mut jac: Vec<Variant> = vec![
        ("jit jac", &jit_jac),
        ("rust jac via pointer", &ptr_jac),
        ("rust jac inlined", &inline_jac),
    ];
    if let Some(f) = &c_jac {
        jac.push(("c jac", f));
    }
    let mut t = time_group(&rhs, reps, calls)?;
    let n_rhs = t.len();
    t.extend(time_group(&jac, reps, calls)?);
    let same = |ts: &[Timing]| ts.iter().all(|x| x.checksum == ts[0].checksum);

    let mut j = Json::default();
    j.str("spike", "s2")
        .str("target", env!("PCKERN_TARGET"))
        .str("profile", env!("PCKERN_PROFILE"))
        .int("calls", calls as i64)
        .int("reps", reps as i64)
        .int("samples", samples as i64)
        .num("compile_ms", jit.compile_ms)
        .int("code_bytes", jit.code_bytes as i64)
        .str("c_compiler", c.map_or("none", |c| c.2))
        .obj("max_ulp_vs_rust", ulps)
        .boolean("rhs_checksums_match", same(&t[..n_rhs]))
        .boolean("jac_checksums_match", same(&t[n_rhs..]));
    let rows = t
        .iter()
        .map(|x| {
            let mut o = Json::default();
            o.str("name", x.name)
                .num("median_ns", median(&x.ns))
                .num("min_ns", x.ns.iter().copied().fold(f64::INFINITY, f64::min))
                .num("max_ns", x.ns.iter().copied().fold(0.0, f64::max))
                .str("checksum", &format!("{:016x}", x.checksum));
            o
        })
        .collect();
    j.objs("timings", rows);
    let ratio_rows = ratios(&t[..n_rhs], &t[n_rhs..])
        .into_iter()
        .map(|x| {
            let mut o = Json::default();
            o.str("name", x.name)
                .num("median", x.median)
                .num("min", x.min)
                .num("max", x.max)
                .nums("per_rep", &x.per_rep);
            o
        })
        .collect();
    j.objs("ratios", ratio_rows);
    Ok(j.finish())
}

/// `run_json("s2", args)` and `spikecli s2`. The C column appears only when
/// build.rs compiled `fhn_aot.c`, which it does for host builds.
pub fn run(args: &str) -> Result<String> {
    suite(args, native_c())
}

/// The same report, timing C functions that the caller passes in. The MEX
/// gateway passes its copy of `fhn_aot.c`, compiled by MSVC through mex.
///
/// # Safety
/// `rhs` and `jac` must follow the `RhsFn` contract (jac writes 4 values),
/// `args` must be NULL or NUL-terminated, and `out` must be NULL or valid
/// for one write.
pub unsafe extern "C" fn bench_extern(
    rhs: Option<RhsFn>,
    jac: Option<RhsFn>,
    args: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    barrier(|| {
        let out = unsafe { out_arg(out, "bench_extern: output") }?;
        *out = std::ptr::null_mut();
        let args = unsafe { str_arg(args, "bench_extern: args") }?;
        let (Some(rhs), Some(jac)) = (rhs, jac) else {
            return Err(bad_arg("bench_extern: rhs and jac must not be NULL"));
        };
        *out = into_c_string(&suite(args, Some((rhs, jac, "passed in by the caller")))?);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn over_v_reps_every_variant_takes_every_position_once() {
        for v in 1..=5 {
            for start in 0..2 * v {
                // seen[variant][position]
                let mut seen = vec![vec![0; v]; v];
                for r in start..start + v {
                    let order: Vec<usize> = rep_order(r, v).collect();
                    let mut sorted = order.clone();
                    sorted.sort();
                    assert_eq!(sorted, (0..v).collect::<Vec<_>>(), "v={v} r={r}");
                    for (position, &variant) in order.iter().enumerate() {
                        seen[variant][position] += 1;
                    }
                }
                assert!(
                    seen.iter().flatten().all(|&n| n == 1),
                    "v={v} start={start}: {seen:?}"
                );
            }
        }
    }

    #[test]
    fn a_group_warms_up_each_variant_then_rotates() {
        let calls = RefCell::new(Vec::new());
        let f0 = |_: usize| {
            calls.borrow_mut().push(0);
            10
        };
        let f1 = |_: usize| {
            calls.borrow_mut().push(1);
            11
        };
        let f2 = |_: usize| {
            calls.borrow_mut().push(2);
            12
        };
        let variants: [Variant; 3] = [("a", &f0), ("b", &f1), ("c", &f2)];
        let t = time_group(&variants, 4, 1).unwrap();
        assert_eq!(
            calls.into_inner(),
            [0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2]
        );
        let names: Vec<_> = t.iter().map(|x| x.name).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert_eq!(
            t.iter().map(|x| x.checksum).collect::<Vec<_>>(),
            [10, 11, 12]
        );
        assert!(t.iter().all(|x| x.ns.len() == 4));
    }

    /// Folding these two outputs 128 times must give different checksums. A
    /// rotate-and-XOR fold repeats every 64 calls and gives 0 for both.
    #[test]
    fn a_difference_on_every_call_changes_the_checksum() {
        let run = |out: &[f64; 2]| (0..128).fold(0u64, |acc, _| fold(acc, out));
        assert_ne!(run(&[1.5, -0.25]), run(&[1.5, -0.5]));
    }

    #[test]
    fn a_group_rejects_a_checksum_that_changes() {
        let n = RefCell::new(0u64);
        let drift = |_: usize| {
            *n.borrow_mut() += 1;
            *n.borrow() / 3
        };
        let steady = |_: usize| 7;
        let variants: [Variant; 2] = [("steady", &steady), ("drift", &drift)];
        let Err(err) = time_group(&variants, 5, 1) else {
            panic!("the changed checksum was not caught");
        };
        assert!(
            err.to_string().contains("s2: drift: checksum changed"),
            "{err}"
        );
    }

    fn timing(name: &'static str, ns: [f64; 3]) -> Timing {
        Timing {
            name,
            ns: ns.to_vec(),
            checksum: 0,
        }
    }

    fn row(name: &'static str, median: f64, min: f64, max: f64, per_rep: [f64; 3]) -> Ratio {
        Ratio {
            name,
            median,
            min,
            max,
            per_rep: per_rep.to_vec(),
        }
    }

    #[test]
    fn ratios_divide_jit_by_the_other_variant_rep_by_rep() {
        let rhs = [
            timing("jit rhs", [2.0, 6.0, 9.0]),
            timing("rust rhs via pointer", [1.0, 2.0, 3.0]),
            timing("rust rhs inlined", [100.0, 100.0, 100.0]),
            timing("c rhs", [4.0, 3.0, 9.0]),
        ];
        let jac = [
            timing("jit jac", [8.0, 8.0, 8.0]),
            timing("rust jac via pointer", [4.0, 2.0, 16.0]),
            timing("rust jac inlined", [100.0, 100.0, 100.0]),
            timing("c jac", [1.0, 8.0, 32.0]),
        ];
        let rhs_ptr = row("rhs jit/pointer", 3.0, 2.0, 3.0, [2.0, 3.0, 3.0]);
        let jac_ptr = row("jac jit/pointer", 2.0, 0.5, 4.0, [2.0, 4.0, 0.5]);
        assert_eq!(
            ratios(&rhs, &jac),
            [
                rhs_ptr,
                row("rhs jit/c", 1.0, 0.5, 2.0, [0.5, 2.0, 1.0]),
                jac_ptr,
                row("jac jit/c", 1.0, 0.25, 8.0, [8.0, 1.0, 0.25]),
            ]
        );
        // Without a C column there are no jit/c rows.
        let rhs_ptr = row("rhs jit/pointer", 3.0, 2.0, 3.0, [2.0, 3.0, 3.0]);
        let jac_ptr = row("jac jit/pointer", 2.0, 0.5, 4.0, [2.0, 4.0, 0.5]);
        assert_eq!(ratios(&rhs[..3], &jac[..3]), [rhs_ptr, jac_ptr]);
    }
}
