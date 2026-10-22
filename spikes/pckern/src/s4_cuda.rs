//! S4: NVRTC saxpy for sm_120 through cudarc.
//!
//! One kernel is compiled two ways, to PTX for `compute_{maj}{min}` and to a
//! cubin for `sm_{maj}{min}`, and each result runs through the driver API.
//! NVRTC fuses `a * x[i] + y[i]` into one FMA by default (`--fmad=true`), so
//! a path passes when it matches `a*x + y` rounded once (`fma_ref`); the
//! distance from the unfused CPU result is reported for information only.

use std::ffi::{CString, c_char};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg, sys as dsys,
};
use cudarc::nvrtc::{self, CompileError, CompileOptions, Ptx, compile_ptx_with_opts, sys as nsys};

use crate::ffi::{Args, Json, Result, bad_arg, catch_panic, internal};
use crate::rng::SplitMix64;
use crate::s2_jit::median;

/// The kernel from contract section 5. Both paths compile this text.
const SRC: &str = r#"extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

const A: f32 = 1.7;
const MAX_N: usize = 1 << 27;
const MAX_REPS: usize = 10_000;

/// `run_json("s4", args)` and `spikecli s4`.
pub fn run(args: &str) -> Result<String> {
    let a = Args::parse(args, &["n", "reps", "seed"])?;
    let n = a.usize_or("n", 16_777_216)?;
    let reps = a.usize_or("reps", 20)?;
    let seed = a.u64_or("seed", 2026)?;
    if !(1..=MAX_N).contains(&n) {
        return Err(bad_arg(format!(
            "s4: n={n} is out of range; expected 1 <= n <= {MAX_N}"
        )));
    }
    if !(1..=MAX_REPS).contains(&reps) {
        return Err(bad_arg(format!(
            "s4: reps={reps} is out of range; expected 1 <= reps <= {MAX_REPS}"
        )));
    }
    // cudarc panics when a library or a symbol is missing. Outside the two
    // paths such a panic becomes an internal error, as contract section 5
    // says.
    match catch_panic(|| report(n, reps, seed)) {
        Ok(r) => r,
        Err(msg) => Err(internal(format!("s4: panic: {msg}"))),
    }
}

fn report(n: usize, reps: usize, seed: u64) -> Result<String> {
    // SAFETY: these only try to open the libraries and close them again.
    let (have_cuda, have_nvrtc) = unsafe { (dsys::is_culib_present(), nsys::is_culib_present()) };
    if !have_cuda || !have_nvrtc {
        return Err(internal(format!(
            "s4: CUDA driver library found: {have_cuda}, NVRTC library found: {have_nvrtc}"
        )));
    }
    let mut driver = 0;
    // SAFETY: the pointer is valid for one write. This works before cuInit.
    unsafe { dsys::cuDriverGetVersion(&mut driver) }
        .result()
        .map_err(|e| internal(format!("s4: cuDriverGetVersion: {:?}", e.0)))?;
    let (mut nv_major, mut nv_minor) = (0, 0);
    // SAFETY: both pointers are valid for one write.
    unsafe { nsys::nvrtcVersion(&mut nv_major, &mut nv_minor) }
        .result()
        .map_err(|e| internal(format!("s4: nvrtcVersion: {:?}", e.0)))?;
    let ctx = CudaContext::new(0).map_err(|e| internal(format!("s4: context: {:?}", e.0)))?;
    let device = ctx
        .name()
        .map_err(|e| internal(format!("s4: device name: {:?}", e.0)))?;
    let (cc_major, cc_minor) = ctx
        .compute_capability()
        .map_err(|e| internal(format!("s4: compute capability: {:?}", e.0)))?;

    let mut rng = SplitMix64::new(seed);
    let x: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    let job = Job {
        stream: ctx.default_stream(),
        ctx: &ctx,
        x: &x,
        y: &y,
        reps,
    };
    let paths = [
        run_path(&job, Kind::Ptx, format!("compute_{cc_major}{cc_minor}")),
        run_path(&job, Kind::Cubin, format!("sm_{cc_major}{cc_minor}")),
    ];
    let chosen = select(driver, nv_major, nv_minor);
    let pass = paths.iter().any(|p| p.name == chosen && p.pass());

    let mut j = Json::default();
    j.str("spike", "s4")
        .str("target", env!("PCKERN_TARGET"))
        .str("profile", env!("PCKERN_PROFILE"))
        .int("driver_version", driver.into())
        .int("nvrtc_major", nv_major.into())
        .int("nvrtc_minor", nv_minor.into())
        .str("nvrtc_path", &nvrtc_path())
        .str("device", &device)
        .int("cc_major", cc_major.into())
        .int("cc_minor", cc_minor.into())
        .int("n", n as i64)
        .num("a", f64::from(A))
        .int("reps", reps as i64)
        .str("chosen", chosen)
        .objs("paths", paths.iter().map(|p| p.json(n)).collect())
        .boolean("pass", pass);
    Ok(j.finish())
}

/// Contract section 5: PTX from NVRTC loads only on a driver at least as
/// new as NVRTC, and an sm_ cubin loads on any 13.x driver.
fn select(driver: i32, nvrtc_major: i32, nvrtc_minor: i32) -> &'static str {
    if driver >= nvrtc_major * 1000 + nvrtc_minor * 10 {
        "ptx"
    } else if driver >= 13000 {
        "cubin"
    } else {
        ""
    }
}

/// Distance in units in the last place. 0 means equal (+0 and -0 count as
/// equal); a NaN on either side gives infinity.
fn ulp_f32(a: f32, b: f32) -> f64 {
    fn ord(x: f32) -> i64 {
        let i = i64::from(x.to_bits() as i32);
        if i < 0 { i64::from(i32::MIN) - i } else { i }
    }
    if a.is_nan() || b.is_nan() {
        f64::INFINITY
    } else {
        ord(a).abs_diff(ord(b)) as f64
    }
}

/// The fused CPU reference: `a*x + y` rounded once, to nearest with ties to
/// even. It does not call `f32::mul_add`, which on x86_64-pc-windows-gnu
/// calls MinGW-w64's `fmaf`; inside MATLAB that missed the FMA result by up
/// to 2 ulp on the S4 inputs.
///
/// `a*x` is exact in f64 (24 by 24 bits fit in 53), and 2Sum (Knuth and
/// Møller, Algorithm 2 in Kornerup, Lefèvre, Louvet and Muller, HAL
/// ensl-00331519) gives the error of the f64 sum exactly. That lets the sum be
/// rounded to odd (Boldo and Melquiond, "Emulation of a FMA and
/// correctly-rounded sums: proved algorithms using rounding to odd", IEEE
/// Trans. Computers 57(4), 2008): an exact sum stays, any other moves to the
/// neighbor whose last bit is 1. Every f32, every midpoint of two f32 values
/// and the overflow threshold has at most 25 significant bits, so each is an
/// f64 whose last bit is 0. Rounding to odd never moves the sum onto or past
/// one of them, so the final `as f32` rounds it the way it rounds the exact
/// value.
fn fma_ref(a: f32, x: f32, y: f32) -> f32 {
    let p = f64::from(a) * f64::from(x);
    let z = f64::from(y);
    let s = p + z;
    let zz = s - p;
    let e = (p - (s - zz)) + (z - zz);
    // A zero sum is exact, so `e == 0` covers it; |s| < 2^257 cannot overflow.
    if e == 0.0 || !s.is_finite() {
        return s as f32;
    }
    let bits = s.to_bits();
    let odd = if bits & 1 == 1 {
        s
    } else if (e > 0.0) == (s > 0.0) {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    };
    odd as f32
}

/// Largest ulp distances of the device result from the fused and from the
/// unfused CPU reference. The unfused value is not an accuracy figure: where
/// `a*x + y` nearly cancels, rounding `a*x` before the add moves the result
/// by up to half an ulp of `a*x`, which can be millions of ulps of the small
/// result, so the value only shows that NVRTC fused the multiply-add.
fn max_ulps(x: &[f32], y: &[f32], out: &[f32]) -> (f64, f64) {
    let (mut fused, mut unfused) = (0.0f64, 0.0f64);
    for ((&xi, &yi), &oi) in x.iter().zip(y).zip(out) {
        fused = fused.max(ulp_f32(oi, fma_ref(A, xi, yi)));
        unfused = unfused.max(ulp_f32(oi, A * xi + yi));
    }
    (fused, unfused)
}

/// What both paths share.
struct Job<'a> {
    ctx: &'a Arc<CudaContext>,
    stream: Arc<CudaStream>,
    x: &'a [f32],
    y: &'a [f32],
    reps: usize,
}

#[derive(Clone, Copy)]
enum Kind {
    Ptx,
    Cubin,
}

/// What one path measured. NaN marks a value it did not reach.
struct PathReport {
    name: &'static str,
    arch: String,
    compile_ms: f64,
    load_ms: f64,
    error: String,
    max_ulp_fma: f64,
    max_ulp_unfused: f64,
    kernel_ns: Vec<f64>,
}

impl PathReport {
    fn pass(&self) -> bool {
        self.error.is_empty() && self.max_ulp_fma <= 1.0
    }

    fn json(&self, n: usize) -> Json {
        let ns = &self.kernel_ns;
        let mid = median(ns);
        let mut rows = Vec::new();
        if !ns.is_empty() {
            let mut t = Json::default();
            t.str("name", "kernel")
                .num("median_ns", mid)
                .num("min_ns", ns.iter().copied().fold(f64::INFINITY, f64::min))
                .num("max_ns", ns.iter().copied().fold(0.0, f64::max));
            rows.push(t);
        }
        let mut o = Json::default();
        o.str("name", self.name)
            .str("arch", &self.arch)
            .num("compile_ms", self.compile_ms)
            .num("load_ms", self.load_ms)
            .str("error", &self.error)
            .num("max_ulp_fma", self.max_ulp_fma)
            .num("max_ulp_unfused", self.max_ulp_unfused)
            // Bytes per nanosecond is GB/s. The median of no samples is NaN.
            .num("gbps", 12.0 * n as f64 / mid)
            .objs("timings", rows)
            .boolean("pass", self.pass());
        o
    }
}

/// Runs one path inside the panic barrier, so its failure ends up in its
/// own `error` and the other path still runs.
fn run_path(job: &Job, kind: Kind, arch: String) -> PathReport {
    let mut r = PathReport {
        name: match kind {
            Kind::Ptx => "ptx",
            Kind::Cubin => "cubin",
        },
        arch,
        compile_ms: f64::NAN,
        load_ms: f64::NAN,
        error: String::new(),
        max_ulp_fma: f64::NAN,
        max_ulp_unfused: f64::NAN,
        kernel_ns: Vec::new(),
    };
    let outcome = catch_panic(|| attempt(job, kind, &mut r));
    // cudarc's Drop code stores its errors in the context, and the next
    // load_module would return them. Take them here and charge them to the
    // path that caused them.
    let leftover = job.ctx.check_err();
    let error = match outcome {
        Ok(Ok(())) => leftover.err().map(|e| format!("cleanup: {:?}", e.0)),
        Ok(Err(msg)) => Some(msg),
        Err(msg) => Some(format!("panic: {msg}")),
    };
    if let Some(msg) = error {
        r.error = msg;
        r.kernel_ns.clear();
    }
    r
}

/// Formats a driver error as "step: CUDA_ERROR_NAME".
fn drv(step: &'static str) -> impl Fn(DriverError) -> String {
    move |e| format!("{step}: {:?}", e.0)
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

/// Compiles, loads, checks and times one path. It fills `r` as it goes, so
/// a failed path keeps what it measured.
fn attempt(job: &Job, kind: Kind, r: &mut PathReport) -> std::result::Result<(), String> {
    let t0 = Instant::now();
    let image = match kind {
        Kind::Ptx => {
            let opts = CompileOptions {
                options: vec![format!("--gpu-architecture={}", r.arch)],
                ..Default::default()
            };
            compile_ptx_with_opts(SRC, opts).map_err(compile_error)?
        }
        Kind::Cubin => Ptx::from_binary(compile_cubin(&r.arch)?),
    };
    r.compile_ms = ms_since(t0);

    let t0 = Instant::now();
    let module = job.ctx.load_module(image).map_err(drv("load_module"))?;
    r.load_ms = ms_since(t0);
    let f = module
        .load_function("saxpy")
        .map_err(drv("load_function"))?;

    let stream = &job.stream;
    let n = job.x.len();
    let x_d = stream.clone_htod(job.x).map_err(drv("copy x to device"))?;
    let mut y_d = stream.clone_htod(job.y).map_err(drv("copy y to device"))?;
    let (a, n_i32) = (A, n as i32);
    let cfg = LaunchConfig::for_num_elems(n as u32);
    // One launch followed by a stream synchronize, so execution errors
    // show up here and each timing covers the whole kernel.
    let launch = |y_d: &mut CudaSlice<f32>| -> std::result::Result<(), String> {
        let mut b = stream.launch_builder(&f);
        b.arg(&a).arg(&x_d).arg(y_d).arg(&n_i32);
        // SAFETY: the arguments match saxpy(float, const float*, float*,
        // int), x_d and y_d hold n floats, and the kernel checks i < n.
        unsafe { b.launch(cfg) }.map_err(drv("launch"))?;
        stream.synchronize().map_err(drv("synchronize"))
    };

    let read = |y_d: &CudaSlice<f32>| -> std::result::Result<Vec<f32>, String> {
        let out = stream.clone_dtoh(y_d).map_err(drv("copy y to host"))?;
        stream
            .synchronize()
            .map_err(drv("synchronize after copy"))?;
        Ok(out)
    };
    check_and_time(job.x, job.y, job.reps, r, &mut y_d, launch, read)
}

/// Checks the first launch against the CPU references, then times `reps`
/// launches after one warm-up. A result that misses, NaN included, shows
/// only in `max_ulp_fma` and `pass()`; `error` is for CUDA and NVRTC
/// failures, so the path keeps its timings. `launch` and `read` do the
/// device work on `y_d`, which lets unit tests run this without a GPU.
fn check_and_time<B>(
    x: &[f32],
    y: &[f32],
    reps: usize,
    r: &mut PathReport,
    y_d: &mut B,
    mut launch: impl FnMut(&mut B) -> std::result::Result<(), String>,
    read: impl FnOnce(&B) -> std::result::Result<Vec<f32>, String>,
) -> std::result::Result<(), String> {
    launch(y_d)?;
    let out = read(y_d)?;
    (r.max_ulp_fma, r.max_ulp_unfused) = max_ulps(x, y, &out);

    launch(y_d)?;
    let mut ns = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        launch(y_d)?;
        ns.push(t0.elapsed().as_secs_f64() * 1e9);
    }
    r.kernel_ns = ns;
    Ok(())
}

/// Formats a failed PTX compile as "step: NVRTC_ERROR_NAME", with the
/// compile log when there is one.
fn compile_error(e: CompileError) -> String {
    match e {
        CompileError::CompileError { nvrtc, log, .. } => {
            format!("compile: {:?}: {}", nvrtc.0, log.to_string_lossy().trim())
        }
        CompileError::CreationError(e) => format!("create_program: {:?}", e.0),
        CompileError::GetLogError(e) => format!("get_program_log: {:?}", e.0),
        CompileError::GetPtxError(e) => format!("get_ptx: {:?}", e.0),
        CompileError::DestroyError(e) => format!("destroy_program: {:?}", e.0),
    }
}

/// Compiles SRC to a cubin with `nvrtc::result`, since cudarc's safe API
/// only returns PTX. The program is destroyed on every way out, including
/// a panic inside cudarc.
fn compile_cubin(arch: &str) -> std::result::Result<Vec<u8>, String> {
    let src = CString::new(SRC).map_err(|e| format!("source: {e}"))?;
    let prog = nvrtc::result::create_program(&src, None)
        .map_err(|e| format!("create_program: {:?}", e.0))?;
    // SAFETY: prog stays live until destroy_program below, and src outlives it.
    let built = catch_panic(|| unsafe { cubin_from(prog, arch) });
    // SAFETY: prog came from create_program and is destroyed only here.
    let destroyed = unsafe { nvrtc::result::destroy_program(prog) };
    let cubin = built.map_err(|msg| format!("panic: {msg}"))??;
    destroyed.map_err(|e| format!("destroy_program: {:?}", e.0))?;
    Ok(cubin)
}

/// # Safety
/// `prog` must come from `create_program` and not be destroyed yet.
unsafe fn cubin_from(prog: nsys::nvrtcProgram, arch: &str) -> std::result::Result<Vec<u8>, String> {
    let opts = [format!("--gpu-architecture={arch}")];
    // SAFETY (all blocks): prog is live, as the caller promises.
    if let Err(e) = unsafe { nvrtc::result::compile_program(prog, &opts) } {
        return Err(format!("compile: {:?}: {}", e.0, unsafe { log_of(prog) }));
    }
    let mut size = 0usize;
    unsafe { nsys::nvrtcGetCUBINSize(prog, &mut size) }
        .result()
        .map_err(|e| format!("nvrtcGetCUBINSize: {:?}", e.0))?;
    let mut cubin = vec![0u8; size];
    unsafe { nsys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast()) }
        .result()
        .map_err(|e| format!("nvrtcGetCUBIN: {:?}", e.0))?;
    Ok(cubin)
}

/// The compile log as trimmed text, or a note when NVRTC cannot return it.
///
/// # Safety
/// `prog` must come from `create_program` and not be destroyed yet.
unsafe fn log_of(prog: nsys::nvrtcProgram) -> String {
    match unsafe { nvrtc::result::get_program_log(prog) } {
        Ok(raw) => c_text(&raw).trim().to_owned(),
        Err(e) => format!("no log ({:?})", e.0),
    }
}

/// Text from a NUL-terminated buffer that NVRTC filled.
fn c_text(raw: &[c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The file this process loaded NVRTC from, read from its memory map. The
/// name test skips libnvrtc-builtins, which NVRTC maps while it compiles.
#[cfg(target_os = "linux")]
fn nvrtc_path() -> String {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines()
        .filter_map(|line| line.find('/').map(|i| &line[i..]))
        .find(|path| {
            path.rsplit('/')
                .next()
                .is_some_and(|file| file.starts_with("libnvrtc.so"))
        })
        .unwrap_or("")
        .to_owned()
}

/// MATLAB asks the gateway's `module_path` on other targets.
#[cfg(not(target_os = "linux"))]
fn nvrtc_path() -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_follows_the_contract() {
        assert_eq!(select(13040, 13, 4), "ptx");
        assert_eq!(select(13020, 13, 4), "cubin");
        assert_eq!(select(12080, 13, 4), "");
    }

    #[test]
    fn ulp_distance_counts_floats_between() {
        assert_eq!(ulp_f32(1.5, 1.5), 0.0);
        assert_eq!(ulp_f32(1.0, 1.0f32.next_up()), 1.0);
        assert_eq!(ulp_f32(-1.0, (-1.0f32).next_down()), 1.0);
        assert_eq!(ulp_f32(-0.0, 0.0), 0.0);
        let tiny = f32::from_bits(1);
        assert_eq!(ulp_f32(-tiny, tiny), 2.0);
        assert_eq!(ulp_f32(f32::NAN, 1.0), f64::INFINITY);
        assert_eq!(ulp_f32(1.0, f32::NAN), f64::INFINITY);
        assert_eq!(ulp_f32(f32::NAN, f32::NAN), f64::INFINITY);
    }

    /// `a*x + y` rounded twice: once to f64, then to f32.
    fn naive(a: f32, x: f32, y: f32) -> f32 {
        (f64::from(a) * f64::from(x) + f64::from(y)) as f32
    }

    /// 2^e for a normal f32 exponent.
    fn pow2(e: i32) -> f32 {
        f32::from_bits(((e + 127) as u32) << 23)
    }

    /// Equal bits, or both NaN (NaN payloads are not compared).
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// The FMA instruction, which rounds `a*x + y` once. Callers check for
    /// FMA first.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    fn hardware(a: f32, x: f32, y: f32) -> f32 {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
        _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(x), _mm_set_ss(y)))
    }

    /// The exact `a*x + y` sits just off a midpoint of two f32 values and
    /// the f64 sum lands on that midpoint, so rounding twice picks the wrong
    /// neighbor. Each case checks that `naive` misses, so the case tests
    /// something.
    #[test]
    fn the_reference_rounds_once_where_double_rounding_misses() {
        let cases = [
            // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24, a midpoint; 2^-60 lifts it.
            (
                1.0 + pow2(-12),
                1.0 + pow2(-12),
                pow2(-60),
                1.0 + pow2(-11) + pow2(-23),
            ),
            // a*x = 2^-24 - 2^-60 puts the sum just under 1 + 2^-23 + 2^-24.
            (
                pow2(-24) * (1.0 + pow2(-18)),
                1.0 - pow2(-18),
                1.0 + pow2(-23),
                1.0 + pow2(-23),
            ),
        ];
        for (a, x, y, want) in cases {
            for (a, y, want) in [(a, y, want), (-a, -y, -want)] {
                assert_eq!(fma_ref(a, x, y), want, "fma_ref({a:e}, {x:e}, {y:e})");
                assert_ne!(naive(a, x, y), want, "naive({a:e}, {x:e}, {y:e})");
            }
        }
    }

    #[test]
    fn the_reference_handles_exact_and_special_values() {
        let tiny = f32::from_bits(1);
        let cases = [
            (1.5, 2.0, 0.25, 3.25),
            (f32::MAX, 2.0, -f32::MAX, f32::MAX),
            (f32::MAX, 2.0, 0.0, f32::INFINITY),
            (-f32::MAX, 2.0, 0.0, f32::NEG_INFINITY),
            (f32::INFINITY, 1.0, 1.0, f32::INFINITY),
            (f32::INFINITY, 0.0, 1.0, f32::NAN),
            (f32::INFINITY, 1.0, f32::NEG_INFINITY, f32::NAN),
            (f32::NAN, 1.0, 1.0, f32::NAN),
            (1.0, -0.0, 0.0, 0.0),
            (-1.0, 0.0, -0.0, -0.0),
            (2.0, 3.0, -6.0, 0.0),
            (-2.0, 3.0, 6.0, 0.0),
            (tiny, 0.5, 0.0, 0.0),
            (-tiny, 0.5, 0.0, -0.0),
            (tiny, 0.75, 0.0, tiny),
        ];
        for (a, x, y, want) in cases {
            let got = fma_ref(a, x, y);
            assert!(
                same(got, want),
                "fma_ref({a:e}, {x:e}, {y:e}) = {got:e}, want {want:e}"
            );
        }
    }

    /// `fma_ref` against the FMA instruction on random inputs of three kinds:
    /// any bit pattern (NaN, infinities, subnormals, overflow), `y` within a
    /// few ulps of `-a*x`, and 13-bit `a` and `x` with a `y` too small to
    /// change the f64 sum, which often lands that sum on a midpoint of two
    /// f32 values. The test also counts the inputs where `fma_ref` and
    /// `naive` differ and requires some, so it covers the double rounding
    /// case. Skips on a CPU without FMA.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_reference_matches_the_fma_instruction_on_random_inputs() {
        if !std::arch::is_x86_feature_detected!("fma") {
            println!("skipped: this CPU has no FMA instruction");
            return;
        }
        fn any(rng: &mut SplitMix64) -> f32 {
            f32::from_bits(rng.next_u64() as u32)
        }
        /// A random sign times 1.f times 2^e, with `frac` random fraction bits.
        fn short(rng: &mut SplitMix64, frac: u32, e: i32) -> f32 {
            let k = (rng.next_u64() >> (64 - frac)) as u32;
            let sign = rng.next_u64() as u32 & 0x8000_0000;
            f32::from_bits(sign | (((e + 127) as u32) << 23) | (k << (23 - frac)))
        }
        fn int(rng: &mut SplitMix64, lo: i32, hi: i32) -> i32 {
            lo + (rng.next_u64() % (hi - lo + 1) as u64) as i32
        }
        let mut rng = SplitMix64::new(754);
        let (mut misses, mut first, mut double_rounding) = (0usize, None, 0usize);
        for i in 0..3_000_000 {
            let (a, x, y) = match i % 3 {
                0 => (any(&mut rng), any(&mut rng), any(&mut rng)),
                1 => {
                    let (a, x) = (any(&mut rng), any(&mut rng));
                    let flip = rng.next_u64() as u32 & 0xf;
                    (a, x, -f32::from_bits((a * x).to_bits() ^ flip))
                }
                _ => {
                    let (ea, ex) = (int(&mut rng, -10, 10), int(&mut rng, -10, 10));
                    let ey = ea + ex - int(&mut rng, 54, 80);
                    (
                        short(&mut rng, 12, ea),
                        short(&mut rng, 12, ex),
                        short(&mut rng, 23, ey),
                    )
                }
            };
            let got = fma_ref(a, x, y);
            // SAFETY: the CPU has FMA, as checked above.
            let want = unsafe { hardware(a, x, y) };
            if !same(got, want) {
                misses += 1;
                first.get_or_insert((a, x, y, got, want));
            }
            double_rounding += usize::from(!same(got, naive(a, x, y)));
        }
        assert_eq!(
            misses, 0,
            "first miss (a, x, y, fma_ref, hardware): {first:?}"
        );
        assert!(double_rounding > 0, "no input separates fma_ref from naive");
        println!("{double_rounding} inputs where rounding twice differs");
    }

    /// The reference must equal the FMA instruction, which rounds `a*x + y`
    /// once, on every input of an S4 run with the default seed and size.
    /// Skips on a CPU without FMA.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_reference_matches_the_fma_instruction_on_the_s4_inputs() {
        if !std::arch::is_x86_feature_detected!("fma") {
            println!("skipped: this CPU has no FMA instruction");
            return;
        }
        let n = 16_777_216;
        let mut rng = SplitMix64::new(2026);
        let x: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let (mut worst, mut misses) = (0.0f64, 0usize);
        for (&xi, &yi) in x.iter().zip(&y) {
            // SAFETY: the CPU has FMA, as checked above.
            let d = ulp_f32(fma_ref(A, xi, yi), unsafe { hardware(A, xi, yi) });
            worst = worst.max(d);
            misses += usize::from(d > 0.0);
        }
        assert_eq!(
            (worst, misses),
            (0.0, 0),
            "largest ulp distance, inputs that differ"
        );
    }

    /// A report with nothing measured yet, as run_path starts it.
    fn blank(name: &'static str) -> PathReport {
        PathReport {
            name,
            arch: "sm_120".to_owned(),
            compile_ms: f64::NAN,
            load_ms: f64::NAN,
            error: String::new(),
            max_ulp_fma: f64::NAN,
            max_ulp_unfused: f64::NAN,
            kernel_ns: Vec::new(),
        }
    }

    /// A device result that holds a NaN makes `max_ulp_fma` infinite.
    /// `check_and_time` still times the kernel and returns Ok, so run_path
    /// leaves `error` empty and the path keeps its timings.
    #[test]
    fn a_nan_result_fails_the_check_and_keeps_the_timings() {
        let (x, y) = ([1.0f32, 2.0], [0.5f32, 0.25]);
        let mut r = blank("ptx");
        let mut launches = 0;
        let outcome = check_and_time(
            &x,
            &y,
            3,
            &mut r,
            &mut (),
            |_| {
                launches += 1;
                Ok(())
            },
            |_| Ok(vec![f32::NAN, fma_ref(A, 2.0, 0.25)]),
        );
        assert_eq!(outcome, Ok(()));
        assert_eq!(launches, 5, "first launch, warm-up, three timed");
        assert_eq!(r.max_ulp_fma, f64::INFINITY);
        assert_eq!(r.kernel_ns.len(), 3);
        assert!(r.error.is_empty());
        assert!(!r.pass());
    }

    /// A path that ran but missed prints an empty `error`, `max_ulp_fma` as
    /// null, one kernel timing row with its `gbps`, and `pass: false`.
    #[test]
    fn a_failed_check_prints_null_and_keeps_its_timing_row() {
        let mut r = blank("ptx");
        (r.compile_ms, r.load_ms) = (12.5, 0.5);
        (r.max_ulp_fma, r.max_ulp_unfused) = (f64::INFINITY, f64::INFINITY);
        r.kernel_ns = vec![2000.0, 1000.0];
        assert!(!r.pass());
        let text = r.json(1000).finish();
        assert!(text.contains(r#""error":"""#), "{text}");
        assert!(text.contains(r#""max_ulp_fma":null"#), "{text}");
        assert!(text.contains(r#""gbps":8e0"#), "{text}");
        assert!(text.contains(r#""timings":[{"name":"kernel","#), "{text}");
        assert_eq!(text.matches(r#""name":"kernel""#).count(), 1, "{text}");
        assert!(text.contains(r#""pass":false"#), "{text}");
    }

    /// A path that failed after its compile prints its error, keeps
    /// `compile_ms`, and has no timings, `gbps: null` and `pass: false`.
    #[test]
    fn a_failed_path_prints_its_error_and_no_timings() {
        let mut r = blank("cubin");
        r.compile_ms = 12.5;
        r.error = "load_module: CUDA_ERROR_INVALID_PTX".to_owned();
        assert!(!r.pass());
        let text = r.json(1000).finish();
        assert!(text.contains(r#""compile_ms":1.25e1"#), "{text}");
        assert!(text.contains(r#""load_ms":null"#), "{text}");
        let error = r#""error":"load_module: CUDA_ERROR_INVALID_PTX""#;
        assert!(text.contains(error), "{text}");
        assert!(text.contains(r#""timings":[]"#), "{text}");
        assert!(text.contains(r#""gbps":null"#), "{text}");
        assert!(text.contains(r#""pass":false"#), "{text}");
    }

    /// Records whether nvrtcGetPTX still returns PTX after an sm_ compile.
    /// The contract does not cover this, so the test asserts only that the
    /// compile works and the cubin is not empty. Run it with
    /// `--show-output`; the answer goes into the S4 report. Skips when
    /// either library is missing or no CUDA context can be created.
    #[test]
    fn nvrtc_get_ptx_after_an_sm_compile() {
        // SAFETY: these only try to open the libraries.
        if !unsafe { dsys::is_culib_present() && nsys::is_culib_present() } {
            println!("skipped: CUDA driver or NVRTC library not found");
            return;
        }
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                println!("skipped: no usable CUDA device ({:?})", e.0);
                return;
            }
        };
        let (maj, min) = ctx.compute_capability().expect("compute capability");
        let src = CString::new(SRC).unwrap();
        let prog = nvrtc::result::create_program(&src, None).expect("create_program");
        let opts = [format!("--gpu-architecture=sm_{maj}{min}")];
        // SAFETY: prog is live until destroy_program, and src outlives it.
        let (compiled, ptx, cubin) = unsafe {
            let compiled = nvrtc::result::compile_program(prog, &opts);
            let ptx = nvrtc::result::get_ptx(prog);
            let mut size = 0usize;
            let cubin = nsys::nvrtcGetCUBINSize(prog, &mut size)
                .result()
                .map(|()| size);
            nvrtc::result::destroy_program(prog).expect("destroy_program");
            (compiled, ptx, cubin)
        };
        compiled.expect("compile");
        println!("options: {opts:?}");
        println!("nvrtcGetCUBINSize: {cubin:?}");
        match ptx {
            Ok(raw) => {
                let text = c_text(&raw);
                let target = text.lines().find(|l| l.starts_with(".target"));
                println!("nvrtcGetPTX: ok, {} bytes, {target:?}", text.len());
            }
            Err(e) => println!("nvrtcGetPTX: {:?}", e.0),
        }
        assert!(cubin.expect("cubin size") > 0);
    }
}
