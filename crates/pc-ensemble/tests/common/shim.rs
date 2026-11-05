//! Compiles the whole ensemble kernel as plain C and runs it from Rust.
//!
//! The C text is the one the GPU backends get, with the C dialect prelude. Its transcendental
//! functions come from a table of Rust callbacks that call the same `Real` methods the CPU
//! backend uses, so every operation matches `run_cpu` and the comparison can be bit for bit.

use std::path::PathBuf;
use std::process::Command;

use libloading::{Library, Symbol};
use pc_codegen::kernel::{Dialect, Scalar};
use pc_ensemble::abi::pack;
use pc_ensemble::job::{Job, Label, Method, Precision, validate};
use pc_ensemble::kernel::{Integrator, kernel_source};
use pc_ensemble::real::Real;
use pc_ensemble::{EnsembleResult, Tape};

const FLAGS: [&str; 9] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-fno-math-errno",
    "-std=c99",
    "-Wall",
    "-Wextra",
    "-Werror",
];

/// Launches before a run that cannot finish is called stuck.
const LAUNCH_LIMIT: u32 = 5_000_000;

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let p = std::env::temp_dir().join(format!("pc-shim-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn compiler() -> Option<&'static str> {
    ["gcc", "clang", "cc"].into_iter().find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

/// The transcendental functions of the kernel, in the order of `C_TABLE`: twelve of one
/// argument, then `atan2`, `pow` and `fmod`.
#[repr(C)]
struct Tbl<T: Real> {
    unary: [extern "C" fn(T) -> T; 12],
    binary: [extern "C" fn(T, T) -> T; 3],
}

macro_rules! unary {
    ($($name:ident => $m:ident),* $(,)?) => {
        $(extern "C" fn $name<T: Real>(x: T) -> T { x.$m() })*
    };
}

unary!(
    u_sin => sin, u_cos => cos, u_tan => tan, u_asin => asin, u_acos => acos,
    u_atan => atan, u_sinh => sinh, u_cosh => cosh, u_tanh => tanh, u_exp => exp,
    u_ln => ln, u_log10 => log10,
);

extern "C" fn b_atan2<T: Real>(x: T, y: T) -> T {
    x.atan2(y)
}

extern "C" fn b_pow<T: Real>(x: T, y: T) -> T {
    x.powf(y)
}

extern "C" fn b_fmod<T: Real>(x: T, y: T) -> T {
    x.fmod(y)
}

fn table<T: Real>() -> Tbl<T> {
    let unary: [extern "C" fn(T) -> T; 12] = [
        u_sin::<T>,
        u_cos::<T>,
        u_tan::<T>,
        u_asin::<T>,
        u_acos::<T>,
        u_atan::<T>,
        u_sinh::<T>,
        u_cosh::<T>,
        u_tanh::<T>,
        u_exp::<T>,
        u_ln::<T>,
        u_log10::<T>,
    ];
    let binary: [extern "C" fn(T, T) -> T; 3] = [b_atan2::<T>, b_pow::<T>, b_fmod::<T>];
    Tbl { unary, binary }
}

/// The kernel of one job, built as a shared library.
pub struct Shim {
    _dir: Dir,
    lib: Library,
    pub source: String,
}

type Ens<T> = unsafe extern "C" fn(
    *const T,
    *const u64,
    *const T,
    *const T,
    *mut T,
    *mut T,
    *mut u32,
    u64,
    u64,
    u32,
    u32,
    u64,
);

impl Shim {
    /// Builds the kernel for the model, method and precision of `job`. `None` after a loud
    /// skip when there is no C compiler.
    pub fn new(tag: &str, job: &Job) -> Option<Shim> {
        let Some(cc) = compiler() else {
            assert!(
                std::env::var_os("PC_REQUIRE_CC").is_none(),
                "no C compiler found and PC_REQUIRE_CC is set"
            );
            println!("SKIPPED: no C compiler");
            return None;
        };
        let tape = Tape::build(job.model).expect("tape");
        let integrator = match job.method {
            Method::Rk4 { .. } => Integrator::Rk4,
            Method::Dp5 { .. } => Integrator::Dp5,
        };
        let scalar = match job.precision {
            Precision::Single => Scalar::F32,
            Precision::Double => Scalar::F64,
        };
        let source = kernel_source(&tape, integrator, scalar, Dialect::C).expect("kernel source");
        let dir = Dir::new(tag);
        let src = dir.0.join("k.c");
        let so = dir.0.join("k.so");
        std::fs::write(&src, &source).unwrap();
        let out = Command::new(cc)
            .args(FLAGS)
            .arg(&src)
            .arg("-o")
            .arg(&so)
            .arg("-lm")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{cc} failed for {tag}:\n{}\n--- source ---\n{source}",
            String::from_utf8_lossy(&out.stderr)
        );
        // SAFETY: the library is the one just built from text this test produced.
        let lib = unsafe { Library::new(&so) }.unwrap();
        Some(Shim {
            _dir: dir,
            lib,
            source,
        })
    }

    /// Runs `job` through the kernel, `steps` step attempts per member and launch.
    pub fn run(&self, job: &Job, steps: u64) -> EnsembleResult {
        match job.precision {
            Precision::Single => self.run_as::<f32>(job, steps),
            Precision::Double => self.run_as::<f64>(job, steps),
        }
    }

    fn run_as<T: Real>(&self, job: &Job, steps: u64) -> EnsembleResult {
        let checked = validate(job).expect("validate");
        let tape = Tape::build(job.model).expect("tape");
        let mut p = pack::<T>(job, &checked, &tape);
        let tbl = table::<T>();
        // SAFETY: both symbols are defined by the text built in `new`, with these types.
        let (init, ens): (Symbol<unsafe extern "C" fn(*const Tbl<T>)>, Symbol<Ens<T>>) = unsafe {
            (
                self.lib.get(b"pc_shim_init").expect("pc_shim_init"),
                self.lib.get(b"pc_ens").expect("pc_ens"),
            )
        };
        // SAFETY: `tbl` outlives the run and `pc_shim_init` copies it.
        unsafe { init(&tbl) };
        let mut launches = 0;
        while !p.done() {
            launches += 1;
            assert!(launches <= LAUNCH_LIMIT, "the kernel never finished");
            // SAFETY: every buffer is sized by `pack` for `members` members and `ns` samples.
            unsafe {
                ens(
                    p.rc.as_ptr(),
                    p.segm.as_ptr(),
                    p.x0.as_ptr(),
                    p.par.as_ptr(),
                    p.work.as_mut_ptr(),
                    p.out.as_mut_ptr(),
                    p.ctl.as_mut_ptr(),
                    p.members as u64,
                    steps,
                    p.max_steps,
                    p.ns as u32,
                    p.pstride as u64,
                );
            }
        }
        let label = Label {
            backend: "c-shim".into(),
            device: "host".into(),
            driver: "cc".into(),
            precision: job.precision,
        };
        p.finish(label).expect("finish")
    }
}

/// Fails unless the two results agree in every status, count and bit.
pub fn assert_same(what: &str, want: &EnsembleResult, got: &EnsembleResult) {
    assert_eq!(want.n, got.n, "{what}: n");
    assert_eq!(want.members, got.members, "{what}: members");
    assert_eq!(want.status, got.status, "{what}: status");
    assert_eq!(want.accepted, got.accepted, "{what}: accepted");
    assert_eq!(want.rejected, got.rejected, "{what}: rejected");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&want.times), bits(&got.times), "{what}: times");
    assert_eq!(
        bits(&want.t_reached),
        bits(&got.t_reached),
        "{what}: t_reached"
    );
    for (i, (a, b)) in want.states.iter().zip(&got.states).enumerate() {
        // NaN marks a sample a member did not reach: the payload does not matter.
        let same = (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits();
        assert!(
            same,
            "{what}: states[{i}] is {b:e} ({:#x}) in the kernel, {a:e} ({:#x}) on the CPU",
            b.to_bits(),
            a.to_bits()
        );
    }
    assert_eq!(want.states.len(), got.states.len(), "{what}: states length");
}

/// Compares the CPU backend with the kernel for several launch sizes. Returns the CPU
/// result, or `None` after a loud skip.
pub fn check(what: &str, job: &Job) -> Option<EnsembleResult> {
    let shim = Shim::new(what, job)?;
    let want = super::run(job);
    for steps in [1, 7, 64, u64::MAX] {
        let got = shim.run(job, steps);
        assert_same(&format!("{what} (steps {steps})"), &want, &got);
    }
    Some(want)
}
