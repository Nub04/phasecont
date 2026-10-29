//! `model_compile` with `emit_c` and the AOT evaluator, through the table from `pc_api_v1()`.

use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;
use std::time::{Duration, Instant};

use pc_engine::{engine_kinds, entry_symbol};
use pc_ir::lower;
use pc_ode::{MapLoader, SourceMap, load};
use pckern::{PcApiV1, pc_api_v1};
use serde_json::Value;

const LORENZ: &str = "par s=10, r=28, b=2.6666666666666665\n\
init x=1,y=1,z=1\n\
x'=s*(y-x)\n\
y'=x*(r-z)-y\n\
z'=x*y-b*z\n\
done\n";

const FHN: &str = "par a=0.7, b=0.8, c=0.08, i=0.5\n\
init v=-1,w=-0.5\n\
v'=v-v^3/3-w+i\n\
w'=c*(v+a-b*w)\n\
done\n";

const FLAGS: [&str; 6] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-std=c99",
    "-Wall",
];

fn api() -> &'static PcApiV1 {
    let p = pc_api_v1();
    assert!(!p.is_null());
    unsafe { &*p }
}

fn last_error() -> String {
    let p = (api().last_error)();
    assert!(!p.is_null());
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

struct Id(u64);

impl Drop for Id {
    fn drop(&mut self) {
        if self.0 != 0 {
            assert_eq!((api().free_handle)(self.0), 0);
        }
    }
}

/// A scratch folder that is removed at the end.
struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Temp {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        // The folder is not created: `emit_c` has to do that.
        Temp(std::env::temp_dir().join(format!("pc-ffi-{tag}-{}-{nanos}", std::process::id())))
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn parse(text: &str) -> Id {
    let mut id = 0u64;
    let rc = unsafe {
        (api().model_from_ode)(text.as_ptr().cast(), text.len(), &mut id, ptr::null_mut())
    };
    assert_eq!(rc, 0, "{}", last_error());
    Id(id)
}

fn compile(m: &Id, opts: &str) -> i32 {
    let c = CString::new(opts).unwrap();
    unsafe { (api().model_compile)(m.0, c.as_ptr()) }
}

fn info(m: &Id) -> Value {
    let mut s: *mut c_char = ptr::null_mut();
    assert_eq!(unsafe { (api().model_info)(m.0, &mut s) }, 0);
    let text = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(s) };
    serde_json::from_str(&text).unwrap()
}

fn rhs(m: &Id, x: &[f64], p: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    let rc = unsafe {
        (api().model_rhs)(
            m.0,
            x.as_ptr(),
            x.len(),
            p.as_ptr(),
            p.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
    assert_eq!(rc, 0, "{}", last_error());
    out
}

fn json_path(p: &Path) -> String {
    serde_json::to_string(p.to_str().unwrap()).unwrap()
}

/// Emits the C text for `text` into `dir` and returns the file path.
fn emit(m: &Id, dir: &Path) -> PathBuf {
    let rc = compile(m, &format!(r#"{{"emit_c":{}}}"#, json_path(dir)));
    assert_eq!(rc, 0, "{}", last_error());
    PathBuf::from(info(m)["c_file"].as_str().expect("c_file").to_owned())
}

/// Compiles the emitted C with gcc, or None (with a note) when gcc is missing.
fn build_lib(c_file: &Path) -> Option<PathBuf> {
    if Command::new("gcc").arg("--version").output().is_err() {
        println!("skipped: gcc is not installed");
        return None;
    }
    let lib = c_file.with_extension(std::env::consts::DLL_EXTENSION);
    let out = Command::new("gcc")
        .args(FLAGS)
        .arg("-o")
        .arg(&lib)
        .arg(c_file)
        .output()
        .expect("run gcc");
    assert!(
        out.status.success(),
        "gcc failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Some(lib)
}

fn aot_model(text: &str, tmp: &Temp) -> Option<Id> {
    let m = parse(text);
    let c = emit(&m, &tmp.0);
    let lib = build_lib(&c)?;
    let rc = compile(
        &m,
        &format!(r#"{{"evaluator":"aot","aot_lib":{}}}"#, json_path(&lib)),
    );
    assert_eq!(rc, 0, "{}", last_error());
    Some(m)
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn ulp_dist(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let key = |x: f64| {
        let u = x.to_bits();
        if u >> 63 == 1 { !u } else { u | (1 << 63) }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn emit_c_writes_the_text_that_emit_c_gives() {
    let tmp = Temp::new("emit");
    let m = parse(LORENZ);
    let path = emit(&m, &tmp.0);
    assert_eq!(path.parent().unwrap(), tmp.0);
    let name = path.file_name().unwrap().to_str().unwrap();
    let hex = name
        .strip_prefix("pc_model_")
        .and_then(|s| s.strip_suffix(".c"))
        .expect("pc_model_<hex>.c");
    assert_eq!(hex.len(), 16);
    assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()));

    let mut map = SourceMap::new();
    let f = map.add("model.ode", LORENZ);
    let l = load(f, &mut map, &mut MapLoader::new());
    let mut ir = lower(&l.expanded, &l.info).expect("lower");
    let kinds = engine_kinds();
    let entries: Vec<_> = kinds.iter().map(|k| ir.entry(k).unwrap()).collect();
    let named: Vec<_> = kinds.iter().map(entry_symbol).zip(entries.iter()).collect();
    let want = pc_codegen::emit_c(&ir, &named);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), want);

    // The same model gives the same file name; without an evaluator key the JIT is built.
    assert_eq!(emit(&m, &tmp.0), path);
    assert_eq!(info(&m)["evaluator"], "jit");
}

fn rhs_matches_jit(text: &str, x: &[f64], p: &[f64], tag: &str) {
    let tmp = Temp::new(tag);
    let Some(aot) = aot_model(text, &tmp) else {
        return;
    };
    let i = info(&aot);
    assert_eq!(i["evaluator"], "aot");
    assert!(i["aot_lib"].as_str().unwrap().ends_with(".so") || cfg!(not(target_os = "linux")));
    assert!(i["c_file"].as_str().unwrap().ends_with(".c"));
    let jit = parse(text);
    assert_eq!(compile(&jit, r#"{"evaluator":"jit"}"#), 0);
    for (xs, ps) in [(x.to_vec(), p.to_vec()), (vec![0.3; x.len()], p.to_vec())] {
        let (a, j) = (rhs(&aot, &xs, &ps), rhs(&jit, &xs, &ps));
        assert!(
            a.iter().zip(&j).all(|(a, j)| same(*a, *j)),
            "{tag}: {a:?} vs {j:?}"
        );
    }
}

#[test]
fn aot_rhs_equals_jit_on_fitzhugh_nagumo() {
    rhs_matches_jit(FHN, &[-1.0, -0.5], &[0.7, 0.8, 0.08, 0.5], "fhn");
}

#[test]
fn aot_rhs_equals_jit_on_lorenz() {
    rhs_matches_jit(LORENZ, &[1.0, 2.0, 3.0], &[10.0, 28.0, 8.0 / 3.0], "lorenz");
}

fn wait(run: u64) -> Value {
    let t0 = Instant::now();
    loop {
        let mut s: *mut c_char = ptr::null_mut();
        assert_eq!(
            unsafe { (api().run_poll)(run, &mut s) },
            0,
            "{}",
            last_error()
        );
        let text = unsafe { CStr::from_ptr(s) }.to_str().unwrap().to_owned();
        unsafe { (api().free_string)(s) };
        let v: Value = serde_json::from_str(&text).unwrap();
        if v["state"] != "running" {
            return v;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "run did not finish");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Runs an equilibrium branch in `v` and returns the state and the rows of branch 0.
fn branch(m: &Id) -> (Value, usize, Vec<f64>) {
    let problem = CString::new(
        r#"{"type":"EP","free":"i","options":{"h0":0.05,"hmax":0.2,"max_steps":120,"bounds":[-1,3]}}"#,
    )
    .unwrap();
    let (u0, p) = ([-1.1994, -0.6243], [0.7, 0.8, 0.08, 0.5]);
    let mut run = 0u64;
    let rc = unsafe {
        (api().run_start)(
            m.0,
            problem.as_ptr(),
            u0.as_ptr(),
            u0.len(),
            p.as_ptr(),
            p.len(),
            &mut run,
        )
    };
    assert_eq!(rc, 0, "{}", last_error());
    let run = Id(run);
    let st = wait(run.0);
    assert_eq!(st["state"], "done", "{st}");
    let count = st["branches"][0]["points"].as_u64().unwrap() as usize;
    let stride = st["stride"].as_u64().unwrap() as usize;
    let mut buf = vec![f64::NAN; count * stride];
    let mut rows = 0usize;
    let rc = unsafe {
        (api().run_copy)(
            run.0,
            0,
            0,
            0,
            count,
            buf.as_mut_ptr(),
            buf.len(),
            &mut rows,
        )
    };
    assert_eq!(rc, 0, "{}", last_error());
    assert_eq!(rows, count);
    (st, count, buf)
}

#[test]
fn a_continuation_run_with_the_aot_model_reproduces_the_jit_branch() {
    let tmp = Temp::new("run");
    let Some(aot) = aot_model(FHN, &tmp) else {
        return;
    };
    let jit = parse(FHN);
    assert_eq!(compile(&jit, r#"{"evaluator":"jit"}"#), 0);
    let (sj, nj, vj) = branch(&jit);
    let (sa, na, va) = branch(&aot);
    assert!(nj > 10, "branch too short: {nj}");
    assert_eq!(na, nj);
    assert_eq!(sa["branches"][0]["stop"], sj["branches"][0]["stop"]);
    let worst = va
        .iter()
        .zip(&vj)
        .map(|(a, j)| ulp_dist(*a, *j))
        .max()
        .unwrap();
    assert!(worst <= 4, "branches differ by {worst} ulp");
}

#[test]
fn a_bad_aot_path_gives_5_and_the_handle_stays_usable() {
    let tmp = Temp::new("bad");
    let m = parse(LORENZ);
    let missing = tmp.0.join("nothing-here.so");
    let rc = compile(
        &m,
        &format!(r#"{{"evaluator":"aot","aot_lib":{}}}"#, json_path(&missing)),
    );
    assert_eq!(rc, 5, "{}", last_error());
    assert!(info(&m)["evaluator"].is_null());
    assert_eq!(compile(&m, r#"{"evaluator":"jit"}"#), 0, "{}", last_error());
    assert_eq!(info(&m)["evaluator"], "jit");
    let out = rhs(&m, &[1.0, 1.0, 1.0], &[10.0, 28.0, 8.0 / 3.0]);
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn aot_without_aot_lib_and_other_bad_options_give_2() {
    let m = parse(LORENZ);
    for opts in [
        r#"{"evaluator":"aot"}"#,
        r#"{"evaluator":"jit","aot_lib":"x"}"#,
        r#"{"evaluator":"gpu"}"#,
        r#"{"evaluator":"aot","aot_lib":3}"#,
        r#"{"emit_c":3}"#,
        r#"{"other":1}"#,
    ] {
        assert_eq!(compile(&m, opts), 2, "{opts}: {}", last_error());
    }
    assert!(info(&m)["evaluator"].is_null());
}
