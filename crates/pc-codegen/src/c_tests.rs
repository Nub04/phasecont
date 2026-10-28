//! The AOT C text is compiled with gcc, loaded, and compared bit for bit with the JIT.

use crate::c::MATH_FIELDS;
use crate::emit_c;
use libloading::{Library, Symbol};
use pc_ir::dag::{Dag, Fn1, Fn2, NodeId};
use pc_ir::eval::eval;
use pc_ir::math::{MATH, MathTable};
use pc_ir::{Entry, EntryKind, IrModel, Slot};
use pc_jit::CompiledModel;
use std::path::PathBuf;
use std::process::Command;

const FLAGS: [&str; 7] = [
    "-O2",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-std=c99",
    "-Wall",
    "-Wextra",
];

fn build_model(src: &str) -> IrModel {
    use pc_ode::{MapLoader, Severity, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    match pc_ir::lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    }
}

fn cc_version(cc: &str) -> Option<String> {
    let o = Command::new(cc).arg("--version").output().ok()?;
    if !o.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&o.stdout);
    Some(s.lines().next()?.trim().to_string())
}

fn gcc_version() -> Option<String> {
    cc_version("gcc")
}

/// Compilers the AOT text is checked with: gcc, and clang when installed.
fn compilers() -> Vec<&'static str> {
    ["gcc", "clang"]
        .into_iter()
        .filter(|cc| cc_version(cc).is_some())
        .collect()
}

/// Writes references/aot/compiler.txt (first line of `gcc --version`, then the flags) when
/// the content changed. A read-only checkout only gets a note.
fn record_compiler(version: &str) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/aot");
    let path = dir.join("compiler.txt");
    let text = format!("{version}\nflags: gcc {} -Werror\n", FLAGS.join(" "));
    if std::fs::read_to_string(&path).is_ok_and(|old| old == text) {
        return;
    }
    let r = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, text));
    if let Err(e) = r {
        println!("note: could not write {}: {e}", path.display());
    }
}

struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Temp {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("pc-codegen-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        Temp(dir)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Compiles `c` with gcc and loads it. The temp dir lives as long as the library.
struct Aot {
    lib: Library,
    _tmp: Temp,
}

fn compile_c(cc: &str, tag: &str, c: &str) -> Aot {
    let tmp = Temp::new(tag);
    let src = tmp.0.join("model.c");
    let so = tmp.0.join("model.so");
    std::fs::write(&src, c).expect("write c");
    let out = Command::new(cc)
        .args(FLAGS)
        .arg("-Werror")
        .arg("-o")
        .arg(&so)
        .arg(&src)
        .output()
        .expect("run gcc");
    assert!(
        out.status.success(),
        "{cc} failed:\n{}\n{c}",
        String::from_utf8_lossy(&out.stderr)
    );
    // SAFETY: loads the library this test just built; its initializers are empty.
    let lib = unsafe { Library::new(&so) }.expect("load so");
    // SAFETY: pc_aot_init takes the table pointer and stores it.
    unsafe {
        let init: Symbol<unsafe extern "C" fn(*const MathTable)> =
            lib.get(b"pc_aot_init").expect("pc_aot_init");
        init(&MATH);
    }
    Aot { lib, _tmp: tmp }
}

impl Aot {
    fn call(&self, name: &str, x: &[f64], p: &[f64], v: Option<&[f64]>, out: &mut [f64]) {
        let sym = name.as_bytes();
        // SAFETY: the emitted signatures are (x, p, out) and (x, p, v, out); the caller
        // sizes every slice from the entry contract.
        unsafe {
            match v {
                None => {
                    let f: Symbol<unsafe extern "C" fn(*const f64, *const f64, *mut f64)> =
                        self.lib.get(sym).expect("symbol");
                    f(x.as_ptr(), p.as_ptr(), out.as_mut_ptr());
                }
                Some(v) => {
                    let f: Symbol<
                        unsafe extern "C" fn(*const f64, *const f64, *const f64, *mut f64),
                    > = self.lib.get(sym).expect("symbol");
                    f(x.as_ptr(), p.as_ptr(), v.as_ptr(), out.as_mut_ptr());
                }
            }
        }
    }
}

/// Distance in units in the last place; `u64::MAX` when exactly one value is NaN.
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

const EDGES: [f64; 13] = [
    0.0,
    -0.0,
    0.5,
    -1.3,
    2.0,
    1e-3,
    7.25,
    f64::NAN,
    -2.5,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1e300,
    5e-324,
];

fn edge(k: usize) -> f64 {
    EDGES[k % EDGES.len()]
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn kinds_for() -> Vec<(String, EntryKind)> {
    use Slot::{Full, P, U};
    vec![
        ("f".into(), EntryKind::F),
        ("aux".into(), EntryKind::Aux),
        ("jac_u".into(), EntryKind::JacU),
        ("jac_p".into(), EntryKind::JacP),
        ("d_u".into(), EntryKind::Dir(vec![U])),
        ("d_uu".into(), EntryKind::Dir(vec![U, U])),
        ("d_fp".into(), EntryKind::Dir(vec![Full, P])),
        ("d_upf".into(), EntryKind::Dir(vec![U, P, Full])),
        ("d_uuuu".into(), EntryKind::Dir(vec![U, U, U, U])),
    ]
}

/// Compares every entry of `src` between the AOT library and the JIT. Returns the
/// maximum ulp distance seen.
fn compare_model(cc: &str, tag: &str, src: &str) -> u64 {
    // Lowering is deterministic, so a second model gives the same node ids as the one
    // whose entries feed the emitter.
    let mut ir = build_model(src);
    let mut jit = CompiledModel::new(build_model(src));
    let (n, n_p) = (ir.n(), ir.n_p());
    let kinds = kinds_for();
    let entries: Vec<Entry> = kinds
        .iter()
        .map(|(_, k)| ir.entry(k).expect("entry"))
        .collect();
    let pairs: Vec<(String, &Entry)> = kinds
        .iter()
        .map(|(nm, _)| nm.clone())
        .zip(entries.iter())
        .collect();
    let c = emit_c(&ir, &pairs);
    let aot = compile_c(cc, tag, &c);
    let mut rng = Lcg(7);
    let mut max_ulp = 0u64;
    let mut finite = 0usize;
    for pt in 0..24usize {
        let x: Vec<f64> = (0..n)
            .map(|i| {
                if pt < 12 {
                    edge(pt + i)
                } else {
                    rng.next() * 6.0 - 3.0
                }
            })
            .collect();
        let p: Vec<f64> = (0..=n_p)
            .map(|j| {
                if pt < 12 {
                    edge(pt * 3 + j + 1)
                } else {
                    rng.next() * 3.0 + 0.1
                }
            })
            .collect();
        for (name, kind) in &kinds {
            let len = entries
                .iter()
                .zip(&kinds)
                .find(|(_, (nm, _))| nm == name)
                .map(|(e, _)| e.outputs.len())
                .unwrap_or(0);
            let v: Vec<f64> = match kind {
                EntryKind::Dir(slots) => {
                    let total: usize = slots.iter().map(|s| s.len(n, n_p)).sum();
                    (0..total)
                        .map(|k| {
                            if pt < 12 {
                                edge(pt + 2 * k + 3)
                            } else {
                                rng.next() * 2.0 - 1.0
                            }
                        })
                        .collect()
                }
                _ => Vec::new(),
            };
            let mut want = vec![f64::MAX; len];
            jit.call(kind, &x, &p, &v, &mut want).expect("jit call");
            let mut got = vec![f64::MIN; len];
            let vv = if matches!(kind, EntryKind::Dir(_)) {
                Some(v.as_slice())
            } else {
                None
            };
            aot.call(name, &x, &p, vv, &mut got);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                let d = ulp_dist(*a, *b);
                finite += usize::from(a.is_finite());
                max_ulp = max_ulp.max(if d == u64::MAX { 1 << 62 } else { d });
                // A NaN matches any NaN: IEEE leaves the sign and payload of a NaN result
                // unspecified, and gcc reorders commutative operands, so which input NaN
                // (or the default NaN) wins can differ from the JIT. The strict tests below
                // compare NaN bits.
                assert!(
                    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan() && d == 0),
                    "{tag}/{name} point {pt} out[{i}]: aot {a:e} ({:#x}) jit {b:e} ({:#x}), x={x:?} p={p:?}",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
    assert!(max_ulp <= 4, "{tag}: max ulp {max_ulp}");
    assert!(finite > 100, "{tag}: only {finite} finite values compared");
    println!("aot {cc} {tag}: max ulp {max_ulp}, {finite} finite values");
    max_ulp
}

const MODELS: [(&str, &str); 9] = [
    (
        "trig",
        "par a=0.8\n\
         x'=sin(x)+cos(y)+tan(a*x)+asin(a*x/2)+acos(a*y/3)+atan(x)\n\
         y'=sinh(x)+cosh(y)+tanh(x*y)+atan2(y,x)\naux s=sin(a*x)\ndone\n",
    ),
    (
        "expln",
        "par a=0.8, b=1.5\n\
         x'=exp(a*x)+ln(abs(y)+1)+log10(abs(x)+2)+sqrt(abs(x*y)+1)+x^y\n\
         y'=abs(x)^b+ln(x)+sqrt(y)+exp(-y)\naux s=log10(x*b)\ndone\n",
    ),
    (
        "special",
        "par a=0.8\n\
         x'=erf(x)+erfc(a*y)+lgamma(abs(x)+1.5)\ny'=erf(x*y)-lgamma(y)+erfc(x)\ndone\n",
    ),
    (
        "bessel",
        "par a=1.2\n\
         x'=besselj(2,x)+bessely(1,abs(x)+0.5)\ny'=besselj(a,y)*bessely(0,abs(y)+0.1)\ndone\n",
    ),
    (
        "piecewise",
        "par a=0.8, b=2\n\
         x'=mod(x*7,a+1)+flr(y*3)+ceil(x*2)+abs(y)+heav(x-0.3)+sign(y-0.1)\n\
         y'=min(x,y)+max(x,b)+sign(x)+heav(y)+mod(y,-b)\naux s=sign(x)\ndone\n",
    ),
    (
        "logic",
        "par a=2, b=3\n\
         x'=if(x>y)then(x*a)else(y+b)\n\
         y'=(x>0)&(y<5)+(x<=y)|(x>=3)+(not(y))+(x==y)\n\
         aux s=if(x)then(1)else(2)\naux q=(x>y)|(a>b)\ndone\n",
    ),
    (
        "table",
        "par a=2\ntable h % 5 0 2 t^2\ntable g % 3 -1 1 a*t\n\
         x'=h(x)+g(y)\ny'=h(y*x)-g(a)\ndone\n",
    ),
    (
        "intpow",
        "par a=1.1\n\
         x'=x^2+x^3+x^5+x^7+x^16+x^17+x^(-1)+x^(-3)+x^(-7)+x^(-16)\n\
         y'=y^4+y^6+y^9+a^3*y^(-2)\naux s=x^11\ndone\n",
    ),
    (
        "mixed",
        "par a=1.3, b=0.7\ninit x=0.4, y=0.3\naux s=x*y+sin(a)\n\
         x'=a*sin(x*y)-x^3+exp(-y)/(1+x^2)+max(x,b*y)\n\
         y'=b*x-y+tanh(a*y)+sqrt(1+x*x)+x^20+atan2(y,x)\ndone\n",
    ),
];

#[test]
fn aot_matches_jit_on_every_model() {
    let Some(v) = gcc_version() else {
        println!("skip: gcc not found, AOT C test not run");
        return;
    };
    record_compiler(&v);
    println!("compiler: {v}");
    println!("flags: gcc {} -Werror", FLAGS.join(" "));
    for cc in compilers() {
        for (tag, src) in MODELS {
            compare_model(cc, tag, src);
        }
    }
}

#[test]
fn math_fields_follow_the_math_table_order() {
    assert_eq!(
        std::mem::size_of::<MathTable>(),
        MATH_FIELDS.len() * std::mem::size_of::<usize>()
    );
    macro_rules! check {
        ($($i:expr => $f:ident),* $(,)?) => {
            $(
                assert_eq!(MATH_FIELDS[$i], stringify!($f));
                assert_eq!(std::mem::offset_of!(MathTable, $f), $i * std::mem::size_of::<usize>());
            )*
        };
    }
    check!(0 => sin, 1 => cos, 2 => tan, 3 => asin, 4 => acos, 5 => atan, 6 => atan2,
        7 => sinh, 8 => cosh, 9 => tanh, 10 => exp, 11 => ln, 12 => log10, 13 => sqrt,
        14 => pow, 15 => erf, 16 => erfc, 17 => lgamma, 18 => besselj, 19 => bessely,
        20 => polygamma, 21 => floor, 22 => ceil, 23 => pc_mod, 24 => table,
        25 => table_slope);
    let ir = build_model("x'=-x\ndone\n");
    let c = emit_c(&ir, &[]);
    let start = c.find("typedef struct pc_math_t").expect("struct");
    let end = c[start..].find("} pc_math_t;").expect("struct end") + start;
    let body = &c[start..end];
    let mut last = 0;
    for f in MATH_FIELDS {
        let pat = format!("(*{f})(");
        let at = body
            .find(&pat)
            .unwrap_or_else(|| panic!("field {f} missing"));
        assert!(at >= last, "field {f} out of order");
        last = at;
    }
    assert_eq!(body.matches("(*").count(), 26);
}

#[test]
fn entry_names_are_sanitized() {
    let mut ir = build_model("x'=-x\ndone\n");
    let e = ir.entry(&EntryKind::F).expect("entry");
    let c = emit_c(
        &ir,
        &[("my-f".into(), &e), ("1st".into(), &e), ("".into(), &e)],
    );
    assert!(c.contains("void my_f(const double *x"), "{c}");
    assert!(c.contains("void _1st(const double *x"), "{c}");
    assert!(c.contains("void _(const double *x"), "{c}");
}

/// Compiles `e` with every available compiler and compares each output with `eval`,
/// bit for bit (NaN sign and payload included), at the given `x` and `v` values.
fn check_against_eval(tag: &str, ir: &IrModel, e: &Entry, xs: &[f64], v: &[f64]) {
    let is_dir = matches!(e.kind, EntryKind::Dir(_));
    let c = emit_c(ir, &[("ent".into(), e)]);
    for cc in compilers() {
        let aot = compile_c(cc, tag, &c);
        for &xv in xs {
            let mut want = vec![f64::MAX; e.outputs.len()];
            eval(&ir.dag, &e.outputs, &[xv], &[0.0], v, &mut want);
            let mut got = vec![f64::MIN; e.outputs.len()];
            aot.call("ent", &[xv], &[0.0], is_dir.then_some(v), &mut got);
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{cc}/{tag} x={xv:?} out[{i}]: aot {a:e} ({:#x}) eval {b:e} ({:#x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
}

const XS: [f64; 8] = [0.0, -0.0, 0.25, 0.75, -3.0, 2.0, f64::NAN, f64::INFINITY];

/// Hand-built DAG: table data with infinities, -0.0 and a negative NaN that the lookups
/// read, a malformed table, a missing table, bit-exact constants (NaN payload, -0.0,
/// subnormal) and an out-of-range output id. Everything is compared bit for bit.
#[test]
fn awkward_constants_and_tables_match_eval() {
    let mut ir = build_model("x'=-x\ndone\n");
    let dag: &mut Dag = &mut ir.dag;
    let t_inf = dag.add_table(vec![
        3.0,
        0.0,
        1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]);
    // The second node is a NaN with the sign bit set; the lookup at x in [0.5, 1] reads it.
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
    let t_nan = dag.add_table(vec![3.0, 0.0, 1.0, -0.0, 1.0, neg_nan]);
    let t_zero = dag.add_table(vec![2.0, 0.0, 1.0, -0.0, -0.0]);
    let t_bad = dag.add_table(vec![1.0, 0.0, 1.0, 5.0]);
    let t_short = dag.add_table(vec![4.0, 0.0, 1.0, 1.0]);
    let x = dag.var(0);
    let mut outs: Vec<NodeId> = vec![dag.table(t_inf, x), dag.table_slope(t_inf, x)];
    outs.push(dag.table(t_nan, x));
    outs.push(dag.table_slope(t_nan, x));
    outs.push(dag.table(t_zero, x));
    outs.push(dag.table_slope(t_zero, x));
    outs.push(dag.table(t_bad, x));
    outs.push(dag.table(t_short, x));
    outs.push(dag.table(99, x));
    outs.push(dag.konst(-0.0));
    outs.push(dag.konst(f64::from_bits(0x7ff8_0000_dead_beef)));
    outs.push(dag.konst(f64::from_bits(1)));
    outs.push(dag.konst(f64::NEG_INFINITY));
    outs.push(dag.konst(1.0 / 3.0));
    let nx = dag.neg(x);
    let ax = dag.call1(Fn1::Abs, nx);
    outs.push(ax);
    let sg = dag.call1(Fn1::Sign, x);
    outs.push(sg);
    let mn = dag.call2(Fn2::Min, x, nx);
    outs.push(mn);
    outs.push(NodeId(u32::MAX));
    let e = Entry {
        kind: EntryKind::F,
        outputs: outs,
    };
    check_against_eval("awk", &ir, &e, &XS, &[]);
    // The negative NaN really reaches an output (guards against a vacuous check).
    let mut out = vec![0.0; e.outputs.len()];
    eval(&ir.dag, &e.outputs, &[0.75], &[0.0], &[], &mut out);
    assert_eq!(out[2].to_bits(), 0xfff8_0000_0000_0000);
}

/// A malformed or missing table lowers to NaN and its argument stays dead, so the C has
/// no unused variable (the tests compile with -Werror).
#[test]
fn bad_table_argument_is_not_emitted() {
    let mut ir = build_model("x'=-x\ndone\n");
    let dag: &mut Dag = &mut ir.dag;
    let t_bad = dag.add_table(vec![1.0, 0.0, 1.0, 5.0]);
    let x = dag.var(0);
    let s = dag.call1(Fn1::Sin, x);
    let c = dag.call1(Fn1::Cos, x);
    let outs = vec![dag.table(t_bad, s), dag.table_slope(99, c)];
    let e = Entry {
        kind: EntryKind::F,
        outputs: outs,
    };
    let text = emit_c(&ir, &[("ent".into(), &e)]);
    assert!(!text.contains("pc_math->sin"), "{text}");
    assert!(!text.contains("pc_math->cos"), "{text}");
    check_against_eval("badtab", &ir, &e, &XS, &[]);
}

/// Leaves outside the entry's arrays read as NaN, and a `Dir` leaf in a plain entry
/// does not need a `v` parameter.
#[test]
fn out_of_range_leaves_are_nan() {
    let mut ir = build_model("x'=-x\ndone\n");
    let dag: &mut Dag = &mut ir.dag;
    let x = dag.var(0);
    let outs = vec![
        x,
        dag.var(5),
        dag.param(0),
        dag.param(9),
        dag.dir(0),
        dag.dir(3),
    ];
    let plain = Entry {
        kind: EntryKind::F,
        outputs: outs.clone(),
    };
    check_against_eval("leaf_plain", &ir, &plain, &XS, &[]);
    let dir = Entry {
        kind: EntryKind::Dir(vec![Slot::U]),
        outputs: outs,
    };
    check_against_eval("leaf_dir", &ir, &dir, &XS, &[2.5]);
}
