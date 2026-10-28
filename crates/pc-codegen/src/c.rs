//! AOT C emitter.
//!
//! [`emit_c`] turns DAG entries into one self-contained C99 file that includes only
//! `<stdint.h>` and calls no libm function. Every transcendental goes through the
//! `pc_math_t` function table that the host passes to `pc_aot_init`, so the file gives the
//! same bits as `pc_ir::eval::eval` and the JIT when the host passes `pc_ir::math::MATH`.
//! Build it with `-ffp-contract=off`; a fused multiply-add would change the rounding.
//!
//! Constants are written from their bit patterns through `pc_bits`, because decimal text
//! cannot spell NaN payloads or tell +0.0 from -0.0 in every compiler. Table data is
//! written as decimal literals (static initializers cannot call `pc_bits`); an infinite
//! entry becomes `PC_INF` and a NaN entry `PC_NAN`, which loses the NaN payload only.
//! A table that the interpreter treats as malformed lowers to a NaN constant, as in the JIT.

use pc_ir::Entry;
use pc_ir::EntryKind;
use pc_ir::IrModel;
use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use std::fmt::Write;

/// One field of `pc_math_t`: its name and the C parameter list.
const FIELDS: [(&str, &str); 26] = [
    ("sin", "double"),
    ("cos", "double"),
    ("tan", "double"),
    ("asin", "double"),
    ("acos", "double"),
    ("atan", "double"),
    ("atan2", "double, double"),
    ("sinh", "double"),
    ("cosh", "double"),
    ("tanh", "double"),
    ("exp", "double"),
    ("ln", "double"),
    ("log10", "double"),
    ("sqrt", "double"),
    ("pow", "double, double"),
    ("erf", "double"),
    ("erfc", "double"),
    ("lgamma", "double"),
    ("besselj", "double, double"),
    ("bessely", "double, double"),
    ("polygamma", "double, double"),
    ("floor", "double"),
    ("ceil", "double"),
    ("pc_mod", "double, double"),
    ("table", "const double *, double"),
    ("table_slope", "const double *, double"),
];

/// The field names of `pc_math_t` in order; the same order as `pc_ir::math::MathTable`.
pub const MATH_FIELDS: [&str; 26] = {
    let mut names = [""; 26];
    let mut i = 0;
    while i < FIELDS.len() {
        names[i] = FIELDS[i].0;
        i += 1;
    }
    names
};

const NAN_BITS: u64 = 0x7ff8_0000_0000_0000;

const PRELUDE: &str = "\
#include <stdint.h>

#ifdef _WIN32
#define PC_EXPORT __declspec(dllexport)
#else
#define PC_EXPORT __attribute__((visibility(\"default\")))
#endif

static inline double pc_bits(uint64_t b) {
    union { uint64_t u; double d; } c;
    c.u = b;
    return c.d;
}

static inline double pc_fabs(double x) {
    union { uint64_t u; double d; } c;
    c.d = x;
    c.u &= 0x7fffffffffffffffULL;
    return c.d;
}

static inline double pc_heav(double x) { return x >= 0.0 ? 1.0 : 0.0; }

static inline double pc_sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    if (x == 0.0) return 0.0;
    return pc_bits(0x7ff8000000000000ULL);
}

static inline double pc_min(double x, double y) { return x < y ? x : y; }

static inline double pc_max(double x, double y) { return x > y ? x : y; }

";

/// A C identifier for `name`: other characters become `_`, and a leading digit (or an
/// empty name) gets a `_` in front. The caller keeps names distinct and away from C
/// keywords and from the `pc_` and `t<digits>` names this file uses.
fn c_ident(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    s
}

/// `pc_bits(...)` for a constant, with the decimal value in a comment.
fn konst(bits: u64) -> String {
    let x = f64::from_bits(bits);
    format!("pc_bits(0x{bits:016x}ULL) /* {x:e} */")
}

/// A decimal literal for a table entry.
fn table_lit(x: f64) -> String {
    if x.is_nan() {
        "PC_NAN".to_string()
    } else if x == f64::INFINITY {
        "PC_INF".to_string()
    } else if x == f64::NEG_INFINITY {
        "(-PC_INF)".to_string()
    } else {
        format!("{x:e}")
    }
}

/// A table the shim can read, as in the JIT: valid header and enough data.
fn table_ok(data: &[f64]) -> bool {
    match data.first() {
        Some(&n) if n.is_finite() && (2.0..=2147483647.0).contains(&n) => {
            data.len() >= 3 + n as usize
        }
        _ => false,
    }
}

fn children(op: &Op, f: &mut dyn FnMut(NodeId)) {
    match *op {
        Op::Const(_) | Op::Var(_) | Op::Param(_) | Op::Dir(_) => {}
        Op::Neg(a)
        | Op::PowI(a, _)
        | Op::Call1(_, a)
        | Op::Not(a)
        | Op::Table(_, a)
        | Op::TableSlope(_, a) => f(a),
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::Call2(_, a, b)
        | Op::Cmp(_, a, b)
        | Op::And(a, b)
        | Op::Or(a, b) => {
            f(a);
            f(b);
        }
        Op::Select(c, a, b) => {
            f(c);
            f(a);
            f(b);
        }
    }
}

/// The nodes reachable from `outputs`, as `eval` marks them.
fn live_set(dag: &Dag, outputs: &[NodeId]) -> Vec<bool> {
    let mut live = vec![false; dag.len()];
    for o in outputs {
        if let Some(l) = live.get_mut(o.0 as usize) {
            *l = true;
        }
    }
    for i in (0..dag.len()).rev() {
        if live[i] {
            children(dag.op(NodeId(i as u32)), &mut |c| {
                if let Some(l) = live.get_mut(c.0 as usize) {
                    *l = true;
                }
            });
        }
    }
    live
}

fn math1(f: Fn1) -> &'static str {
    match f {
        Fn1::Sin => "pc_math->sin",
        Fn1::Cos => "pc_math->cos",
        Fn1::Tan => "pc_math->tan",
        Fn1::Asin => "pc_math->asin",
        Fn1::Acos => "pc_math->acos",
        Fn1::Atan => "pc_math->atan",
        Fn1::Sinh => "pc_math->sinh",
        Fn1::Cosh => "pc_math->cosh",
        Fn1::Tanh => "pc_math->tanh",
        Fn1::Exp => "pc_math->exp",
        Fn1::Ln => "pc_math->ln",
        Fn1::Log10 => "pc_math->log10",
        Fn1::Sqrt => "pc_math->sqrt",
        Fn1::Abs => "pc_fabs",
        Fn1::Erf => "pc_math->erf",
        Fn1::Erfc => "pc_math->erfc",
        Fn1::Lgamma => "pc_math->lgamma",
        Fn1::Floor => "pc_math->floor",
        Fn1::Ceil => "pc_math->ceil",
        Fn1::Heav => "pc_heav",
        Fn1::Sign => "pc_sign",
    }
}

fn math2(f: Fn2) -> &'static str {
    match f {
        Fn2::Atan2 => "pc_math->atan2",
        Fn2::Pow => "pc_math->pow",
        Fn2::Mod => "pc_math->pc_mod",
        Fn2::BesselJ => "pc_math->besselj",
        Fn2::BesselY => "pc_math->bessely",
        Fn2::Polygamma => "pc_math->polygamma",
        Fn2::Min => "pc_min",
        Fn2::Max => "pc_max",
    }
}

fn var(id: NodeId) -> String {
    format!("t{}", id.0)
}

/// Statements for `PowI`, in the multiplication order documented on `Op::PowI`.
fn powi(out: &mut String, id: NodeId, a: NodeId, n: i32) {
    let mut m = n.unsigned_abs();
    let mut base = var(a);
    let mut acc: Option<String> = None;
    let mut k = 0;
    let mut step = |out: &mut String, expr: String| {
        k += 1;
        let name = format!("t{}_{k}", id.0);
        let _ = writeln!(out, "    const double {name} = {expr};");
        name
    };
    loop {
        if m == 0 {
            break;
        }
        if m & 1 == 1 {
            acc = Some(match acc {
                None => base.clone(),
                Some(a) => step(out, format!("{a} * {base}")),
            });
        }
        m >>= 1;
        if m == 0 {
            break;
        }
        base = step(out, format!("{base} * {base}"));
    }
    let r = acc.unwrap_or_else(|| "1.0".to_string());
    let expr = if n < 0 { format!("1.0 / {r}") } else { r };
    let _ = writeln!(out, "    const double {} = {expr};", var(id));
}

fn node(out: &mut String, dag: &Dag, id: NodeId, tables_ok: &dyn Fn(u32) -> bool) {
    let t = var(id);
    let expr = match *dag.op(id) {
        Op::Const(b) => konst(b),
        Op::Var(i) => format!("x[{i}]"),
        Op::Param(j) => format!("p[{j}]"),
        Op::Dir(k) => format!("v[{k}]"),
        Op::Neg(a) => format!("-{}", var(a)),
        Op::Add(a, b) => format!("{} + {}", var(a), var(b)),
        Op::Sub(a, b) => format!("{} - {}", var(a), var(b)),
        Op::Mul(a, b) => format!("{} * {}", var(a), var(b)),
        Op::Div(a, b) => format!("{} / {}", var(a), var(b)),
        Op::PowI(a, n) => {
            powi(out, id, a, n);
            return;
        }
        Op::Call1(f, a) => format!("{}({})", math1(f), var(a)),
        Op::Call2(f, a, b) => format!("{}({}, {})", math2(f), var(a), var(b)),
        Op::Cmp(c, a, b) => {
            let o = match c {
                Cmp::Lt => "<",
                Cmp::Le => "<=",
                Cmp::Gt => ">",
                Cmp::Ge => ">=",
                Cmp::Eq => "==",
            };
            format!("({} {o} {}) ? 1.0 : 0.0", var(a), var(b))
        }
        Op::And(a, b) => format!("({} != 0.0 && {} != 0.0) ? 1.0 : 0.0", var(a), var(b)),
        Op::Or(a, b) => format!("({} != 0.0 || {} != 0.0) ? 1.0 : 0.0", var(a), var(b)),
        Op::Not(a) => format!("({} == 0.0) ? 1.0 : 0.0", var(a)),
        Op::Select(c, a, b) => format!("({} != 0.0) ? {} : {}", var(c), var(a), var(b)),
        Op::Table(tid, a) if tables_ok(tid) => {
            format!("pc_math->table(pc_table_{tid}, {})", var(a))
        }
        Op::TableSlope(tid, a) if tables_ok(tid) => {
            format!("pc_math->table_slope(pc_table_{tid}, {})", var(a))
        }
        Op::Table(..) | Op::TableSlope(..) => konst(NAN_BITS),
    };
    let _ = writeln!(out, "    const double {t} = {expr};");
}

/// Emits a C99 file with one exported function per `(name, entry)` pair.
///
/// Plain entries (`F`, `Aux`, `JacU`, `JacP`) get `void name(const double *x, const double
/// *p, double *out)`; `Dir` entries get `void name(const double *x, const double *p, const
/// double *v, double *out)`. The array lengths are the contract in `pc_ir::entry`. The
/// host must call `pc_aot_init` with a `pc_math_t` table before any entry. Names are
/// sanitized with c_ident rules: other characters become `_` and a leading digit gets
/// a `_` prefix.
pub fn emit_c(ir: &IrModel, entries: &[(String, &Entry)]) -> String {
    let dag = &ir.dag;
    let lives: Vec<Vec<bool>> = entries
        .iter()
        .map(|(_, e)| live_set(dag, &e.outputs))
        .collect();
    // Tables that some live node reads and that the shim can use.
    let mut used = vec![false; dag.tables.len()];
    for live in &lives {
        for (i, l) in live.iter().enumerate() {
            if !*l {
                continue;
            }
            if let Op::Table(t, _) | Op::TableSlope(t, _) = *dag.op(NodeId(i as u32))
                && let Some(u) = used.get_mut(t as usize)
            {
                *u = true;
            }
        }
    }
    let tables_ok = |t: u32| {
        dag.tables
            .get(t as usize)
            .is_some_and(|d| table_ok(d) && used.get(t as usize).copied().unwrap_or(false))
    };

    let mut s = String::from(PRELUDE);
    s.push_str("typedef struct pc_math_t {\n");
    for (name, args) in FIELDS {
        let _ = writeln!(s, "    double (*{name})({args});");
    }
    s.push_str("} pc_math_t;\n\nstatic const pc_math_t *pc_math;\n\n");
    s.push_str("PC_EXPORT void pc_aot_init(const pc_math_t *m) { pc_math = m; }\n\n");

    let mut need_special = false;
    let mut arrays = String::new();
    for (id, data) in dag.tables.iter().enumerate() {
        if !tables_ok(id as u32) {
            continue;
        }
        let lits: Vec<String> = data.iter().map(|&x| table_lit(x)).collect();
        need_special |= data.iter().any(|x| !x.is_finite());
        let _ = writeln!(
            arrays,
            "static const double pc_table_{id}[] = {{\n    {}\n}};\n",
            lits.join(", ")
        );
    }
    if need_special {
        s.push_str("#define PC_INF (1e308 * 10.0)\n#define PC_NAN (PC_INF - PC_INF)\n\n");
    }
    s.push_str(&arrays);

    for ((name, e), live) in entries.iter().zip(&lives) {
        let name = c_ident(name);
        let _ = write!(
            s,
            "PC_EXPORT void {name}(const double *x, const double *p, "
        );
        let is_dir = matches!(e.kind, EntryKind::Dir(_));
        if is_dir {
            s.push_str("const double *v, ");
        }
        s.push_str("double *out) {\n    (void)x;\n    (void)p;\n");
        if is_dir {
            s.push_str("    (void)v;\n");
        }
        s.push_str("    (void)out;\n");
        for (i, l) in live.iter().enumerate() {
            if *l {
                node(&mut s, dag, NodeId(i as u32), &tables_ok);
            }
        }
        for (k, o) in e.outputs.iter().enumerate() {
            if (o.0 as usize) < dag.len() {
                let _ = writeln!(s, "    out[{k}] = {};", var(*o));
            } else {
                let _ = writeln!(s, "    out[{k}] = {};", konst(NAN_BITS));
            }
        }
        s.push_str("}\n\n");
    }
    s
}
