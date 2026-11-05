//! Text shared by the ensemble kernels: a prelude per dialect, float literals that survive
//! the round trip through any C-like compiler, and the right-hand side as straight-line code.
//!
//! The same text is built for plain C, OpenCL C, CUDA C and Metal Shading Language. Every
//! construct goes through a `pc_*` macro so the integrator text on top of it is written once.
//! The C dialect is what the tests compile with the system compiler; the other three are
//! checked on a device (CUDA, OpenCL) or on a Mac (Metal).
//!
//! The right-hand side follows `Tape::eval` of `pc-ensemble` op by op, with no fused
//! multiply-add and no reordering, so the kernel and the CPU loop round the same way.

use core::fmt;
use core::fmt::Write as _;

use pc_ir::dag::{Cmp, Fn1, Fn2, Op};

/// The language a kernel is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// Plain C99. Transcendentals go through a table the host fills in.
    C,
    OpenCl,
    Cuda,
    Msl,
}

impl Dialect {
    pub fn name(self) -> &'static str {
        match self {
            Dialect::C => "c",
            Dialect::OpenCl => "opencl",
            Dialect::Cuda => "cuda",
            Dialect::Msl => "msl",
        }
    }
}

/// The scalar type of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    F32,
    F64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// The right-hand side uses a construct no kernel can run. The text names it.
    Unsupported(String),
    /// Metal Shading Language has no double precision type.
    NoDouble,
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Unsupported(m) => write!(f, "unsupported construct: {m}"),
            KernelError::NoDouble => write!(f, "Metal Shading Language has no double precision"),
        }
    }
}

impl std::error::Error for KernelError {}

/// The name of the construct of `op` that no kernel can run, if there is one.
pub fn unsupported_op(op: &Op) -> Option<&'static str> {
    match *op {
        Op::Dir(_) => Some("directional derivative input"),
        Op::Table(..) => Some("table"),
        Op::TableSlope(..) => Some("table slope"),
        Op::Call1(f, _) => match f {
            Fn1::Erf => Some("erf"),
            Fn1::Erfc => Some("erfc"),
            Fn1::Lgamma => Some("lgamma"),
            _ => None,
        },
        Op::Call2(f, ..) => match f {
            Fn2::BesselJ => Some("besselj"),
            Fn2::BesselY => Some("bessely"),
            Fn2::Polygamma => Some("polygamma"),
            _ => None,
        },
        _ => None,
    }
}

/// `v` as a literal of the scalar type that a C compiler reads back to the same value.
///
/// Finite values are hexadecimal floats, so no decimal rounding sits between the host and the
/// device. For `F32` the value is rounded to `f32` first, the way `v as f32` rounds. A
/// negative value is wrapped in parentheses. Infinities and NaN use the `PC_INF` and
/// `PC_NAN` macros of the prelude; a NaN payload is dropped.
pub fn float_literal(v: f64, s: Scalar) -> String {
    if v.is_nan() {
        return "PC_NAN".into();
    }
    let neg = v.is_sign_negative();
    let body = match s {
        Scalar::F64 => {
            if v.is_infinite() {
                "PC_INF".to_string()
            } else {
                hex_body(v.abs().to_bits(), 52, 11, 1023, "")
            }
        }
        Scalar::F32 => {
            let f = v as f32;
            if f.is_infinite() {
                "PC_INF".to_string()
            } else {
                // 23 mantissa bits are 5.75 hex digits: shifting left by one makes six.
                hex_body(u64::from(f.abs().to_bits()), 23, 8, 127, "f")
            }
        }
    };
    if neg { format!("(-{body})") } else { body }
}

/// The hexadecimal float of a non-negative finite value with the given field widths.
fn hex_body(bits: u64, mant_bits: u32, exp_bits: u32, bias: i64, suffix: &str) -> String {
    let mant = bits & ((1u64 << mant_bits) - 1);
    let exp = ((bits >> mant_bits) & ((1u64 << exp_bits) - 1)) as i64;
    if exp == 0 && mant == 0 {
        return format!("0x0p+0{suffix}");
    }
    // Align the mantissa to whole hex digits.
    let digits = mant_bits.div_ceil(4) as usize;
    let aligned = mant << (digits as u32 * 4 - mant_bits);
    let mut frac = format!("{aligned:0digits$x}");
    while frac.ends_with('0') {
        frac.pop();
    }
    let (lead, e) = if exp == 0 {
        (0, 1 - bias)
    } else {
        (1, exp - bias)
    };
    let dot = if frac.is_empty() {
        String::new()
    } else {
        format!(".{frac}")
    };
    format!("0x{lead}{dot}p{e:+}{suffix}")
}

/// The names of the 15 transcendental functions, in the order of the C dialect's table.
pub const C_TABLE: [&str; 15] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "ln", "log10",
    "atan2", "pow", "fmod",
];

/// The macros, types and helpers every kernel text starts with.
pub fn prelude(d: Dialect, s: Scalar) -> Result<String, KernelError> {
    if d == Dialect::Msl && s == Scalar::F64 {
        return Err(KernelError::NoDouble);
    }
    let f32_ = s == Scalar::F32;
    let mut t = String::new();
    let real = if f32_ { "float" } else { "double" };
    let sfx = if f32_ { "f" } else { "" };
    match d {
        Dialect::C => {
            t.push_str("#include <math.h>\n");
            let _ = writeln!(t, "typedef {real} pc_real;");
            t.push_str("typedef unsigned int pc_uint;\ntypedef unsigned long long pc_ulong;\n");
            t.push_str(
                "#define PC_FN static inline\n#define PC_GLOBAL\n#define PC_IN const\n#define PC_CONSTANT static const\n",
            );
            let _ = writeln!(t, "#define PC_INF ((pc_real)INFINITY)");
            let _ = writeln!(t, "#define PC_NAN ((pc_real)NAN)");
            let _ = writeln!(
                t,
                "#define PC_ZERO 0.0{sfx}\n#define PC_ONE 1.0{sfx}\n#define pc_isfinite(x) isfinite(x)"
            );
            for (pc, c) in [
                ("sqrt", "sqrt"),
                ("fabs", "fabs"),
                ("floor", "floor"),
                ("ceil", "ceil"),
            ] {
                let _ = writeln!(t, "#define pc_{pc} {c}{sfx}");
            }
            t.push_str("typedef struct {\n");
            for name in C_TABLE {
                if name == "atan2" || name == "pow" || name == "fmod" {
                    let _ = writeln!(t, "    pc_real (*m_{name})(pc_real, pc_real);");
                } else {
                    let _ = writeln!(t, "    pc_real (*m_{name})(pc_real);");
                }
            }
            t.push_str("} pc_tbl_t;\nstatic pc_tbl_t pc_tbl;\n");
            for name in C_TABLE {
                let _ = writeln!(t, "#define pc_{name} pc_tbl.m_{name}");
            }
            t.push_str("void pc_shim_init(const pc_tbl_t *tbl) { pc_tbl = *tbl; }\n");
        }
        Dialect::OpenCl => {
            if !f32_ {
                t.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
            }
            // A compiler may fuse a multiply and an add on its own. The ensemble kernel must
            // round every operation, so it matches the CPU backend operation for operation.
            t.push_str("#pragma OPENCL FP_CONTRACT OFF\n");
            let _ = writeln!(t, "typedef {real} pc_real;");
            t.push_str("typedef uint pc_uint;\ntypedef ulong pc_ulong;\n");
            t.push_str(
                "#define PC_FN\n#define PC_GLOBAL __global\n#define PC_IN __global const\n#define PC_CONSTANT __constant\n",
            );
            t.push_str("#define PC_INF ((pc_real)INFINITY)\n#define PC_NAN ((pc_real)NAN)\n");
            let _ = writeln!(
                t,
                "#define PC_ZERO 0.0{sfx}\n#define PC_ONE 1.0{sfx}\n#define pc_isfinite(x) isfinite(x)"
            );
            math_names(&mut t, "", false);
        }
        Dialect::Cuda => {
            let _ = writeln!(t, "typedef {real} pc_real;");
            t.push_str("typedef unsigned int pc_uint;\ntypedef unsigned long long pc_ulong;\n");
            t.push_str(
                "#define PC_FN __device__\n#define PC_GLOBAL\n#define PC_IN const\n#define PC_CONSTANT __constant__\n",
            );
            if f32_ {
                t.push_str(
                    "#define PC_INF __int_as_float(0x7f800000)\n#define PC_NAN __int_as_float(0x7fc00000)\n",
                );
            } else {
                t.push_str(
                    "#define PC_INF __longlong_as_double(0x7ff0000000000000LL)\n#define PC_NAN __longlong_as_double(0x7ff8000000000000LL)\n",
                );
            }
            let _ = writeln!(
                t,
                "#define PC_ZERO 0.0{sfx}\n#define PC_ONE 1.0{sfx}\n#define pc_isfinite(x) isfinite(x)"
            );
            math_names(&mut t, sfx, false);
        }
        Dialect::Msl => {
            t.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
            t.push_str("typedef float pc_real;\ntypedef uint pc_uint;\ntypedef ulong pc_ulong;\n");
            t.push_str(
                "#define PC_FN inline\n#define PC_GLOBAL device\n#define PC_IN device const\n#define PC_CONSTANT constant\n",
            );
            t.push_str("#define PC_INF INFINITY\n#define PC_NAN NAN\n");
            t.push_str(
                "#define PC_ZERO 0.0f\n#define PC_ONE 1.0f\n#define pc_isfinite(x) isfinite(x)\n",
            );
            math_names(&mut t, "", true);
        }
    }
    Ok(t)
}

/// `#define`s that map the `pc_*` math names to a dialect's functions. `sfx` is the `f`
/// of CUDA's single precision names. `precise` selects Metal's `metal::precise::` versions.
fn math_names(t: &mut String, sfx: &str, precise: bool) {
    let ns = if precise { "metal::precise::" } else { "" };
    for (pc, c) in [
        ("sin", "sin"),
        ("cos", "cos"),
        ("tan", "tan"),
        ("asin", "asin"),
        ("acos", "acos"),
        ("atan", "atan"),
        ("sinh", "sinh"),
        ("cosh", "cosh"),
        ("tanh", "tanh"),
        ("exp", "exp"),
        ("ln", "log"),
        ("log10", "log10"),
        ("atan2", "atan2"),
        ("pow", "pow"),
        ("fmod", "fmod"),
        ("sqrt", "sqrt"),
    ] {
        let _ = writeln!(t, "#define pc_{pc} {ns}{c}{sfx}");
    }
    // Exact in every dialect, so these are not the precise variants.
    for c in ["fabs", "floor", "ceil"] {
        let _ = writeln!(t, "#define pc_{c} {c}{sfx}");
    }
}

fn v(i: u32) -> String {
    format!("v{i}")
}

fn fn1_name(f: Fn1) -> &'static str {
    match f {
        Fn1::Sin => "pc_sin",
        Fn1::Cos => "pc_cos",
        Fn1::Tan => "pc_tan",
        Fn1::Asin => "pc_asin",
        Fn1::Acos => "pc_acos",
        Fn1::Atan => "pc_atan",
        Fn1::Sinh => "pc_sinh",
        Fn1::Cosh => "pc_cosh",
        Fn1::Tanh => "pc_tanh",
        Fn1::Exp => "pc_exp",
        Fn1::Ln => "pc_ln",
        Fn1::Log10 => "pc_log10",
        Fn1::Sqrt => "pc_sqrt",
        Fn1::Abs => "pc_fabs",
        Fn1::Floor => "pc_floor",
        Fn1::Ceil => "pc_ceil",
        Fn1::Heav | Fn1::Sign | Fn1::Erf | Fn1::Erfc | Fn1::Lgamma => "",
    }
}

/// The right-hand side `out[k] = f_k(t, x, p)` as the function `name`.
///
/// `ops` and `outputs` are a tape: each op reads the slots before it. A parameter index equal
/// to `n_p` is the time `t`. The function takes `(pc_real t, PC_IN pc_real *x, PC_IN pc_real
/// *p, PC_GLOBAL pc_real *out)`.
pub fn emit_rhs(
    ops: &[Op],
    outputs: &[u32],
    n_p: usize,
    scalar: Scalar,
    name: &str,
) -> Result<String, KernelError> {
    let mut names: Vec<&'static str> = Vec::new();
    for op in ops {
        if let Some(n) = unsupported_op(op)
            && !names.contains(&n)
        {
            names.push(n);
        }
    }
    if !names.is_empty() {
        return Err(KernelError::Unsupported(names.join(", ")));
    }
    let mut b = String::new();
    let _ = writeln!(
        b,
        "PC_FN void {name}(pc_real t, PC_IN pc_real *x, PC_IN pc_real *p, PC_GLOBAL pc_real *out) {{"
    );
    b.push_str("    (void)t;\n    (void)x;\n    (void)p;\n");
    for (i, op) in ops.iter().enumerate() {
        let i = i as u32;
        let id = v(i);
        let expr = match *op {
            Op::Const(bits) => float_literal(f64::from_bits(bits), scalar),
            Op::Var(k) => format!("x[{k}]"),
            Op::Param(j) => {
                if j as usize == n_p {
                    "t".to_string()
                } else {
                    format!("p[{j}]")
                }
            }
            Op::Neg(a) => format!("-{}", v(a.0)),
            Op::Add(a, c) => format!("{} + {}", v(a.0), v(c.0)),
            Op::Sub(a, c) => format!("{} - {}", v(a.0), v(c.0)),
            Op::Mul(a, c) => format!("{} * {}", v(a.0), v(c.0)),
            Op::Div(a, c) => format!("{} / {}", v(a.0), v(c.0)),
            Op::PowI(a, n) => {
                emit_powi(&mut b, i, &v(a.0), n);
                continue;
            }
            Op::Call1(f, a) => {
                let x = v(a.0);
                match f {
                    Fn1::Heav => format!("({x} >= PC_ZERO) ? PC_ONE : PC_ZERO"),
                    Fn1::Sign => format!(
                        "({x} > PC_ZERO) ? PC_ONE : (({x} < PC_ZERO) ? (-PC_ONE) : (({x} == PC_ZERO) ? PC_ZERO : PC_NAN))"
                    ),
                    _ => format!("{}({x})", fn1_name(f)),
                }
            }
            Op::Call2(f, a, c) => {
                let (x, y) = (v(a.0), v(c.0));
                match f {
                    Fn2::Atan2 => format!("pc_atan2({x}, {y})"),
                    Fn2::Pow => format!("pc_pow({x}, {y})"),
                    Fn2::Mod => {
                        let _ = writeln!(b, "    const pc_real {id}_r = pc_fmod({x}, {y});");
                        format!("({id}_r < PC_ZERO) ? ({id}_r + {y}) : {id}_r")
                    }
                    Fn2::Min => format!("({x} < {y}) ? {x} : {y}"),
                    Fn2::Max => format!("({x} > {y}) ? {x} : {y}"),
                    Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma => unreachable!("refused above"),
                }
            }
            Op::Cmp(c, a, d) => {
                let sym = match c {
                    Cmp::Lt => "<",
                    Cmp::Le => "<=",
                    Cmp::Gt => ">",
                    Cmp::Ge => ">=",
                    Cmp::Eq => "==",
                };
                format!("({} {sym} {}) ? PC_ONE : PC_ZERO", v(a.0), v(d.0))
            }
            Op::And(a, c) => format!(
                "(({} != PC_ZERO) && ({} != PC_ZERO)) ? PC_ONE : PC_ZERO",
                v(a.0),
                v(c.0)
            ),
            Op::Or(a, c) => format!(
                "(({} != PC_ZERO) || ({} != PC_ZERO)) ? PC_ONE : PC_ZERO",
                v(a.0),
                v(c.0)
            ),
            Op::Not(a) => format!("({} == PC_ZERO) ? PC_ONE : PC_ZERO", v(a.0)),
            Op::Select(c, a, d) => format!("({} != PC_ZERO) ? {} : {}", v(c.0), v(a.0), v(d.0)),
            Op::Dir(_) | Op::Table(..) | Op::TableSlope(..) => unreachable!("refused above"),
        };
        let _ = writeln!(b, "    const pc_real {id} = {expr};");
    }
    for (k, &o) in outputs.iter().enumerate() {
        let _ = writeln!(b, "    out[{k}] = {};", v(o));
    }
    b.push_str("}\n");
    Ok(b)
}

/// `v{i} = base^n` in the order documented on `Op::PowI`.
fn emit_powi(b: &mut String, i: u32, base0: &str, n: i32) {
    let mut m = n.unsigned_abs();
    let mut base = base0.to_string();
    let mut acc: Option<String> = None;
    let mut k = 0;
    if m == 0 {
        let _ = writeln!(b, "    const pc_real {} = PC_ONE;", v(i));
        return;
    }
    loop {
        if m & 1 == 1 {
            acc = Some(match acc {
                None => base.clone(),
                Some(a) => {
                    let nm = format!("v{i}_{k}");
                    k += 1;
                    let _ = writeln!(b, "    const pc_real {nm} = {a} * {base};");
                    nm
                }
            });
        }
        m >>= 1;
        if m == 0 {
            break;
        }
        let nb = format!("v{i}_{k}");
        k += 1;
        let _ = writeln!(b, "    const pc_real {nb} = {base} * {base};");
        base = nb;
    }
    let r = acc.unwrap_or_else(|| "PC_ONE".into());
    if n < 0 {
        let _ = writeln!(b, "    const pc_real {} = PC_ONE / {r};", v(i));
    } else {
        let _ = writeln!(b, "    const pc_real {} = {r};", v(i));
    }
}
