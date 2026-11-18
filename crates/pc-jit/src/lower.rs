//! Lowers the live nodes of one entry to a Cranelift function.
//!
//! Every op copies the rule in `pc_ir::eval::eval_op` (double precision) or in the `f32`
//! tape of pc-ensemble (single precision). Nodes are lowered once each in increasing id
//! order. No `fma` is emitted.

use crate::{Float, JitError};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{self, InstBuilder, MemFlagsData, Type, types};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use pc_ir::math::table_ok;
use std::collections::{BTreeSet, HashMap};

/// The import name of the shim behind `f`, or `None` for ops lowered in line and, in
/// single precision, for the ops the `f32` tape refuses.
pub(crate) fn shim1(f: Fn1, float: Float) -> Option<&'static str> {
    if float == Float::F32 {
        return Some(match f {
            Fn1::Sin => "pc_sinf",
            Fn1::Cos => "pc_cosf",
            Fn1::Tan => "pc_tanf",
            Fn1::Asin => "pc_asinf",
            Fn1::Acos => "pc_acosf",
            Fn1::Atan => "pc_atanf",
            Fn1::Sinh => "pc_sinhf",
            Fn1::Cosh => "pc_coshf",
            Fn1::Tanh => "pc_tanhf",
            Fn1::Exp => "pc_expf",
            Fn1::Ln => "pc_lnf",
            Fn1::Log10 => "pc_log10f",
            Fn1::Floor => "pc_floorf",
            Fn1::Ceil => "pc_ceilf",
            Fn1::Erf | Fn1::Erfc | Fn1::Lgamma => return None,
            Fn1::Sqrt | Fn1::Abs | Fn1::Heav | Fn1::Sign => return None,
        });
    }
    Some(match f {
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
        Fn1::Erf => "pc_erf",
        Fn1::Erfc => "pc_erfc",
        Fn1::Lgamma => "pc_lgamma",
        Fn1::Floor => "pc_floor",
        Fn1::Ceil => "pc_ceil",
        Fn1::Sqrt | Fn1::Abs | Fn1::Heav | Fn1::Sign => return None,
    })
}

pub(crate) fn shim2(f: Fn2, float: Float) -> Option<&'static str> {
    if float == Float::F32 {
        return Some(match f {
            Fn2::Atan2 => "pc_atan2f",
            Fn2::Pow => "pc_powf",
            Fn2::Mod => "pc_modf",
            Fn2::BesselJ | Fn2::BesselY | Fn2::Polygamma => return None,
            Fn2::Min | Fn2::Max => return None,
        });
    }
    Some(match f {
        Fn2::Atan2 => "pc_atan2",
        Fn2::Pow => "pc_pow",
        Fn2::Mod => "pc_mod",
        Fn2::BesselJ => "pc_besselj",
        Fn2::BesselY => "pc_bessely",
        Fn2::Polygamma => "pc_polygamma",
        Fn2::Min | Fn2::Max => return None,
    })
}

/// Import names the live nodes call.
pub(crate) fn imports_needed(dag: &Dag, live: &[bool], float: Float) -> BTreeSet<&'static str> {
    let mut set = BTreeSet::new();
    for (i, &l) in live.iter().enumerate() {
        if !l {
            continue;
        }
        match *dag.op(NodeId(i as u32)) {
            Op::Call1(f, _) => set.extend(shim1(f, float)),
            Op::Call2(f, _, _) => set.extend(shim2(f, float)),
            // `table` refuses these in single precision.
            Op::Table(..) | Op::TableSlope(..) if float == Float::F32 => {}
            Op::Table(..) => {
                set.insert("pc_table");
            }
            Op::TableSlope(..) => {
                set.insert("pc_table_slope");
            }
            _ => {}
        }
    }
    set
}

/// Addresses of the table copies that the code reads, keyed by table id.
pub(crate) struct Tables {
    pub store: Vec<Box<[f64]>>,
    ids: HashMap<u32, Option<*const f64>>,
}

impl Tables {
    pub fn new() -> Tables {
        Tables {
            store: Vec::new(),
            ids: HashMap::new(),
        }
    }

    /// The address of the copy of table `id`, or `None` when it is missing or malformed.
    pub(crate) fn ptr(&mut self, dag: &Dag, id: u32) -> Option<*const f64> {
        if let Some(p) = self.ids.get(&id) {
            return *p;
        }
        let p = match dag.tables.get(id as usize) {
            Some(d) if table_ok(d) => {
                let copy: Box<[f64]> = d.clone().into_boxed_slice();
                let p = copy.as_ptr();
                self.store.push(copy);
                Some(p)
            }
            _ => None,
        };
        self.ids.insert(id, p);
        p
    }
}

pub(crate) struct Ctx<'a> {
    pub dag: &'a Dag,
    pub outputs: &'a [NodeId],
    pub live: &'a [bool],
    pub refs: &'a HashMap<&'static str, ir::FuncRef>,
    pub table_ptrs: &'a HashMap<u32, Option<*const f64>>,
    /// True for a `DirFn` signature (4 parameters).
    pub has_dir: bool,
    pub ptr: Type,
    pub float: Float,
    pub target: TargetFrontendConfig,
}

/// The constant `v` in the float type; single precision rounds it as `v as f32` does.
fn fconst(b: &mut FunctionBuilder, float: Float, v: f64) -> ir::Value {
    match float {
        Float::F64 => b.ins().f64const(v),
        Float::F32 => b.ins().f32const(v as f32),
    }
}

/// The NaN of the interpreter in each precision: `f64::NAN` or `f32::NAN`.
fn fnan(b: &mut FunctionBuilder, float: Float) -> ir::Value {
    match float {
        Float::F64 => b.ins().f64const(f64::NAN),
        Float::F32 => b.ins().f32const(f32::NAN),
    }
}

/// log2 of the element size, so element `k` sits at byte `k << shift`.
fn stride_shift(float: Float) -> u32 {
    match float {
        Float::F64 => 3,
        Float::F32 => 2,
    }
}

// x, p, v and out are aligned arrays of the float type that the caller guarantees, so the
// accesses cannot trap.
fn load(b: &mut FunctionBuilder, float: Float, base: ir::Value, k: u32) -> ir::Value {
    let ty = float.ty();
    let off = i64::from(k) << stride_shift(float);
    match i32::try_from(off) {
        Ok(o) => b.ins().load(ty, MemFlagsData::trusted(), base, o),
        Err(_) => {
            let a = b.ins().iadd_imm_s(base, off);
            b.ins().load(ty, MemFlagsData::trusted(), a, 0)
        }
    }
}

fn store(b: &mut FunctionBuilder, float: Float, v: ir::Value, base: ir::Value, k: usize) {
    let off = (k as i64) << stride_shift(float);
    match i32::try_from(off) {
        Ok(o) => {
            b.ins().store(MemFlagsData::trusted(), v, base, o);
        }
        Err(_) => {
            let a = b.ins().iadd_imm_s(base, off);
            b.ins().store(MemFlagsData::trusted(), v, a, 0);
        }
    }
}

/// 1.0 when the `i8` condition `c` is set, else +0.0.
///
/// Single precision builds the bits without a branch: `bmask` turns the condition into all
/// ones or all zeros, and `band` with the bits of 1.0 leaves 1.0 or +0.0, the same two
/// values the select gives.
fn b2f(b: &mut FunctionBuilder, float: Float, c: ir::Value) -> ir::Value {
    match float {
        Float::F64 => {
            let one = b.ins().f64const(1.0);
            let zero = b.ins().f64const(0.0);
            b.ins().select(c, one, zero)
        }
        Float::F32 => {
            let m = b.ins().bmask(types::I32, c);
            let k = b.ins().iconst(types::I32, i64::from(1.0f32.to_bits()));
            let bits = b.ins().band(m, k);
            b.ins().bitcast(types::F32, MemFlagsData::new(), bits)
        }
    }
}

/// The bits of `x` with the sign bit shifted out: zero exactly for +0.0 and -0.0.
fn magnitude_bits(b: &mut FunctionBuilder, x: ir::Value) -> ir::Value {
    let bits = b.ins().bitcast(types::I32, MemFlagsData::new(), x);
    b.ins().ishl_imm_u(bits, 1)
}

/// `x != 0.0` as Rust evaluates it: true for NaN (Cranelift `ne` is unordered or not equal).
/// Single precision tests the magnitude bits, which are nonzero for NaN too.
fn nonzero(b: &mut FunctionBuilder, float: Float, x: ir::Value) -> ir::Value {
    match float {
        Float::F64 => {
            let zero = b.ins().f64const(0.0);
            b.ins().fcmp(FloatCC::NotEqual, x, zero)
        }
        Float::F32 => {
            let m = magnitude_bits(b, x);
            b.ins().icmp_imm_u(IntCC::NotEqual, m, 0)
        }
    }
}

/// `x == 0.0`: true for either zero, false for NaN.
fn is_zero(b: &mut FunctionBuilder, float: Float, x: ir::Value) -> ir::Value {
    match float {
        Float::F64 => {
            let zero = b.ins().f64const(0.0);
            b.ins().fcmp(FloatCC::Equal, x, zero)
        }
        Float::F32 => {
            let m = magnitude_bits(b, x);
            b.ins().icmp_imm_u(IntCC::Equal, m, 0)
        }
    }
}

/// `+0.0` when `x == 0.0` (either zero), else `r`: the rule of `Op::ZMul` and `Op::ZDiv`,
/// with `r` the product or quotient, which is always computed.
///
/// A float `select` is a compare and a jump on x64: Cranelift lowers it to a conditional
/// move that it emits as a branch. The jump was kept on purpose. Integer-mask forms (the
/// magnitude bits through `bmask` and `band`, and a vector `fcmp` with `band`) took 145 ns
/// and 120 ns against 102 ns for the select on the fourth derivative of a two-variable
/// model, because the compare and the `band` sit on the dependency chain and a Jacobian
/// sees the same zero pattern on every call, so the jump predicts.
fn zero_guard(b: &mut FunctionBuilder, float: Float, x: ir::Value, r: ir::Value) -> ir::Value {
    let zero = fconst(b, float, 0.0);
    let x_is_zero = is_zero(b, float, x);
    b.ins().select(x_is_zero, zero, r)
}

/// The order documented on `Op::PowI`.
fn powi(b: &mut FunctionBuilder, float: Float, x: ir::Value, n: i32) -> ir::Value {
    let mut m = n.unsigned_abs();
    if m == 0 {
        return fconst(b, float, 1.0);
    }
    let mut base = x;
    let mut acc: Option<ir::Value> = None;
    loop {
        if m & 1 == 1 {
            acc = Some(match acc {
                None => base,
                Some(a) => b.ins().fmul(a, base),
            });
        }
        m >>= 1;
        if m == 0 {
            break;
        }
        base = b.ins().fmul(base, base);
    }
    let r = acc.unwrap_or(base);
    if n < 0 {
        let one = fconst(b, float, 1.0);
        b.ins().fdiv(one, r)
    } else {
        r
    }
}

/// Builds the body of one entry into `func`, whose signature is already set.
pub(crate) fn build(
    func: &mut ir::Function,
    fbc: &mut FunctionBuilderContext,
    c: Ctx,
) -> Result<(), JitError> {
    let mut b = FunctionBuilder::new(func, fbc);
    let block = b.create_block();
    b.append_block_params_for_function_params(block);
    b.switch_to_block(block);
    b.seal_block(block);
    let params: Vec<ir::Value> = b.block_params(block).to_vec();
    let at = |i: usize| params.get(i).copied();
    let (x, p) = (at(0), at(1));
    let (v, out) = if c.has_dir {
        (at(2), at(3))
    } else {
        (None, at(2))
    };
    let fl = c.float;
    let mut vals: Vec<Option<ir::Value>> = vec![None; c.dag.len()];

    for i in 0..c.dag.len() {
        if !c.live.get(i).copied().unwrap_or(false) {
            continue;
        }
        let arg = |b: &mut FunctionBuilder, vals: &[Option<ir::Value>], id: NodeId| match vals
            .get(id.0 as usize)
            .copied()
            .flatten()
        {
            Some(v) => v,
            None => fnan(b, fl),
        };
        let leaf = |b: &mut FunctionBuilder, base: Option<ir::Value>, k: u32| match base {
            Some(base) => load(b, fl, base, k),
            None => fnan(b, fl),
        };
        let r = match *c.dag.op(NodeId(i as u32)) {
            Op::Const(bits) => fconst(&mut b, fl, f64::from_bits(bits)),
            Op::Var(k) => leaf(&mut b, x, k),
            Op::Param(k) => leaf(&mut b, p, k),
            Op::Dir(k) => leaf(&mut b, v, k),
            Op::Neg(a) => {
                let a = arg(&mut b, &vals, a);
                b.ins().fneg(a)
            }
            Op::Add(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                b.ins().fadd(l, r)
            }
            Op::Sub(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                b.ins().fsub(l, r)
            }
            Op::Mul(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                b.ins().fmul(l, r)
            }
            // `t == 0` gives +0 whatever `p` is, else `t * p`; see `Op::ZMul`.
            Op::ZMul(t, p) => {
                let (t, p) = (arg(&mut b, &vals, t), arg(&mut b, &vals, p));
                let prod = b.ins().fmul(t, p);
                zero_guard(&mut b, fl, t, prod)
            }
            Op::Div(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                b.ins().fdiv(l, r)
            }
            // `n == 0` gives +0 whatever `d` is, else `n / d`; see `Op::ZDiv`.
            Op::ZDiv(n, d) => {
                let (n, d) = (arg(&mut b, &vals, n), arg(&mut b, &vals, d));
                let quot = b.ins().fdiv(n, d);
                zero_guard(&mut b, fl, n, quot)
            }
            Op::PowI(a, n) => {
                let a = arg(&mut b, &vals, a);
                powi(&mut b, fl, a, n)
            }
            Op::Call1(f, a) => {
                let a = arg(&mut b, &vals, a);
                call1(&mut b, &c, f, a)?
            }
            Op::Call2(f, l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                call2(&mut b, &c, f, l, r)?
            }
            Op::Cmp(k, l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                let cc = match k {
                    Cmp::Lt => FloatCC::LessThan,
                    Cmp::Le => FloatCC::LessThanOrEqual,
                    Cmp::Gt => FloatCC::GreaterThan,
                    Cmp::Ge => FloatCC::GreaterThanOrEqual,
                    Cmp::Eq => FloatCC::Equal,
                };
                let t = b.ins().fcmp(cc, l, r);
                b2f(&mut b, fl, t)
            }
            Op::And(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                let (l, r) = (nonzero(&mut b, fl, l), nonzero(&mut b, fl, r));
                let t = b.ins().band(l, r);
                b2f(&mut b, fl, t)
            }
            Op::Or(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                let (l, r) = (nonzero(&mut b, fl, l), nonzero(&mut b, fl, r));
                let t = b.ins().bor(l, r);
                b2f(&mut b, fl, t)
            }
            Op::Not(a) => {
                let a = arg(&mut b, &vals, a);
                let t = is_zero(&mut b, fl, a);
                b2f(&mut b, fl, t)
            }
            Op::Select(k, l, r) => {
                let (k, l, r) = (
                    arg(&mut b, &vals, k),
                    arg(&mut b, &vals, l),
                    arg(&mut b, &vals, r),
                );
                let t = nonzero(&mut b, fl, k);
                b.ins().select(t, l, r)
            }
            Op::Table(id, a) => {
                let a = arg(&mut b, &vals, a);
                table(&mut b, &c, "pc_table", id, a)?
            }
            Op::TableSlope(id, a) => {
                let a = arg(&mut b, &vals, a);
                table(&mut b, &c, "pc_table_slope", id, a)?
            }
        };
        if let Some(slot) = vals.get_mut(i) {
            *slot = Some(r);
        }
    }

    if let Some(out) = out {
        for (k, o) in c.outputs.iter().enumerate() {
            let r = match vals.get(o.0 as usize).copied().flatten() {
                Some(v) => v,
                None => fnan(&mut b, fl),
            };
            store(&mut b, fl, r, out, k);
        }
    }
    b.ins().return_(&[]);
    b.finalize(c.target);
    Ok(())
}

fn call_shim(
    b: &mut FunctionBuilder,
    c: &Ctx,
    name: &str,
    args: &[ir::Value],
) -> Result<ir::Value, JitError> {
    let Some(&f) = c.refs.get(name) else {
        return Err(JitError::MissingImport(name.to_string()));
    };
    let call = b.ins().call(f, args);
    match b.inst_results(call).first() {
        Some(&r) => Ok(r),
        None => Err(JitError::Cranelift(format!("{name} returned no value"))),
    }
}

fn call1(b: &mut FunctionBuilder, c: &Ctx, f: Fn1, x: ir::Value) -> Result<ir::Value, JitError> {
    let fl = c.float;
    Ok(match f {
        Fn1::Sqrt => b.ins().sqrt(x),
        Fn1::Abs => b.ins().fabs(x),
        // 1 when x >= 0, else 0 (NaN gives 0).
        Fn1::Heav => {
            let zero = fconst(b, fl, 0.0);
            let t = b.ins().fcmp(FloatCC::GreaterThanOrEqual, x, zero);
            b2f(b, fl, t)
        }
        // 1, -1, +0.0 for either zero, and the canonical NaN for NaN, as math::sign.
        Fn1::Sign => {
            let zero = fconst(b, fl, 0.0);
            let one = fconst(b, fl, 1.0);
            let minus = fconst(b, fl, -1.0);
            let nan = fnan(b, fl);
            let is_zero = b.ins().fcmp(FloatCC::Equal, x, zero);
            let neg = b.ins().fcmp(FloatCC::LessThan, x, zero);
            let pos = b.ins().fcmp(FloatCC::GreaterThan, x, zero);
            let rest = b.ins().select(is_zero, zero, nan);
            let rest = b.ins().select(neg, minus, rest);
            b.ins().select(pos, one, rest)
        }
        other => match shim1(other, fl) {
            Some(name) => call_shim(b, c, name, &[x])?,
            None => return Err(JitError::MissingImport(format!("{other:?} in {fl:?}"))),
        },
    })
}

fn call2(
    b: &mut FunctionBuilder,
    c: &Ctx,
    f: Fn2,
    x: ir::Value,
    y: ir::Value,
) -> Result<ir::Value, JitError> {
    Ok(match f {
        // x < y ? x : y
        Fn2::Min => {
            let t = b.ins().fcmp(FloatCC::LessThan, x, y);
            b.ins().select(t, x, y)
        }
        // x > y ? x : y
        Fn2::Max => {
            let t = b.ins().fcmp(FloatCC::GreaterThan, x, y);
            b.ins().select(t, x, y)
        }
        other => match shim2(other, c.float) {
            Some(name) => call_shim(b, c, name, &[x, y])?,
            None => {
                return Err(JitError::MissingImport(format!(
                    "{other:?} in {:?}",
                    c.float
                )));
            }
        },
    })
}

fn table(
    b: &mut FunctionBuilder,
    c: &Ctx,
    name: &str,
    id: u32,
    x: ir::Value,
) -> Result<ir::Value, JitError> {
    // The `f32` tape refuses tables, so there is no single precision rule to follow.
    if c.float == Float::F32 {
        return Err(JitError::MissingImport(format!("{name} in F32")));
    }
    match c.table_ptrs.get(&id).copied().flatten() {
        Some(ptr) => {
            let addr = b.ins().iconst(c.ptr, ptr as usize as i64);
            call_shim(b, c, name, &[addr, x])
        }
        // A missing or malformed table reads as NaN, like the interpreter.
        None => Ok(b.ins().f64const(f64::NAN)),
    }
}
