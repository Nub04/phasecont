//! Lowers the live nodes of one entry to a Cranelift function.
//!
//! Every op copies the rule in `pc_ir::eval::eval_op`. Nodes are lowered once each in
//! increasing id order. No `fma` is emitted.

use crate::JitError;
use cranelift_codegen::ir::condcodes::FloatCC;
use cranelift_codegen::ir::{self, InstBuilder, MemFlagsData, Type, types};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use pc_ir::dag::{Cmp, Dag, Fn1, Fn2, NodeId, Op};
use std::collections::{BTreeSet, HashMap};

/// The import name of the shim behind `f`, or `None` for ops lowered in line.
pub(crate) fn shim1(f: Fn1) -> Option<&'static str> {
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

pub(crate) fn shim2(f: Fn2) -> Option<&'static str> {
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

/// Marks the nodes reachable from `outputs`.
pub(crate) fn live_set(dag: &Dag, outputs: &[NodeId]) -> Vec<bool> {
    let n = dag.len();
    let mut live = vec![false; n];
    for o in outputs {
        if let Some(l) = live.get_mut(o.0 as usize) {
            *l = true;
        }
    }
    for i in (0..n).rev() {
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

/// Import names the live nodes call.
pub(crate) fn imports_needed(dag: &Dag, live: &[bool]) -> BTreeSet<&'static str> {
    let mut set = BTreeSet::new();
    for (i, &l) in live.iter().enumerate() {
        if !l {
            continue;
        }
        match *dag.op(NodeId(i as u32)) {
            Op::Call1(f, _) => set.extend(shim1(f)),
            Op::Call2(f, _, _) => set.extend(shim2(f)),
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

/// A table block is usable when the header is valid and the data is long enough for the
/// shim to read; the shim trusts the header. Anything else lowers to NaN, like the
/// interpreter's answer for a malformed block.
pub(crate) fn table_ok(data: &[f64]) -> bool {
    match data.first() {
        Some(&n) if n.is_finite() && (2.0..=2147483647.0).contains(&n) => {
            data.len() >= 3 + n as usize
        }
        _ => false,
    }
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
    pub target: TargetFrontendConfig,
}

// x, p, v and out are aligned f64 arrays that the caller guarantees, so the accesses
// cannot trap.
fn load(b: &mut FunctionBuilder, base: ir::Value, k: u32) -> ir::Value {
    let off = 8 * i64::from(k);
    match i32::try_from(off) {
        Ok(o) => b.ins().load(types::F64, MemFlagsData::trusted(), base, o),
        Err(_) => {
            let a = b.ins().iadd_imm_s(base, off);
            b.ins().load(types::F64, MemFlagsData::trusted(), a, 0)
        }
    }
}

fn store(b: &mut FunctionBuilder, v: ir::Value, base: ir::Value, k: usize) {
    let off = 8 * k as i64;
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

fn b2f(b: &mut FunctionBuilder, c: ir::Value) -> ir::Value {
    let one = b.ins().f64const(1.0);
    let zero = b.ins().f64const(0.0);
    b.ins().select(c, one, zero)
}

/// `x != 0.0` as Rust evaluates it: true for NaN (Cranelift `ne` is unordered or not equal).
fn nonzero(b: &mut FunctionBuilder, x: ir::Value) -> ir::Value {
    let zero = b.ins().f64const(0.0);
    b.ins().fcmp(FloatCC::NotEqual, x, zero)
}

/// The order documented on `Op::PowI`.
fn powi(b: &mut FunctionBuilder, x: ir::Value, n: i32) -> ir::Value {
    let mut m = n.unsigned_abs();
    if m == 0 {
        return b.ins().f64const(1.0);
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
        let one = b.ins().f64const(1.0);
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
    let nan = f64::NAN;
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
            None => b.ins().f64const(nan),
        };
        let leaf = |b: &mut FunctionBuilder, base: Option<ir::Value>, k: u32| match base {
            Some(base) => load(b, base, k),
            None => b.ins().f64const(nan),
        };
        let r = match *c.dag.op(NodeId(i as u32)) {
            Op::Const(bits) => b.ins().f64const(f64::from_bits(bits)),
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
            Op::Div(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                b.ins().fdiv(l, r)
            }
            Op::PowI(a, n) => {
                let a = arg(&mut b, &vals, a);
                powi(&mut b, a, n)
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
                b2f(&mut b, t)
            }
            Op::And(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                let (l, r) = (nonzero(&mut b, l), nonzero(&mut b, r));
                let t = b.ins().band(l, r);
                b2f(&mut b, t)
            }
            Op::Or(l, r) => {
                let (l, r) = (arg(&mut b, &vals, l), arg(&mut b, &vals, r));
                let (l, r) = (nonzero(&mut b, l), nonzero(&mut b, r));
                let t = b.ins().bor(l, r);
                b2f(&mut b, t)
            }
            Op::Not(a) => {
                let a = arg(&mut b, &vals, a);
                let zero = b.ins().f64const(0.0);
                let t = b.ins().fcmp(FloatCC::Equal, a, zero);
                b2f(&mut b, t)
            }
            Op::Select(k, l, r) => {
                let (k, l, r) = (
                    arg(&mut b, &vals, k),
                    arg(&mut b, &vals, l),
                    arg(&mut b, &vals, r),
                );
                let t = nonzero(&mut b, k);
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
                None => b.ins().f64const(nan),
            };
            store(&mut b, r, out, k);
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
    Ok(match f {
        Fn1::Sqrt => b.ins().sqrt(x),
        Fn1::Abs => b.ins().fabs(x),
        // 1 when x >= 0, else 0 (NaN gives 0).
        Fn1::Heav => {
            let zero = b.ins().f64const(0.0);
            let t = b.ins().fcmp(FloatCC::GreaterThanOrEqual, x, zero);
            b2f(b, t)
        }
        // 1, -1, +0.0 for either zero, and the canonical NaN for NaN, as math::sign.
        Fn1::Sign => {
            let zero = b.ins().f64const(0.0);
            let one = b.ins().f64const(1.0);
            let minus = b.ins().f64const(-1.0);
            let nan = b.ins().f64const(f64::NAN);
            let is_zero = b.ins().fcmp(FloatCC::Equal, x, zero);
            let neg = b.ins().fcmp(FloatCC::LessThan, x, zero);
            let pos = b.ins().fcmp(FloatCC::GreaterThan, x, zero);
            let rest = b.ins().select(is_zero, zero, nan);
            let rest = b.ins().select(neg, minus, rest);
            b.ins().select(pos, one, rest)
        }
        other => match shim1(other) {
            Some(name) => call_shim(b, c, name, &[x])?,
            None => return Err(JitError::MissingImport(format!("{other:?}"))),
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
        other => match shim2(other) {
            Some(name) => call_shim(b, c, name, &[x, y])?,
            None => return Err(JitError::MissingImport(format!("{other:?}"))),
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
    match c.table_ptrs.get(&id).copied().flatten() {
        Some(ptr) => {
            let addr = b.ins().iconst(c.ptr, ptr as usize as i64);
            call_shim(b, c, name, &[addr, x])
        }
        // A missing or malformed table reads as NaN, like the interpreter.
        None => Ok(b.ins().f64const(f64::NAN)),
    }
}
