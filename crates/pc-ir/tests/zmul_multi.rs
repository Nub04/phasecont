//! Multi-slot directional derivatives with a zero tangent component and a non-finite factor.
//!
//! The interpreter nests one `Dual` per slot, and a `Dual` product, quotient or call skips a
//! tangent that is zero in every component of the inner levels (`Scalar::is_zero`). The
//! subset expansion in `multi.rs` guards a product term on one component of the tangent.
//! These tests compare the two on right-hand sides where the tangent of a factor has a
//! zero first-order component and a nonzero mixed one (`x*x` or `q*x` at `x = 0`), and the
//! other factor is infinite (`1/max(a-x-10,0)` where the clamp is 0).
//!
//! The comparison is element by element: NaN equals NaN, +0 equals -0, an infinity equals
//! the same infinity, and two finite values agree within 1e-9 relative plus 1e-12 absolute.
//! The two sides sum the terms in a different order, so finite values can differ in the last
//! bits (the zt2 report measured up to 40 ulp on the nine corpus models), and a mathematical
//! zero that comes from cancelling terms of order 1 is a few 1e-16 on one side and exactly 0
//! on the other. The absolute floor is for that case. A zero against a nonzero beyond the
//! floor, a NaN against a number or an infinity against a finite value is a divergence
//! whatever its size.

use pc_ir::eval::eval_entry;
use pc_ir::{EntryKind, Fallback, IrModel, Slot, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn build(src: &str) -> (IrModel, Fallback) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    let ir = match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    };
    let fb = Fallback::new(&l.expanded, &l.info, &ir);
    (ir, fb)
}

/// NaN equals NaN, +0 equals -0 (`==`), and finite values agree within 1e-9 relative plus
/// 1e-12 absolute.
fn agree(dag: f64, fallback: f64) -> bool {
    if dag.is_nan() || fallback.is_nan() {
        return dag.is_nan() && fallback.is_nan();
    }
    if dag == fallback {
        return true;
    }
    if !dag.is_finite() || !fallback.is_finite() {
        return false;
    }
    (dag - fallback).abs() <= 1e-9 * dag.abs().max(fallback.abs()) + 1e-12
}

/// `m` is 0 where `a - x - 10 <= 0`, so `1/m` is infinite there.
const M: &str = "max(a-x-10,0)";

/// Tangent carriers: each has a zero first-order component at `x = 0` (or `y = 0`) and a
/// nonzero mixed second-order one. The value stays above 0, so the model value is finite
/// inside `exp`.
const TANGENTS: [&str; 5] = ["(2+x*x)", "(2+q*x)", "(2+x*y)", "(2+x*y*q)", "(3+q*x*x)"];

/// Right-hand-side shapes. `{T}` is a tangent carrier from [`TANGENTS`] and `{M}` the clamp.
/// Every shape puts the infinite factor where the model value stays finite (inside `exp`,
/// or in a `tanh`). None of them takes a power, `cosh` or `exp` of the infinite factor
/// itself, whose partial is infinite (the zt2 report lists that as a separate open item).
const SHAPES: [(&str, &str); 8] = [
    ("tangent carrier times a reciprocal", "exp(-{T}*(1/{M}))"),
    ("reciprocal times a tangent carrier", "exp(-(1/{M})*{T})"),
    ("tangent carrier over the clamp", "exp(-{T}/{M})"),
    ("tangent carrier times a log", "exp({T}*ln({M}))"),
    ("tanh of a quotient", "tanh({T}/{M})"),
    (
        "carrier, reciprocal and a second factor",
        "exp(-{T}*(1/{M})*(1+y*y))",
    ),
    ("carrier times a negative power", "exp(-{T}*{M}^(-1))"),
    (
        "carrier times a reciprocal plus a quotient",
        "exp(-{T}*(1/{M})-{T}/({M}+0))",
    ),
];

/// Slot lists of order 2 and 3 over U and P, and a few with `Full`.
fn slot_lists() -> Vec<Vec<Slot>> {
    use Slot::{Full, P, U};
    vec![
        vec![U, U],
        vec![U, P],
        vec![P, U],
        vec![P, P],
        vec![U, U, U],
        vec![U, U, P],
        vec![U, P, U],
        vec![P, U, U],
        vec![U, P, P],
        vec![P, U, P],
        vec![P, P, U],
        vec![P, P, P],
        vec![Full, U],
        vec![U, Full],
        vec![Full, P, U],
        vec![P, U, Full],
    ]
}

/// States x and y, then the parameters a and q and t. The clamp `max(a-x-10,0)` is 0 at
/// the first six points (the pole), and positive at the last two.
const POINTS: [([f64; 2], [f64; 3]); 8] = [
    ([0.0, 0.0], [1.0, 0.5, 0.0]),
    ([-0.0, 0.7], [1.0, 0.5, 0.0]),
    ([0.0, -0.4], [1.0, -1.5, 0.0]),
    ([0.3, 0.2], [1.0, 0.5, 0.0]),
    ([-1.5, 0.0], [8.0, 0.5, 0.0]),
    ([0.0, 0.0], [10.0, 0.5, 0.0]),
    ([0.0, 0.5], [11.5, 0.5, 0.0]),
    ([-0.4, 0.3], [12.0, -0.5, 0.0]),
];

const NONZERO: [f64; 12] = [
    1.0, -0.5, 2.0, 0.25, -1.5, 0.75, 1.25, -2.0, 0.5, -0.75, 1.5, -0.25,
];

/// Directions for a packed vector of length `len`. The first `2^len` zero patterns when
/// `len <= 6`, else 64 patterns from a fixed generator, one of them all zero but the last.
fn directions(len: usize) -> Vec<Vec<f64>> {
    let masks: Vec<u32> = if len <= 6 {
        (0..1u32 << len).collect()
    } else {
        let mut s = 0x9E37_79B9_7F4A_7C15_u64;
        (0..64)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as u32) & ((1u32 << len) - 1)
            })
            .collect()
    };
    masks
        .into_iter()
        .map(|mask| {
            (0..len)
                .map(|i| if mask >> i & 1 == 1 { NONZERO[i] } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Compares every output of the entry for `slots` on the points and directions above.
/// Returns the number of values compared and the divergences.
fn compare(
    rhs_x: &str,
    slots: &[Slot],
    points: &[([f64; 2], [f64; 3])],
    out: &mut Vec<String>,
) -> usize {
    let src = format!("par a=1, q=0.5\ninit x=0, y=0\nx'=-x+{rhs_x}\ny'=-y+x\ndone\n");
    let (mut ir, fb) = build(&src);
    let (n, n_p) = (ir.n(), ir.n_p());
    let kind = EntryKind::Dir(slots.to_vec());
    let entry = ir.entry(&kind).expect("entry");
    let len: usize = slots.iter().map(|s| s.len(n, n_p)).sum();
    let dirs = directions(len);
    let mut compared = 0;
    for (x, p) in points {
        for v in &dirs {
            let mut got = vec![f64::MAX; entry.outputs.len()];
            eval_entry(&ir.dag, &entry, x, p, v, &mut got);
            let mut want = vec![f64::MAX; entry.outputs.len()];
            fb.eval(&kind, x, p, v, &mut want);
            for i in 0..got.len() {
                compared += 1;
                if !agree(got[i], want[i]) {
                    out.push(format!(
                        "x'=-x+{rhs_x} {slots:?} x={x:?} p={p:?} v={v:?} out {i}: \
                         dag {:?} ({:016x}), fallback {:?} ({:016x})",
                        got[i],
                        got[i].to_bits(),
                        want[i],
                        want[i].to_bits()
                    ));
                }
            }
        }
    }
    compared
}

fn run_table(points: &[([f64; 2], [f64; 3])]) -> (usize, Vec<String>) {
    let mut divs = Vec::new();
    let mut compared = 0;
    for (_what, shape) in SHAPES {
        for t in TANGENTS {
            let rhs = shape.replace("{T}", t).replace("{M}", M);
            for slots in slot_lists() {
                compared += compare(&rhs, &slots, points, &mut divs);
            }
        }
    }
    (compared, divs)
}

fn check(points: &[([f64; 2], [f64; 3])]) {
    let (compared, divs) = run_table(points);
    assert!(compared > 20_000, "only {compared} values compared");
    assert!(
        divs.is_empty(),
        "{} of {compared} values differ, first: {:#?}",
        divs.len(),
        divs.iter().take(10).collect::<Vec<_>>()
    );
}

/// Where the clamp is positive every factor is finite, so a zero tangent component
/// (`x*x` at `x = 0`, a parameter slot of a state-only term) only has to give 0 in both.
#[test]
fn multi_slot_derivatives_with_a_zero_tangent_component_equal_the_fallback_off_the_pole() {
    check(&POINTS[6..]);
}

/// On the pole `1/max(a-x-10,0)` is infinite. The interpreter skips a tangent that is zero
/// in every component of the inner levels. The subset expansion builds the product of the
/// per-slot components of the operand with a plain multiplication, so an infinite component
/// times a zero one gives NaN where the interpreter gives 0.
///
/// Smallest case, `x'=-x+exp(-(2+x*x)*(1/max(a-x-10,0)))`, `y'=-y+x`, entry `Dir [U, U]`,
/// `x=[0.3, 0.2]`, `p=[1.0, 0.5, 0.0]`, `v=[1, 0, 0, 0]` (slot 0 along `(1, 0)`, slot 1
/// along `(0, 0)`): output 0 is NaN in the DAG and +0 in the interpreter. The operand
/// `T = 2+x*x` has `T_0 = 0.6` and `T_1 = T_01 = 0`. Which side is right is an open owner
/// question (rule 9), kept in docs/STATUS.md as "Multi-slot derivatives on a pole".
#[test]
#[ignore = "DAG gives NaN where the interpreter gives 0, owner question in docs/STATUS.md"]
fn multi_slot_derivatives_with_a_zero_tangent_component_equal_the_fallback_on_the_pole() {
    check(&POINTS[..6]);
}
