//! The Bessel order cap through whole models: `x' = besselj(n, x)` and `x' = bessely(n, x)`
//! with `par n`, evaluated by the DAG `eval` and by the `Fallback` evaluator. The JIT and AOT
//! paths have the same checks in `pc-jit` and `pc-codegen`.

use pc_ir::eval::eval;
use pc_ir::{Entry, EntryKind, Fallback, IrModel, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

const ORDERS: [f64; 4] = [10000.0, -10000.0, 10001.0, -10001.0];
/// Points where `Y_10000` is finite. Below about 9000 it overflows to -inf, see
/// `y_overflows_below_the_representable_range`.
const XS: [f64; 3] = [9990.0, 10000.0, 12000.0];

fn build(src: &str) -> (IrModel, pc_ode::Loaded) {
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
    (ir, l)
}

fn model(f: &str) -> String {
    format!("par n=1\ninit x=1\nx'={f}(n,x)\ndone\n")
}

/// Evaluates the entry through the DAG `eval`; `p` is `[n, t]`.
fn dag_value(ir: &mut IrModel, kind: &EntryKind, x: f64, n: f64) -> Vec<f64> {
    let e: Entry = ir.entry(kind).expect("entry");
    let mut out = vec![f64::NAN; e.outputs.len()];
    eval(&ir.dag, &e.outputs, &[x], &[n, 0.0], &[], &mut out);
    out
}

#[test]
fn order_cap_through_dag_and_fallback() {
    for f in ["besselj", "bessely"] {
        let (mut ir, l) = build(&model(f));
        let fb = Fallback::new(&l.expanded, &l.info, &ir);
        for n in ORDERS {
            let capped = n.abs() > 10000.0;
            for x in XS {
                let dag = dag_value(&mut ir, &EntryKind::F, x, n);
                let mut fbv = [f64::NAN];
                fb.eval(&EntryKind::F, &[x], &[n, 0.0], &[], &mut fbv);
                if capped {
                    assert!(dag[0].is_nan(), "{f} dag n={n} x={x}: {}", dag[0]);
                    assert!(fbv[0].is_nan(), "{f} fallback n={n} x={x}: {}", fbv[0]);
                } else {
                    assert!(dag[0].is_finite(), "{f} dag n={n} x={x}: {}", dag[0]);
                    assert_eq!(dag[0].to_bits(), fbv[0].to_bits(), "{f} n={n} x={x}");
                }
            }
        }
    }
}

/// `Y_n(x)` for a fixed small `x` is larger than the largest double (about 10^35000 for
/// `n = 10000, x = 0.5`), so the shim returns -inf there. That is the true value outside the
/// range of `f64`, not a NaN from the cap. The cap does not change it.
#[test]
fn y_overflows_below_the_representable_range() {
    for n in [10000.0, -10000.0] {
        for x in [0.5, 3.0, 1000.0] {
            assert_eq!(pc_ir::math::bessely(n, x), f64::NEG_INFINITY, "n={n} x={x}");
        }
    }
    assert_eq!(pc_ir::math::bessely(9999.0, 0.5), f64::NEG_INFINITY);
}

/// The x-derivative of `J_n` and `Y_n` uses orders `n - 1` and `n + 1` (DLMF 10.6.1). At the cap
/// `n + 1` is out of range, so the Jacobian entry is NaN. One order below the cap it is finite.
/// The order has no tangent, so `d/dn` is 0 wherever the value is finite.
#[test]
fn x_derivative_at_the_cap_is_nan() {
    for f in ["besselj", "bessely"] {
        let (mut ir, _) = build(&model(f));
        for n in [9999.0, -9999.0] {
            for x in XS {
                let j = dag_value(&mut ir, &EntryKind::JacU, x, n);
                assert!(j[0].is_finite(), "{f} JacU n={n} x={x}: {}", j[0]);
            }
        }
        for n in [10000.0, -10000.0] {
            for x in XS {
                let j = dag_value(&mut ir, &EntryKind::JacU, x, n);
                assert!(j[0].is_nan(), "{f} JacU n={n} x={x}: {}", j[0]);
            }
        }
    }
}
