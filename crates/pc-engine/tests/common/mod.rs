//! Helpers shared by the pc-engine integration tests.
#![allow(dead_code)]

use pc_engine::ModelEval;
use pc_ir::EntryKind;
use pc_ode::{FsLoader, Model, ModelInfo, Severity, SourceMap, load as ode_load};
use std::path::Path;

/// One RK4 step of x' = T f(x) in rescaled time.
pub fn rk4(m: &mut dyn ModelEval, p: &[f64], t: f64, x: &mut [f64], dt: f64) {
    let n = x.len();
    let mut f = |y: &[f64]| {
        let mut out = vec![0.0; n];
        m.eval(&EntryKind::F, y, p, &[], &mut out).unwrap();
        out.iter().map(|v| t * v).collect::<Vec<f64>>()
    };
    let add = |y: &[f64], k: &[f64], s: f64| -> Vec<f64> {
        y.iter().zip(k).map(|(a, b)| a + s * b).collect()
    };
    let k1 = f(x);
    let k2 = f(&add(x, &k1, dt / 2.0));
    let k3 = f(&add(x, &k2, dt / 2.0));
    let k4 = f(&add(x, &k3, dt));
    for i in 0..n {
        x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// Loads `tests/models/<name>.ode`. Panics with the diagnostics when the file does not
/// read or has an error.
pub fn load(name: &str) -> (Model, ModelInfo) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/models")
        .join(format!("{name}.ode"));
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let top = loader
        .load_top(&mut map, &path)
        .unwrap_or_else(|e| panic!("cannot read {e}"));
    let loaded = ode_load(top, &mut map, &mut loader);
    let errors: Vec<String> = loaded
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{} {}", d.code, d.message))
        .collect();
    if !errors.is_empty() {
        panic!("{name}.ode has errors:\n{}", errors.join("\n"));
    }
    (loaded.expanded, loaded.info)
}

/// Distance in units in the last place; two NaN match, one NaN is `u64::MAX`.
pub fn ulp_dist(a: f64, b: f64) -> u64 {
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
