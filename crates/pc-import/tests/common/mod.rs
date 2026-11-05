//! Loads imported `.ode` text through the real parser and evaluates its right-hand side.
#![allow(dead_code)]

use pc_engine::{Evaluator, ModelEval, build_allowing_t};
use pc_ir::EntryKind;
use pc_ode::{MapLoader, Severity, SourceMap};

/// A loaded model that evaluates `f(x, p, t)`.
pub struct Loaded {
    eval: Box<dyn ModelEval>,
}

impl Loaded {
    pub fn n(&self) -> usize {
        self.eval.n()
    }

    pub fn n_p(&self) -> usize {
        self.eval.n_p()
    }

    /// The right-hand side at `x` with parameters `par` (declaration order) and time `t`.
    pub fn f(&mut self, x: &[f64], par: &[f64], t: f64) -> Vec<f64> {
        let mut p = par.to_vec();
        p.push(t);
        let mut out = vec![0.0; self.eval.n()];
        self.eval
            .eval(&EntryKind::F, x, &p, &[], &mut out)
            .expect("evaluation failed");
        out
    }
}

/// Parses `ode`, fails the test on any error diagnostic and builds the fallback evaluator.
pub fn load(ode: &str) -> Loaded {
    let mut map = SourceMap::new();
    let id = map.add("imported.ode", ode.to_string());
    let mut loader = MapLoader::new();
    let loaded = pc_ode::load(id, &mut map, &mut loader);
    let errors: Vec<_> = loaded
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errors.is_empty(), "errors {errors:?} in:\n{ode}");
    let eval = build_allowing_t(&loaded.expanded, &loaded.info, Evaluator::Fallback)
        .unwrap_or_else(|e| panic!("build failed: {e:?} in:\n{ode}"));
    Loaded { eval }
}
