//! Fault injection for the recursion cap (SPEC 8af-5, Errors).
//!
//! The cap is the constant `MAX_ACTIVE_CALLS` with no knob, and a derivative unit makes the
//! same number of calls as the value, so a real recursive model cannot make a chosen call
//! fail. `Inject` wraps any model and returns an error on the calls that a predicate picks,
//! which puts the failure at one pass of one driver.

use pc_engine::{EngineError, ModelEval};
use pc_ir::{EntryKind, Slot};
use std::cell::Cell;
use std::rc::Rc;

/// What the predicate sees of one `eval` call.
pub struct Call<'a> {
    /// Number of calls made before this one, counting from 0.
    pub index: usize,
    pub kind: &'a EntryKind,
    /// The kind of the call before this one, `None` for the first.
    pub prev: Option<&'a EntryKind>,
    pub x: &'a [f64],
    pub p: &'a [f64],
    /// The direction vectors of a `Dir` entry, one block of `n` values per slot.
    pub v: &'a [f64],
}

type Rule<'a> = Box<dyn FnMut(&Call) -> bool + 'a>;

/// A model that fails the calls picked by `rule` with `error` and passes every other call to
/// `inner`. `hits` counts the failures, so a test can check that the fault fired.
pub struct Inject<'a> {
    inner: &'a mut dyn ModelEval,
    rule: Rule<'a>,
    error: EngineError,
    calls: usize,
    prev: Option<EntryKind>,
    pub hits: Rc<Cell<usize>>,
}

/// The error the recursion cap gives, as `EvalError::Depth` maps to it.
pub fn cap_error() -> EngineError {
    EngineError::CallLimit("test: sum, cap 1000000".into())
}

impl<'a> Inject<'a> {
    /// Fails with the recursion cap error.
    pub fn call_limit(
        inner: &'a mut dyn ModelEval,
        rule: impl FnMut(&Call) -> bool + 'a,
    ) -> Inject<'a> {
        Inject::with_error(inner, cap_error(), rule)
    }

    /// Fails with `error`, for the test that other errors keep their old handling.
    pub fn with_error(
        inner: &'a mut dyn ModelEval,
        error: EngineError,
        rule: impl FnMut(&Call) -> bool + 'a,
    ) -> Inject<'a> {
        Inject {
            inner,
            rule: Box::new(rule),
            error,
            calls: 0,
            prev: None,
            hits: Rc::new(Cell::new(0)),
        }
    }
}

impl ModelEval for Inject<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn n_p(&self) -> usize {
        self.inner.n_p()
    }
    fn map_residual(&self) -> bool {
        self.inner.map_residual()
    }
    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        let call = Call {
            index: self.calls,
            kind,
            prev: self.prev.as_ref(),
            x,
            p,
            v,
        };
        let fail = (self.rule)(&call);
        self.calls += 1;
        self.prev = Some(kind.clone());
        if fail {
            self.hits.set(self.hits.get() + 1);
            return Err(self.error.clone());
        }
        self.inner.eval(kind, x, p, v, out)
    }
}

/// True for a `Dir` entry whose slots are all `U` and number at least `k`: the multilinear
/// forms of the normal form code. The curve systems also call `Dir`, with a `Full` slot or with
/// mixed `U`/`P` slots, and the cycle and map curve systems call `Dir([U, U])`.
pub fn is_form(kind: &EntryKind, k: usize) -> bool {
    matches!(kind, EntryKind::Dir(s) if s.len() >= k && s.iter().all(|c| *c == Slot::U))
}

/// True for a bilinear form `B(a, b)` of two different vectors. The test functions of the LP
/// curve call `B(q, q)`, so a mixed pair comes only from the Bogdanov-Takens coefficients.
pub fn is_mixed_pair(c: &Call) -> bool {
    let h = c.v.len() / 2;
    is_form(c.kind, 2) && c.v.len() == 2 * h && c.v[..h] != c.v[h..]
}
