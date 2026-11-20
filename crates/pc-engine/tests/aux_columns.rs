//! Aux columns of IVP and ITER rows (SPEC section 10, round E). With `n_aux > 0` every
//! point holds the state followed by the `Aux` entry evaluated at the point's own t, state
//! and parameters. With `n_aux = 0` the points are the ones the integrator and the
//! iteration gave before the columns existed.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, Evaluator, IterSetup, IvpOptions, IvpSetup, ModelEval, Point, StopReason,
    build_allowing_t, integrate, iterate,
};
use pc_ir::EntryKind;
use std::sync::atomic::AtomicBool;

/// The states of `ivp_aux.ode`.
const N_IVP: usize = 2;
/// Its aux quantities: energy, tx and lnx.
const AUX_IVP: usize = 3;
/// The state and the aux quantities of `map_aux.ode`: dist, kx and lg.
const N_MAP: usize = 1;
const AUX_MAP: usize = 3;
const A: f64 = 0.1;
const R: f64 = 2.5;

fn model(name: &str, which: Evaluator) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    build_allowing_t(&m, &info, which).unwrap()
}

fn ivp_setup(t0: f64, t1: f64, n_aux: usize) -> IvpSetup {
    IvpSetup {
        x0: vec![1.0, 0.0],
        params: vec![A],
        t0,
        t1,
        n_aux,
    }
}

fn iter_setup(n0: i64, n1: i64, n_aux: usize) -> IterSetup {
    IterSetup {
        x0: vec![0.1],
        params: vec![R],
        n0,
        n1,
        n_aux,
    }
}

fn ivp(m: &mut dyn ModelEval, s: &IvpSetup) -> Branch {
    integrate(
        m,
        s,
        &IvpOptions::default(),
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

fn iter(m: &mut dyn ModelEval, s: &IterSetup) -> Branch {
    iterate(m, s, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

/// FNV-1a over little-endian bytes, as in `hopf_curve_pin.rs`.
struct Digest(u64);

impl Digest {
    fn word(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// The number of points and a hash of the bits of every time, step and state.
fn digest(b: &Branch) -> (usize, u64) {
    let mut d = Digest(0xcbf2_9ce4_8422_2325);
    for p in &b.points {
        d.word(p.alpha.to_bits());
        d.word(p.h.to_bits());
        d.word(p.u.len() as u64);
        for v in &p.u {
            d.word(v.to_bits());
        }
    }
    (b.points.len(), d.0)
}

/// Same bits, or both NaN (a NaN keeps no payload that the tests rely on).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The golden hashes below were taken from the integrator and the iteration before the
/// aux columns, on the machine of `hopf_curve_pin.rs`. The CI runners can give other last
/// bits through `powf` and `ln`, so they compare the point counts only.
fn check_golden(name: &str, got: (usize, u64), want: (usize, u64)) {
    assert_eq!(got.0, want.0, "{name}: point count");
    if std::env::var_os("CI").is_some() {
        eprintln!("{name}: hash {:#x} not compared on CI", got.1);
        return;
    }
    assert_eq!(
        got, want,
        "{name}: (points, hash) is {got:#x?}, the golden is {want:#x?}"
    );
}

const EVALUATORS: [Evaluator; 2] = [Evaluator::Jit, Evaluator::Fallback];

// ---------- the aux count 0 changes nothing ----------

#[test]
fn an_ivp_run_with_no_aux_gives_the_points_of_before_bit_for_bit() {
    for (name, t0, t1, want) in [
        ("forward", 0.0, 6.0, (82, 0x9478_f9cc_66f2_a943)),
        ("backward", 3.0, -4.0, (96, 0x42bc_cb3b_5b85_9c27)),
    ] {
        let mut m = model("ivp_aux", Evaluator::Jit);
        let b = ivp(&mut *m, &ivp_setup(t0, t1, 0));
        assert_eq!(b.stop, StopReason::End, "{name}");
        assert!(b.points.iter().all(|p| p.u.len() == N_IVP), "{name}");
        check_golden(name, digest(&b), want);
    }
}

#[test]
fn an_iter_run_with_no_aux_gives_the_points_of_before_bit_for_bit() {
    for (name, n0, n1, want) in [
        ("from 0", 0, 40, (41, 0x7401_1563_07e8_97b0)),
        ("from 5", 5, 12, (8, 0x086d_4f02_2ea3_6a68)),
    ] {
        let mut m = model("map_aux", Evaluator::Jit);
        let b = iter(&mut *m, &iter_setup(n0, n1, 0));
        assert_eq!(b.stop, StopReason::End, "{name}");
        assert!(b.points.iter().all(|p| p.u.len() == N_MAP), "{name}");
        check_golden(name, digest(&b), want);
    }
}

// ---------- IVP ----------

#[test]
fn ivp_aux_entries_equal_a_direct_evaluation_at_the_points_time_and_state() {
    for which in EVALUATORS {
        for (t0, t1) in [(0.0, 6.0), (3.0, -4.0)] {
            let mut m = model("ivp_aux", which.clone());
            let with = ivp(&mut *m, &ivp_setup(t0, t1, AUX_IVP));
            let mut plain_model = model("ivp_aux", which.clone());
            let plain = ivp(&mut *plain_model, &ivp_setup(t0, t1, 0));
            // The columns do not change the integration: same stop, same points, same
            // state bits, same times and steps.
            assert_eq!(with.stop, plain.stop, "{which:?}");
            assert_eq!(with.points.len(), plain.points.len(), "{which:?}");
            for (w, p) in with.points.iter().zip(&plain.points) {
                assert_eq!(w.alpha.to_bits(), p.alpha.to_bits());
                assert_eq!(w.h.to_bits(), p.h.to_bits());
                assert_eq!(w.u.len(), N_IVP + AUX_IVP);
                for i in 0..N_IVP {
                    assert_eq!(w.u[i].to_bits(), p.u[i].to_bits(), "{which:?} state {i}");
                }
                // The extra entries are what a direct call of Aux gives there.
                let mut want = [0.0; AUX_IVP];
                let r = m.eval(
                    &EntryKind::Aux,
                    &w.u[..N_IVP],
                    &[A, w.alpha],
                    &[],
                    &mut want,
                );
                assert!(
                    matches!(r, Ok(()) | Err(EngineError::NotFinite(_))),
                    "{r:?}"
                );
                for (j, v) in want.iter().enumerate() {
                    assert!(
                        same(w.u[N_IVP + j], *v),
                        "{which:?} t = {}: aux {j} is {} and a direct call gives {v}",
                        w.alpha,
                        w.u[N_IVP + j]
                    );
                }
                // tx = t x reads the time slot, which is the row's own time.
                assert_eq!(w.u[N_IVP + 1].to_bits(), (w.alpha * w.u[0]).to_bits());
                let energy = (w.u[0] * w.u[0] + w.u[1] * w.u[1]) / 2.0;
                assert!((w.u[N_IVP] - energy).abs() <= 1e-12 * energy.max(1.0));
            }
        }
    }
}

#[test]
fn a_non_finite_ivp_aux_value_is_kept_and_does_not_stop_the_run() {
    for which in EVALUATORS {
        let mut m = model("ivp_aux", which.clone());
        let with = ivp(&mut *m, &ivp_setup(0.0, 6.0, AUX_IVP));
        let mut plain_model = model("ivp_aux", which.clone());
        let plain = ivp(&mut *plain_model, &ivp_setup(0.0, 6.0, 0));
        assert_eq!(with.stop, StopReason::End, "{which:?}");
        // lnx is NaN wherever x is negative, which the oscillator reaches after t = pi / 2.
        let nan_rows: Vec<&Point> = with
            .points
            .iter()
            .filter(|p| p.u[N_IVP + 2].is_nan())
            .collect();
        assert!(
            nan_rows.len() > 10,
            "{which:?}: {} NaN rows",
            nan_rows.len()
        );
        assert!(nan_rows.iter().all(|p| p.u[0] <= 0.0));
        // The other aux entries of such a row are still the values computed.
        assert!(nan_rows.iter().all(|p| p.u[N_IVP].is_finite()));
        // The run ends on t1 with the states of the run without columns.
        let (l, q) = (with.points.last().unwrap(), plain.points.last().unwrap());
        assert_eq!(l.alpha.to_bits(), 6.0_f64.to_bits());
        assert_eq!(l.u[0].to_bits(), q.u[0].to_bits());
        assert_eq!(l.u[1].to_bits(), q.u[1].to_bits());
    }
}

#[test]
fn the_streamed_ivp_points_carry_the_aux_columns() {
    let mut m = model("ivp_aux", Evaluator::Jit);
    let mut lens = Vec::new();
    let b = integrate(
        &mut *m,
        &ivp_setup(0.0, 1.0, AUX_IVP),
        &IvpOptions::default(),
        &AtomicBool::new(false),
        &mut |p| lens.push(p.u.len()),
    )
    .unwrap();
    assert_eq!(lens.len(), b.points.len());
    assert!(lens.iter().all(|&l| l == N_IVP + AUX_IVP), "{lens:?}");
}

/// A model that fails its `Aux` entry for every time at or after `from`.
struct AuxFailsFrom {
    inner: Box<dyn ModelEval>,
    from: f64,
}

impl ModelEval for AuxFailsFrom {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn n_p(&self) -> usize {
        self.inner.n_p()
    }
    fn eval(
        &mut self,
        kind: &EntryKind,
        x: &[f64],
        p: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), EngineError> {
        if matches!(kind, EntryKind::Aux) && p[p.len() - 1] >= self.from {
            return Err(EngineError::Model("aux failed".into()));
        }
        self.inner.eval(kind, x, p, v, out)
    }
}

#[test]
fn an_aux_error_other_than_non_finite_stops_the_ivp_run_and_keeps_the_rows_before_it() {
    let mut plain_model = model("ivp_aux", Evaluator::Jit);
    let plain = ivp(&mut *plain_model, &ivp_setup(0.0, 6.0, 0));
    let mut m = AuxFailsFrom {
        inner: model("ivp_aux", Evaluator::Jit),
        from: 2.0,
    };
    let b = ivp(&mut m, &ivp_setup(0.0, 6.0, AUX_IVP));
    assert!(
        matches!(&b.stop, StopReason::Error(EngineError::Model(s)) if s == "aux failed"),
        "{:?}",
        b.stop
    );
    assert!(b.points.len() > 2 && b.points.len() < plain.points.len());
    assert!(b.points.iter().all(|p| p.alpha < 2.0));
    for (w, p) in b.points.iter().zip(&plain.points) {
        assert_eq!(w.alpha.to_bits(), p.alpha.to_bits());
        assert_eq!(w.u[0].to_bits(), p.u[0].to_bits());
    }
    // A failure on the start row leaves no row, since there is no row without its aux.
    let mut m = AuxFailsFrom {
        inner: model("ivp_aux", Evaluator::Jit),
        from: -1.0,
    };
    let b = ivp(&mut m, &ivp_setup(0.0, 6.0, AUX_IVP));
    assert!(b.points.is_empty());
    assert!(matches!(b.stop, StopReason::Error(EngineError::Model(_))));
}

#[test]
fn an_aux_count_the_model_does_not_have_stops_the_run_with_an_error() {
    let mut m = model("ivp_aux", Evaluator::Jit);
    let b = ivp(&mut *m, &ivp_setup(0.0, 6.0, AUX_IVP + 1));
    assert!(b.points.is_empty());
    assert!(matches!(b.stop, StopReason::Error(_)), "{:?}", b.stop);
}

// ---------- ITER ----------

#[test]
fn iter_aux_entries_equal_a_direct_evaluation_at_the_iterate_number_and_state() {
    for which in EVALUATORS {
        for (n0, n1) in [(0, 40), (5, 12)] {
            let mut m = model("map_aux", which.clone());
            let with = iter(&mut *m, &iter_setup(n0, n1, AUX_MAP));
            let mut plain_model = model("map_aux", which.clone());
            let plain = iter(&mut *plain_model, &iter_setup(n0, n1, 0));
            assert_eq!(with.stop, StopReason::End);
            assert_eq!(with.points.len(), plain.points.len());
            for (w, p) in with.points.iter().zip(&plain.points) {
                assert_eq!(w.alpha.to_bits(), p.alpha.to_bits());
                assert_eq!(w.h.to_bits(), p.h.to_bits());
                assert_eq!(w.u.len(), N_MAP + AUX_MAP);
                assert_eq!(w.u[0].to_bits(), p.u[0].to_bits(), "{which:?}");
                let mut want = [0.0; AUX_MAP];
                let r = m.eval(
                    &EntryKind::Aux,
                    &w.u[..N_MAP],
                    &[R, w.alpha],
                    &[],
                    &mut want,
                );
                assert!(
                    matches!(r, Ok(()) | Err(EngineError::NotFinite(_))),
                    "{r:?}"
                );
                for (j, v) in want.iter().enumerate() {
                    assert!(
                        same(w.u[N_MAP + j], *v),
                        "{which:?} iterate {}: aux {j} is {} and a direct call gives {v}",
                        w.alpha,
                        w.u[N_MAP + j]
                    );
                }
                // dist = x - 0.5 and kx = k x, with k the number of the row's iterate.
                assert_eq!(w.u[N_MAP].to_bits(), (w.u[0] - 0.5).to_bits());
                assert_eq!(w.u[N_MAP + 1].to_bits(), (w.alpha * w.u[0]).to_bits());
            }
        }
    }
}

#[test]
fn a_non_finite_iter_aux_value_is_kept_and_does_not_stop_the_run() {
    let mut m = model("map_aux", Evaluator::Jit);
    let with = iter(&mut *m, &iter_setup(0, 40, AUX_MAP));
    assert_eq!(with.stop, StopReason::End);
    assert_eq!(with.points.len(), 41);
    // lg = ln(x - 0.4) is NaN for the first two iterates, 0.1 and 0.225, and finite after.
    let nan: Vec<bool> = with
        .points
        .iter()
        .map(|p| p.u[N_MAP + 2].is_nan())
        .collect();
    assert_eq!(&nan[..4], &[true, true, false, false]);
    assert!(nan[2..].iter().all(|n| !n));
    // The states are those of the run without columns.
    let mut plain_model = model("map_aux", Evaluator::Jit);
    let plain = iter(&mut *plain_model, &iter_setup(0, 40, 0));
    for (w, p) in with.points.iter().zip(&plain.points) {
        assert_eq!(w.u[0].to_bits(), p.u[0].to_bits());
    }
}

#[test]
fn an_aux_error_other_than_non_finite_stops_the_iter_run_and_keeps_the_rows_before_it() {
    let mut m = AuxFailsFrom {
        inner: model("map_aux", Evaluator::Jit),
        from: 5.0,
    };
    let b = iter(&mut m, &iter_setup(0, 40, AUX_MAP));
    assert!(
        matches!(&b.stop, StopReason::Error(EngineError::Model(_))),
        "{:?}",
        b.stop
    );
    let k: Vec<f64> = b.points.iter().map(|p| p.alpha).collect();
    assert_eq!(k, [0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn the_streamed_iter_points_carry_the_aux_columns() {
    let mut m = model("map_aux", Evaluator::Jit);
    let mut lens = Vec::new();
    let b = iterate(
        &mut *m,
        &iter_setup(0, 6, AUX_MAP),
        &AtomicBool::new(false),
        &mut |p| lens.push(p.u.len()),
    )
    .unwrap();
    assert_eq!(lens.len(), b.points.len());
    assert!(lens.iter().all(|&l| l == N_MAP + AUX_MAP), "{lens:?}");
}
