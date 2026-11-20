//! Iteration of a map (SPEC section 8u, acceptance test 1). Each test writes out its map
//! in Rust with the association the IR keeps: `x^2` is `x*x` and `r*x*(1-x)` is
//! `(r*x)*(1-x)`, so the engine and the hand loop agree bit for bit.

mod common;

use common::load;
use pc_engine::{
    Branch, EngineError, EpSetup, Evaluator, IterSetup, ModelEval, Options, Run, RunKind, RunSpec,
    RunState, StopReason, build_allowing_t, iterate,
};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

fn model(name: &str) -> Box<dyn ModelEval> {
    let (m, info) = load(name);
    build_allowing_t(&m, &info, Evaluator::Jit).unwrap()
}

fn setup(x0: &[f64], params: &[f64], n0: i64, n1: i64) -> IterSetup {
    IterSetup {
        x0: x0.to_vec(),
        params: params.to_vec(),
        n0,
        n1,
        n_aux: 0,
    }
}

fn run(name: &str, s: &IterSetup) -> Branch {
    iterate(
        model(name).as_mut(),
        s,
        &AtomicBool::new(false),
        &mut |_| {},
    )
    .unwrap()
}

/// The rows of a branch as [k, x(k)...] bits.
fn rows(b: &Branch) -> Vec<Vec<u64>> {
    b.points
        .iter()
        .map(|p| {
            let mut r = vec![p.alpha.to_bits()];
            r.extend(p.u.iter().map(|v| v.to_bits()));
            r
        })
        .collect()
}

fn expected(n0: i64, xs: &[Vec<f64>]) -> Vec<Vec<u64>> {
    xs.iter()
        .enumerate()
        .map(|(j, x)| {
            let mut r = vec![((n0 + j as i64) as f64).to_bits()];
            r.extend(x.iter().map(|v| v.to_bits()));
            r
        })
        .collect()
}

#[test]
fn the_logistic_map_matches_a_hand_loop_bit_for_bit() {
    let r = 2.5;
    let mut x = 0.1_f64;
    let mut xs = vec![vec![x]];
    for _ in 0..50 {
        x = (r * x) * (1.0 - x);
        xs.push(vec![x]);
    }
    for name in ["map_logistic", "map_logistic_discrete"] {
        let b = run(name, &setup(&[0.1], &[r], 0, 50));
        assert_eq!(b.stop, StopReason::End, "{name}");
        assert_eq!(rows(&b), expected(0, &xs), "{name}");
        assert_eq!(b.points[0].h, 0.0);
        assert!(b.points[1..].iter().all(|p| p.h == 1.0));
    }
}

#[test]
fn t_inside_the_map_is_the_number_of_the_iterate_mapped_from() {
    let b = run("map_shift", &setup(&[0.0], &[], 0, 4));
    let xs: Vec<Vec<f64>> = [0.0, 0.0, 1.0, 3.0, 6.0].map(|v| vec![v]).into();
    assert_eq!(rows(&b), expected(0, &xs));
    let b = run("map_shift", &setup(&[0.0], &[], 5, 9));
    let xs: Vec<Vec<f64>> = [0.0, 5.0, 11.0, 18.0, 26.0].map(|v| vec![v]).into();
    assert_eq!(rows(&b), expected(5, &xs));
}

#[test]
fn the_henon_map_matches_a_hand_loop_bit_for_bit() {
    let (a, b) = (1.4, 0.3);
    let (mut x, mut y) = (0.0_f64, 0.0_f64);
    let mut xs = vec![vec![x, y]];
    for _ in 0..100 {
        (x, y) = (1.0 - a * (x * x) + y, b * x);
        xs.push(vec![x, y]);
    }
    let br = run("map_henon", &setup(&[0.0, 0.0], &[a, b], 0, 100));
    assert_eq!(br.stop, StopReason::End);
    assert_eq!(rows(&br), expected(0, &xs));
}

#[test]
fn an_overflow_stops_the_run_and_keeps_the_rows() {
    let b = run("map_square", &setup(&[10.0], &[], 0, 20));
    assert!(
        matches!(b.stop, StopReason::Error(EngineError::NotFinite(_))),
        "{:?}",
        b.stop
    );
    assert_eq!(b.points.len(), 9);
    let last = b.points.last().unwrap();
    // Eight squarings of 10 by hand, about 1e256; the ninth overflows.
    let x8 = (0..8).fold(10.0_f64, |x, _| x * x);
    assert_eq!(last.alpha, 8.0);
    assert_eq!(last.u[0].to_bits(), x8.to_bits());
    assert!((x8 * x8).is_infinite());
}

#[test]
fn a_cancel_before_the_start_is_an_error() {
    let r = iterate(
        model("map_logistic").as_mut(),
        &setup(&[0.1], &[2.5], 0, 5),
        &AtomicBool::new(true),
        &mut |_| {},
    );
    assert!(matches!(r, Err(EngineError::Cancelled)), "{r:?}");
}

#[test]
fn bad_starts_are_refused() {
    for s in [
        setup(&[0.1, 0.2], &[2.5], 0, 5),
        setup(&[0.1], &[], 0, 5),
        setup(&[f64::INFINITY], &[2.5], 0, 5),
    ] {
        let r = iterate(
            model("map_logistic").as_mut(),
            &s,
            &AtomicBool::new(false),
            &mut |_| {},
        );
        assert!(matches!(r, Err(EngineError::BadProblem(_))), "{s:?}: {r:?}");
    }
}

#[test]
fn a_background_run_iterates_a_map_that_uses_t() {
    let (m, info) = load("map_shift");
    let mut spec = RunSpec::new(
        m,
        info,
        Evaluator::Jit,
        EpSetup::new(0, Vec::new(), vec![0.0]),
        Options::default(),
    );
    spec.kind = RunKind::Iter {
        n: [5, 9],
        n_aux: 0,
    };
    let run = Run::start(spec);
    let t0 = Instant::now();
    while run.poll().state == RunState::Running {
        assert!(t0.elapsed() < Duration::from_secs(30));
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(run.poll().state, RunState::Done);
    assert_eq!(run.stop(0), Some(StopReason::End));
    // An ITER branch keeps flat rows [k, x] (SPEC 8af-1).
    assert_eq!(run.row_stride(0), Some(2));
    let rows = run.poll().points[0];
    let mut buf = vec![f64::NAN; 2 * rows];
    assert_eq!(run.copy_rows(0, 0, rows, &mut buf).unwrap(), rows);
    let x: Vec<(f64, f64)> = buf
        .as_chunks::<2>()
        .0
        .iter()
        .map(|r| (r[0], r[1]))
        .collect();
    // A flat branch has no points to clone, and a wrong buffer or branch is an error.
    let p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run.points(0, 0, 1)));
    assert!(p.is_err(), "points() on a flat branch panics");
    assert!(run.copy_rows(0, 0, rows, &mut buf[1..]).is_err());
    assert!(run.copy_rows(1, 0, 0, &mut []).is_err());
    assert_eq!(run.row_stride(1), None);
    assert_eq!(run.copy_rows(0, rows, 1, &mut buf[..2]).unwrap(), 0);
    assert_eq!(
        x,
        [
            (5.0, 0.0),
            (6.0, 5.0),
            (7.0, 11.0),
            (8.0, 18.0),
            (9.0, 26.0)
        ]
    );
}
