//! The device list and the self-test on the CPU backend.

use pc_ensemble::Precision;
use pc_ensemble::device::{Backend, BackendState, list_devices};
use pc_ensemble::selftest::{Selected, SelfTestError, run_selftest};

#[test]
fn the_cpu_backend_is_listed_first() {
    let list = list_devices();
    let first = &list.devices[0];
    assert_eq!(first.backend, Backend::Cpu);
    assert_eq!(first.index, 0);
    assert!(first.fp64 && first.fp32);
    assert!(first.name.contains("rayon"), "{}", first.name);
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cpu)
        .map(|(_, s)| s.clone());
    assert_eq!(state, Some(BackendState::Available));
}

#[test]
fn every_backend_has_a_state() {
    let list = list_devices();
    for b in [Backend::Cpu, Backend::Cuda, Backend::OpenCl, Backend::Metal] {
        assert!(
            list.backends.iter().any(|(x, _)| *x == b),
            "{} has no state",
            b.name()
        );
    }
}

#[test]
fn backend_names_round_trip() {
    for b in [Backend::Cpu, Backend::Cuda, Backend::OpenCl, Backend::Metal] {
        assert_eq!(Backend::from_name(b.name()), Some(b));
    }
    assert_eq!(Backend::from_name("hip"), None);
    assert_eq!(Backend::from_name("CPU"), None);
}

#[test]
fn the_cpu_selftest_passes_in_both_precisions() {
    for p in [Precision::Single, Precision::Double] {
        let r = run_selftest(Selected {
            backend: Backend::Cpu,
            device: 0,
            precision: p,
        })
        .expect("cpu self-test");
        assert!(r.pass, "{}: {:#?}", p.name(), r.tiers);
        assert_eq!(r.label.backend, "cpu");
        assert_eq!(r.label.precision, p);
        // Comparing the CPU backend with itself proves nothing, so the report says so.
        assert_eq!(r.bit_exact, None);
        for t in &r.tiers {
            assert!(t.max_error <= t.limit, "{}: {t:?}", t.name);
            assert!(t.max_error.is_finite(), "{}: {t:?}", t.name);
        }
        let names: Vec<&str> = r.tiers.iter().map(|t| t.name).collect();
        for want in ["analytic-rk4", "analytic-dp5", "determinism"] {
            assert!(names.contains(&want), "{names:?}");
        }
    }
}

#[test]
fn the_analytic_errors_are_small_and_nonzero() {
    // A zero error would mean the check compares a result with itself.
    let r = run_selftest(Selected {
        backend: Backend::Cpu,
        device: 0,
        precision: Precision::Double,
    })
    .unwrap();
    for t in r.tiers.iter().filter(|t| t.name.starts_with("analytic")) {
        assert!(t.max_error > 0.0, "{t:?}");
        assert!(t.max_error < 1e-6, "{t:?}");
    }
}

#[test]
fn a_missing_device_is_not_a_pass() {
    let e = run_selftest(Selected {
        backend: Backend::Cpu,
        device: 1,
        precision: Precision::Single,
    })
    .unwrap_err();
    assert!(matches!(e, SelfTestError::NoDevice(_)), "{e:?}");
}

#[test]
fn a_backend_that_is_not_built_has_no_device() {
    for b in [Backend::Cuda, Backend::OpenCl, Backend::Metal] {
        // The Metal backend is built on macOS, where it finds a device.
        if cfg!(target_os = "macos") && b == Backend::Metal {
            continue;
        }
        let e = run_selftest(Selected {
            backend: b,
            device: 0,
            precision: Precision::Single,
        })
        .unwrap_err();
        let SelfTestError::NoDevice(text) = e else {
            panic!("{b:?}: not NoDevice");
        };
        assert!(text.contains(b.name()), "{text}");
    }
}

mod run_with {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use pc_ensemble::device::{Backend, DeviceInfo};
    use pc_ensemble::selftest::{SelfTestError, run_with};
    use pc_ensemble::{EnsembleError, EnsembleResult, Job, Label, Precision, run_cpu};

    fn device() -> DeviceInfo {
        DeviceInfo {
            backend: Backend::Cuda,
            index: 0,
            name: "fake".into(),
            driver: "0".into(),
            fp32: true,
            fp64: true,
            memory_bytes: None,
        }
    }

    fn label(p: Precision) -> Label {
        Label {
            backend: "cuda".into(),
            device: "fake".into(),
            driver: "0".into(),
            precision: p,
        }
    }

    fn cpu(j: &Job) -> Result<EnsembleResult, EnsembleError> {
        run_cpu(j, u64::MAX, &AtomicBool::new(false))
    }

    #[test]
    fn a_runner_that_matches_the_cpu_is_bit_exact() {
        for p in [Precision::Single, Precision::Double] {
            let r = run_with(&cpu, device(), label(p), true).unwrap();
            assert_eq!(r.bit_exact, Some(true));
            assert!(r.pass, "{:#?}", r.tiers);
        }
    }

    #[test]
    fn one_wrong_bit_fails_the_comparison_and_the_report() {
        // The last bit of the first state of every result is flipped: the analytic tiers
        // still pass, the bit comparison must not.
        let flip = |j: &Job| {
            let mut r = cpu(j)?;
            let b = r.states[0].to_bits() ^ 1;
            r.states[0] = f64::from_bits(b);
            Ok(r)
        };
        let r = run_with(&flip, device(), label(Precision::Double), true).unwrap();
        assert_eq!(r.bit_exact, Some(false));
        assert!(!r.pass);
        assert!(
            r.tiers
                .iter()
                .filter(|t| t.name.starts_with("analytic"))
                .all(|t| t.pass())
        );
    }

    #[test]
    fn the_comparison_is_skipped_when_not_asked_for() {
        let r = run_with(&cpu, device(), label(Precision::Double), false).unwrap();
        assert_eq!(r.bit_exact, None);
        assert!(r.pass);
    }

    #[test]
    fn a_runner_that_changes_between_calls_fails_determinism() {
        let calls = AtomicUsize::new(0);
        let drift = |j: &Job| {
            let mut r = cpu(j)?;
            if calls.fetch_add(1, Ordering::SeqCst) % 2 == 1 {
                r.states[0] = f64::from_bits(r.states[0].to_bits() ^ 1);
            }
            Ok(r)
        };
        let r = run_with(&drift, device(), label(Precision::Double), false).unwrap();
        let det = r.tiers.iter().find(|t| t.name == "determinism").unwrap();
        assert!(det.max_error > 0.0, "{det:?}");
        assert!(!r.pass);
    }

    #[test]
    fn a_runner_that_returns_a_wrong_answer_fails_the_analytic_tier() {
        let wrong = |j: &Job| {
            let mut r = cpu(j)?;
            for v in r.states.iter_mut() {
                *v *= 1.01;
            }
            Ok(r)
        };
        let r = run_with(&wrong, device(), label(Precision::Double), false).unwrap();
        assert!(!r.pass);
        assert!(
            r.tiers
                .iter()
                .any(|t| t.name.starts_with("analytic") && !t.pass())
        );
    }

    #[test]
    fn a_runner_error_is_reported_as_an_error_not_a_pass() {
        let fail = |_: &Job| Err(EnsembleError::Cancelled);
        let e = run_with(&fail, device(), label(Precision::Single), true).unwrap_err();
        assert!(
            matches!(e, SelfTestError::Run(EnsembleError::Cancelled)),
            "{e:?}"
        );
    }
}
