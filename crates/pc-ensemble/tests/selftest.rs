//! The device list and the self-test on the CPU backend.

use pc_ensemble::Precision;
use pc_ensemble::device::{Backend, BackendState, list_devices};
mod common;

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
        // The CPU backend does not take the fma tier, so there is no fma report.
        assert_eq!(r.fma, None);
        for t in &r.tiers {
            assert!(t.max_error <= t.limit, "{}: {t:?}", t.name);
            assert!(t.max_error.is_finite(), "{}: {t:?}", t.name);
        }
        let names: Vec<&str> = r.tiers.iter().map(|t| t.name).collect();
        for want in [
            "analytic-rk4",
            "analytic-dp5",
            "iterate",
            "determinism",
            "recursion",
        ] {
            assert!(names.contains(&want), "{names:?}");
        }
    }
}

/// On a CUDA device the double report carries the fma comparison, the single report does
/// not, and the comparison never changes `pass`. The difference is printed, not bounded.
#[test]
fn a_cuda_double_selftest_reports_the_fma_tier() {
    if let Err(why) = pc_ensemble::device::find_device(Backend::Cuda, 0) {
        common::skip_gpu("cuda", &why);
        return;
    }
    for p in [Precision::Single, Precision::Double] {
        let r = run_selftest(Selected {
            backend: Backend::Cuda,
            device: 0,
            precision: p,
        })
        .expect("cuda self-test");
        let tiers_pass = r.tiers.iter().all(|t| t.pass()) && r.bit_exact != Some(false);
        assert_eq!(r.pass, tiers_pass, "{}: {r:#?}", p.name());
        match p {
            Precision::Single => assert_eq!(r.fma, None),
            Precision::Double => {
                let f = r.fma.clone().expect("an fma report on CUDA double");
                assert!(f.max_difference.is_finite(), "{f:?}");
                eprintln!("cuda double fma: {f:?}");
            }
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
fn the_single_precision_iterate_error_is_small_and_nonzero() {
    // The start states and the sums round to f32, so a zero error would mean the check
    // compares a result with itself. In double precision the error is 0 by design.
    let r = run_selftest(Selected {
        backend: Backend::Cpu,
        device: 0,
        precision: Precision::Single,
    })
    .unwrap();
    let t = r.tiers.iter().find(|t| t.name == "iterate").unwrap();
    assert!(t.max_error > 0.0, "{t:?}");
    assert!(t.max_error < 1e-6, "{t:?}");
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
    use pc_ensemble::tape::{KERNEL_FRAMES, with_max_calls};
    use pc_ensemble::{
        EnsembleError, EnsembleResult, Job, Label, Math, Method, Precision, Status, run_cpu,
    };

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
            math: pc_ensemble::Math::Exact,
        }
    }

    /// The CPU backend under the call cap of the fake CUDA device, so it ends a member past
    /// 4,096 frames as a CUDA kernel does.
    fn cpu(j: &Job) -> Result<EnsembleResult, EnsembleError> {
        with_max_calls(KERNEL_FRAMES, || uncapped(j))
    }

    /// The CPU backend at its own cap of 1,000,000 frames.
    fn uncapped(j: &Job) -> Result<EnsembleResult, EnsembleError> {
        run_cpu(j, u64::MAX, &AtomicBool::new(false))
    }

    #[test]
    fn a_runner_that_does_not_stop_at_the_cap_fails_the_recursion_tier() {
        let r = run_with(&uncapped, device(), label(Precision::Double), false, false).unwrap();
        let t = r.tiers.iter().find(|t| t.name == "recursion").unwrap();
        assert_eq!(t.max_error, f64::INFINITY, "{t:?}");
        assert!(!r.pass);
        let t = r.tiers.iter().find(|t| t.name == "analytic-rk4").unwrap();
        assert!(t.pass(), "{t:?}");
    }

    #[test]
    fn a_backend_that_does_not_run_calls_has_no_recursion_tier() {
        for b in [Backend::OpenCl, Backend::Metal] {
            let d = DeviceInfo {
                backend: b,
                ..device()
            };
            let r = run_with(&cpu, d, label(Precision::Single), false, false).unwrap();
            assert!(r.pass, "{b:?}: {:#?}", r.tiers);
            assert!(r.tiers.iter().all(|t| t.name != "recursion"), "{b:?}");
        }
        let r = run_with(&cpu, device(), label(Precision::Single), false, false).unwrap();
        assert!(r.tiers.iter().any(|t| t.name == "recursion"));
    }

    #[test]
    fn a_runner_that_matches_the_cpu_is_bit_exact() {
        for p in [Precision::Single, Precision::Double] {
            let r = run_with(&cpu, device(), label(p), true, false).unwrap();
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
        let r = run_with(&flip, device(), label(Precision::Double), true, false).unwrap();
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
        let r = run_with(&cpu, device(), label(Precision::Double), false, false).unwrap();
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
        let r = run_with(&drift, device(), label(Precision::Double), false, false).unwrap();
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
        let r = run_with(&wrong, device(), label(Precision::Double), false, false).unwrap();
        assert!(!r.pass);
        assert!(
            r.tiers
                .iter()
                .any(|t| t.name.starts_with("analytic") && !t.pass())
        );
    }

    fn flip_first(mut r: EnsembleResult) -> EnsembleResult {
        r.states[0] = f64::from_bits(r.states[0].to_bits() ^ 1);
        r
    }

    #[test]
    fn a_wrong_bit_in_the_map_jobs_alone_fails_the_comparison() {
        // Only the Henon job of the comparison uses `iter`, so the comparison fails only
        // if that job is in it. One bit of a state near 16 is far below the iterate limit.
        let flip = |j: &Job| {
            let r = cpu(j)?;
            Ok(if j.method == Method::Iter {
                flip_first(r)
            } else {
                r
            })
        };
        for p in [Precision::Single, Precision::Double] {
            let r = run_with(&flip, device(), label(p), true, false).unwrap();
            assert_eq!(r.bit_exact, Some(false), "{}", p.name());
            assert!(!r.pass);
            for name in ["analytic-rk4", "analytic-dp5", "iterate"] {
                let t = r.tiers.iter().find(|t| t.name == name).unwrap();
                assert!(t.pass(), "{}: {t:?}", p.name());
            }
        }
    }

    #[test]
    fn a_map_job_that_changes_between_calls_fails_determinism() {
        // Only the Henon job of the determinism tier uses `iter`, so the tier fails only
        // if that job is in it.
        for p in [Precision::Single, Precision::Double] {
            let calls = AtomicUsize::new(0);
            let drift = |j: &Job| {
                let r = cpu(j)?;
                let odd = j.method == Method::Iter && calls.fetch_add(1, Ordering::SeqCst) % 2 == 1;
                Ok(if odd { flip_first(r) } else { r })
            };
            let r = run_with(&drift, device(), label(p), false, false).unwrap();
            let det = r.tiers.iter().find(|t| t.name == "determinism").unwrap();
            assert!(det.max_error > 0.0, "{}: {det:?}", p.name());
            assert!(!r.pass);
            for name in ["analytic-rk4", "analytic-dp5", "iterate"] {
                let t = r.tiers.iter().find(|t| t.name == name).unwrap();
                assert!(t.pass(), "{}: {t:?}", p.name());
            }
        }
    }

    #[test]
    fn a_runner_that_passes_the_wrong_time_to_a_map_fails_the_iterate_tier() {
        // Iterating over [t0 + 1, t1 + 1] evaluates F at k + 1 in place of k, the error the
        // tier is there to catch. The ODE tiers and determinism still pass.
        let late = |j: &Job| {
            if j.method == Method::Iter {
                let mut shifted = j.clone();
                shifted.t0 += 1.0;
                shifted.t1 += 1.0;
                cpu(&shifted)
            } else {
                cpu(j)
            }
        };
        for p in [Precision::Single, Precision::Double] {
            let r = run_with(&late, device(), label(p), false, false).unwrap();
            assert!(!r.pass);
            for t in &r.tiers {
                assert_eq!(t.pass(), t.name != "iterate", "{}: {t:?}", p.name());
            }
        }
    }

    #[test]
    fn a_runner_error_is_reported_as_an_error_not_a_pass() {
        let fail = |_: &Job| Err(EnsembleError::Cancelled);
        let e = run_with(&fail, device(), label(Precision::Single), true, false).unwrap_err();
        assert!(
            matches!(e, SelfTestError::Run(EnsembleError::Cancelled)),
            "{e:?}"
        );
    }

    #[test]
    fn the_fma_report_is_off_unless_asked_for() {
        let r = run_with(&cpu, device(), label(Precision::Double), true, false).unwrap();
        assert_eq!(r.fma, None);
    }

    #[test]
    fn a_runner_that_ignores_the_tier_reports_no_fma_difference() {
        let r = run_with(&cpu, device(), label(Precision::Double), true, true).unwrap();
        let f = r.fma.expect("asked for");
        assert_eq!((f.max_difference, f.status_differ), (0.0, 0));
        assert!(r.pass, "{:#?}", r.tiers);
    }

    /// The fma jobs come back moved by a relative 1e-6, and one of them with a different
    /// status. The report shows both, and `pass` stays what the exact tiers say.
    #[test]
    fn an_fma_difference_is_reported_and_never_fails_the_report() {
        let nudge = |j: &Job| {
            let mut r = cpu(j)?;
            if j.math == Math::Fma {
                for x in &mut r.states {
                    *x *= 1.0 + 1e-6;
                }
                r.status[0] = Status::Bounds;
            }
            Ok(r)
        };
        let r = run_with(&nudge, device(), label(Precision::Double), true, true).unwrap();
        let f = r.fma.expect("asked for");
        assert_eq!(
            f.status_differ, 2,
            "one member of each of the two jobs: {f:?}"
        );
        assert!(f.max_difference > 5e-7 && f.max_difference < 1e-6, "{f:?}");
        assert_eq!(r.bit_exact, Some(true));
        assert!(r.pass, "{:#?}", r.tiers);
    }

    #[test]
    fn a_device_that_refuses_the_fma_jobs_is_an_error() {
        let refuse = |j: &Job| {
            if j.math == Math::Fma {
                Err(EnsembleError::BadJob("no fma".into()))
            } else {
                cpu(j)
            }
        };
        let e = run_with(&refuse, device(), label(Precision::Double), true, true).unwrap_err();
        assert!(
            matches!(e, SelfTestError::Fma(EnsembleError::BadJob(_))),
            "{e:?}"
        );
        // The message names the tier, so the CLI error (exit 1) says which jobs failed.
        let text = e.to_string();
        assert!(text.contains("fma tier"), "{text}");
    }
}
