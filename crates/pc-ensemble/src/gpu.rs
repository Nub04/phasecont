//! What the GPU backends share: the panic guard, the launch sizing and the launch loop.
//!
//! A run is cut into launches that each advance every member by at most K step attempts. K
//! starts small and follows the measured time of the last launch so that one launch takes
//! about 100 ms, well under the 2 s Windows display watchdog. The loop here knows nothing
//! about a driver: a backend hands it a closure that runs one launch and reads the control
//! words back.

use std::any::Any;
use std::mem::ManuallyDrop;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use pc_codegen::kernel::KernelError;

use crate::abi::ctl_done;
use crate::job::EnsembleError;

/// How long one launch should take.
pub const TARGET: Duration = Duration::from_millis(100);
/// Step attempts of the first launch, which also times a cold kernel.
pub const FIRST_STEPS: u64 = 16;
/// Most step attempts one launch may ask for.
pub const MAX_STEPS_PER_LAUNCH: u64 = 1 << 32;

/// Runs `f` and returns its panic as text.
///
/// Every call into a GPU library goes through this, because a panic must not cross the C ABI
/// (rule 7) and some of the Rust wrappers panic when a library is missing.
pub fn guard<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let msg = payload_text(&*payload);
        // Dropping the payload runs its Drop code, which could panic again. A second panic
        // leaves the payload undropped for good, which is what we want.
        let mut payload = ManuallyDrop::new(payload);
        // SAFETY: the payload is dropped at most once, here, and never used afterwards.
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            ManuallyDrop::drop(&mut payload)
        }));
        msg
    })
}

/// A kernel the emitter cannot write is an unsupported model, whichever dialect was asked for.
pub fn kernel_err(e: KernelError) -> EnsembleError {
    match e {
        KernelError::Unsupported(m) => EnsembleError::Unsupported(m),
        other => EnsembleError::Unsupported(format!("{other:?}")),
    }
}

fn payload_text(p: &(dyn Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "a panic with no text".to_owned()
    }
}

/// The step attempts of the next launch, given that `steps` took `took`.
///
/// It aims at [`TARGET`], grows at most fourfold per launch (the first launches time a cold
/// kernel) and shrinks at once.
pub fn next_steps(steps: u64, took: Duration) -> u64 {
    let secs = took.as_secs_f64().max(1e-6);
    let want = steps as f64 * TARGET.as_secs_f64() / secs;
    let capped = want
        .min(steps as f64 * 4.0)
        .min(MAX_STEPS_PER_LAUNCH as f64);
    (capped as u64).max(1)
}

/// Launches until no member is running.
///
/// `launch(steps)` must run one launch of at most `steps` step attempts per member, wait for
/// it to finish and return how long it took with the control words as they are afterwards.
/// `ctl` holds the control words on entry and the last ones read on a normal return.
///
/// The cancel flag is read before every launch. A launch that changes no control word is an
/// error, since a launch gives every running member at least one step attempt and an attempt
/// changes a counter. A launch that comes back with the wrong number of words is an error too.
pub fn drive(
    ctl: &mut Vec<u32>,
    members: usize,
    cancel: &AtomicBool,
    mut launch: impl FnMut(u64) -> Result<(Duration, Vec<u32>), EnsembleError>,
) -> Result<(), EnsembleError> {
    let mut steps = FIRST_STEPS;
    let mut before = ctl.clone();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(EnsembleError::Cancelled);
        }
        let (took, now) = launch(steps)?;
        if now.len() != ctl.len() {
            return Err(EnsembleError::GpuLaunch(format!(
                "the kernel returned {} control words, expected {}",
                now.len(),
                ctl.len()
            )));
        }
        *ctl = now;
        if ctl_done(ctl, members) {
            return Ok(());
        }
        if *ctl == before {
            return Err(EnsembleError::GpuLaunch(
                "no member advanced in a launch of the kernel".into(),
            ));
        }
        before.clone_from(ctl);
        steps = next_steps(steps, took);
    }
}

/// The global size of a launch: `members` rounded up to a whole number of groups.
pub fn global_size(members: usize, group: usize) -> usize {
    members.div_ceil(group) * group
}

/// Checks the buffers of a run against what the device allows.
///
/// `buffers` are the buffer names and byte sizes. A single buffer above `max_alloc`, or a
/// total above `memory`, is an error that says how to shrink the run. A `max_alloc` of 0 means
/// the driver gave no per-buffer limit.
pub fn check_alloc(
    buffers: &[(&str, u64)],
    max_alloc: u64,
    memory: Option<u64>,
) -> Result<(), EnsembleError> {
    if let Some(&(name, bytes)) = buffers
        .iter()
        .filter(|(_, b)| max_alloc > 0 && *b > max_alloc)
        .max_by_key(|(_, b)| *b)
    {
        return Err(EnsembleError::BadJob(format!(
            "the {name} buffer needs {bytes} bytes and the device allows {max_alloc} in one \
             buffer: run fewer members at a time"
        )));
    }
    let total: u64 = buffers.iter().map(|(_, b)| *b).sum();
    if let Some(mem) = memory
        && total > mem
    {
        return Err(EnsembleError::BadJob(format!(
            "the run needs {total} bytes on the device, which has {mem}: run fewer members at a \
             time"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::Status;
    use crate::kernel::CTL;

    #[test]
    fn the_global_size_covers_every_member_in_whole_groups() {
        assert_eq!(global_size(1, 64), 64);
        assert_eq!(global_size(64, 64), 64);
        assert_eq!(global_size(65, 64), 128);
        assert_eq!(global_size(1000, 32), 1024);
        assert_eq!(global_size(7, 1), 7);
    }

    #[test]
    fn a_buffer_above_the_device_limit_is_named() {
        let e = check_alloc(&[("rc", 100), ("out", 5000), ("work", 900)], 1000, None).unwrap_err();
        let EnsembleError::BadJob(m) = e else {
            panic!("expected BadJob");
        };
        assert!(
            m.contains("out") && m.contains("5000") && m.contains("1000"),
            "{m}"
        );
    }

    #[test]
    fn a_run_above_the_device_memory_is_an_error() {
        let buffers = [("rc", 400), ("out", 400), ("work", 400)];
        // Each buffer fits, the sum does not.
        let e = check_alloc(&buffers, 500, Some(1000)).unwrap_err();
        assert!(
            matches!(&e, EnsembleError::BadJob(m) if m.contains("1200")),
            "{e}"
        );
        assert!(check_alloc(&buffers, 500, Some(1200)).is_ok());
        // A device that reports no limit passes.
        assert!(check_alloc(&buffers, 0, None).is_ok());
    }

    fn words(members: usize, status: Status, count: u32) -> Vec<u32> {
        let mut c = vec![0; members * CTL];
        for m in 0..members {
            c[m * CTL] = u32::from(status.code());
            c[m * CTL + 2] = count;
        }
        c
    }

    #[test]
    fn a_panic_becomes_text_and_the_guard_stays_usable() {
        let msg = guard::<()>(|| panic!("inner {}", 7)).unwrap_err();
        assert_eq!(msg, "inner 7");
        let msg = guard::<()>(|| panic!("plain")).unwrap_err();
        assert_eq!(msg, "plain");
        assert_eq!(guard(|| 5), Ok(5));
    }

    #[test]
    fn the_launch_size_aims_at_the_target() {
        // A launch that took half the target doubles; one that took far less grows fourfold.
        assert_eq!(next_steps(1000, TARGET / 2), 2000);
        assert_eq!(next_steps(1000, Duration::from_micros(10)), 4000);
        // A launch that took twice the target halves, and the size never reaches zero.
        assert_eq!(next_steps(1000, TARGET * 2), 500);
        assert_eq!(next_steps(1, TARGET * 100), 1);
        assert_eq!(
            next_steps(MAX_STEPS_PER_LAUNCH, Duration::from_micros(1)),
            MAX_STEPS_PER_LAUNCH
        );
    }

    #[test]
    fn the_loop_launches_until_every_member_is_done() {
        let mut ctl = words(3, Status::Running, 0);
        let mut seen = Vec::new();
        let cancel = AtomicBool::new(false);
        drive(&mut ctl, 3, &cancel, |steps| {
            seen.push(steps);
            let n = seen.len() as u32;
            let status = if n == 3 { Status::Ok } else { Status::Running };
            Ok((TARGET / 2, words(3, status, n)))
        })
        .unwrap();
        assert_eq!(seen.len(), 3);
        // The first launch is small, and a launch at half the target doubles the next one.
        assert_eq!(seen[0], FIRST_STEPS);
        assert_eq!(seen[1], FIRST_STEPS * 2);
        assert_eq!(seen[2], FIRST_STEPS * 4);
        assert_eq!(ctl, words(3, Status::Ok, 3));
    }

    #[test]
    fn a_run_that_is_done_at_once_launches_once() {
        let mut ctl = words(2, Status::Running, 0);
        let mut launches = 0;
        let cancel = AtomicBool::new(false);
        drive(&mut ctl, 2, &cancel, |_| {
            launches += 1;
            Ok((TARGET, words(2, Status::Ok, 1)))
        })
        .unwrap();
        assert_eq!(launches, 1);
    }

    #[test]
    fn a_launch_that_changes_nothing_is_an_error() {
        let mut ctl = words(2, Status::Running, 0);
        let cancel = AtomicBool::new(false);
        let e = drive(&mut ctl, 2, &cancel, |_| {
            Ok((TARGET, words(2, Status::Running, 0)))
        })
        .unwrap_err();
        assert!(matches!(e, EnsembleError::GpuLaunch(m) if m.contains("no member advanced")));
    }

    #[test]
    fn a_stall_after_progress_is_still_caught() {
        let mut ctl = words(1, Status::Running, 0);
        let cancel = AtomicBool::new(false);
        let mut n = 0;
        let e = drive(&mut ctl, 1, &cancel, |_| {
            n += 1;
            // The first launch moves a counter, the second changes nothing.
            Ok((TARGET, words(1, Status::Running, n.min(1))))
        })
        .unwrap_err();
        assert!(matches!(e, EnsembleError::GpuLaunch(_)));
        assert_eq!(n, 2);
    }

    #[test]
    fn a_set_cancel_flag_stops_before_any_launch() {
        let mut ctl = words(1, Status::Running, 0);
        let cancel = AtomicBool::new(true);
        let mut launched = false;
        let e = drive(&mut ctl, 1, &cancel, |_| {
            launched = true;
            Ok((TARGET, words(1, Status::Ok, 1)))
        })
        .unwrap_err();
        assert_eq!(e, EnsembleError::Cancelled);
        assert!(!launched);
    }

    #[test]
    fn a_cancel_between_launches_stops_the_run() {
        let mut ctl = words(1, Status::Running, 0);
        let cancel = AtomicBool::new(false);
        let mut n = 0;
        let e = drive(&mut ctl, 1, &cancel, |_| {
            n += 1;
            cancel.store(true, Ordering::Relaxed);
            Ok((TARGET, words(1, Status::Running, n)))
        })
        .unwrap_err();
        assert_eq!(e, EnsembleError::Cancelled);
        assert_eq!(n, 1);
    }

    #[test]
    fn a_launch_error_comes_back_unchanged() {
        let mut ctl = words(1, Status::Running, 0);
        let cancel = AtomicBool::new(false);
        let e = drive(&mut ctl, 1, &cancel, |_| {
            Err(EnsembleError::GpuTimeout("watchdog".into()))
        })
        .unwrap_err();
        assert_eq!(e, EnsembleError::GpuTimeout("watchdog".into()));
    }

    #[test]
    fn control_words_of_the_wrong_length_are_an_error() {
        let mut ctl = words(2, Status::Running, 0);
        let cancel = AtomicBool::new(false);
        let e = drive(&mut ctl, 2, &cancel, |_| {
            Ok((TARGET, words(1, Status::Ok, 1)))
        })
        .unwrap_err();
        assert!(matches!(e, EnsembleError::GpuLaunch(m) if m.contains("control words")));
    }
}
