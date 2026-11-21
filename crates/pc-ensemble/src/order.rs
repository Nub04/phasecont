//! Member order and stack layout choice for the `auto` stack layout (SPEC section 8af-5).
//!
//! A kernel with calls runs fastest with the warp stack layout when the 32 lanes of a warp
//! recurse to nearly the same depth, and up to about four times slower than the member
//! layout when they do not. `auto` evaluates every member's right-hand side once on the CPU
//! at `(x0, t0)`, records how deep it went, sorts the members by that depth so that each warp
//! holds members of similar depth, and predicts from the sorted depths whether warp beats
//! member. The GPU run then checks the prediction against the depths the kernel reports.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::job::{EnsembleError, Job, Params, Precision};
use crate::real::Real;
use crate::tape::{CpuTape, KERNEL_FRAMES, Tape};

/// How deep one right-hand side of a member recursed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Depth {
    /// The most frames active at once.
    pub peak: u32,
    /// The frames entered.
    pub calls: u64,
}

/// Lanes of one warp, the group the prediction looks at.
pub const WARP: usize = 32;

/// Warp time over member time of a group whose lanes recurse to nearly the same depth, and
/// of one whose lanes do not: the best and the worst ratios measured on the RTX 5060 (SPEC
/// section 8af-5).
pub const NARROW_RATIO: f64 = 0.6;
pub const WIDE_RATIO: f64 = 4.1;

/// `auto` picks warp when the predicted ratio is at most this, and the GPU run falls back to
/// member when the ratio from the depths the kernel reports goes above [`FALLBACK_RATIO`].
pub const CHOOSE_RATIO: f64 = 0.9;
pub const FALLBACK_RATIO: f64 = 1.0;

/// The widest spread of peak depth inside a group of a single-chain recursion (every member
/// enters as many frames as its peak depth) that still counts as narrow.
pub fn chain_spread(precision: Precision) -> u32 {
    match precision {
        Precision::Double => 12,
        Precision::Single => 16,
    }
}

/// The depth of every member's right-hand side at `(x0, t0)`, in the job's precision, with
/// the kernel's frame cap. `None` for a tape without calls. A member that passes the cap
/// reports the depth where it stopped, which is the cap.
pub fn depth_keys(
    tape: &Tape,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<Option<Vec<Depth>>, EnsembleError> {
    if tape.units.is_empty() {
        return Ok(None);
    }
    match job.precision {
        Precision::Double => keys_as::<f64>(tape, job, cancel),
        Precision::Single => keys_as::<f32>(tape, job, cancel),
    }
    .map(Some)
}

fn keys_as<T: Real>(
    tape: &Tape,
    job: &Job,
    cancel: &AtomicBool,
) -> Result<Vec<Depth>, EnsembleError> {
    let (n, n_p) = (tape.n, tape.n_p);
    let mut cpu = CpuTape::<T>::new(tape);
    cpu.set_max_calls(KERNEL_FRAMES);
    let t0 = T::from_f64(job.t0);
    let keys: Vec<Depth> = (0..job.members)
        .into_par_iter()
        .map_init(
            || {
                (
                    cpu.slab(),
                    vec![T::ZERO; n],
                    vec![T::ZERO; n_p],
                    vec![T::ZERO; n],
                )
            },
            |(slab, x, p, out), m| {
                if cancel.load(Ordering::Relaxed) {
                    return Depth::default();
                }
                for (d, &s) in x.iter_mut().zip(&job.x0[m * n..(m + 1) * n]) {
                    *d = T::from_f64(s);
                }
                let src = match job.params {
                    Params::Shared(v) => &v[..n_p],
                    Params::PerMember(v) => &v[m * n_p..(m + 1) * n_p],
                };
                for (d, &s) in p.iter_mut().zip(src) {
                    *d = T::from_f64(s);
                }
                cpu.bind(p, slab);
                cpu.eval_xt_with(x, t0, slab, out, Some(cancel));
                let (calls, peak) = slab.depth().expect("a tape with calls");
                Depth { peak, calls }
            },
        )
        .collect();
    if cancel.load(Ordering::Relaxed) {
        return Err(EnsembleError::Cancelled);
    }
    Ok(keys)
}

/// The stable order that sorts `keys` by peak depth, then by frames entered, or `None` when
/// they are in that order already.
pub fn sort_order(keys: &[Depth]) -> Option<Vec<u32>> {
    if keys.windows(2).all(|w| w[0] <= w[1]) {
        return None;
    }
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_by_key(|&m| keys[m as usize]);
    Some(order)
}

/// Whether one warp group is narrow, so the warp layout is expected to beat member on it. A
/// single chain (every lane enters as many frames as its peak depth) is narrow when its peak
/// depths differ by at most [`chain_spread`]; any other recursion only when every lane has
/// the same depth.
pub fn narrow(group: &[Depth], precision: Precision) -> bool {
    let Some(first) = group.first() else {
        return true;
    };
    if group.iter().all(|d| d.calls == u64::from(d.peak)) {
        let (lo, hi) = group.iter().fold((u32::MAX, 0), |(lo, hi), d| {
            (lo.min(d.peak), hi.max(d.peak))
        });
        hi - lo <= chain_spread(precision)
    } else {
        group.iter().all(|d| d == first)
    }
}

/// Predicted warp time over member time of members in this order: the ratio of each group
/// of [`WARP`] members, [`NARROW_RATIO`] or [`WIDE_RATIO`], weighted by the most frames a
/// lane of the group enters. `None` when no member enters a frame, so the layouts cost the
/// same.
pub fn predicted_ratio(keys: &[Depth], precision: Precision) -> Option<f64> {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for g in keys.chunks(WARP) {
        let w = g.iter().map(|d| d.calls).max().unwrap_or(0) as f64;
        let r = if narrow(g, precision) {
            NARROW_RATIO
        } else {
            WIDE_RATIO
        };
        num += w * r;
        den += w;
    }
    (den > 0.0).then(|| num / den)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(peaks: &[u32]) -> Vec<Depth> {
        peaks
            .iter()
            .map(|&p| Depth {
                peak: p,
                calls: u64::from(p),
            })
            .collect()
    }

    #[test]
    fn sorted_keys_need_no_order_and_the_order_is_stable() {
        assert_eq!(sort_order(&chain(&[1, 1, 2, 5])), None);
        let k = chain(&[3, 1, 3, 1, 2]);
        assert_eq!(sort_order(&k), Some(vec![1, 3, 4, 0, 2]));
        // Peak first, then frames entered.
        let t = vec![
            Depth { peak: 2, calls: 9 },
            Depth { peak: 2, calls: 3 },
            Depth { peak: 1, calls: 50 },
        ];
        assert_eq!(sort_order(&t), Some(vec![2, 1, 0]));
    }

    #[test]
    fn a_chain_is_narrow_up_to_its_spread_and_a_tree_only_when_equal() {
        let base: Vec<u32> = (0..32).map(|i| 2048 + i % 13).collect();
        assert!(narrow(&chain(&base), Precision::Double));
        let wide: Vec<u32> = (0..32).map(|i| 2048 + i % 14).collect();
        assert!(!narrow(&chain(&wide), Precision::Double));
        assert!(narrow(&chain(&wide), Precision::Single));
        let tree = Depth {
            peak: 10,
            calls: 177,
        };
        assert!(narrow(&[tree; 32], Precision::Double));
        let mut mixed = vec![tree; 32];
        mixed[5].calls = 287;
        mixed[5].peak = 11;
        assert!(!narrow(&mixed, Precision::Single));
    }

    #[test]
    fn the_prediction_weights_groups_by_their_deepest_lane() {
        assert_eq!(predicted_ratio(&chain(&[0; 64]), Precision::Double), None);
        let mut k = chain(&[100; 32]);
        assert_eq!(predicted_ratio(&k, Precision::Double), Some(NARROW_RATIO));
        // A third group with one lane that enters no frame spreads its chain by 100, so it is
        // wide, and its tenth of the weight lifts the ratio past the choice.
        k.extend(chain(&[800; 32]));
        let mut w = chain(&[100; 32]);
        w[0].peak = 0;
        w[0].calls = 0;
        k.extend(w);
        let r = predicted_ratio(&k, Precision::Double).expect("frames");
        let want = (100.0 * 0.6 + 800.0 * 0.6 + 100.0 * 4.1) / 1000.0;
        assert!((r - want).abs() < 1e-15, "{r} vs {want}");
        assert!(r > CHOOSE_RATIO);
    }
}
