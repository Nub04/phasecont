//! Times the `auto` stack layout against `member` and `warp` on CUDA (SPEC section 8af-5,
//! tests): the key pass, the sorting and every switch count, since each run goes through
//! `run_on` from the job to the result in the caller's order.
//!
//! It is a measurement, not a check, so it is ignored in a normal run:
//!
//! ```text
//! PC_REQUIRE_GPU=cuda cargo test --release -p pc-ensemble --test auto_time -- --ignored --nocapture
//! ```
//!
//! Settings, all optional: `PC_AT_MEMBERS` (comma list, default `4096,10000,16384`; 10,000 is
//! not a whole number of blocks, so the last block and the last warp group run part full),
//! `PC_AT_PRECISION` (`single`, `double`, default both), `PC_AT_REPEATS` (default 5),
//! `PC_AT_CASES` (comma list of case names, default all). Every case prints one `AT` line with
//! the fastest run of each layout in milliseconds, `auto` over `member`, `auto` over the faster
//! of the two, and the `stack_ran` and `sorted` of the `auto` run. The first run of each layout
//! compiles its kernels and is not timed. The timed runs take turns, one run of each layout per
//! round, so a clock that drifts during a case slows every layout alike. All three layouts
//! must give the same bits.

mod common;

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use common::*;
use pc_ensemble::device::{Backend, BackendState, DeviceInfo, find_device, list_devices};
use pc_ensemble::run::run_on;
use pc_ensemble::{EnsembleResult, Job, Method, Params, Precision, StackLayout};

const SUM: &str = "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/(1+n)\ndone\n";
const PW: &str = "par n=0\npw(a,m)=if(m<1)then(1)else(a*pw(a,m-1))\nx'=-0.01*pw(x,n)\ndone\n";
const FIB: &str =
    "par n=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\nx'=-x*fib(n)/(1+fib(n))\ndone\n";
/// A short tree beside a chain: the call count spreads wide while no depth passes 201.
const TREE: &str = "par n=2,k=0\nfib(m)=if(m<2)then(m)else(fib(m-1)+fib(m-2))\n\
s(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(fib(n)+s(k))/(1+fib(n)+s(k))\ndone\n";
/// The depth follows the state, which moves at the member's own speed `w`.
const DRIFT: &str =
    "par w=1\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=w+1e-12*s(flr(2048*(1+sin(x))))\ndone\n";

/// A fixed number in [0, 1) for member `i`.
fn u(i: usize) -> f64 {
    ((i as u64 * 2_654_435_761) % 4_294_967_296) as f64 / 4_294_967_296.0
}

/// Depths `base ± spread / 2`, scattered over the members, sorted when `sorted`.
fn depths(n: usize, base: f64, spread: f64, sorted: bool) -> Vec<f64> {
    let mut d: Vec<f64> = (0..n)
        .map(|i| {
            (base - spread / 2.0 + u(i) * spread)
                .floor()
                .clamp(0.0, 4095.0)
        })
        .collect();
    if sorted {
        d.sort_by(f64::total_cmp);
    }
    d
}

/// 90 % of the members at `base`, 10 % scattered over 0 to 4,095.
fn ninety(n: usize, base: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if u(i * 7 + 3) < 0.1 {
                (u(i) * 4095.0).floor()
            } else {
                base
            }
        })
        .collect()
}

struct Case {
    name: &'static str,
    src: &'static str,
    x0: f64,
    params: fn(usize) -> Vec<f64>,
    t1: f64,
}

fn cases() -> Vec<Case> {
    let c = |name, src, x0, params, t1| Case {
        name,
        src,
        x0,
        params,
        t1,
    };
    vec![
        c(
            "drift-scattered",
            DRIFT,
            0.0,
            |n| (0..n).map(|i| 0.5 + 2.5 * u(i)).collect(),
            0.5,
        ),
        c(
            "drift-ordered",
            DRIFT,
            0.0,
            |n| (0..n).map(|i| 0.5 + 2.5 * i as f64 / n as f64).collect(),
            0.5,
        ),
        c(
            "short-sum-scattered",
            SUM,
            1.0,
            |n| depths(n, 2048.0, 2048.0, false),
            0.05,
        ),
        c("short-sum-equal", SUM, 1.0, |n| vec![2048.0; n], 0.05),
        c("ninety-ten", SUM, 1.0, |n| ninety(n, 2048.0), 0.2),
        c("sum-equal", SUM, 1.0, |n| vec![2048.0; n], 0.2),
        c(
            "sum-sorted",
            SUM,
            1.0,
            |n| depths(n, 2048.0, 2048.0, true),
            0.2,
        ),
        c(
            "sum-scattered",
            SUM,
            1.0,
            |n| depths(n, 2048.0, 2048.0, false),
            0.2,
        ),
        c(
            "sum-scattered-narrow",
            SUM,
            1.0,
            |n| depths(n, 256.0, 8.0, false),
            0.2,
        ),
        c("pw-equal", PW, 0.999, |n| vec![2048.0; n], 0.2),
        c(
            "pw-sorted",
            PW,
            0.999,
            |n| depths(n, 2048.0, 2048.0, true),
            0.2,
        ),
        c(
            "pw-scattered",
            PW,
            0.999,
            |n| depths(n, 2048.0, 2048.0, false),
            0.2,
        ),
        c(
            "tree-wide",
            TREE,
            1.0,
            |n| {
                (0..n)
                    .flat_map(|i| {
                        [
                            (2.0 + 13.0 * u(i * 5 + 1)).floor(),
                            (201.0 * u(i * 11 + 2)).floor(),
                        ]
                    })
                    .collect()
            },
            0.2,
        ),
        c("fib-equal", FIB, 1.0, |n| vec![14.0; n], 0.2),
        c("fib-sorted", FIB, 1.0, |n| depths(n, 10.0, 16.0, true), 0.2),
        c(
            "fib-scattered",
            FIB,
            1.0,
            |n| depths(n, 10.0, 16.0, false),
            0.2,
        ),
    ]
}

fn env_list(name: &str) -> Option<Vec<String>> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_owned()).collect())
}

/// The fastest of `repeats` runs of each layout in `stacks`, in milliseconds, with the result
/// of its last run. Each layout runs once untimed first, so its kernels are compiled. The
/// timed runs go in rounds, one run of each layout per round, so a clock or power state that
/// drifts during the measurement falls on every layout alike.
fn timed<const K: usize>(
    device: &DeviceInfo,
    j: &Job,
    stacks: [StackLayout; K],
    repeats: usize,
) -> [(f64, EnsembleResult); K] {
    let cancel = AtomicBool::new(false);
    let jobs = stacks.map(|stack| {
        let mut o = j.clone();
        o.stack = stack;
        o
    });
    let mut out = jobs.each_ref().map(|o| {
        (
            f64::INFINITY,
            run_on(device, o, &cancel).expect("the warm-up run"),
        )
    });
    for _ in 0..repeats {
        for (o, (best, r)) in jobs.iter().zip(out.iter_mut()) {
            let start = Instant::now();
            *r = run_on(device, o, &cancel).expect("a timed run");
            *best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
    }
    out
}

#[test]
#[ignore = "a timing measurement that needs a CUDA device and takes minutes"]
fn auto_against_member_and_warp() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    eprintln!("cuda device 0: {} ({})", device.name, device.driver);
    let members: Vec<usize> = env_list("PC_AT_MEMBERS")
        .map(|v| {
            v.iter()
                .map(|s| s.parse().expect("a member count"))
                .collect()
        })
        .unwrap_or(vec![4096, 10000, 16384]);
    let precisions: Vec<Precision> = env_list("PC_AT_PRECISION")
        .map(|v| {
            v.iter()
                .map(|s| match s.as_str() {
                    "single" => Precision::Single,
                    "double" => Precision::Double,
                    _ => panic!("PC_AT_PRECISION takes single and double"),
                })
                .collect()
        })
        .unwrap_or(vec![Precision::Single, Precision::Double]);
    let repeats: usize = std::env::var("PC_AT_REPEATS")
        .ok()
        .map_or(5, |s| s.parse().expect("a repeat count"));
    let only = env_list("PC_AT_CASES");
    eprintln!(
        "AT case precision members member_ms warp_ms auto_ms auto/member auto/best stack_ran sorted"
    );
    for case in cases() {
        if only
            .as_ref()
            .is_some_and(|o| !o.iter().any(|c| c == case.name))
        {
            continue;
        }
        let model = build_model(case.src);
        for &p in &precisions {
            if p == Precision::Double && !device.fp64 {
                continue;
            }
            for &n in &members {
                let x0 = vec![case.x0; n];
                let params = (case.params)(n);
                let mut j = job(&model, n, &x0, &[], case.t1, Method::Rk4 { dt: 0.01 }, p);
                j.params = Params::PerMember(&params);
                let [(m, rm), (w, rw), (a, ra)] = timed(
                    &device,
                    &j,
                    [StackLayout::Member, StackLayout::Warp, StackLayout::Auto],
                    repeats,
                );
                assert_eq!(
                    bits(&rm.states),
                    bits(&rw.states),
                    "{}: warp bits",
                    case.name
                );
                assert_eq!(
                    bits(&rm.states),
                    bits(&ra.states),
                    "{}: auto bits",
                    case.name
                );
                eprintln!(
                    "AT {} {} {n} {m:.2} {w:.2} {a:.2} {:.3} {:.3} {} {}",
                    case.name,
                    p.name(),
                    a / m,
                    a / m.min(w),
                    ra.label.stack_ran.name(),
                    ra.label.sorted
                );
            }
        }
    }
}

/// The warp over member time of single-chain recursions against the spread of depth inside a
/// warp group, for the constants of the `auto` prediction. `PC_SW_BASES` (default
/// `256,2048`) and `PC_SW_SPREADS` (default `0,4,8,16,32,64,128,512,2048`) set the grid; the
/// members and precisions follow `PC_AT_MEMBERS` and `PC_AT_PRECISION`. Every line is
/// `SW model precision members base spread member_ms warp_ms warp/member`.
#[test]
#[ignore = "a timing measurement that needs a CUDA device and takes minutes"]
fn warp_ratio_against_the_spread() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    let nums = |name: &str, d: &str| -> Vec<f64> {
        env_list(name)
            .unwrap_or(d.split(',').map(str::to_owned).collect())
            .iter()
            .map(|s| s.parse().expect("a number"))
            .collect()
    };
    let members: Vec<usize> = nums("PC_AT_MEMBERS", "4096,16384")
        .iter()
        .map(|&v| v as usize)
        .collect();
    let precisions: Vec<Precision> = match std::env::var("PC_AT_PRECISION").as_deref() {
        Ok("single") => vec![Precision::Single],
        Ok("double") => vec![Precision::Double],
        _ => vec![Precision::Single, Precision::Double],
    };
    let bases = nums("PC_SW_BASES", "256,2048");
    let spreads = nums("PC_SW_SPREADS", "0,4,8,16,32,64,128,512,2048");
    let repeats: usize = std::env::var("PC_AT_REPEATS")
        .ok()
        .map_or(3, |s| s.parse().expect("a repeat count"));
    eprintln!("SW model precision members base spread member_ms warp_ms warp/member");
    for (name, src, x0) in [("sum", SUM, 1.0), ("pw", PW, 0.999)] {
        let model = build_model(src);
        for &p in &precisions {
            if p == Precision::Double && !device.fp64 {
                continue;
            }
            for &n in &members {
                for &b in &bases {
                    for &s in &spreads {
                        let x = vec![x0; n];
                        let params = depths(n, b, s, false);
                        let mut j = job(&model, n, &x, &[], 0.2, Method::Rk4 { dt: 0.01 }, p);
                        j.params = Params::PerMember(&params);
                        let [(m, _), (w, _)] = timed(
                            &device,
                            &j,
                            [StackLayout::Member, StackLayout::Warp],
                            repeats,
                        );
                        eprintln!(
                            "SW {name} {} {n} {b} {s} {m:.2} {w:.2} {:.3}",
                            p.name(),
                            w / m
                        );
                    }
                }
            }
        }
    }
}

/// The CPU cost of the `auto` plan alone (the key pass and the sort) for the equal-depth
/// cases, to split the fixed overhead of `auto` from its per-step cost. Every line is
/// `PA case precision members plan_ms`.
#[test]
#[ignore = "a timing measurement"]
fn plan_auto_alone() {
    use pc_ensemble::Tape;
    use pc_ensemble::order::plan_auto;
    let members: Vec<usize> = env_list("PC_AT_MEMBERS")
        .map(|v| {
            v.iter()
                .map(|s| s.parse().expect("a member count"))
                .collect()
        })
        .unwrap_or(vec![4096, 16384]);
    for case in cases() {
        let model = build_model(case.src);
        let tape = Tape::build(&model).expect("a tape");
        for p in [Precision::Single, Precision::Double] {
            for &n in &members {
                let x0 = vec![case.x0; n];
                let params = (case.params)(n);
                let mut j = job(&model, n, &x0, &[], case.t1, Method::Rk4 { dt: 0.01 }, p);
                j.params = Params::PerMember(&params);
                let cancel = AtomicBool::new(false);
                plan_auto(&tape, &j, &cancel).expect("a plan");
                let mut best = f64::INFINITY;
                for _ in 0..5 {
                    let start = Instant::now();
                    plan_auto(&tape, &j, &cancel).expect("a plan");
                    best = best.min(start.elapsed().as_secs_f64() * 1e3);
                }
                eprintln!("PA {} {} {n} {best:.2}", case.name, p.name());
            }
        }
    }
}

/// The first call of a model, kernel compiles included, under `member` and under `auto`, on a
/// short run of scattered depths. Each layout gets its own copy of the model (a constant that
/// differs in the text), so neither finds the kernels of the other in the cache. It prints one
/// `COLD` line per precision: milliseconds of the first and of a warm call of each layout.
#[test]
#[ignore = "a timing measurement that needs a CUDA device"]
fn cold_first_call_of_auto_and_member() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    let members = 4096;
    let x0: Vec<f64> = (0..members)
        .map(|i| 1.0 + i as f64 / members as f64)
        .collect();
    let d = depths(members, 1024.0, 2048.0, false);
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    }
    eprintln!("COLD precision member_cold member_warm auto_cold auto_warm stack_ran");
    for (k, &p) in precisions.iter().enumerate() {
        let mut ms = Vec::new();
        let mut ran = "";
        for (l, stack) in [StackLayout::Member, StackLayout::Auto]
            .into_iter()
            .enumerate()
        {
            // A different divisor per precision and layout gives a new kernel text.
            let text = format!(
                "par n=0\ns(m)=if(m<1)then(0)else(1+s(m-1))\nx'=-x*(1+s(n))/({}+n)\ndone\n",
                1 + 2 * k + l
            );
            let model = build_model(&text);
            let mut j = job(&model, members, &x0, &[], 0.5, Method::Rk4 { dt: 0.05 }, p);
            j.params = Params::PerMember(&d);
            j.stack = stack;
            let cancel = AtomicBool::new(false);
            for _ in 0..2 {
                let start = Instant::now();
                let r = run_on(&device, &j, &cancel).expect("a run");
                ms.push(start.elapsed().as_secs_f64() * 1e3);
                ran = r.label.stack_ran.name();
            }
        }
        eprintln!(
            "COLD {} {:.1} {:.1} {:.1} {:.1} {ran}",
            p.name(),
            ms[0],
            ms[1],
            ms[2],
            ms[3]
        );
    }
}

/// The first call under `member` and under `auto` on the cases where `member` wins, kernel
/// compiles included, at a size whose launches fall below the timing floor of the judge and
/// at one above it. Each layout gets its own copy of the model (a constant that differs in the
/// text), so neither finds the kernels of the other in the cache. `PC_AT_CASES` and
/// `PC_AT_MEMBERS` choose the cases (default `drift-scattered,tree-wide`) and sizes (default
/// `64,512,4096`). It prints one `COLDCASE` line per case, size and precision: milliseconds of
/// the first and of a warm call of each layout, `auto` over `member` for each, and the kernel
/// compiles of each layout. The constants start from the clock, so a rerun misses the
/// driver's JIT cache as a new model would.
#[test]
#[ignore = "a timing measurement that needs a CUDA device"]
fn cold_first_call_where_member_wins() {
    let state = list_devices()
        .backends
        .iter()
        .find(|(b, _)| *b == Backend::Cuda)
        .map(|(_, s)| s.clone())
        .expect("cuda is listed");
    let device = match state {
        BackendState::Available => find_device(Backend::Cuda, 0).expect("device 0"),
        BackendState::Unavailable(why) => return skip_gpu("cuda", &why),
        BackendState::NotBuilt => return skip_gpu("cuda", "not built into this binary"),
    };
    let names = env_list("PC_AT_CASES")
        .unwrap_or_else(|| vec!["drift-scattered".into(), "tree-wide".into()]);
    let sizes: Vec<usize> = env_list("PC_AT_MEMBERS")
        .map(|v| {
            v.iter()
                .map(|s| s.parse().expect("a member count"))
                .collect()
        })
        .unwrap_or_else(|| vec![64, 512, 4096]);
    let mut precisions = vec![Precision::Single];
    if device.fp64 {
        precisions.push(Precision::Double);
    }
    eprintln!(
        "COLDCASE case members precision member_cold member_warm auto_cold auto_warm \
         cold_ratio warm_ratio stack_ran member_compiles auto_compiles"
    );
    // The tags start from the clock, so a second process does not send the kernel texts of
    // the first one again and get them from the driver's JIT cache.
    let mut tag = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() % 1_000_000);
    for case in cases()
        .into_iter()
        .filter(|c| names.iter().any(|n| n == c.name))
    {
        for &members in &sizes {
            let params = (case.params)(members);
            let x0 = vec![case.x0; members];
            for &p in &precisions {
                let mut ms = Vec::new();
                let mut compiles = Vec::new();
                let mut ran = "";
                for stack in [StackLayout::Member, StackLayout::Auto] {
                    // A constant that differs per run gives a new kernel text; at 1e-30 it
                    // does not change the sum it is added to.
                    tag += 1;
                    let text = case.src.replace("\ndone", &format!("+{tag}e-30*x\ndone"));
                    let model = build_model(&text);
                    let mut j = job(
                        &model,
                        members,
                        &x0,
                        &[],
                        case.t1,
                        Method::Rk4 { dt: 0.01 },
                        p,
                    );
                    j.params = Params::PerMember(&params);
                    j.stack = stack;
                    let cancel = AtomicBool::new(false);
                    let before = pc_ensemble::cuda::kernel_compiles();
                    for _ in 0..2 {
                        let start = Instant::now();
                        let r = run_on(&device, &j, &cancel).expect("a run");
                        ms.push(start.elapsed().as_secs_f64() * 1e3);
                        ran = r.label.stack_ran.name();
                    }
                    compiles.push(pc_ensemble::cuda::kernel_compiles() - before);
                }
                eprintln!(
                    "COLDCASE {} {members} {} {:.1} {:.1} {:.1} {:.1} {:.3} {:.3} {ran} {} {}",
                    case.name,
                    p.name(),
                    ms[0],
                    ms[1],
                    ms[2],
                    ms[3],
                    ms[2] / ms[0],
                    ms[3] / ms[1],
                    compiles[0],
                    compiles[1]
                );
            }
        }
    }
}
