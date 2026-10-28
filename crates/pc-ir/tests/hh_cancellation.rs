//! Cancellation in the Hodgkin-Huxley directional derivatives (task 14).
//!
//! On four of the vectors that `check_model` draws for `hodgkin_huxley.ode` (seed 1, 6
//! points), the DAG and the fallback differ by more than `1e-13 * s` (`s` is the largest
//! magnitude of the output vector) although both are within a few ulp of the term
//! magnitude. The running error bound of `eval_err` covers them. Each vector below is one
//! failing output, with the exact `x`, `p` and `v` of the draw.

use pc_ir::check::{allowed, close, close_err, scale};
use pc_ir::eval::eval_err;
use pc_ir::{EntryKind, Fallback, Slot, lower};
use pc_ode::{FsLoader, SourceMap, load};
use std::path::Path;

type Vector = (
    &'static [Slot],
    usize,
    &'static [f64],
    &'static [f64],
    &'static [f64],
);

const VECTORS: [Vector; 4] = [
    (
        &[Slot::P, Slot::P, Slot::Full, Slot::P],
        0,
        &[
            -64.58381491374249,
            0.5226571802330722,
            0.756499313539546,
            0.3658381582753055,
        ],
        &[
            9.1118738929838,
            0.9702068964161106,
            112.89713270996403,
            33.49727238652892,
            0.2959940980511429,
            53.97959973231453,
            -71.8981381907624,
            -51.79737799603904,
            0.6298523334291815,
        ],
        &[
            0.6686172156762191,
            0.5638415964691326,
            0.7367627979319054,
            0.4484776772111865,
            0.3886259590921879,
            -0.8777642782793216,
            0.8688524701777831,
            -0.17354728689193988,
            0.935557224666699,
            0.2742638143710039,
            -0.13545922412402156,
            -0.02400218136151211,
            -0.11918971711207704,
            0.6646504150138455,
            0.1831238663904029,
            0.8225517002613116,
            0.8973219106452162,
            0.7221137875147703,
            -0.751038694619804,
            0.8385660385344105,
            -0.6813931075085167,
            0.7532538140807061,
            -0.1779030846754288,
            -0.8946027327284896,
            0.4016567230414243,
            0.365296802331357,
            -0.32437446012498494,
            0.2927225316084705,
            0.4126125340894575,
            0.6874034886009603,
            -0.26388564463894637,
            0.13312693011220378,
            0.14341031723242037,
            -0.8613662862362303,
            0.29902411094009773,
            -0.7113315043556478,
        ],
    ),
    (
        &[Slot::U, Slot::P, Slot::Full, Slot::P, Slot::P],
        0,
        &[
            -65.11546892752453,
            -0.2427260144336117,
            0.7992730451253531,
            0.0732975537960775,
        ],
        &[
            9.91589540411643,
            1.0442810223678662,
            115.30834698229383,
            34.83241321091939,
            0.31375896835070227,
            51.87861945283443,
            -81.08934178705479,
            -52.82167251967595,
            0.43927676721767195,
        ],
        &[
            0.9779904439346387,
            0.2011368937003788,
            0.937484731400613,
            -0.01403026683526365,
            -0.9440824145597977,
            -0.37527616144904363,
            0.36680657116774773,
            0.8106469265248253,
            -0.8607158674768134,
            0.2962101900171459,
            -0.9834777275960607,
            0.16489130131147234,
            0.17236577827336141,
            -0.28200901269867273,
            -0.38489936565544514,
            -0.16093053266038115,
            0.7525645425125453,
            0.10285949561785923,
            -0.08486410376770914,
            -0.77229434141369,
            -0.8520293175502309,
            -0.5216147305068701,
            -0.6490152958806521,
            -0.5423750360560815,
            0.46354108300047314,
            0.330696130352496,
            -0.9963576682835849,
            -0.8808528077732509,
            0.9745330376171726,
            0.40601982932045044,
            0.869917698283176,
            -0.24196652806209107,
            0.5665537342954159,
            0.43368413470278666,
            -0.44853592361321026,
            0.7789915658769304,
            -0.9895663392509919,
            -0.16760792168317273,
            0.8039433185349898,
            0.5019349950583947,
        ],
    ),
    (
        &[Slot::U, Slot::Full, Slot::P, Slot::U, Slot::P],
        0,
        &[
            -64.92058244288141,
            0.4195174740623034,
            0.5695037837811329,
            0.8157975900559846,
        ],
        &[
            9.57705389127218,
            0.981394927929472,
            121.82760155277771,
            33.0620078041178,
            0.28876106564300247,
            54.53013086794706,
            -77.63296714773747,
            -56.07747309661514,
            0.9119792094629998,
        ],
        &[
            -0.1440597600212543,
            0.7243156058384919,
            0.588010529918201,
            0.7431989348237975,
            -0.9404183109424822,
            0.27849016795154813,
            0.1454984673099693,
            -0.9025808031328137,
            0.4932839077151656,
            -0.3942223279405419,
            0.3232859477189052,
            -0.7672295501225428,
            -0.4477670827460709,
            -0.5936812471352935,
            0.833115812083099,
            -0.9798853503388365,
            -0.45992527862647226,
            0.9235203134971708,
            -0.9504267935495161,
            -0.7822922573101259,
            -0.46021794312615083,
            0.15786251448174848,
            -0.6279612555145935,
            0.8361598842067879,
            -0.21669185381712808,
            0.26697352718257283,
            -0.8713354118510928,
            0.25493921277741305,
            -0.11970523365937913,
            -0.5402096586886427,
            0.13643441170413428,
            -0.42070788262333103,
            -0.3842380534692904,
            0.015513644631967916,
            0.9411661374728311,
            0.10609325121510249,
        ],
    ),
    (
        &[Slot::Full, Slot::U, Slot::P, Slot::Full, Slot::P],
        0,
        &[
            -64.98388259570585,
            0.3224702217492978,
            0.9682832115377623,
            0.6361220017965562,
        ],
        &[
            9.573806056761807,
            0.9117287640835552,
            120.55168205321137,
            34.04125177765791,
            0.3055112411919278,
            46.47498468695268,
            -70.77567430664722,
            -55.9901521152549,
            0.7096885233277589,
        ],
        &[
            0.03037378283693748,
            0.7308085403051552,
            0.13745556343292686,
            -0.7781978874083206,
            -0.18209738110778906,
            0.8718834488718954,
            0.4044970627365285,
            -0.8878545498579395,
            0.17925973054091782,
            -0.8035714028687349,
            -0.901560733097629,
            -0.8706967129962309,
            0.09396334993735378,
            -0.6411177883495747,
            -0.8028527499849869,
            0.7896718071963753,
            -0.5493997903995447,
            -0.612047709615567,
            0.7052593963505747,
            -0.3862681489673341,
            0.04723493444064175,
            0.028466096681920128,
            0.5082367780044461,
            -0.9002934991951834,
            0.49735972850540944,
            0.4127578145181936,
            0.7772537516495932,
            -0.08892511008076731,
            -0.39528368370846323,
            0.6999085849756197,
            -0.3728133559057776,
            0.4759928468226762,
            -0.4329441356830175,
            0.12814418000600192,
            -0.16716637820756275,
            0.4622522773469593,
            -0.7299630785830835,
            -0.06987525106470538,
            0.3141762531216299,
            0.7568738317437507,
            0.5159968733584279,
            -0.4173798592031077,
            -0.9156233458025376,
            -0.08382881342088089,
        ],
    ),
];

struct Case {
    got: f64,
    want: f64,
    s: f64,
    err: f64,
}

/// Runs the entry of each vector and returns the failing output with its bound.
fn cases() -> Vec<Case> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-ode/tests/corpus/hodgkin_huxley.ode");
    let src = std::fs::read_to_string(&path).expect("read hodgkin_huxley.ode");
    let mut map = SourceMap::new();
    let top = map.add("hodgkin_huxley.ode", &src);
    let l = load(top, &mut map, &mut FsLoader::new());
    let mut ir = lower(&l.expanded, &l.info).expect("lower");
    let fb = Fallback::new(&l.expanded, &l.info, &ir);
    VECTORS
        .iter()
        .map(|&(slots, i, x, p, v)| {
            let kind = EntryKind::Dir(slots.to_vec());
            let e = ir.entry(&kind).expect("entry");
            let mut got = vec![f64::NAN; e.outputs.len()];
            let mut err = vec![f64::NAN; e.outputs.len()];
            eval_err(&ir.dag, &e.outputs, x, p, v, &mut got, &mut err);
            let mut want = vec![f64::NAN; e.outputs.len()];
            fb.eval(&kind, x, p, v, &mut want);
            Case {
                got: got[i],
                want: want[i],
                s: scale(&want),
                err: err[i],
            }
        })
        .collect()
}

#[test]
fn the_old_rule_fails_and_the_running_error_bound_passes() {
    for (k, c) in cases().iter().enumerate() {
        assert!(!close(c.got, c.want, c.s), "vector {k} passes the old rule");
        assert!(
            close_err(c.got, c.want, c.s, c.err),
            "vector {k}: diff {:e}, allowed {:e}, err {:e}",
            (c.got - c.want).abs(),
            allowed(c.got, c.want, c.s, c.err),
            c.err
        );
        // The bound is far from tight here, so the margin is wide.
        assert!((c.got - c.want).abs() < 0.1 * allowed(c.got, c.want, c.s, c.err));
    }
}

#[test]
fn a_sign_flip_or_a_one_percent_error_in_the_principal_output_still_fails() {
    for (k, c) in cases().iter().enumerate() {
        // The failing output is the one of largest magnitude, so `want == +-s`.
        assert_eq!(c.want.abs(), c.s, "vector {k} is not the principal output");
        for (name, w) in [("sign flip", -c.want), ("relative 1e-2", c.want * 1.01)] {
            assert!(
                !close_err(c.got, w, c.s, c.err),
                "vector {k}: {name} passes"
            );
        }
    }
}

#[test]
fn small_perturbations_fail_wherever_the_bound_is_below_them() {
    let cs = cases();
    // The allowed difference is 8 eps E, a bound of the rounding error of the DAG
    // evaluation. It is about 2e-10 for vector 0, which is 1.6e-9 of that output, so a
    // relative 1e-10 change in the fallback value is below the bound there and passes. A
    // relative 1e-8 change fails on every vector, and 1e-10 fails on the other three.
    for (k, c) in cs.iter().enumerate() {
        let w = c.want * (1.0 + 1e-8);
        assert!(
            !close_err(c.got, w, c.s, c.err),
            "vector {k}: relative 1e-8 passes"
        );
        let w = c.want * (1.0 + 1e-10);
        let passes = close_err(c.got, w, c.s, c.err);
        assert_eq!(passes, k == 0, "vector {k}: relative 1e-10 gives {passes}");
    }
    let c = &cs[0];
    let rel = allowed(c.got, c.want, c.s, c.err) / c.want.abs();
    assert!(
        (1e-10..1e-8).contains(&rel),
        "undetected relative error {rel:e}"
    );
}
