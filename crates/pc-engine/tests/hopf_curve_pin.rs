//! Bit pin for the ODE Hopf curve.
//!
//! P34 slice 3 added opt-in ZH and HH coefficients to the delay Hopf run and touched the
//! Hopf curve entry points and the run front end. The ODE `H` curve was meant to stay as it
//! was. These tests run three ODE Hopf curves, hash every `f64` of every point and special
//! point, and compare the hash with a golden value.
//!
//! The hash covers the raw bits, so a change in the last digit of any value fails the test.
//! The values were produced on an Intel Core Ultra 7 255H. If another CPU gives a different
//! last bit from the same code, the failure message prints the new hash and the point count;
//! that is a portability finding to look at, not a reason to update the number blindly.
//!
//! How the goldens relate to slice 3: the engine before slice 3 (3a1673e) and the slice 3
//! commit (10b9d07) give the same hash for all three curves. The Bautin and double-Hopf
//! goldens below are those values and have not moved since. The Brusselator hash moved once
//! after slice 3, at bcbcea6 (column scaling in `start_tangent`, shared by every curve): 2 of
//! the 520 numbers of its curve change by 4.4e-16 in the point and 42 tangent entries change by
//! at most 3.3e-16. Before that commit the hash was 3147689500935688442. The golden below is
//! the value after bcbcea6, so a later change shows up as a failure.

mod common;

use common::load;
use pc_engine::{
    Branch, CurveSetup, EpSetup, Evaluator, ModelEval, Options, Point, SpecialPoint, build,
    continue_ep, continue_hopf_curve,
};
use std::sync::atomic::AtomicBool;

/// FNV-1a over little-endian bytes. The numbers it produces are only compared with each
/// other, so any fixed hash would do.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Digest(0xcbf2_9ce4_8422_2325)
    }
    fn byte(&mut self, b: u8) {
        self.0 ^= u64::from(b);
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fn word(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.byte(b);
        }
    }
    fn f(&mut self, v: f64) {
        self.word(v.to_bits());
    }
    fn fs(&mut self, v: &[f64]) {
        self.word(v.len() as u64);
        for x in v {
            self.f(*x);
        }
    }
    fn text(&mut self, s: &str) {
        self.word(s.len() as u64);
        for b in s.bytes() {
            self.byte(b);
        }
    }
    fn opt(&mut self, v: Option<f64>) {
        match v {
            Some(x) => {
                self.byte(1);
                self.f(x);
            }
            None => self.byte(0),
        }
    }
    fn point(&mut self, p: &Point) {
        self.fs(&p.u);
        self.f(p.alpha);
        self.fs(&p.tangent);
        self.word(p.eigenvalues.len() as u64);
        for (re, im) in &p.eigenvalues {
            self.f(*re);
            self.f(*im);
        }
        self.fs(&p.tests);
        self.fs(&p.codim2);
        self.f(p.h);
        self.word(p.newton_iters as u64);
        self.fs(&p.mesh);
    }
    fn special(&mut self, s: &SpecialPoint) {
        self.text(&s.label);
        self.word(s.index as u64);
        self.point(&s.point);
        self.opt(s.extras.coefficient);
        self.opt(s.extras.coefficient2);
        self.opt(s.extras.omega);
        match &s.extras.normalization {
            Some(n) => {
                self.byte(1);
                self.text(n);
            }
            None => self.byte(0),
        }
        self.word(s.extras.values.len() as u64);
        for (name, v) in &s.extras.values {
            self.text(name);
            self.f(*v);
        }
        self.byte(u8::from(s.extras.refined));
    }
}

/// Points, specials and the hash of the whole branch, including stop reason and warnings.
fn summary(br: &Branch) -> (usize, usize, u64) {
    let mut d = Digest::new();
    d.word(br.points.len() as u64);
    for p in &br.points {
        d.point(p);
    }
    d.word(br.specials.len() as u64);
    for s in &br.specials {
        d.special(s);
    }
    d.text(&format!("{:?}", br.stop));
    d.word(br.warnings.len() as u64);
    for w in &br.warnings {
        d.text(w);
    }
    (br.points.len(), br.specials.len(), d.0)
}

fn model_of(name: &str) -> Box<dyn ModelEval> {
    let (model, info) = load(name);
    build(&model, &info, Evaluator::Jit).unwrap()
}

/// The Hopf point of the equilibrium branch in parameter 0 of `name`, as the start of a curve in
/// parameters (0, 1). `p` is the full parameter list and `u0` the start state.
fn hopf_start(
    m: &mut dyn ModelEval,
    p: Vec<f64>,
    u0: Vec<f64>,
    bounds: (f64, f64),
    free: usize,
) -> CurveSetup {
    let cancel = AtomicBool::new(false);
    let opts = Options {
        bounds,
        ..Options::default()
    };
    let ep = continue_ep(
        m,
        &EpSetup::new(free, p.clone(), u0),
        &opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let h: Vec<_> = ep.specials.iter().filter(|s| s.label == "H").collect();
    assert_eq!(h.len(), 1, "specials {:?}", ep.specials);
    let mut q = p;
    q[free] = h[0].point.alpha;
    CurveSetup::new([0, 1], q, h[0].point.u.clone())
}

fn labels(br: &Branch) -> Vec<&str> {
    br.specials.iter().map(|s| s.label.as_str()).collect()
}

fn check(name: &str, got: (usize, usize, u64), want: (usize, usize, u64)) {
    assert_eq!(
        got, want,
        "{name}: (points, specials, hash) is {got:#x?}, the golden is {want:#x?}"
    );
}

#[test]
fn the_brusselator_hopf_curve_is_unchanged_bit_for_bit() {
    let mut m = model_of("brusselator");
    let cancel = AtomicBool::new(false);
    let a0 = 1.0;
    let ep_opts = Options {
        bounds: (0.5, 4.0),
        ..Options::default()
    };
    let ep = continue_ep(
        &mut *m,
        &EpSetup::new(1, vec![a0, 1.2], vec![a0, 1.2 / a0]),
        &ep_opts,
        &cancel,
        &mut |_| {},
    )
    .unwrap();
    let h: Vec<_> = ep.specials.iter().filter(|s| s.label == "H").collect();
    assert_eq!(h.len(), 1);
    let setup = CurveSetup::new([0, 1], vec![a0, h[0].point.alpha], h[0].point.u.clone());
    let opts = Options {
        bounds: (1.5, 5.0),
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    check("brusselator", summary(&br), BRUSSELATOR);
}

#[test]
fn the_bautin_hopf_curve_with_its_gh_point_is_unchanged_bit_for_bit() {
    let mut m = model_of("bautin");
    let cancel = AtomicBool::new(false);
    let setup = hopf_start(
        &mut *m,
        vec![-0.5, 0.5, 1.0],
        vec![0.0, 0.0],
        (-1.0, 1.0),
        0,
    );
    let opts = Options {
        direction: -1.0,
        bounds: (-0.5, 0.8),
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    assert_eq!(labels(&br), ["GH"], "stop {:?}", br.stop);
    check("bautin", summary(&br), BAUTIN);
}

#[test]
fn the_double_hopf_curve_with_its_hh_point_is_unchanged_bit_for_bit() {
    let mut m = model_of("hh");
    let cancel = AtomicBool::new(false);
    let mut p = vec![-0.3, 0.4, 2.0, 0.9];
    p.extend_from_slice(&[-0.8, 0.3, 1.7, -0.4, 0.6, 0.5, -1.1, 0.2]);
    let setup = hopf_start(&mut *m, p, vec![0.0; 4], (-0.5, 0.5), 0);
    let opts = Options {
        bounds: (-0.5, 1.0),
        direction: -1.0,
        ..Options::default()
    };
    let br = continue_hopf_curve(&mut *m, &setup, &opts, &cancel, &mut |_| {}).unwrap();
    assert_eq!(labels(&br), ["HH"], "stop {:?}", br.stop);
    check("hh", summary(&br), HH);
}

// Golden values: (points, specials, hash). BAUTIN and HH were taken before slice 3 and still
// hold. BRUSSELATOR was taken after bcbcea6, see the header.
const BRUSSELATOR: (usize, usize, u64) = (52, 0, 12_734_820_752_016_515_302);
const BAUTIN: (usize, usize, u64) = (17, 1, 13_246_036_168_963_318_012);
const HH: (usize, usize, u64) = (16, 1, 15_915_677_381_896_240_780);
