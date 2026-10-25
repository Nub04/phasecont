use crate::complex::dir_complex;
use crate::entry::{EntryKind, Slot};
use crate::eval::eval;
use crate::interp::Fallback;
use crate::lower::{IrModel, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn setup(src: &str) -> (Fallback, IrModel) {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let errs: Vec<_> = l
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "check errors: {errs:?}");
    let ir = match lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    };
    (Fallback::new(&l.expanded, &l.info, &ir), ir)
}

fn ulp(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    f64::from_bits(x.abs().to_bits() + 1) - x.abs()
}

/// Section 3 tolerance for one output vector; `want` is the reference.
fn within(got: &[f64], want: &[f64]) -> Result<(), String> {
    within_floor(got, want, 0.0)
}

/// As [`within`], with the scale `s` raised to at least `floor`. Used where the true value
/// is 0 and the reference holds rounding noise (second derivative of `mod` in its divisor).
fn within_floor(got: &[f64], want: &[f64], floor: f64) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("length {} vs {}", got.len(), want.len()));
    }
    let s = want
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(floor)
        .max(f64::MIN_POSITIVE);
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        if (a.is_nan() && b.is_nan()) || a == b {
            continue;
        }
        let diff = (a - b).abs();
        let m = a.abs().max(b.abs());
        if diff <= 4.0 * ulp(m) || diff <= 1e-13 * s {
            continue;
        }
        return Err(format!("entry {i}: {a:?} vs {b:?} (scale {s:e})"));
    }
    Ok(())
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn dir_out(ir: &mut IrModel, slots: &[Slot], x: &[f64], p: &[f64], v: &[f64]) -> Vec<f64> {
    let e = match ir.entry(&EntryKind::Dir(slots.to_vec())) {
        Ok(e) => e,
        Err(e) => panic!("entry {slots:?} failed: {e}"),
    };
    let mut out = vec![0.0; e.outputs.len()];
    eval(&ir.dag, &e.outputs, x, p, v, &mut out);
    out
}

fn all_mixes(k: usize) -> Vec<Vec<Slot>> {
    let mut all = vec![vec![]];
    for _ in 0..k {
        let mut next = Vec::new();
        for m in &all {
            for s in [Slot::U, Slot::P, Slot::Full] {
                let mut m2: Vec<Slot> = m.clone();
                m2.push(s);
                next.push(m2);
            }
        }
        all = next;
    }
    all
}

/// Every slot mix for k = 1..=kmax against the fallback at `npts` random points.
fn check_model(src: &str, xr: (f64, f64), p_base: &[f64], kmax: usize, npts: usize, seed: u64) {
    check_model_floor(src, xr, p_base, kmax, npts, seed, 0.0);
}

fn check_model_floor(
    src: &str,
    xr: (f64, f64),
    p_base: &[f64],
    kmax: usize,
    npts: usize,
    seed: u64,
    floor: f64,
) {
    let (fb, mut ir) = setup(src);
    let (n, n_p) = (ir.n(), ir.n_p());
    assert_eq!(p_base.len(), n_p + 1);
    let mut rng = Rng(seed);
    for k in 1..=kmax {
        for slots in all_mixes(k) {
            for _ in 0..npts {
                let x: Vec<f64> = (0..n).map(|_| rng.range(xr.0, xr.1)).collect();
                let p: Vec<f64> = p_base.iter().map(|&b| b * rng.range(0.9, 1.1)).collect();
                let len: usize = slots.iter().map(|s| s.len(n, n_p)).sum();
                let v: Vec<f64> = (0..len).map(|_| rng.range(-1.0, 1.0)).collect();
                let got = dir_out(&mut ir, &slots, &x, &p, &v);
                let mut want = vec![0.0; n];
                fb.eval(&EntryKind::Dir(slots.clone()), &x, &p, &v, &mut want);
                if let Err(m) = within_floor(&got, &want, floor) {
                    panic!("{slots:?} at x={x:?} p={p:?} v={v:?}: {m}");
                }
            }
        }
    }
}

#[test]
fn polynomial_all_mixes_k1_to_5() {
    check_model(
        "par a=1.3, b=0.7\nx'=a*x*y-x^3+b*y^2-x\ny'=x^2*y-y^4+a*b*x\ndone\n",
        (-1.0, 1.0),
        &[1.3, 0.7, 0.0],
        5,
        2,
        11,
    );
}

#[test]
fn transcendental_all_mixes_k1_to_5() {
    check_model(
        "x'=sin(x)*cos(y)+tan(y/3)+asin(x/4)+acos(y/4)+atan(x*y)+sinh(x)+cosh(y)+tanh(x*y)\n\
         y'=exp(x)*ln(1+y*y)+log10(2+x*x)+sqrt(1+x*x)+abs(x-y)+erf(x)+erfc(y)+lgamma(2+x*x)\
         +heav(x)+sign(y)+flr(x)+ceil(y)\n\
         done\n",
        (0.2, 0.9),
        &[0.0],
        5,
        1,
        12,
    );
}

#[test]
fn pow_and_division_all_mixes_k1_to_5() {
    check_model(
        "par a=2, b=0.5\n\
         x'=(1+x*x)^1.5+(1+y*y)^a+2^x+x^3+y^(-2)+(1+x*x)^(0.5+y*y)+x^20+x/(1+y*y)\
         +(y+2)/(x*x+1)/(a+y)\n\
         y'=(2+x)^b*y+(3+y)**a+x^a*y^b+x^(y+0.3)\n\
         done\n",
        (0.4, 1.4),
        &[2.0, 0.5, 0.0],
        5,
        1,
        13,
    );
}

#[test]
fn branches_and_special_functions_all_mixes_k1_to_5() {
    // The reference holds rounding noise where the true value is 0 (second derivative of `mod`
    // in its divisor) or cancels heavily (order-5 `atan2`, see the closed-form test below);
    // the scale floor of 1 suits these O(1) outputs.
    check_model_floor(
        "par a=2, b=0.5\n\
         x'=atan2(y,x)+mod(x*7,a+1)+min(x,y)+max(x,b*y)+besselj(2,x)+bessely(1,abs(x)+0.5)\n\
         y'=if(x>y)then(x*a)else(y*b)+if(x<0.5&y>0)then(sin(y))else(x*x)\n\
         done\n",
        (0.3, 0.8),
        &[2.0, 0.5, 0.0],
        5,
        2,
        14,
        1.0,
    );
}

#[test]
fn tables_all_mixes_k1_to_5() {
    check_model(
        "par a=1.5\ntable h % 9 -2 2 sin(2*t)\n\
         x'=h(x)+h(y)*x\ny'=-y+a*x*h(x*y)\ndone\n",
        (0.3, 0.9),
        &[1.5, 0.0],
        5,
        2,
        15,
    );
}

/// Non-autonomous with three parameters: `t` is never seeded, so P and Full slots carry
/// no time component and the result still equals the fallback.
#[test]
fn nonautonomous_parameter_model_all_mixes_k1_to_5() {
    check_model(
        "par a=1.2, b=-0.4, c=0.9\n\
         x'=a*x*sin(t)+b*y*y+c*exp(-x*a)+cos(t)*x\n\
         y'=x*y*b-c*a*y+t*x^2\n\
         done\n",
        (0.3, 1.1),
        &[1.2, -0.4, 0.9, 0.7],
        5,
        2,
        16,
    );
}

#[test]
fn pow_zero_base_matches_the_interpreter() {
    // pow(0, 3.5) = 0 with finite x-partials up to order 3 and `ln 0` in the y-partials,
    // which the interpreter and the select wrap both drop. (Order 4 in x needs x^-0.5 = inf,
    // where the two evaluators may differ in inf versus NaN.)
    let (fb, mut ir) = setup("x'=x^y\ny'=x^(y-0.5)*y\ndone\n");
    for k in 1..=3usize {
        for slots in all_mixes(k)
            .into_iter()
            .filter(|s| s.iter().all(|t| *t == Slot::U))
        {
            let len: usize = slots.iter().map(|s| s.len(2, 0)).sum();
            let v: Vec<f64> = (0..len).map(|i| 0.3 + 0.17 * i as f64).collect();
            let (x, p) = ([0.0, 3.5], [0.0]);
            let got = dir_out(&mut ir, &slots, &x, &p, &v);
            let mut want = vec![0.0; 2];
            fb.eval(&EntryKind::Dir(slots.clone()), &x, &p, &v, &mut want);
            if let Err(m) = within(&got, &want) {
                panic!("{slots:?}: {m}");
            }
        }
    }
}

#[test]
fn lgamma_order_five_stays_finite() {
    let (fb, mut ir) = setup("x'=lgamma(2+x*x)\ndone\n");
    let slots = vec![Slot::U; 5];
    let v = [0.9, -0.4, 0.7, 0.3, -0.8];
    let got = dir_out(&mut ir, &slots, &[0.6], &[0.0], &v);
    assert!(got[0].is_finite(), "{got:?}");
    let mut want = [0.0];
    fb.eval(&EntryKind::Dir(slots), &[0.6], &[0.0], &v, &mut want);
    assert!(within(&got, &want).is_ok(), "{got:?} vs {want:?}");
}

// ---- hand-derived values ----

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn pack(dirs: &[&[f64]]) -> Vec<f64> {
    dirs.iter().flat_map(|d| d.iter().copied()).collect()
}

#[test]
fn fold_f_uu_is_two() {
    // x' = a + x^2 gives f_uu = 2, so B(v, w) = 2 v w.
    let (_fb, mut ir) = setup("par a=0.4\nx'=a+x^2\ndone\n");
    let slots = [Slot::U, Slot::U];
    for (x, v, w) in [(0.3, 1.0, 1.0), (-1.7, 0.5, -2.0), (0.0, 3.0, 0.25)] {
        let out = dir_out(&mut ir, &slots, &[x], &[0.4, 0.0], &[v, w]);
        assert!(close(out[0], 2.0 * v * w, 1e-15), "{out:?}");
    }
    // The second derivative of x^2 does not depend on x: the output has no state input.
    let e = ir.entry(&EntryKind::Dir(slots.to_vec())).unwrap();
    let a = dir_out(&mut ir, &slots, &[0.1], &[0.4, 0.0], &[1.0, 1.0]);
    let b = dir_out(&mut ir, &slots, &[9.0], &[0.4, 0.0], &[1.0, 1.0]);
    assert_eq!(a, b);
    assert_eq!(e.outputs.len(), 1);
}

/// Hopf: x' = a x - y - x r2, y' = x + a y - y r2 with r2 = x^2 + y^2.
/// With g(u) = -(u.u) u (the cubic part) the multilinear forms are, for each i:
///   Dg_i[v]       = -v_i (u.u) - 2 u_i (u.v)
///   D2g_i[v,w]    = -2 [v_i (u.w) + w_i (u.v) + u_i (v.w)]
///   D3g_i[v,w,z]  = -2 [v_i (w.z) + w_i (v.z) + z_i (v.w)]
/// The linear part a x - y, x + a y gives A v and, in the parameter slot, f_a = (x, y), so
/// D2f[(da), (v)] = da * v. The form B vanishes at u = 0 and C is constant.
#[test]
fn hopf_b_c_and_mixed_entries() {
    let (_fb, mut ir) = setup("par a=0.3\nx'=a*x-y-x*(x*x+y*y)\ny'=x+a*y-y*(x*x+y*y)\ndone\n");
    let a = 0.3;
    let p = [a, 0.0];
    let (v, w, z) = ([0.7, 0.2], [-0.5, 1.1], [0.9, 0.4]);
    let u = [0.3, -0.4];
    let uu = dot(u, u);

    // Order 1.
    let got = dir_out(&mut ir, &[Slot::U], &u, &p, &v);
    let want = [
        a * v[0] - v[1] + (-v[0] * uu - 2.0 * u[0] * dot(u, v)),
        v[0] + a * v[1] + (-v[1] * uu - 2.0 * u[1] * dot(u, v)),
    ];
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");

    // Order 2 at the origin: B = 0.
    let got = dir_out(&mut ir, &[Slot::U; 2], &[0.0, 0.0], &p, &pack(&[&v, &w]));
    assert_eq!(got, vec![0.0, 0.0]);
    // Order 2 at u.
    let got = dir_out(&mut ir, &[Slot::U; 2], &u, &p, &pack(&[&v, &w]));
    let mut want = [0.0; 2];
    for i in 0..2 {
        want[i] = -2.0 * (v[i] * dot(u, w) + w[i] * dot(u, v) + u[i] * dot(v, w));
    }
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");

    // Order 3: C is constant.
    let mut want = [0.0; 2];
    for i in 0..2 {
        want[i] = -2.0 * (v[i] * dot(w, z) + w[i] * dot(v, z) + z[i] * dot(v, w));
    }
    for at in [[0.0, 0.0], u] {
        let got = dir_out(&mut ir, &[Slot::U; 3], &at, &p, &pack(&[&v, &w, &z]));
        assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");
    }
    // Order 4 and above vanish.
    let got = dir_out(&mut ir, &[Slot::U; 4], &u, &p, &pack(&[&v, &w, &z, &v]));
    assert_eq!(got, vec![0.0, 0.0]);

    // Mixed entries: A_alpha v = D2f[(da), (v)] = da * v, in either order, with a P slot and
    // with a Full slot (u part v, parameter part da).
    let da = 1.7;
    let want = [da * v[0], da * v[1]];
    let got = dir_out(&mut ir, &[Slot::P, Slot::U], &u, &p, &pack(&[&[da], &v]));
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");
    let got = dir_out(&mut ir, &[Slot::U, Slot::P], &u, &p, &pack(&[&v, &[da]]));
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");
    // Full slot (v, 0) with P slot da: same value. Full slot (0, db) with U slot v: db * v.
    let got = dir_out(
        &mut ir,
        &[Slot::Full, Slot::Full],
        &u,
        &p,
        &pack(&[&[v[0], v[1], 0.0], &[0.0, 0.0, da]]),
    );
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");
    // Parameter-only second derivative is zero (f is linear in a).
    let got = dir_out(&mut ir, &[Slot::P, Slot::P], &u, &p, &[1.0, 2.0]);
    assert_eq!(got, vec![0.0, 0.0]);
}

/// Bautin: x' = a x - y + s x r2 + c x r2^2, y' = x + a y + s y r2 + c y r2^2.
/// The cubic part s (u.u) u gives C_i = 2 s [v_i (w.z) + w_i (v.z) + z_i (v.w)] (Hopf is
/// s = -1). The quintic part h = c (u.u)^2 u: the coefficient of t1...t5 in
/// q(u)^2 u_i with u = sum t_j v_j is, summing over the vector m that supplies u_i and the
/// three pairings (jk)(ln) of the other four, 8 (vj.vk)(vl.vn) v_m,i. So
/// D5h_i = 8 c sum_m v_m,i sum_pairings (.)(.). Check: all five equal gives
/// 8 * 5 * 3 = 120 = 5! times c |v|^4 v. All derivatives of order 2 and 4 vanish at 0.
#[test]
fn bautin_c_and_fifth_order() {
    let (_fb, mut ir) = setup(
        "par a=0.3, s=-0.8, c=0.25\n\
         x'=a*x-y+s*x*(x*x+y*y)+c*x*(x*x+y*y)^2\n\
         y'=x+a*y+s*y*(x*x+y*y)+c*y*(x*x+y*y)^2\n\
         done\n",
    );
    let (s, c) = (-0.8, 0.25);
    let p = [0.3, s, c, 0.0];
    let o = [0.0, 0.0];
    let vs: [[f64; 2]; 5] = [
        [0.7, 0.2],
        [-0.5, 1.1],
        [0.9, 0.4],
        [0.3, -0.6],
        [-0.2, 0.8],
    ];

    for k in [2usize, 4] {
        let dirs: Vec<&[f64]> = vs.iter().take(k).map(|d| &d[..]).collect();
        let got = dir_out(&mut ir, &vec![Slot::U; k], &o, &p, &pack(&dirs));
        assert_eq!(got, vec![0.0, 0.0], "order {k}");
    }

    let (v, w, z) = (vs[0], vs[1], vs[2]);
    let mut want = [0.0; 2];
    for i in 0..2 {
        want[i] = 2.0 * s * (v[i] * dot(w, z) + w[i] * dot(v, z) + z[i] * dot(v, w));
    }
    let got = dir_out(&mut ir, &[Slot::U; 3], &o, &p, &pack(&[&v, &w, &z]));
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");

    let mut want = [0.0; 2];
    for m in 0..5 {
        let rest: Vec<usize> = (0..5).filter(|&j| j != m).collect();
        let (j, k, l, n) = (rest[0], rest[1], rest[2], rest[3]);
        let pairs = dot(vs[j], vs[k]) * dot(vs[l], vs[n])
            + dot(vs[j], vs[l]) * dot(vs[k], vs[n])
            + dot(vs[j], vs[n]) * dot(vs[k], vs[l]);
        for i in 0..2 {
            want[i] += 8.0 * c * vs[m][i] * pairs;
        }
    }
    let dirs: Vec<&[f64]> = vs.iter().map(|d| &d[..]).collect();
    let got = dir_out(&mut ir, &[Slot::U; 5], &o, &p, &pack(&dirs));
    assert!(within(&got, &want).is_ok(), "{got:?} {want:?}");
}

// ---- structure ----

#[test]
fn dir_entries_are_cached_and_bad_slot_counts_refused() {
    let (_fb, mut ir) = setup("x'=sin(x)*y\ny'=x*x\ndone\n");
    let k = EntryKind::Dir(vec![Slot::U, Slot::U, Slot::U]);
    let e1 = ir.entry(&k).unwrap();
    let len = ir.dag.len();
    let e2 = ir.entry(&k).unwrap();
    assert_eq!(e1, e2);
    assert_eq!(ir.dag.len(), len);
    for bad in [vec![], vec![Slot::U; 6]] {
        let e = ir.entry(&EntryKind::Dir(bad)).unwrap_err();
        assert_eq!(e.kind, crate::IrErrorKind::Unsupported);
    }
}

#[test]
fn dir_entry_reports_too_large() {
    let src = "x'=sin(x)*cos(x)+x*x\ndone\n";
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let full = lower(&l.expanded, &l.info).unwrap().dag.len();
    let mut ir = match crate::lower::lower_with_cap(&l.expanded, &l.info, full) {
        Ok(m) => m,
        Err(e) => panic!("{e:?}"),
    };
    let e = ir.entry(&EntryKind::Dir(vec![Slot::U; 5])).unwrap_err();
    assert_eq!(e.kind, crate::IrErrorKind::TooLarge);
}

#[test]
fn node_counts_by_order() {
    let src = "par a=1.3\nx'=sin(x)*y+a*x*x/(1+y*y)\ny'=exp(x)*y-x^3+tanh(a*y)\ndone\n";
    let mut counts = Vec::new();
    for k in 1..=5usize {
        let (_fb, mut ir) = setup(src);
        let before = ir.dag.len();
        ir.entry(&EntryKind::Dir(vec![Slot::U; k])).unwrap();
        counts.push((k, ir.dag.len() - before));
    }
    println!("node counts (new nodes by k): {counts:?}");
    for w in counts.windows(2) {
        assert!(w[1].1 > w[0].1, "{counts:?}");
    }
}

// ---- complex split ----

#[test]
fn complex_split_k2_hand_expanded() {
    // B(v, w) = 2 v w on x' = x^2. v = 1 + 2i, w = 3 - i:
    // v w = (3 + 2) + (-1 + 6) i = 5 + 5i, so B = 10 + 10i.
    let mut calls = 0;
    let (mut re, mut im) = ([0.0], [0.0]);
    dir_complex(
        |v, out| {
            calls += 1;
            out[0] = 2.0 * v[0] * v[1];
        },
        &[&[1.0], &[3.0]],
        &[&[2.0], &[-1.0]],
        &mut re,
        &mut im,
    );
    assert_eq!(calls, 4);
    assert_eq!((re[0], im[0]), (10.0, 10.0));
}

#[derive(Clone, Copy)]
struct C(f64, f64);

impl std::ops::Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
}

#[test]
fn complex_split_k3_against_complex_arithmetic() {
    // x' = x^3: D3 f[v, w, z] = 6 v w z, with a vector state of two copies and a real
    // coupling so the closure is a genuine real trilinear map: out_i = 6 v_i w_i z_i.
    let vs = [C(0.5, -1.2), C(1.5, 0.3)];
    let ws = [C(-0.7, 0.4), C(0.2, 0.9)];
    let zs = [C(0.8, 1.1), C(-1.3, 0.6)];
    let split = |d: &[C; 2]| (vec![d[0].0, d[1].0], vec![d[0].1, d[1].1]);
    let ((vr, vi), (wr, wi), (zr, zi)) = (split(&vs), split(&ws), split(&zs));
    let (mut re, mut im) = ([0.0; 2], [0.0; 2]);
    dir_complex(
        |v, out| {
            for i in 0..2 {
                out[i] = 6.0 * v[i] * v[2 + i] * v[4 + i];
            }
        },
        &[&vr, &wr, &zr],
        &[&vi, &wi, &zi],
        &mut re,
        &mut im,
    );
    for i in 0..2 {
        let p = vs[i] * ws[i] * zs[i];
        assert!(close(re[i], 6.0 * p.0, 1e-14), "{re:?}");
        assert!(close(im[i], 6.0 * p.1, 1e-14), "{im:?}");
    }
}

#[test]
fn complex_split_matches_dag_entry_with_real_directions_only() {
    // Zero imaginary parts: one real call, imaginary output zero.
    let (_fb, mut ir) = setup("x'=x^3\ndone\n");
    let e = ir.entry(&EntryKind::Dir(vec![Slot::U; 3])).unwrap();
    let (mut re, mut im) = ([0.0], [1.0]);
    let mut calls = 0;
    dir_complex(
        |v, out| {
            calls += 1;
            eval(&ir.dag, &e.outputs, &[0.4], &[0.0], v, out);
        },
        &[&[1.0], &[2.0], &[3.0]],
        &[&[0.0], &[0.0], &[0.0]],
        &mut re,
        &mut im,
    );
    assert_eq!(calls, 1);
    assert!(close(re[0], 36.0, 1e-14));
    assert_eq!(im[0], 0.0);
}

/// atan2(y, x) = Im log(x + i y), so D^5 along d_j = (dx_j, dy_j) is
/// Im( 24 prod(dx_j + i dy_j) / (x + i y)^5 ). The partials of atan2 cancel heavily at order 5:
/// the DAG and the interpreter each carry rounding noise near 1e-13 relative, so both are
/// checked against this closed form with 1e-12.
#[test]
fn atan2_order_five_against_complex_log() {
    let src = "par a=2, b=0.5\nx'=atan2(y,x)\ny'=x\ndone\n";
    let (fb, mut ir) = setup(src);
    let slots = vec![Slot::Full, Slot::Full, Slot::Full, Slot::Full, Slot::U];
    let x = [0.47718156927854066, 0.3094687482141233];
    let p = [1.8885506396727643, 0.5499736958187871, 0.0];
    let v = [
        -0.9857503191048591,
        -0.5590871966474522,
        0.29404276860943224,
        0.43376470813130075,
        0.23635516791372946,
        0.9709782248671888,
        -0.8265853801172849,
        -0.4889539916285419,
        -0.32730073382390734,
        -0.6108923841928711,
        0.8334961082323151,
        -0.3665237342329375,
        0.19221327731121907,
        0.6550669398569247,
        -0.09901495401099747,
        0.07540851928620529,
        0.15285771847377494,
        -0.6365123173097458,
    ];
    let got = dir_out(&mut ir, &slots, &x, &p, &v);
    let mut want = vec![0.0; 2];
    fb.eval(&EntryKind::Dir(slots.clone()), &x, &p, &v, &mut want);
    // offsets: Full len 4 each: u at off, off+1
    let offs = [0usize, 4, 8, 12, 16];
    let mut prod = C(1.0, 0.0);
    for o in offs {
        prod = prod * C(v[o], v[o + 1]);
    }
    // w = x + i y with state (x, y) = (u0, u1): atan2(y, x)
    let w = C(x[0], x[1]);
    let mut w5 = C(1.0, 0.0);
    for _ in 0..5 {
        w5 = w5 * w;
    }
    let den = w5.0 * w5.0 + w5.1 * w5.1;
    let q = C(w5.0 / den, -w5.1 / den);
    let z = prod * q;
    let truth = 24.0 * z.1;
    assert!(close(got[0], truth, 1e-12), "dag {:e} vs {truth:e}", got[0]);
    assert!(
        close(want[0], truth, 1e-12),
        "interp {:e} vs {truth:e}",
        want[0]
    );
}
