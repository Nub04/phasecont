use super::*;
use crate::eval::eval;
use crate::lower::{IrModel, lower};
use pc_ode::{MapLoader, Severity, SourceMap, load};

fn loaded(src: &str) -> pc_ode::Loaded {
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    load(top, &mut map, &mut MapLoader::new())
}

/// Loads, checks and lowers `src`; returns the fallback and the lowered model.
fn setup(src: &str) -> (Fallback, IrModel) {
    let l = loaded(src);
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

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Hand-derived and finite-difference comparisons with a stated relative tolerance;
/// the comparison itself is `check::within`.
fn close(a: f64, b: f64, tol: f64) -> bool {
    crate::check::within(a, b, 1.0 + b.abs(), tol)
}

/// F and aux of the fallback against `eval.rs`, bit for bit.
fn compare(src: &str, points: &[(&[f64], &[f64])]) {
    let (fb, ir) = setup(src);
    for (x, p) in points {
        let mut want = vec![0.0; ir.f.len()];
        eval(&ir.dag, &ir.f, x, p, &[], &mut want);
        let mut got = vec![0.0; ir.f.len()];
        fb.eval(&EntryKind::F, x, p, &[], &mut got);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(same(*g, *w), "f[{i}] at {x:?} {p:?}: {g:?} vs {w:?}");
        }
        let mut want = vec![0.0; ir.aux.len()];
        eval(&ir.dag, &ir.aux, x, p, &[], &mut want);
        let mut got = vec![0.0; ir.aux.len()];
        fb.eval(&EntryKind::Aux, x, p, &[], &mut got);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(same(*g, *w), "aux[{i}] at {x:?} {p:?}: {g:?} vs {w:?}");
        }
    }
}

#[test]
fn f_matches_eval_every_statement_kind() {
    compare(
        "par a=2, b=0.5\nnumber c=3\n!d=a*b\nw=x*c\nf(u,v)=u+v*arg1\n\
         table h % 3 0 2 t^2\nx'=-a*x+w+d+h(1.5)+f(x,y)\n\
         dy/dt=y^3+x^(-2)+y^a\naux s=x+y\nx(0)=1\ninit y=2\ndone\n",
        &[
            (&[1.0, 2.0], &[2.0, 0.5, 0.0]),
            (&[-0.7, 1.3], &[1.1, 0.2, 2.0]),
            (&[3.1, -2.2], &[4.0, 0.5, 0.0]),
        ],
    );
}

#[test]
fn f_matches_eval_builtins() {
    compare(
        "par a=0.8\n\
         x'=sin(x)+cos(y)+tan(a*x)+asin(a*x/2)+acos(a*y/3)+atan(x)\n\
         y'=sinh(x)+cosh(y)+tanh(x*y)+exp(a*x)+ln(abs(y)+1)+log10(abs(x)+2)+sqrt(abs(x*y)+1)\n\
         z'=erf(x)+erfc(y)+lgamma(abs(x)+1.5)+flr(y*3)+ceil(x*2)+heav(x-0.3)+sign(y-0.1)\n\
         w'=atan2(y,x)+mod(x*7,a+1)+min(x,y)+max(x,y)+besselj(2,x)+bessely(1,abs(x)+0.5)\
         +x^3+y**(-2)+abs(x)^2.5+log(abs(y)+2)\n\
         done\n",
        &[
            (&[0.7, 1.3, 0.2, -0.4], &[0.8, 0.0]),
            (&[-0.9, 0.45, 1.7, 2.5], &[0.3, 1.0]),
            (&[2.2, -1.1, -0.6, 0.9], &[1.4, 0.0]),
        ],
    );
}

#[test]
fn f_matches_eval_scoping_logic_and_time() {
    compare(
        "par a=2, b=3\nnumber k=5\n!d=a*b+k\nw=x*d+t\n\
         f(x,y)=x-y*arg2+a\ng(u)=f(u,a)*u\nh(a)=a*2+b\n\
         x'=if(x>y)then(f(x,y)+g(x))else(h(y)+w)\n\
         y'=(x>0)&(y<5)+(x<=y)|(x>=3)+(not(y))+(x==y)\n\
         aux s=w*d+h(x)+t\naux q=g(y)\ndone\n",
        &[
            (&[1.5, 0.5], &[2.0, 3.0, 0.7]),
            (&[-1.5, 2.5], &[1.2, 0.4, 3.1]),
            (&[3.0, 3.0], &[2.0, 3.0, 0.0]),
        ],
    );
}

#[test]
fn f_matches_eval_powers() {
    compare(
        "par a=2\nx'=x^3\ny'=x**16\nz'=x^17\nu'=x^2.5\nv'=x^a\nq'=x^(-2)\nw'=x^0\n\
         r'=x^1\ns'=x^(-1)\nm'=x^7+x^(2*2)+x^(a-1)\ndone\n",
        &[
            (&[2.0; 11], &[2.0, 0.0]),
            (&[1.37; 11], &[3.0, 0.0]),
            (&[-0.83; 11], &[2.0, 0.0]),
        ],
    );
}

#[test]
fn f_matches_eval_tables_map_and_fixed() {
    compare(
        "par a=2\ntable h % 3 0 2 t^2\ntable g % 3 0 1 a*t\ntable k % 3 0 2 h(t)\n\
         x'=h(x)+g(x)+k(x)\ndone\n",
        &[
            (&[1.5], &[2.0, 0.0]),
            (&[0.3], &[2.0, 0.0]),
            (&[-0.5], &[2.0, 0.0]),
        ],
    );
    compare(
        "par r=3\nx(t+1)=r*x*(1-x)\nx(0)=0.5\ndone\n",
        &[(&[0.5], &[3.0, 0.0]), (&[0.21], &[3.7, 4.0])],
    );
    compare(
        "w=x*2\ntable h % 3 0 1 t+w\nx'=h(0.5)+w\nx(0)=4\ndone\n",
        &[(&[4.0], &[0.0]), (&[1.1], &[0.0])],
    );
    compare(
        "%[1..3]\nx[j]'=-x[j]+[j]+x[j]^2\n%\ndone\n",
        &[(&[1.0, 0.5, -2.0], &[0.0])],
    );
}

fn d(re: f64, eps: f64) -> Dual<f64> {
    Dual { re, eps }
}

fn fd(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5 * x.abs().max(1.0);
    (f(x + h) - f(x - h)) / (2.0 * h)
}

macro_rules! u {
    ($m:ident, $pts:expr) => {
        (
            stringify!($m),
            <Dual<f64> as Scalar>::$m as fn(Dual<f64>) -> Dual<f64>,
            <f64 as Scalar>::$m as fn(f64) -> f64,
            &$pts as &[f64],
        )
    };
}

#[test]
fn dual_unary_rules_match_central_differences() {
    let cases = [
        u!(sin, [0.3, -1.2, 2.0]),
        u!(cos, [0.3, -1.2, 2.0]),
        u!(tan, [0.3, -1.0]),
        u!(asin, [0.3, -0.6]),
        u!(acos, [0.3, -0.6]),
        u!(atan, [0.3, -2.5, 7.0]),
        u!(sinh, [0.3, -1.4]),
        u!(cosh, [0.3, -1.4]),
        u!(tanh, [0.3, -1.4]),
        u!(exp, [0.3, -1.4, 2.0]),
        u!(ln, [0.5, 3.0]),
        u!(log10, [0.5, 3.0]),
        u!(sqrt, [0.5, 3.0]),
        u!(abs, [-1.5, 2.0]),
        u!(erf, [0.3, -0.8, 1.9]),
        u!(erfc, [0.3, -0.8, 1.9]),
        u!(lgamma, [0.7, 3.5, -1.5]),
    ];
    for (name, fd_, ff, pts) in cases {
        for &x in pts {
            let r = fd_(d(x, 1.0));
            assert!(same(r.re, ff(x)), "{name} value at {x}");
            let want = fd(ff, x);
            assert!(
                close(r.eps, want, 1e-6),
                "{name}' at {x}: {} vs {want}",
                r.eps
            );
            // The tangent scales linearly.
            let r3 = fd_(d(x, 3.0));
            assert!(close(r3.eps, 3.0 * r.eps, 1e-12), "{name} scaling at {x}");
        }
    }
}

#[test]
fn dual_non_smooth_unary_rules() {
    for x in [0.3, -1.7, 2.0] {
        let t = d(x, 1.0);
        assert_eq!(Scalar::floor(t).eps, 0.0);
        assert_eq!(Scalar::ceil(t).eps, 0.0);
        assert_eq!(Scalar::heav(t).eps, 0.0);
        assert_eq!(Scalar::sign(t).eps, 0.0);
        assert_eq!(Scalar::abs(t).eps, x.signum());
    }
    assert_eq!(Scalar::heav(d(0.5, 1.0)).re, 1.0);
    assert_eq!(Scalar::sign(d(-0.5, 1.0)).re, -1.0);
}

#[test]
fn dual_binary_rules_match_central_differences() {
    type B = fn(Dual<f64>, Dual<f64>) -> Dual<f64>;
    type F = fn(f64, f64) -> f64;
    type Case = (&'static str, B, F, &'static [(f64, f64)], bool, bool);
    let cases: [Case; 6] = [
        (
            "pow",
            Scalar::pow,
            Scalar::pow,
            &[(1.7, 2.3), (0.4, -1.5), (3.0, 0.5)],
            true,
            true,
        ),
        (
            "atan2",
            Scalar::atan2,
            Scalar::atan2,
            &[(1.2, 0.7), (-0.5, 2.0), (0.3, -1.1)],
            true,
            true,
        ),
        (
            "mod",
            Scalar::modulo,
            Scalar::modulo,
            &[(7.3, 2.1), (-7.3, 2.1), (0.4, 5.0)],
            true,
            true,
        ),
        (
            "besselj",
            Scalar::besselj,
            Scalar::besselj,
            &[(2.0, 1.7), (0.0, 0.6), (3.0, 4.2)],
            false,
            true,
        ),
        (
            "bessely",
            Scalar::bessely,
            Scalar::bessely,
            &[(1.0, 1.7), (0.0, 0.6), (2.0, 4.2)],
            false,
            true,
        ),
        (
            "min",
            Scalar::min,
            Scalar::min,
            &[(1.0, 2.0), (3.0, -2.0)],
            true,
            true,
        ),
    ];
    for (name, fdual, ff, pts, first, second) in cases {
        for &(a, b) in pts {
            let dx = fdual(d(a, 1.0), d(b, 0.0));
            let dy = fdual(d(a, 0.0), d(b, 1.0));
            assert!(same(dx.re, ff(a, b)), "{name} value");
            if first {
                let want = fd(|t| ff(t, b), a);
                assert!(
                    close(dx.eps, want, 1e-6),
                    "{name} d/dx at ({a},{b}): {} vs {want}",
                    dx.eps
                );
            } else {
                assert_eq!(dx.eps, 0.0, "{name} d/dn");
            }
            if second {
                let want = fd(|t| ff(a, t), b);
                assert!(
                    close(dy.eps, want, 1e-6),
                    "{name} d/dy at ({a},{b}): {} vs {want}",
                    dy.eps
                );
            }
        }
    }
    // max follows the larger operand's tangent.
    let r = Scalar::max(d(1.0, 5.0), d(2.0, 7.0));
    assert_eq!((r.re, r.eps), (2.0, 7.0));
    let r = Scalar::min(d(1.0, 5.0), d(2.0, 7.0));
    assert_eq!((r.re, r.eps), (1.0, 5.0));
}

#[test]
fn dual_pow_ignores_zero_tangents() {
    // 0 * ln(0) must not appear.
    let r = Scalar::pow(d(0.0, 0.0), d(0.0, 0.0));
    assert_eq!((r.re, r.eps), (1.0, 0.0));
    let r = Scalar::pow(d(0.0, 1.0), d(2.0, 0.0));
    assert_eq!((r.re, r.eps), (0.0, 0.0));
    let r = Scalar::pow(d(3.0, 0.0), d(2.0, 1.0));
    assert!(close(r.eps, 9.0 * 3.0f64.ln(), 1e-14));
    // Both tangents add.
    let r = Scalar::pow(d(2.0, 1.0), d(3.0, 1.0));
    assert!(close(r.eps, 12.0 + 8.0 * 2.0f64.ln(), 1e-14));
}

#[test]
fn dual_arithmetic_powi_table_and_polygamma() {
    let a = d(3.0, 1.0);
    let b = d(2.0, 0.5);
    assert_eq!((a + b).eps, 1.5);
    assert_eq!((a - b).eps, 0.5);
    assert_eq!((a * b).eps, 1.0 * 2.0 + 3.0 * 0.5);
    assert!(close((a / b).eps, (1.0 * 2.0 - 3.0 * 0.5) / 4.0, 1e-15));
    assert_eq!((-a).eps, -1.0);
    for n in [-5, -2, -1, 0, 1, 2, 3, 7, 16] {
        let r = Scalar::powi(d(1.3, 1.0), n);
        assert!(same(r.re, Scalar::powi(1.3f64, n)));
        let want = fd(|t| Scalar::powi(t, n), 1.3);
        assert!(close(r.eps, want, 1e-6), "powi {n}: {} vs {want}", r.eps);
    }
    // f64 powi follows the DAG order exactly.
    let x = 1.37f64;
    let x2 = x * x;
    assert_eq!(Scalar::powi(x, 5).to_bits(), (x * (x2 * x2)).to_bits());
    assert_eq!(Scalar::powi(x, -3).to_bits(), (1.0 / (x * x2)).to_bits());
    assert_eq!(Scalar::powi(x, 1).to_bits(), x.to_bits());
    // Table: value and slope of the segment; the slope has no derivative.
    let t = [3.0, 0.0, 2.0, 0.0, 1.0, 4.0];
    let r = Scalar::table(d(1.5, 2.0), &t);
    assert_eq!((r.re, r.eps), (2.5, 6.0));
    let s = Scalar::table_slope(d(1.5, 2.0), &t);
    assert_eq!((s.re, s.eps), (3.0, 0.0));
    // Polygamma chain: d/dx lgamma = psi0, d/dx psi_m = psi_{m+1}.
    for m in 0..4 {
        let r = Scalar::polygamma(d(2.3, 1.0), m);
        assert!(same(r.re, Scalar::polygamma(2.3f64, m)));
        assert!(close(r.eps, Scalar::polygamma(2.3f64, m + 1), 1e-15));
    }
    let r = Scalar::lgamma(d(2.3, 1.0));
    assert!(close(r.eps, Scalar::polygamma(2.3f64, 0), 1e-15));
}

fn second_order(
    name: &str,
    f1: impl Fn(Dual<f64>) -> Dual<f64>,
    f2: impl Fn(Dual<Dual<f64>>) -> Dual<Dual<f64>>,
    pts: &[f64],
) {
    for &x in pts {
        let seed = Dual {
            re: d(x, 1.0),
            eps: d(1.0, 0.0),
        };
        let r = f2(seed);
        let first = fd(|t| f1(d(t, 1.0)).re, x);
        let second = fd(|t| f1(d(t, 1.0)).eps, x);
        assert!(close(r.re.eps, first, 1e-6), "{name}' at {x}");
        assert!(close(r.eps.re, first, 1e-6), "{name}' (outer) at {x}");
        assert!(
            close(r.eps.eps, second, 1e-5),
            "{name}'' at {x}: {} vs {second}",
            r.eps.eps
        );
    }
}

#[test]
fn nested_duals_give_second_derivatives() {
    second_order("sin", |s| s.sin(), |s| s.sin(), &[0.4, -1.3]);
    second_order("exp", |s| s.exp(), |s| s.exp(), &[0.4, -1.3]);
    second_order("ln", |s| s.ln(), |s| s.ln(), &[0.4, 2.3]);
    second_order("tanh", |s| s.tanh(), |s| s.tanh(), &[0.4, -1.3]);
    second_order("erf", |s| s.erf(), |s| s.erf(), &[0.4, -1.3]);
    second_order("lgamma", |s| s.lgamma(), |s| s.lgamma(), &[0.9, 3.3]);
    second_order(
        "pow",
        |s| s.pow(Dual::from_f64(2.5)),
        |s| s.pow(Dual::from_f64(2.5)),
        &[0.7, 2.2],
    );
    second_order(
        "atan2",
        |s| s.atan2(Dual::from_f64(1.3)),
        |s| s.atan2(Dual::from_f64(1.3)),
        &[0.7, -2.2],
    );
    second_order("powi", |s| s.powi(5), |s| s.powi(5), &[0.7, -1.2]);
    second_order(
        "recip",
        |s| Dual::from_f64(1.0) / s,
        |s| Dual::from_f64(1.0) / s,
        &[0.7, -1.2],
    );
    second_order(
        "besselj",
        |s| Dual::from_f64(2.0).besselj(s),
        |s| Dual::from_f64(2.0).besselj(s),
        &[1.7, 4.2],
    );
}

fn hopf() -> (Fallback, IrModel) {
    setup("par m=0.2, b=0.5\nx'=m*x-y+b*x*x-x*(x*x+y*y)\ny'=x+m*y-y*(x*x+y*y)\ndone\n")
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Third derivative of g(u) = -(u.u) u along (v, w, z): -2[(v.w)z + (v.z)w + (w.z)v].
fn cubic3(v: [f64; 2], w: [f64; 2], z: [f64; 2]) -> [f64; 2] {
    let mut r = [0.0; 2];
    for (i, ri) in r.iter_mut().enumerate() {
        *ri = -2.0 * (dot(v, w) * z[i] + dot(v, z) * w[i] + dot(w, z) * v[i]);
    }
    r
}

#[test]
fn hopf_normal_form_b_and_c() {
    let (fb, _) = hopf();
    let p = [0.2, 0.5, 0.0];
    let (v, w, z) = ([0.7, 0.2], [-0.5, 1.1], [0.9, 0.4]);
    let pack = |dirs: &[[f64; 2]]| dirs.iter().flatten().copied().collect::<Vec<f64>>();
    let b = 0.5;

    // At the origin: B = (2 b vx wx, 0) and C = cubic3.
    let o = [0.0, 0.0];
    let mut out = [0.0; 2];
    fb.eval(
        &EntryKind::Dir(vec![Slot::U; 2]),
        &o,
        &p,
        &pack(&[v, w]),
        &mut out,
    );
    assert!(close(out[0], 2.0 * b * v[0] * w[0], 1e-14));
    assert!(close(out[1], 0.0, 1e-14));
    fb.eval(
        &EntryKind::Dir(vec![Slot::U; 3]),
        &o,
        &p,
        &pack(&[v, w, z]),
        &mut out,
    );
    let want = cubic3(v, w, z);
    assert!(
        close(out[0], want[0], 1e-13) && close(out[1], want[1], 1e-13),
        "{out:?} {want:?}"
    );

    // At a general point.
    let u = [0.3, -0.4];
    fb.eval(
        &EntryKind::Dir(vec![Slot::U; 2]),
        &u,
        &p,
        &pack(&[v, w]),
        &mut out,
    );
    let mut want = [2.0 * b * v[0] * w[0], 0.0];
    for (i, wi) in want.iter_mut().enumerate() {
        *wi -= 2.0 * dot(v, w) * u[i] + 2.0 * dot(u, v) * w[i] + 2.0 * dot(u, w) * v[i];
    }
    assert!(
        close(out[0], want[0], 1e-13) && close(out[1], want[1], 1e-13),
        "{out:?} {want:?}"
    );
    fb.eval(
        &EntryKind::Dir(vec![Slot::U; 3]),
        &u,
        &p,
        &pack(&[v, w, z]),
        &mut out,
    );
    let want = cubic3(v, w, z);
    assert!(close(out[0], want[0], 1e-13) && close(out[1], want[1], 1e-13));

    // First derivative with a Full slot: Df_u v + Df_p dp.
    let (dm, db) = (2.0, -1.0);
    let dir = [v[0], v[1], dm, db];
    fb.eval(&EntryKind::Dir(vec![Slot::Full]), &u, &p, &dir, &mut out);
    let r2 = dot(u, u);
    let want = [
        0.2 * v[0] - v[1] + 2.0 * b * u[0] * v[0] - (2.0 * dot(u, v) * u[0] + r2 * v[0])
            + u[0] * dm
            + u[0] * u[0] * db,
        v[0] + 0.2 * v[1] - (2.0 * dot(u, v) * u[1] + r2 * v[1]) + u[1] * dm,
    ];
    assert!(
        close(out[0], want[0], 1e-13) && close(out[1], want[1], 1e-13),
        "{out:?} {want:?}"
    );

    // Mixed second derivative: P direction then U direction, d/dm of Df_u v = v.
    let mix = [2.0, 0.0, v[0], v[1]];
    fb.eval(
        &EntryKind::Dir(vec![Slot::P, Slot::U]),
        &u,
        &p,
        &mix,
        &mut out,
    );
    assert!(
        close(out[0], 2.0 * v[0], 1e-14) && close(out[1], 2.0 * v[1], 1e-14),
        "{out:?}"
    );
    // d/db of Df_u v = (2 x vx, 0).
    let mix = [0.0, 1.0, v[0], v[1]];
    fb.eval(
        &EntryKind::Dir(vec![Slot::P, Slot::U]),
        &u,
        &p,
        &mix,
        &mut out,
    );
    assert!(
        close(out[0], 2.0 * u[0] * v[0], 1e-14) && close(out[1], 0.0, 1e-14),
        "{out:?}"
    );
}

#[test]
fn jacobians_match_hand_derivatives_and_differences() {
    let (fb, ir) = hopf();
    let u = [0.3, -0.4];
    let p = [0.2, 0.5, 0.0];
    let mut ju = [0.0; 4];
    fb.eval(&EntryKind::JacU, &u, &p, &[], &mut ju);
    let (m, b) = (0.2, 0.5);
    let r2 = dot(u, u);
    let want = [
        // column 0: df/dx
        m + 2.0 * b * u[0] - (3.0 * u[0] * u[0] + u[1] * u[1]),
        1.0 - 2.0 * u[0] * u[1],
        // column 1: df/dy
        -1.0 - 2.0 * u[0] * u[1],
        m - (u[0] * u[0] + 3.0 * u[1] * u[1]),
    ];
    for (g, w) in ju.iter().zip(&want) {
        assert!(close(*g, *w, 1e-13), "{ju:?} vs {want:?}");
    }
    assert!(r2 > 0.0);
    let mut jp = [0.0; 4];
    fb.eval(&EntryKind::JacP, &u, &p, &[], &mut jp);
    let want = [u[0], u[1], u[0] * u[0], 0.0];
    for (g, w) in jp.iter().zip(&want) {
        assert!(close(*g, *w, 1e-14), "{jp:?} vs {want:?}");
    }
    // Central differences of F.
    for j in 0..2 {
        let col = |t: f64| {
            let mut x = u;
            x[j] += t;
            let mut o = [0.0; 2];
            fb.eval(&EntryKind::F, &x, &p, &[], &mut o);
            o
        };
        let h = 1e-6;
        let (a, c) = (col(h), col(-h));
        for i in 0..2 {
            assert!(close(ju[i + 2 * j], (a[i] - c[i]) / (2.0 * h), 1e-6));
        }
    }
    assert_eq!(ir.n_p(), 2);
}

#[test]
fn aux_jacobian_with_time_and_fixed_quantities() {
    let (fb, _) = setup("par a=1.5\nw=a*x*t\nx'=w+sin(x)\naux q=w*x\ndone\n");
    let mut out = [0.0];
    fb.eval(&EntryKind::Aux, &[2.0], &[1.5, 0.5], &[], &mut out);
    assert_eq!(out, [1.5 * 2.0 * 0.5 * 2.0]);
    fb.eval(&EntryKind::JacU, &[2.0], &[1.5, 0.5], &[], &mut out);
    assert!(close(out[0], 1.5 * 0.5 + 2.0f64.cos(), 1e-14));
    fb.eval(&EntryKind::JacP, &[2.0], &[1.5, 0.5], &[], &mut out);
    assert!(close(out[0], 2.0 * 0.5, 1e-14));
}

#[test]
fn five_slot_directional_derivative() {
    let (fb, _) = setup("x'=x^5\ndone\n");
    let mut out = [0.0];
    let ones = |k: usize| vec![1.0; k];
    for (k, want) in [
        (1, 5.0 * 16.0),
        (2, 20.0 * 8.0),
        (3, 60.0 * 4.0),
        (4, 120.0 * 2.0),
        (5, 120.0),
    ] {
        fb.eval(
            &EntryKind::Dir(vec![Slot::U; k]),
            &[2.0],
            &[0.0],
            &ones(k),
            &mut out,
        );
        assert!(close(out[0], want, 1e-12), "k = {k}: {} vs {want}", out[0]);
    }
    // Scaling one direction scales the result.
    fb.eval(
        &EntryKind::Dir(vec![Slot::U; 3]),
        &[2.0],
        &[0.0],
        &[1.0, 2.0, 3.0],
        &mut out,
    );
    assert!(close(out[0], 6.0 * 240.0, 1e-12));
}

#[test]
fn bad_dir_requests_give_nan() {
    let (fb, _) = setup("par a=1\nx'=a*x\ndone\n");
    let mut out = [0.0];
    fb.eval(
        &EntryKind::Dir(vec![Slot::U; 6]),
        &[1.0],
        &[1.0, 0.0],
        &[1.0; 6],
        &mut out,
    );
    assert!(out[0].is_nan());
    fb.eval(&EntryKind::Dir(vec![]), &[1.0], &[1.0, 0.0], &[], &mut out);
    assert!(out[0].is_nan());
    // v too short for two Full slots.
    fb.eval(
        &EntryKind::Dir(vec![Slot::Full; 2]),
        &[1.0],
        &[1.0, 0.0],
        &[1.0; 3],
        &mut out,
    );
    assert!(out[0].is_nan());
    // x too short.
    fb.eval(&EntryKind::F, &[], &[1.0, 0.0], &[], &mut out);
    assert!(out[0].is_nan());
}

#[test]
fn deep_user_function_nesting_gives_nan_not_a_panic() {
    // The model that reaches the interpreter must have lowered, so lower a small one and
    // pass a model nested far past the limit.
    let (_, ir) = setup("x'=-x\ndone\n");
    let mut src = String::from("f0(u)=u+1\n");
    for i in 1..1500 {
        src.push_str(&format!("f{i}(u)=f{}(u)+1\n", i - 1));
    }
    src.push_str("x'=f1499(x)\ndone\n");
    let l = loaded(&src);
    let it = Interp::new(&l.expanded, &l.info, &ir);
    let mut f = [0.0];
    let mut aux: [f64; 0] = [];
    it.eval(&[1.0], &[0.0], &mut f, &mut aux);
    assert!(f[0].is_nan());
    // The same through duals and the fallback.
    let fb = Fallback::from_interp(it);
    let mut out = [0.0];
    fb.eval(&EntryKind::JacU, &[1.0], &[0.0], &[], &mut out);
    assert!(out[0].is_nan());
    // A moderate nesting still works.
    let mut src = String::from("f0(u)=u+1\n");
    for i in 1..300 {
        src.push_str(&format!("f{i}(u)=f{}(u)+1\n", i - 1));
    }
    src.push_str("x'=f299(x)\ndone\n");
    compare(&src, &[(&[0.5], &[0.0])]);
    let (fb, _) = setup(&src);
    fb.eval(&EntryKind::F, &[0.5], &[0.0], &[], &mut out);
    assert_eq!(out[0], 300.5);
}

/// Fallback for a chain model; the lowering of `x'=-x` stands in, since a chain past the
/// lowering limit does not lower.
fn deep_fb(levels: usize) -> Fallback {
    let (_, ir) = setup("x'=-x\ndone\n");
    let l = loaded(&chain_src(levels));
    Fallback::new(&l.expanded, &l.info, &ir)
}

fn chain_src(levels: usize) -> String {
    let mut src = String::from("f0(u)=u+1\n");
    for i in 1..levels {
        src.push_str(&format!("f{i}(u)=f{}(u)+1\n", i - 1));
    }
    src.push_str(&format!("x'=f{}(x)\ndone\n", levels - 1));
    src
}

/// Runs `body` on a 1 MiB stack, the Windows main-thread default.
fn on_small_stack(body: impl FnOnce() + Send + 'static) {
    let h = std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(body)
        .unwrap();
    h.join().expect("the evaluation overflowed or panicked");
}

#[test]
fn depth_limit_shrinks_with_scalar_size() {
    let l1 = depth_limit::<f64>();
    let l2 = depth_limit::<Dual<f64>>();
    let l5 = depth_limit::<Dual<Dual<Dual<Dual<Dual<f64>>>>>>();
    assert!(l1 >= 600, "{l1}");
    assert!(l5 < l2 && l2 <= l1 && l5 >= 20, "{l1} {l2} {l5}");
}

#[test]
fn deep_chains_fit_a_one_mib_stack() {
    on_small_stack(|| {
        // Each chain level costs two nesting units (the call and the body sum).
        let five = vec![Slot::U; 5];
        let lim = depth_limit::<Dual<Dual<Dual<Dual<Dual<f64>>>>>>() / 2;
        for (levels, finite) in [(lim - 3, true), (lim + 3, false)] {
            let fb = deep_fb(levels);
            let mut out = [0.0];
            fb.eval(
                &EntryKind::Dir(five.clone()),
                &[0.5],
                &[0.0],
                &[1.0; 5],
                &mut out,
            );
            assert_eq!(out[0].is_finite(), finite, "dir, {levels} levels: {out:?}");
        }
        let lim = depth_limit::<f64>() / 2;
        for (levels, finite) in [(lim - 3, true), (lim + 3, false)] {
            let fb = deep_fb(levels);
            let mut out = [0.0];
            fb.eval(&EntryKind::F, &[0.5], &[0.0], &[], &mut out);
            assert_eq!(out[0].is_finite(), finite, "F, {levels} levels: {out:?}");
            if finite {
                assert_eq!(out[0], levels as f64 + 0.5);
            }
        }
    });
}

#[test]
fn pow_with_zero_primal_has_no_exponent_tangent() {
    let r = Scalar::pow(d(0.0, 0.0), d(2.0, 1.0));
    assert_eq!((r.re, r.eps), (0.0, 0.0));
    let r = Scalar::pow(d(0.0, 1.0), d(2.0, 1.0));
    assert_eq!((r.re, r.eps), (0.0, 0.0));
}

#[test]
fn jacobian_writes_every_output_slot() {
    let (fb, _) = setup("par a=2\nx'=a*x\ndone\n");
    let mut out = [f64::NAN; 5];
    fb.eval(&EntryKind::JacU, &[3.0], &[2.0, 0.0], &[], &mut out);
    assert_eq!(out, [2.0, 0.0, 0.0, 0.0, 0.0]);
    let mut out = [f64::NAN; 3];
    fb.eval(&EntryKind::JacP, &[3.0], &[2.0, 0.0], &[], &mut out);
    assert_eq!(out, [3.0, 0.0, 0.0]);
}
