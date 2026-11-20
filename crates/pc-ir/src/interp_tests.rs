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

/// The values of F from the fallback at `x` and `p`.
fn fallback_f(src: &str, x: &[f64], p: &[f64]) -> Vec<f64> {
    let (fb, ir) = setup(src);
    let mut got = vec![0.0; ir.f.len()];
    fb.eval(&EntryKind::F, x, p, &[], &mut got);
    got
}

/// A declared formal named `argK` binds by name in the fallback as in the DAG: `f(5,3)` of
/// `f(arg2,arg1)=arg1-arg2` is -2, where positional binding gives +2.
#[test]
fn a_declared_arg_formal_binds_by_name() {
    assert_eq!(
        fallback_f("f(arg2,arg1)=arg1-arg2\nx'=f(5,3)\ndone\n", &[0.0], &[0.0]),
        [-2.0]
    );
    let src = "par a=5, b=3\nf(arg2,arg1)=arg1-arg2\nx'=f(a,b)\ndone\n";
    assert_eq!(fallback_f(src, &[0.0], &[5.0, 3.0, 0.0]), [-2.0]);
    let src = "par a=2, b=3\nf(u,arg1)=u+arg1\nx'=f(a,b)\ndone\n";
    assert_eq!(fallback_f(src, &[0.0], &[2.0, 3.0, 0.0]), [5.0]);
    // A declared `arg3` in a two-argument function.
    let src = "par a=2, b=3\nf(u,arg3)=u*10+arg3\nx'=f(a,b)\ndone\n";
    assert_eq!(fallback_f(src, &[0.0], &[2.0, 3.0, 0.0]), [23.0]);
    // An undeclared `argK` stays positional next to a declared one.
    let src = "par a=1, b=2, c=3\ng(arg2,v,w)=arg1*100+arg2*10+arg3\nx'=g(a,b,c)\ndone\n";
    assert_eq!(fallback_f(src, &[0.0], &[1.0, 2.0, 3.0, 0.0]), [113.0]);
    // The frame of the caller is back after a call that declares its own `argK`.
    let src = "h(arg2,arg1)=arg1-arg2\nk(u,v)=h(v,u)*10+arg1\nx'=k(1,2)\ndone\n";
    assert_eq!(fallback_f(src, &[0.0], &[0.0]), [-9.0]);
}

#[test]
fn f_matches_eval_with_declared_arg_formals() {
    compare(
        "par a=2, b=3\nf(arg2,arg1)=arg1-arg2*a\ng(u,arg3)=u*arg3+arg1\n\
         h(arg2,v,w)=arg1*v+arg2*w+arg3\nx'=f(x,y)+g(y,x)\ny'=h(x,y,a)*b\n\
         aux s=f(y,x)\ndone\n",
        &[
            (&[1.5, 0.5], &[2.0, 3.0, 0.0]),
            (&[-0.7, 1.3], &[1.1, 0.2, 2.0]),
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

// ---- The heap machine (SPEC section 8af-5) ----

/// A function that recurses without end; defining it, called or not, puts every evaluation
/// of the model on the heap machine.
const RUNAWAY: &str = "zz(n)=zz(n+1)+1\n";

/// Interpreter for `src`, which may hold recursive functions.
fn machine_interp(src: &str) -> Interp {
    let (_, ir) = setup(src);
    let l = loaded(src);
    Interp::new(&l.expanded, &l.info, &ir)
}

fn out_len(fb: &Fallback, kind: &EntryKind) -> usize {
    match kind {
        EntryKind::F | EntryKind::Dir(_) => fb.n,
        EntryKind::Aux => fb.n_aux,
        EntryKind::JacU => fb.n * fb.n,
        EntryKind::JacP => fb.n * fb.n_p,
    }
}

#[test]
fn machine_matches_the_walker_bitwise() {
    let models = [
        "par a=2, b=0.5\nnumber c=3\n!d=a*b\nw=x*c\nf(u,v)=u+v*arg1\n\
         table h % 3 0 2 t^2\nx'=-a*x+w+d+h(1.5)+f(x,y)\n\
         dy/dt=y^3+x^(-2)+y^a\naux s=x+y\nx(0)=1\ninit y=2\ndone\n",
        "par a=2, b=3\nf(arg2,arg1)=arg1-arg2*a\ng(u,arg3)=u*arg3+arg1\n\
         h(arg2,v,w)=arg1*v+arg2*w+arg3\nx'=f(x,y)+g(y,x)\ny'=h(x,y,a)*b\n\
         aux s=f(y,x)\ndone\n",
        "par a=0.8\n\
         x'=sin(x)+cos(y)+tan(a*x)+asin(a*x/2)+acos(a*y/3)+atan(x)\n\
         y'=sinh(x)+cosh(y)+tanh(x*y)+exp(a*x)+ln(abs(y)+1)+log10(abs(x)+2)+sqrt(abs(x*y)+1)\n\
         z'=erf(x)+erfc(y)+lgamma(abs(x)+1.5)+flr(y*3)+ceil(x*2)+heav(x-0.3)+sign(y-0.1)\n\
         w'=atan2(y,x)+mod(x*7,a+1)+min(x,y)+max(x,y)+besselj(2,x)+bessely(1,abs(x)+0.5)\
         +x^3+y**(-2)+abs(x)^2.5+log(abs(y)+2)\n\
         done\n",
        "par a=2, b=3\nnumber k=5\n!d=a*b+k\nw=x*d+t\n\
         f(x,y)=x-y*arg2+a\ng(u)=f(u,a)*u\nh(a)=a*2+b\n\
         x'=if(x>y)then(f(x,y)+g(x))else(h(y)+w)\n\
         y'=(x>0)&(y<5)+(x<=y)|(x>=3)+(not(y))+(x==y)\n\
         aux s=w*d+h(x)+t\naux q=g(y)\ndone\n",
        "par a=2\ntable h % 3 0 2 t^2\ntable g % 3 0 1 a*t\ntable k % 3 0 2 h(t)\n\
         x'=h(x)+g(x)+k(x)+x^17+x^(-2)+x^a\ndone\n",
        "w=x*2\ntable h % 3 0 1 t+w\nx'=h(0.5)+w\nx(0)=4\ndone\n",
        "%[1..3]\nx[j]'=-x[j]+[j]+x[j]^2\n%\ndone\n",
    ];
    let points: [&[f64]; 3] = [
        &[0.7, 1.3, 0.2, -0.4],
        &[-0.9, 0.45, 1.7, 2.5],
        &[2.2, -1.1, -0.6, 0.9],
    ];
    let pars: [&[f64]; 2] = [&[0.8, 0.4, 0.3, 0.0], &[2.0, 3.0, 0.7, 1.0]];
    for src in models {
        let (walker, _) = setup(src);
        assert!(!walker.interp.has_recursive);
        let machine = Fallback::from_interp(machine_interp(&format!("{RUNAWAY}{src}")));
        assert!(machine.interp.has_recursive);
        let n = walker.n;
        let kinds = [
            EntryKind::F,
            EntryKind::Aux,
            EntryKind::JacU,
            EntryKind::JacP,
            EntryKind::Dir(vec![Slot::U, Slot::U]),
            EntryKind::Dir(vec![Slot::U, Slot::P, Slot::U]),
        ];
        for x in points {
            for p in pars {
                let x = &x[..n];
                let p = &p[..walker.n_p];
                let v: Vec<f64> = (0..3 * (n + walker.n_p)).map(|i| 0.3 + i as f64).collect();
                for kind in &kinds {
                    let len = out_len(&walker, kind);
                    let mut want = vec![0.0; len];
                    walker.eval(kind, x, p, &v, &mut want);
                    let mut got = vec![0.0; len];
                    machine.try_eval(kind, x, p, &v, &mut got).unwrap();
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(same(*g, *w), "{kind:?}[{i}] of {src:?}: {g:?} vs {w:?}");
                    }
                }
            }
        }
    }
}

fn sum_interp() -> Interp {
    machine_interp(
        "par k=0\nsum(n)=if(n<=0)then(0)else(n+sum(n-1))\nx'=sum(k)\naux r=sum(k-1)\ndone\n",
    )
}

#[test]
fn recursion_cap_counts_active_frames() {
    let it = sum_interp();
    let (mut f, mut aux) = ([0.0], [0.0]);
    // sum(9) holds 10 frames at its deepest: 10 fit a cap of 10, 11 do not.
    it.try_eval_capped(&[0.0], &[9.0, 0.0], &mut f, &mut aux, 10)
        .unwrap();
    assert_eq!((f[0], aux[0]), (45.0, 36.0));
    let e = it
        .try_eval_capped(&[0.0], &[10.0, 0.0], &mut f, &mut aux, 10)
        .unwrap_err();
    assert_eq!(
        e,
        EvalError::Depth {
            func: "sum".into(),
            cap: 10
        }
    );
    assert!(f[0].is_nan() && aux[0].is_nan());
    // The frames of one output are gone before the next starts.
    it.try_eval_capped(&[0.0], &[10.0, 0.0], &mut f[..0], &mut aux, 10)
        .unwrap();
    assert_eq!(aux[0], 45.0);
}

#[test]
fn recursion_reaches_the_cap_on_a_one_mib_stack() {
    on_small_stack(|| {
        let it = sum_interp();
        let (mut f, mut aux) = ([0.0], [0.0]);
        it.try_eval(&[0.0], &[999_999.0, 0.0], &mut f, &mut aux)
            .unwrap();
        assert_eq!(f[0], 499_999_500_000.0);
        let e = it
            .try_eval(&[0.0], &[1_000_000.0, 0.0], &mut f, &mut aux)
            .unwrap_err();
        assert_eq!(
            e,
            EvalError::Depth {
                func: "sum".into(),
                cap: MAX_ACTIVE_CALLS
            }
        );
        assert!(f[0].is_nan() && aux[0].is_nan());
        // eval keeps its old contract: NaN, no error.
        it.eval(&[0.0], &[1_000_000.0, 0.0], &mut f, &mut aux);
        assert!(f[0].is_nan());
    });
}

#[test]
fn deep_recursion_through_wrappers_fits_a_one_mib_stack() {
    on_small_stack(|| {
        for body in ["g(f(n-1))", "max(f(n-1),0)+1", "g(abs(f(n-1)))"] {
            let it = machine_interp(&format!(
                "par k=0\ng(u)=u+1\nf(n)=if(n<=0)then(0)else({body})\nx'=f(k)\ndone\n"
            ));
            let mut f = [0.0];
            it.try_eval(&[0.0], &[999_999.0, 0.0], &mut f, &mut [])
                .unwrap();
            assert_eq!(f[0], 999_999.0, "{body}");
        }
    });
}

#[test]
fn recursion_in_duals_on_the_machine() {
    let it = machine_interp("par k=0\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n");
    let fb = Fallback::from_interp(it);
    let mut out = [0.0];
    for (x, n) in [(1.01f64, 50.0f64), (0.999, 20_000.0), (-1.2, 7.0)] {
        fb.try_eval(&EntryKind::F, &[x], &[n, 0.0], &[], &mut out)
            .unwrap();
        assert!(close(out[0], x.powi(n as i32), 1e-12), "{x} {n}: {out:?}");
        fb.try_eval(&EntryKind::JacU, &[x], &[n, 0.0], &[], &mut out)
            .unwrap();
        let want = n * x.powi(n as i32 - 1);
        assert!(close(out[0], want, 1e-11), "{x} {n}: {out:?} vs {want}");
        fb.try_eval(
            &EntryKind::Dir(vec![Slot::U, Slot::U]),
            &[x],
            &[n, 0.0],
            &[1.0, 1.0],
            &mut out,
        )
        .unwrap();
        let want = n * (n - 1.0) * x.powi(n as i32 - 2);
        assert!(close(out[0], want, 1e-10), "{x} {n}: {out:?} vs {want}");
    }
    // The cap error reaches the derivative entries too.
    let e = fb
        .try_eval(&EntryKind::JacU, &[1.0], &[1_000_000.0, 0.0], &[], &mut out)
        .unwrap_err();
    assert!(matches!(e, EvalError::Depth { .. }), "{e:?}");
    assert!(out[0].is_nan());
}

#[test]
fn running_out_of_memory_is_an_error() {
    let it = sum_interp();
    let (mut f, mut aux) = ([0.0], [0.0]);
    fail_next_grow();
    let e = it
        .try_eval(&[0.0], &[50.0, 0.0], &mut f, &mut aux)
        .unwrap_err();
    assert_eq!(e, EvalError::OutOfMemory);
    assert!(f[0].is_nan() && aux[0].is_nan());
    // The next call has the memory again.
    it.try_eval(&[0.0], &[50.0, 0.0], &mut f, &mut aux).unwrap();
    assert_eq!(f[0], 1275.0);
}

#[test]
fn an_untaken_branch_makes_no_call() {
    let it = machine_interp(&format!(
        "par k=0\n{RUNAWAY}x'=if(k<0)then(0)else(zz(k))\ndone\n"
    ));
    let mut f = [7.0];
    it.try_eval(&[0.0], &[-1.0, 0.0], &mut f, &mut []).unwrap();
    assert_eq!(f[0], 0.0);
    let e = it
        .try_eval_capped(&[0.0], &[1.0, 0.0], &mut f, &mut [], 100)
        .unwrap_err();
    assert_eq!(
        e,
        EvalError::Depth {
            func: "zz".into(),
            cap: 100
        }
    );
}

#[test]
fn mutual_recursion_and_named_quantities_on_the_machine() {
    let it = machine_interp(
        "par k=0\nw=x*2\nev(n)=if(n==0)then(1)else(od(n-1))\nod(n)=if(n==0)then(0)else(ev(n-1))\n\
         h(n)=if(n<=0)then(w)else(h(n-1)+1)\nx'=ev(k)\naux o=od(k)\naux q=h(k)\ndone\n",
    );
    let (mut f, mut aux) = ([0.0], [0.0, 0.0]);
    for n in [0.0, 1.0, 6.0, 7.0, 100_001.0] {
        it.try_eval(&[1.5], &[n, 0.0], &mut f, &mut aux).unwrap();
        let even = (n as u64).is_multiple_of(2);
        assert_eq!(f[0], if even { 1.0 } else { 0.0 }, "ev({n})");
        assert_eq!(aux[0], if even { 0.0 } else { 1.0 }, "od({n})");
        assert_eq!(aux[1], 3.0 + n, "h({n})");
    }
}

/// Every entry of the fallback against the DAG evaluator, bit for bit, on recursive models
/// that `lower` turns into units.
#[test]
fn machine_matches_the_dag_on_recursive_models() {
    let models = [
        "par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n",
        "par k=3\nf(n)=if(n<1)then(1)else(x*f(n-1))\nx'=f(k)\ndone\n",
        "par k=4\nfact(n)=if(n<=1)then(1)else(n*fact(n-1))\nx'=fact(k)*x\ndone\n",
        "par k=5\nev(n)=if(n==0)then(1)else(od(n-1))\nod(n)=if(n==0)then(0)else(ev(n-1)*x)\n\
         x'=ev(k)+od(k)*x^2\naux q=od(k+1)\ndone\n",
        "par k=7\nh(n)=if(n<1)then(0)else(if(n>5)then(h(n-1))else(x+h(n-1)))\nx'=h(k)\ndone\n",
        "par k=6\ng(n)=(if(n>0)then(g(n-1))else(0))+(if(n>5)then(g(n-1))else(sin(x)))\n\
         x'=g(k)\ndone\n",
        // A formal used as an exponent: the DAG reads it as `Arg`, never a constant, so
        // `x^e` is `pow` inside the body even when the caller passes 7. The exponent must
        // be a small integer (|e| <= 16), or the interpreter takes `pow` either way.
        "par k=2\nf(e,n)=if(n<=0)then(x^e)else(f(e,n-1))\nx'=f(7,k)+f(7,0)\ndone\n",
    ];
    let kinds = [
        EntryKind::F,
        EntryKind::Aux,
        EntryKind::JacU,
        EntryKind::JacP,
        EntryKind::Dir(vec![Slot::U, Slot::U]),
        EntryKind::Dir(vec![Slot::U, Slot::P, Slot::U]),
    ];
    for src in models {
        let (fb, mut ir) = setup(src);
        assert!(fb.interp.has_recursive, "{src}");
        assert!(!ir.dag.units().is_empty(), "{src}");
        for x in [0.7, 1.1, -1.3] {
            // The declared parameter, then `t`.
            let p = &[ir_par(&ir, src), 0.0][..];
            let v: Vec<f64> = (0..3 * (fb.n + fb.n_p)).map(|i| 0.3 + i as f64).collect();
            for kind in &kinds {
                if *kind == EntryKind::Aux && fb.n_aux == 0 {
                    continue;
                }
                let len = out_len(&fb, kind);
                let e = ir
                    .entry(kind)
                    .unwrap_or_else(|err| panic!("{kind:?} of {src:?}: {err:?}"));
                let mut want = vec![0.0; len];
                crate::eval::try_eval_entry(&ir.dag, &e, &[x], p, &v, &mut want).unwrap();
                let mut got = vec![0.0; len];
                fb.try_eval(kind, &[x], p, &v, &mut got).unwrap();
                // F and the first derivatives agree bit for bit. A Dir row may sum its
                // terms in another order: the non-recursive `x'=(x*x*x)*x^2` differs by
                // 2 ulp there too, so the bound is a few ulp, not 0.
                let ulps = if matches!(kind, EntryKind::Dir(_)) {
                    4
                } else {
                    0
                };
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        same(*g, *w) || ulp_gap(*g, *w) <= ulps,
                        "{kind:?}[{i}] of {src:?} at {x}: {g:?} vs {w:?}"
                    );
                }
            }
        }
    }
    // The exponent case is only a test when the interpreter's `powi` and `pow` differ at
    // one of the points.
    assert!(
        [0.7f64, 1.1, -1.3]
            .iter()
            .any(|&x| Scalar::powi(x, 7).to_bits() != x.powf(7.0).to_bits())
    );
}

/// The distance in units in the last place between two finite values of one sign.
fn ulp_gap(a: f64, b: f64) -> u64 {
    if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
        return u64::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

/// The declared value of the one parameter of a test model.
fn ir_par(ir: &IrModel, src: &str) -> f64 {
    assert_eq!(ir.n_p(), 1, "{src}");
    let at = src.find("par k=").unwrap() + 6;
    src[at..].split('\n').next().unwrap().parse().unwrap()
}

/// The interpreter evaluates the arguments of a wrapper before its body, while the DAG
/// inlines the wrapper and evaluates an argument only where the body reads it. In
/// `g(k, zz(k))` with `g(a,b)=if(a<0)then(0)else(b)` the DAG never calls `zz` at k = -1,
/// and the interpreter does. An owner item in docs/STATUS.md asks for the SPEC sentence.
#[test]
fn wrapper_arguments_are_eager_in_the_interpreter() {
    let src = format!("par k=0\n{RUNAWAY}g(a,b)=if(a<0)then(0)else(b)\nx'=g(k,zz(k))\ndone\n");
    let (fb, mut ir) = setup(&src);
    let e = ir.entry(&EntryKind::F).unwrap();
    let mut want = [7.0];
    let st =
        crate::eval::try_eval_entry(&ir.dag, &e, &[0.0], &[-1.0, 0.0], &[], &mut want).unwrap();
    assert_eq!((want[0], st.calls), (0.0, 0));
    let mut got = [7.0];
    let err = fb
        .interp
        .try_eval_capped(&[0.0], &[-1.0, 0.0], &mut got, &mut [], 100)
        .unwrap_err();
    assert!(matches!(err, EvalError::Depth { .. }), "{err:?}");
}
