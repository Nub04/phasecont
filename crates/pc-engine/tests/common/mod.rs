//! Helpers shared by the pc-engine integration tests.
#![allow(dead_code)]

use pc_engine::ModelEval;
use pc_ir::EntryKind;
use pc_ode::{FsLoader, Model, ModelInfo, Severity, SourceMap, load as ode_load};
use std::path::Path;

/// One RK4 step of x' = T f(x) in rescaled time.
pub fn rk4(m: &mut dyn ModelEval, p: &[f64], t: f64, x: &mut [f64], dt: f64) {
    let n = x.len();
    let mut f = |y: &[f64]| {
        let mut out = vec![0.0; n];
        m.eval(&EntryKind::F, y, p, &[], &mut out).unwrap();
        out.iter().map(|v| t * v).collect::<Vec<f64>>()
    };
    let add = |y: &[f64], k: &[f64], s: f64| -> Vec<f64> {
        y.iter().zip(k).map(|(a, b)| a + s * b).collect()
    };
    let k1 = f(x);
    let k2 = f(&add(x, &k1, dt / 2.0));
    let k3 = f(&add(x, &k2, dt / 2.0));
    let k4 = f(&add(x, &k3, dt));
    for i in 0..n {
        x[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// RK4 in `steps` steps over time `t` on x' = f(x) and on the variational equation
/// Phi' = f_x(x) Phi with Phi(0) = I. Returns x(t) and Phi(t), column-major.
pub fn flow(
    m: &mut dyn ModelEval,
    p: &[f64],
    x0: &[f64],
    t: f64,
    steps: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = x0.len();
    let mut rhs = |z: &[f64]| {
        let (x, phi) = z.split_at(n);
        let mut out = vec![0.0; n + n * n];
        let mut jac = vec![0.0; n * n];
        m.eval(&EntryKind::F, x, p, &[], &mut out[..n]).unwrap();
        m.eval(&EntryKind::JacU, x, p, &[], &mut jac).unwrap();
        for c in 0..n {
            for i in 0..n {
                out[n + i + n * c] = (0..n).map(|k| jac[i + n * k] * phi[k + n * c]).sum();
            }
        }
        out
    };
    let mut z = x0.to_vec();
    z.extend((0..n * n).map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 }));
    let dt = t / steps as f64;
    let add = |y: &[f64], k: &[f64], s: f64| -> Vec<f64> {
        y.iter().zip(k).map(|(a, b)| a + s * b).collect()
    };
    for _ in 0..steps {
        let k1 = rhs(&z);
        let k2 = rhs(&add(&z, &k1, dt / 2.0));
        let k3 = rhs(&add(&z, &k2, dt / 2.0));
        let k4 = rhs(&add(&z, &k3, dt));
        for i in 0..z.len() {
            z[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    let phi = z.split_off(n);
    (z, phi)
}

/// Solves a x = b by Gaussian elimination with partial pivoting, a row-major and square.
/// Returns x and det(a).
pub fn gauss(mut a: Vec<f64>, mut b: Vec<f64>) -> (Vec<f64>, f64) {
    let n = b.len();
    let mut det = 1.0;
    for c in 0..n {
        let piv = (c..n)
            .max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))
            .unwrap();
        if piv != c {
            for k in 0..n {
                a.swap(c * n + k, piv * n + k);
            }
            b.swap(c, piv);
            det = -det;
        }
        det *= a[c * n + c];
        for i in c + 1..n {
            let l = a[i * n + c] / a[c * n + c];
            for k in c..n {
                a[i * n + k] -= l * a[c * n + k];
            }
            b[i] -= l * b[c];
        }
    }
    for c in (0..n).rev() {
        let s: f64 = (c + 1..n).map(|k| a[c * n + k] * b[k]).sum();
        b[c] = (b[c] - s) / a[c * n + c];
    }
    (b, det)
}

/// The condition that defines a curve of cycles in the shooting oracle.
#[derive(Clone, Copy, Debug)]
pub enum Defining {
    /// A fold of cycles: the shooting Jacobian [M - I, f(u0); f(u_ref)^T, 0] in (u0, T)
    /// is singular.
    Fold,
    /// A period doubling: det(M + I) = 0.
    Flip,
}

/// Residual of the shooting system in z = (u0, T, a1), with a1 parameter `free` of `p`:
/// the periodicity phi_T(u0) - u0, the section f(u_ref).(u0 - u_ref) and the condition.
fn shooting_residual(
    m: &mut dyn ModelEval,
    p: &[f64],
    free: usize,
    (u_ref, f_ref): (&[f64], &[f64]),
    z: &[f64],
    steps: usize,
    cond: Defining,
) -> Vec<f64> {
    let n = u_ref.len();
    let mut q = p.to_vec();
    q[free] = z[n + 1];
    let (end, mono) = flow(m, &q, &z[..n], z[n], steps);
    let mut r: Vec<f64> = (0..n).map(|i| end[i] - z[i]).collect();
    r.push((0..n).map(|i| f_ref[i] * (z[i] - u_ref[i])).sum());
    let g = match cond {
        Defining::Fold => {
            let mut f0 = vec![0.0; n];
            m.eval(&EntryKind::F, &z[..n], &q, &[], &mut f0).unwrap();
            let k = n + 1;
            let mut a = vec![0.0; k * k];
            for i in 0..n {
                for j in 0..n {
                    a[i * k + j] = mono[i + n * j] - if i == j { 1.0 } else { 0.0 };
                }
                a[i * k + n] = f0[i];
                a[n * k + i] = f_ref[i];
            }
            gauss(a, vec![0.0; k]).1
        }
        Defining::Flip => {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = mono[i + n * j] + if i == j { 1.0 } else { 0.0 };
                }
            }
            gauss(a, vec![0.0; n]).1
        }
    };
    r.push(g);
    r
}

/// The shooting oracle of a curve of cycles: Newton, with a central-difference Jacobian,
/// on the shooting system of [`shooting_residual`] with `steps` RK4 steps per period,
/// from the guess (u0, T, a1 = p[free]) and with the section through u0. Returns
/// (u0, T, a1); panics when Newton does not converge in 40 iterations.
pub fn shoot(
    m: &mut dyn ModelEval,
    p: &[f64],
    free: usize,
    u0: &[f64],
    period: f64,
    steps: usize,
    cond: Defining,
) -> (Vec<f64>, f64, f64) {
    let n = u0.len();
    let mut f_ref = vec![0.0; n];
    m.eval(&EntryKind::F, u0, p, &[], &mut f_ref).unwrap();
    let mut z = u0.to_vec();
    z.extend([period, p[free]]);
    let k = n + 2;
    for _ in 0..40 {
        let r = shooting_residual(m, p, free, (u0, &f_ref), &z, steps, cond);
        let mut jac = vec![0.0; k * k];
        for j in 0..k {
            let h = 1e-6 * z[j].abs().max(1.0);
            let (mut zp, mut zm) = (z.clone(), z.clone());
            zp[j] += h;
            zm[j] -= h;
            let rp = shooting_residual(m, p, free, (u0, &f_ref), &zp, steps, cond);
            let rm = shooting_residual(m, p, free, (u0, &f_ref), &zm, steps, cond);
            for i in 0..k {
                jac[i * k + j] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        let (dz, _) = gauss(jac, r.iter().map(|v| -v).collect());
        let mut big = 0.0_f64;
        for j in 0..k {
            z[j] += dz[j];
            big = big.max(dz[j].abs() / z[j].abs().max(1.0));
        }
        if big <= 1e-12 {
            let a1 = z.pop().unwrap();
            let t = z.pop().unwrap();
            return (z, t, a1);
        }
    }
    panic!("the shooting oracle did not converge in 40 Newton iterations");
}

/// Loads `tests/models/<name>.ode`. Panics with the diagnostics when the file does not
/// read or has an error.
pub fn load(name: &str) -> (Model, ModelInfo) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/models")
        .join(format!("{name}.ode"));
    let mut map = SourceMap::new();
    let mut loader = FsLoader::new();
    let top = loader
        .load_top(&mut map, &path)
        .unwrap_or_else(|e| panic!("cannot read {e}"));
    let loaded = ode_load(top, &mut map, &mut loader);
    let errors: Vec<String> = loaded
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{} {}", d.code, d.message))
        .collect();
    if !errors.is_empty() {
        panic!("{name}.ode has errors:\n{}", errors.join("\n"));
    }
    (loaded.expanded, loaded.info)
}

/// Distance in units in the last place; two NaN match, one NaN is `u64::MAX`.
pub fn ulp_dist(a: f64, b: f64) -> u64 {
    if a.is_nan() && b.is_nan() {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let key = |x: f64| {
        let u = x.to_bits();
        if u >> 63 == 1 { !u } else { u | (1 << 63) }
    };
    key(a).abs_diff(key(b))
}

/// RK4 with step `dt` from `x` for time `settle` onto the attractor at `params`, then
/// on to two upward crossings of x[coord] = level, which give the period. Returns the
/// orbit from the first crossing at the N m + 1 points of a uniform mesh with N m =
/// `nm`, and the period.
pub fn attractor_orbit(
    m: &mut dyn ModelEval,
    params: &[f64],
    mut x: Vec<f64>,
    (settle, dt): (f64, f64),
    (coord, level): (usize, f64),
    nm: usize,
) -> (Vec<f64>, f64) {
    let mut p = params.to_vec();
    p.push(0.0);
    for _ in 0..(settle / dt) as usize {
        rk4(m, &p, 1.0, &mut x, dt);
    }
    let (mut t, mut at, mut first) = (0.0, Vec::new(), Vec::new());
    while at.len() < 2 {
        let y0 = x[coord];
        rk4(m, &p, 1.0, &mut x, dt);
        t += dt;
        if y0 < level && x[coord] >= level {
            at.push(t);
            if first.is_empty() {
                first = x.clone();
            }
        }
        assert!(t <= settle, "no cycle at {params:?}");
    }
    let period = at[1] - at[0];
    let sub = 20;
    let mut y = first;
    let mut orbit = y.clone();
    for _ in 0..nm {
        for _ in 0..sub {
            rk4(m, &p, period, &mut y, 1.0 / (nm * sub) as f64);
        }
        orbit.extend(&y);
    }
    (orbit, period)
}
