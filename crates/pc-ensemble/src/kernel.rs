//! The text of an ensemble kernel: one thread integrates one member.
//!
//! The kernel is a transcription of [`crate::traj`], so a backend that runs it without fused
//! multiply-add reproduces the CPU backend operation for operation. The text is built from
//! the right-hand side of the model (`pc_codegen::kernel::emit_rhs`), a dialect prelude and
//! one of two fixed integrator templates. The templates hold no sizes: the model sizes come
//! in as macros (`PC_N`, `PC_WL`), so the templates are plain strings.
//!
//! One launch advances every member by at most `steps` step attempts. The state lives in
//! the buffers described in [`crate::abi`], so the host relaunches until no member is
//! running.
//!
//! The C and CUDA dialects have an entry point so far. The OpenCL and Metal entries come
//! with their backends.

use pc_codegen::kernel::{Dialect, KernelError, Scalar, emit_rhs, float_literal, prelude};

use crate::real::Real;
use crate::tape::Tape;
use crate::traj::Coef;

/// Control words per member, see [`crate::abi`].
pub const CTL: usize = 8;

/// The integrator a kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    Rk4,
    Dp5,
}

/// Workspace values per member: `y`, `y1`, `yt`, seven stages, then `t`, `h` and `facold`.
pub fn work_len(n: usize) -> usize {
    10 * n + 3
}

/// The kernel text for `tape`, or the reason the dialect or model cannot be lowered.
pub fn kernel_source(
    tape: &Tape,
    integrator: Integrator,
    scalar: Scalar,
    dialect: Dialect,
) -> Result<String, KernelError> {
    // The prelude comes first so that a dialect without the scalar reports that first.
    let mut s = prelude(dialect, scalar)?;
    let entry = match dialect {
        Dialect::C => ENTRY_C,
        Dialect::Cuda => ENTRY_CUDA,
        Dialect::OpenCl | Dialect::Msl => {
            return Err(KernelError::Unsupported(format!(
                "{} entry point",
                dialect.name()
            )));
        }
    };
    s.push_str(&format!("#define PC_N {}u\n", tape.n));
    s.push_str(&format!("#define PC_WL {}u\n", work_len(tape.n)));
    s.push_str(&format!("#define PC_CTL {CTL}u\n"));
    s.push_str(STATUS);
    s.push_str(&match scalar {
        Scalar::F32 => consts::<f32>(integrator, scalar),
        Scalar::F64 => consts::<f64>(integrator, scalar),
    });
    s.push_str(&emit_rhs(
        &tape.ops,
        &tape.outputs,
        tape.n_p,
        scalar,
        "pc_rhs",
    )?);
    s.push_str(BAD_STATE);
    match integrator {
        Integrator::Rk4 => s.push_str(RK4),
        Integrator::Dp5 => {
            s.push_str(DP5_HELPERS);
            s.push_str(DP5);
        }
    }
    s.push_str(entry);
    Ok(s)
}

fn def(out: &mut String, name: &str, lit: &str) {
    out.push_str("#define PC_");
    out.push_str(name);
    out.push(' ');
    out.push_str(lit);
    out.push('\n');
}

fn array(out: &mut String, name: &str, values: &[String]) {
    out.push_str(&format!(
        "PC_CONSTANT pc_real {name}[{}] = {{{}}};\n",
        values.len(),
        values.join(", ")
    ));
}

/// The constants of the integrator, rounded to `T` the way the CPU backend rounds them.
fn consts<T: Real>(integrator: Integrator, scalar: Scalar) -> String {
    let co = Coef::<T>::new();
    let lit = |v: T| float_literal(v.to_f64(), scalar);
    let mut s = String::new();
    match integrator {
        Integrator::Rk4 => {
            def(&mut s, "HALF", &lit(co.half));
            def(&mut s, "TWO", &lit(co.two));
            def(&mut s, "SIX", &lit(co.six));
        }
        Integrator::Dp5 => {
            def(&mut s, "SAFE", &lit(co.safe));
            def(&mut s, "INV_FAC1", &lit(co.inv_fac1));
            def(&mut s, "INV_FAC2", &lit(co.inv_fac2));
            def(&mut s, "BETA", &lit(co.beta));
            def(&mut s, "EXPO1", &lit(co.expo1));
            def(&mut s, "FACOLD0", &lit(co.facold0));
            def(&mut s, "FACOLD_MIN", &lit(co.facold_min));
            def(&mut s, "UROUND", &lit(co.uround));
            def(&mut s, "TENTH", &lit(co.tenth));
            def(&mut s, "C101", &lit(co.c101));
            def(&mut s, "C1E10", &lit(co.c1e10));
            def(&mut s, "C1E6", &lit(co.c1e6));
            def(&mut s, "C1E3", &lit(co.c1e3));
            def(&mut s, "C1E15", &lit(co.c1e15));
            def(&mut s, "C001", &lit(co.c001));
            def(&mut s, "C100", &lit(co.c100));
            def(&mut s, "FIFTH", &lit(co.fifth));
            let c: Vec<String> = co.c.iter().map(|&v| lit(v)).collect();
            let a: Vec<String> = co.a.iter().flatten().map(|&v| lit(v)).collect();
            let e: Vec<String> = co.e.iter().map(|&v| lit(v)).collect();
            array(&mut s, "pc_cn", &c);
            array(&mut s, "pc_am", &a);
            array(&mut s, "pc_em", &e);
        }
    }
    s
}

const STATUS: &str = r#"
#define PC_OK 0u
#define PC_BOUNDS 1u
#define PC_STEP_SMALL 2u
#define PC_MAX_STEPS 3u
#define PC_NONFINITE 4u
#define PC_RUNNING 5u

"#;

/// Returns `PC_NONFINITE` or `PC_BOUNDS` when the new state is unusable, `PC_RUNNING` when
/// it is fine. A non-finite value wins over the bound, as on the CPU.
const BAD_STATE: &str = r#"
PC_FN pc_uint pc_bad_state(PC_GLOBAL pc_real *y1, pc_real bounds)
{
    pc_uint i;
    for (i = 0u; i < PC_N; i++) {
        if (!pc_isfinite(y1[i])) {
            return PC_NONFINITE;
        }
    }
    for (i = 0u; i < PC_N; i++) {
        if (pc_fabs(y1[i]) > bounds) {
            return PC_BOUNDS;
        }
    }
    return PC_RUNNING;
}

"#;

const RK4: &str = r#"
PC_FN void pc_member(PC_IN pc_real *rc, PC_IN pc_ulong *segm, PC_IN pc_real *x0,
                     PC_IN pc_real *par, PC_GLOBAL pc_real *w, PC_GLOBAL pc_real *out,
                     PC_GLOBAL pc_uint *c, pc_ulong budget, pc_uint max_steps, pc_uint ns)
{
    PC_GLOBAL pc_real *y = w;
    PC_GLOBAL pc_real *y1 = w + PC_N;
    PC_GLOBAL pc_real *yt = w + 2u * PC_N;
    PC_GLOBAL pc_real *k = w + 3u * PC_N;
    PC_IN pc_real *seg_a = rc + 5u + ns;
    PC_IN pc_real *seg_b = rc + 5u + 2u * ns;
    PC_IN pc_real *seg_h = rc + 5u + 3u * ns;
    pc_real bounds = rc[1];
    pc_uint st = c[0];
    pc_uint accepted = c[2];
    pc_uint si = c[4];
    pc_uint ii = c[5];
    pc_real t = w[10u * PC_N];
    pc_uint i;

    if (st != PC_RUNNING) {
        return;
    }
    if (c[1] == 0u) {
        c[1] = 1u;
        t = rc[0];
        for (i = 0u; i < PC_N; i++) {
            y[i] = x0[i];
        }
    }
    while (budget > 0u && st == PC_RUNNING) {
        pc_ulong m = segm[si];
        pc_real sa = seg_a[si];
        pc_real sb = seg_b[si];
        pc_real h = seg_h[si];
        pc_uint last = ((pc_ulong)ii + 1u == m) ? 1u : 0u;
        pc_real ts = sa + (pc_real)ii * h;
        pc_real t_next = (last != 0u) ? sb : sa + (pc_real)(ii + 1u) * h;
        pc_real hh = h * PC_HALF;
        pc_real h6 = h / PC_SIX;
        pc_uint bad;
        budget -= 1u;
        pc_rhs(ts, y, par, k);
        for (i = 0u; i < PC_N; i++) {
            yt[i] = y[i] + hh * k[i];
        }
        pc_rhs(ts + hh, yt, par, k + PC_N);
        for (i = 0u; i < PC_N; i++) {
            yt[i] = y[i] + hh * k[PC_N + i];
        }
        pc_rhs(ts + hh, yt, par, k + 2u * PC_N);
        for (i = 0u; i < PC_N; i++) {
            yt[i] = y[i] + h * k[2u * PC_N + i];
        }
        pc_rhs(t_next, yt, par, k + 3u * PC_N);
        for (i = 0u; i < PC_N; i++) {
            pc_real sum = k[i] + PC_TWO * k[PC_N + i];
            sum = sum + PC_TWO * k[2u * PC_N + i];
            sum = sum + k[3u * PC_N + i];
            y1[i] = y[i] + h6 * sum;
        }
        bad = pc_bad_state(y1, bounds);
        if (bad != PC_RUNNING) {
            st = bad;
        } else {
            for (i = 0u; i < PC_N; i++) {
                y[i] = y1[i];
            }
            t = t_next;
            accepted += 1u;
            ii += 1u;
            if (last != 0u) {
                for (i = 0u; i < PC_N; i++) {
                    out[si * PC_N + i] = y[i];
                }
                si += 1u;
                ii = 0u;
                if (si == ns) {
                    st = PC_OK;
                }
            }
            if (st == PC_RUNNING && accepted >= max_steps) {
                st = PC_MAX_STEPS;
            }
        }
    }
    w[10u * PC_N] = t;
    c[0] = st;
    c[2] = accepted;
    c[4] = si;
    c[5] = ii;
}

"#;

const DP5_HELPERS: &str = r#"
PC_FN pc_real pc_fmin2(pc_real a, pc_real b)
{
    return (a != a) ? b : ((b != b) ? a : ((b < a) ? b : a));
}

PC_FN pc_real pc_fmax2(pc_real a, pc_real b)
{
    return (a != a) ? b : ((b != b) ? a : ((b > a) ? b : a));
}

PC_FN pc_real pc_clamp(pc_real x, pc_real lo, pc_real hi)
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

PC_FN pc_real pc_hinit(pc_real t0, pc_real rtol, pc_real atol, pc_real hmax,
                       PC_GLOBAL pc_real *y, PC_GLOBAL pc_real *yt, PC_GLOBAL pc_real *k,
                       PC_IN pc_real *par)
{
    pc_uint i;
    pc_real dnf = PC_ZERO;
    pc_real dny = PC_ZERO;
    pc_real s2 = PC_ZERO;
    pc_real h;
    pc_real der2;
    pc_real der12;
    pc_real h1;
    for (i = 0u; i < PC_N; i++) {
        pc_real sk = atol + rtol * pc_fabs(y[i]);
        pc_real a = k[i] / sk;
        pc_real b = y[i] / sk;
        dnf = dnf + a * a;
        dny = dny + b * b;
    }
    h = (dnf <= PC_C1E10 || dny <= PC_C1E10) ? PC_C1E6 : pc_sqrt(dny / dnf) * PC_C001;
    h = pc_fmin2(h, hmax);
    for (i = 0u; i < PC_N; i++) {
        yt[i] = y[i] + h * k[i];
    }
    pc_rhs(t0 + h, yt, par, k + PC_N);
    for (i = 0u; i < PC_N; i++) {
        pc_real sk = atol + rtol * pc_fabs(y[i]);
        pc_real d = (k[PC_N + i] - k[i]) / sk;
        s2 = s2 + d * d;
    }
    der2 = pc_sqrt(s2) / pc_fabs(h);
    der12 = pc_fmax2(pc_fabs(der2), pc_sqrt(dnf));
    if (der12 <= PC_C1E15) {
        h1 = pc_fmax2(PC_C1E6, pc_fabs(h) * PC_C1E3);
    } else {
        h1 = pc_pow(PC_C001 / der12, PC_FIFTH);
    }
    return pc_fmin2(pc_fmin2(PC_C100 * pc_fabs(h), h1), hmax);
}

PC_FN pc_real pc_dp5_step(PC_GLOBAL pc_real *y, PC_GLOBAL pc_real *y1, PC_GLOBAL pc_real *yt,
                          PC_GLOBAL pc_real *k, PC_IN pc_real *par, pc_real t, pc_real h,
                          pc_real t_end, pc_real rtol, pc_real atol)
{
    pc_uint s;
    pc_uint i;
    pc_uint j;
    pc_real acc;
    pc_real ts;
    pc_real sum = PC_ZERO;
    for (s = 1u; s <= 5u; s++) {
        for (i = 0u; i < PC_N; i++) {
            acc = PC_ZERO;
            for (j = 0u; j < s; j++) {
                acc = acc + pc_am[(s - 1u) * 6u + j] * k[j * PC_N + i];
            }
            yt[i] = y[i] + h * acc;
        }
        ts = (pc_cn[s - 1u] == PC_ONE) ? t_end : t + pc_cn[s - 1u] * h;
        pc_rhs(ts, yt, par, k + s * PC_N);
    }
    for (i = 0u; i < PC_N; i++) {
        acc = PC_ZERO;
        for (j = 0u; j < 6u; j++) {
            acc = acc + pc_am[30u + j] * k[j * PC_N + i];
        }
        y1[i] = y[i] + h * acc;
    }
    pc_rhs(t_end, y1, par, k + 6u * PC_N);
    for (i = 0u; i < PC_N; i++) {
        pc_real e = PC_ZERO;
        pc_real sk;
        pc_real r;
        for (j = 0u; j < 7u; j++) {
            e = e + pc_em[j] * k[j * PC_N + i];
        }
        sk = atol + rtol * pc_fmax2(pc_fabs(y[i]), pc_fabs(y1[i]));
        r = h * e / sk;
        sum = sum + r * r;
    }
    return pc_sqrt(sum / (pc_real)PC_N);
}

"#;

const DP5: &str = r#"
PC_FN void pc_member(PC_IN pc_real *rc, PC_IN pc_ulong *segm, PC_IN pc_real *x0,
                     PC_IN pc_real *par, PC_GLOBAL pc_real *w, PC_GLOBAL pc_real *out,
                     PC_GLOBAL pc_uint *c, pc_ulong budget, pc_uint max_steps, pc_uint ns)
{
    PC_GLOBAL pc_real *y = w;
    PC_GLOBAL pc_real *y1 = w + PC_N;
    PC_GLOBAL pc_real *yt = w + 2u * PC_N;
    PC_GLOBAL pc_real *k = w + 3u * PC_N;
    PC_IN pc_real *times = rc + 5u;
    pc_real bounds = rc[1];
    pc_real rtol = rc[2];
    pc_real atol = rc[3];
    pc_real hmax = rc[4];
    pc_uint st = c[0];
    pc_uint accepted = c[2];
    pc_uint rejected = c[3];
    pc_uint si = c[4];
    pc_uint reject = c[6];
    pc_uint nan_err = c[7];
    pc_real t = w[10u * PC_N];
    pc_real h = w[10u * PC_N + 1u];
    pc_real facold = w[10u * PC_N + 2u];
    pc_uint i;

    (void)segm;
    if (st != PC_RUNNING) {
        return;
    }
    if (c[1] == 0u) {
        c[1] = 1u;
        t = rc[0];
        for (i = 0u; i < PC_N; i++) {
            y[i] = x0[i];
        }
        pc_rhs(t, y, par, k);
        for (i = 0u; i < PC_N; i++) {
            if (!pc_isfinite(k[i])) {
                st = PC_NONFINITE;
            }
        }
        if (st == PC_RUNNING) {
            h = pc_hinit(rc[0], rtol, atol, hmax, y, yt, k, par);
            if (!(pc_isfinite(h) && h > PC_ZERO)) {
                st = PC_NONFINITE;
            } else {
                facold = PC_FACOLD0;
            }
        }
    }
    while (budget > 0u && st == PC_RUNNING) {
        budget -= 1u;
        if (!(PC_TENTH * pc_fabs(h) > pc_fabs(t) * PC_UROUND)) {
            st = (nan_err != 0u) ? PC_NONFINITE : PC_STEP_SMALL;
        } else {
            pc_real s = times[si];
            pc_real h_free = h;
            pc_uint landing = 0u;
            pc_uint last;
            pc_real t_end;
            pc_real err;
            pc_real fac11;
            pc_real fac;
            pc_real hnew;
            if (t + PC_C101 * h > s) {
                h = s - t;
                landing = 1u;
            }
            last = (landing != 0u && si + 1u == ns) ? 1u : 0u;
            t_end = (landing != 0u) ? s : t + h;
            err = pc_dp5_step(y, y1, yt, k, par, t, h, t_end, rtol, atol);
            fac11 = pc_pow(err, PC_EXPO1);
            fac = pc_clamp(fac11 / pc_pow(facold, PC_BETA) / PC_SAFE, PC_INV_FAC2, PC_INV_FAC1);
            hnew = h / fac;
            if (err <= PC_ONE) {
                st = pc_bad_state(y1, bounds);
                if (st == PC_RUNNING) {
                    facold = pc_fmax2(err, PC_FACOLD_MIN);
                    accepted += 1u;
                    t = t_end;
                    nan_err = 0u;
                    for (i = 0u; i < PC_N; i++) {
                        y[i] = y1[i];
                    }
                    for (i = 0u; i < PC_N; i++) {
                        k[i] = k[6u * PC_N + i];
                    }
                    if (landing != 0u) {
                        for (i = 0u; i < PC_N; i++) {
                            out[si * PC_N + i] = y[i];
                        }
                        si += 1u;
                        if (last != 0u) {
                            st = PC_OK;
                        }
                    }
                    if (st == PC_RUNNING && accepted >= max_steps) {
                        st = PC_MAX_STEPS;
                    }
                    if (st == PC_RUNNING) {
                        if (pc_fabs(hnew) > hmax) {
                            hnew = hmax;
                        }
                        if (reject != 0u) {
                            hnew = pc_fmin2(pc_fabs(hnew), pc_fabs(h));
                        }
                        if (landing != 0u && h < h_free && hnew < h_free) {
                            hnew = h_free;
                        }
                        reject = 0u;
                        h = hnew;
                    }
                }
            } else {
                hnew = h / pc_fmin2(PC_INV_FAC1, fac11 / PC_SAFE);
                reject = 1u;
                rejected += 1u;
                nan_err = (err != err) ? 1u : 0u;
                h = hnew;
            }
        }
    }
    w[10u * PC_N] = t;
    w[10u * PC_N + 1u] = h;
    w[10u * PC_N + 2u] = facold;
    c[0] = st;
    c[2] = accepted;
    c[3] = rejected;
    c[4] = si;
    c[6] = reject;
    c[7] = nan_err;
}

"#;

const ENTRY_C: &str = r#"
void pc_ens(PC_IN pc_real *rc, PC_IN pc_ulong *segm, PC_IN pc_real *x0, PC_IN pc_real *par,
            PC_GLOBAL pc_real *work, PC_GLOBAL pc_real *out, PC_GLOBAL pc_uint *ctl,
            pc_ulong members, pc_ulong steps, pc_uint max_steps, pc_uint ns, pc_ulong pstride)
{
    pc_ulong m;
    for (m = 0u; m < members; m++) {
        pc_member(rc, segm, x0 + m * PC_N, par + m * pstride, work + m * PC_WL,
                  out + m * (pc_ulong)ns * PC_N, ctl + m * PC_CTL, steps, max_steps, ns);
    }
}
"#;

/// One thread per member. The grid may hold more threads than members, so the guard comes
/// first. `extern "C"` keeps the name that the host looks up.
const ENTRY_CUDA: &str = r#"
extern "C" __global__ void pc_ens(PC_IN pc_real *rc, PC_IN pc_ulong *segm, PC_IN pc_real *x0,
                                  PC_IN pc_real *par, PC_GLOBAL pc_real *work,
                                  PC_GLOBAL pc_real *out, PC_GLOBAL pc_uint *ctl,
                                  pc_ulong members, pc_ulong steps, pc_uint max_steps,
                                  pc_uint ns, pc_ulong pstride)
{
    pc_ulong m = (pc_ulong)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= members) {
        return;
    }
    pc_member(rc, segm, x0 + m * PC_N, par + m * pstride, work + m * PC_WL,
              out + m * (pc_ulong)ns * PC_N, ctl + m * PC_CTL, steps, max_steps, ns);
}
"#;
