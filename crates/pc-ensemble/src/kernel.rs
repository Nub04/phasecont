//! The text of an ensemble kernel: one thread integrates one member.
//!
//! The kernel is a transcription of [`crate::traj`], so a backend that runs it without fused
//! multiply-add reproduces the CPU backend operation for operation. The text is built from
//! the right-hand side of the model (`pc_codegen::kernel::emit_rhs`), a dialect prelude and
//! one of three fixed integrator templates (RK4, Dormand-Prince and the iteration of a map). The templates hold no sizes: the model sizes come
//! in as macros (`PC_N`, `PC_WL`), so the templates are plain strings.
//!
//! One launch advances every member by at most `steps` step attempts. The state lives in
//! the buffers described in [`crate::abi`], so the host relaunches until no member is
//! running.
//!
//! Every dialect has an entry point: C loops over the members, the others run one thread or
//! work item per member. Metal has no double, so its kernels exist for `f32` only.
//!
//! The scratch of a step (`y`, `y1`, and for the integrators `yt` and the stage vectors)
//! lives in one of two places,
//! chosen by [`Layout`]. The private layout keeps it in arrays of the thread. The global
//! layout keeps it in the member's slice of `work`, behind the words a launch hands on, for
//! the models whose arrays do not fit in the local memory of a CUDA thread. Only the
//! declarations differ, so both layouts run the same operations in the same order.

use pc_codegen::kernel::{Dialect, KernelError, Scalar, emit_rhs, float_literal, prelude};

use crate::job::StackLayout;
use crate::real::Real;
use crate::tape::{KERNEL_FRAMES, Tape, emit_calls_rhs, kernel_frame, kernel_stride};
use crate::traj::Coef;

/// Control words per member, see [`crate::abi`].
pub const CTL: usize = 8;

/// The integrator a kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    Rk4,
    Dp5,
    /// The iteration of a map, one call of F per iterate (SPEC section 8aa).
    Iter,
}

/// Values per member in the `work` buffer: what a launch hands to the next one.
///
/// RK4 and the iteration of a map keep `t`, then `y`. DP5 keeps `t`, `h` and `facold`, then `y`
/// and the first stage `k0`, which is the last stage of the previous step. The trial state and the other stages are
/// private to the thread and live only inside a launch.
pub fn work_len(n: usize, integrator: Integrator) -> usize {
    match integrator {
        Integrator::Rk4 | Integrator::Iter => n + 1,
        Integrator::Dp5 => 2 * n + 3,
    }
}

/// Where the scratch of a step lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Arrays of the thread: registers or local memory.
    Private,
    /// The member's slice of `work`, after the words a launch hands on. `t` stays first.
    Global,
}

impl Layout {
    /// Values per member in the `work` buffer under this layout.
    ///
    /// The global layout adds the private arrays to what a launch hands on: `8n + 1` for RK4,
    /// `12n + 3` for Dormand-Prince and `3n + 1` for the iteration of a map.
    pub fn work_len(self, n: usize, integrator: Integrator) -> usize {
        match self {
            Layout::Private => work_len(n, integrator),
            Layout::Global => work_len(n, integrator).saturating_add(private_reals(n, integrator)),
        }
    }
}

/// Reals that one thread of a kernel holds in private arrays: for RK4 and Dormand-Prince the
/// state, the trial state, a second trial state and the stage vectors (4 for RK4, 7 for
/// Dormand-Prince), and for the iteration of a map the state and the next state. A test reads
/// the declarations in the kernel text and checks this count.
pub fn private_reals(n: usize, integrator: Integrator) -> usize {
    match integrator {
        Integrator::Rk4 => n.saturating_mul(7),
        Integrator::Dp5 => n.saturating_mul(10),
        Integrator::Iter => n.saturating_mul(2),
    }
}

/// The kernel text for `tape` with private scratch, or the reason the dialect or model cannot
/// be lowered.
pub fn kernel_source(
    tape: &Tape,
    integrator: Integrator,
    scalar: Scalar,
    dialect: Dialect,
) -> Result<String, KernelError> {
    kernel_source_in(tape, integrator, scalar, dialect, Layout::Private)
}

/// The kernel text for `tape` with the scratch in `layout`. Under the global layout the
/// prelude's `PC_SCR` is redefined as `PC_GLOBAL` before the first function that takes
/// scratch, so the same text serves all four dialects, including OpenCL C 1.2 and Metal,
/// which cannot pass a global pointer where a private one is declared.
pub fn kernel_source_in(
    tape: &Tape,
    integrator: Integrator,
    scalar: Scalar,
    dialect: Dialect,
    layout: Layout,
) -> Result<String, KernelError> {
    kernel_source_with(
        tape,
        integrator,
        scalar,
        dialect,
        layout,
        StackLayout::Member,
    )
}

/// [`kernel_source_in`] with the call stack in `stack`. A tape without calls gives the same
/// text under either stack layout.
pub fn kernel_source_with(
    tape: &Tape,
    integrator: Integrator,
    scalar: Scalar,
    dialect: Dialect,
    layout: Layout,
    stack: StackLayout,
) -> Result<String, KernelError> {
    // The prelude comes first so that a dialect without the scalar reports that first.
    let mut s = prelude(dialect, scalar)?;
    let entry = match dialect {
        Dialect::C => ENTRY_C,
        Dialect::Cuda => ENTRY_CUDA,
        Dialect::OpenCl => ENTRY_OPENCL,
        Dialect::Msl => ENTRY_MSL,
    };
    s.push_str(&format!("#define PC_N {}u\n", tape.n));
    s.push_str(&format!(
        "#define PC_WL {}u\n",
        layout.work_len(tape.n, integrator)
    ));
    if layout == Layout::Global {
        s.push_str(&format!(
            "#define PC_PERSIST {}u\n",
            work_len(tape.n, integrator)
        ));
    }
    s.push_str(&format!("#define PC_CTL {CTL}u\n"));
    if layout == Layout::Global {
        s.push_str("#undef PC_SCR\n#define PC_SCR PC_GLOBAL\n");
    }
    s.push_str(STATUS);
    s.push_str(&match scalar {
        Scalar::F32 => consts::<f32>(integrator, scalar),
        Scalar::F64 => consts::<f64>(integrator, scalar),
    });
    let body = match integrator {
        Integrator::Rk4 => RK4.replace(ARRAYS, arrays(integrator, layout)),
        Integrator::Dp5 => {
            let mut b = DP5_HELPERS.to_string();
            b.push_str(&DP5.replace(ARRAYS, arrays(integrator, layout)));
            b
        }
        Integrator::Iter => ITER.replace(ARRAYS, arrays(integrator, layout)),
    };
    if tape.units.is_empty() {
        s.push_str(&emit_rhs(
            &tape.ops,
            &tape.outputs,
            tape.n_p,
            scalar,
            "pc_rhs",
        )?);
        s.push_str(BAD_STATE);
        s.push_str(&body);
        s.push_str(entry);
    } else {
        // A tape with calls runs its functions on a stack of `PC_STK` reals per member.
        let (rhs, stride) = emit_calls_rhs(tape, scalar, stack)?;
        s.push_str("#define PC_RECURSION_DEPTH 6u\n");
        s.push_str(&format!("#define PC_STK {}u\n", KERNEL_FRAMES * stride));
        s.push_str(&rhs);
        s.push_str(BAD_STATE);
        s.push_str(&with_calls(integrator, &body, entry, stack));
    }
    Ok(s)
}

/// Reals of the per-member stack that the kernel of `tape` needs: [`KERNEL_FRAMES`] frames
/// of the largest frame of its functions, or 0 for a tape without calls (SPEC section 8af-5).
pub fn stack_len(tape: &Tape) -> usize {
    KERNEL_FRAMES * kernel_stride(tape)
}

/// The per-member stack of a kernel with calls (SPEC section 8af-5): [`KERNEL_FRAMES`] frames
/// of `frame_reals` reals, the frame of `function`, the widest function of the tape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stack {
    pub function: String,
    pub frame_reals: usize,
    pub frames: usize,
}

impl Stack {
    /// Reals of the stack of one member, the same as [`stack_len`].
    pub fn reals(&self) -> usize {
        self.frames.saturating_mul(self.frame_reals)
    }
}

/// The stack the kernel of `tape` needs, or `None` for a tape without calls.
pub fn stack_of(tape: &Tape) -> Option<Stack> {
    kernel_frame(tape).map(|(function, frame_reals)| Stack {
        function,
        frame_reals,
        frames: KERNEL_FRAMES,
    })
}

/// Replaces each of the `count` copies of `from` in `s`. The templates are fixed text, so a
/// count that does not match is a bug in this file, which the tests of every template catch.
fn swap(s: &str, from: &str, to: &str, count: usize) -> String {
    debug_assert_eq!(s.matches(from).count(), count, "{from}");
    s.replace(from, to)
}

/// The integrator and entry of a tape with calls: every right-hand side also takes the
/// member's stack and error word, and a set word ends the member with `PC_RECURSION_DEPTH`
/// where the CPU backend tests its slab, before the state checks and before a DP5 step is
/// judged, so a failed stage is never a rejected step.
fn with_calls(integrator: Integrator, body: &str, entry: &str, stack: StackLayout) -> String {
    let mut b = String::new();
    for line in body.split_inclusive('\n') {
        let call = ["pc_rhs(", "pc_hinit(", "pc_dp5_step("]
            .iter()
            .any(|f| line.contains(f) && !line.starts_with("PC_FN"));
        match line.strip_suffix(");\n") {
            Some(head) if call => {
                b.push_str(head);
                b.push_str(", stk, cerr);\n");
            }
            _ => b.push_str(line),
        }
    }
    let auto = stack == StackLayout::Auto;
    let mut b = if auto {
        // The auto variant keeps the peak and the calls of the last right-hand side in words 1
        // and 2 of the error word's array and writes them to `dep` when the launch ends.
        let mut b = swap(
            &b,
            "                     pc_uint ns)\n{\n",
            "                     pc_uint ns, PC_GLOBAL pc_real *stk, PC_GLOBAL pc_uint *dep)\n{\n    pc_uint cerr_w[3] = {0u, 0u, 0u};\n    PC_PRIV pc_uint *cerr = cerr_w;\n",
            1,
        );
        b = swap(
            &b,
            "    c[0] = st;\n",
            "    dep[0] = cerr_w[1];\n    dep[1] = cerr_w[2];\n    c[0] = st;\n",
            1,
        );
        b
    } else {
        swap(
            &b,
            "                     pc_uint ns)\n{\n",
            "                     pc_uint ns, PC_GLOBAL pc_real *stk)\n{\n    pc_uint cerr_w = 0u;\n    PC_PRIV pc_uint *cerr = &cerr_w;\n",
            1,
        )
    };
    match integrator {
        Integrator::Rk4 | Integrator::Iter => {
            b = swap(
                &b,
                "        bad = pc_bad_state(y1, lim);\n",
                "        bad = (*cerr != 0u) ? PC_RECURSION_DEPTH : pc_bad_state(y1, lim);\n",
                1,
            );
        }
        Integrator::Dp5 => {
            b = swap(
                &b,
                "                       PC_IN pc_real *par)\n",
                "                       PC_IN pc_real *par, PC_GLOBAL pc_real *stk, PC_PRIV pc_uint *cerr)\n",
                1,
            );
            b = swap(
                &b,
                "                          pc_real t_end, pc_real rtol, pc_real atol)\n",
                "                          pc_real t_end, pc_real rtol, pc_real atol,\n                          PC_GLOBAL pc_real *stk, PC_PRIV pc_uint *cerr)\n",
                1,
            );
            b = swap(
                &b,
                "                st = PC_NONFINITE;\n            }\n        }\n        if (st == PC_RUNNING) {\n",
                "                st = PC_NONFINITE;\n            }\n        }\n        if (*cerr != 0u) {\n            st = PC_RECURSION_DEPTH;\n        }\n        if (st == PC_RUNNING) {\n",
                1,
            );
            b = swap(
                &b,
                "            if (!(pc_isfinite(h) && h > PC_ZERO)) {\n",
                "            if (*cerr != 0u) {\n                st = PC_RECURSION_DEPTH;\n            } else if (!(pc_isfinite(h) && h > PC_ZERO)) {\n",
                1,
            );
            b = swap(
                &b,
                "            if (err <= PC_ONE) {\n",
                "            if (*cerr != 0u) {\n                st = PC_RECURSION_DEPTH;\n            } else if (err <= PC_ONE) {\n",
                1,
            );
        }
    }
    // Member layout: each member owns PC_STK consecutive reals. Warp layout: a group of 32
    // members owns 32 * PC_STK reals, and slot e of lane l sits at e * 32 + l, so the lanes
    // of a warp that touch the same slot touch adjacent reals. Auto runs the warp layout.
    let base = match stack {
        StackLayout::Member => "stk + m * PC_STK",
        StackLayout::Warp | StackLayout::Auto => {
            "stk + (m / 32u) * (32u * (pc_ulong)PC_STK) + (m % 32u)"
        }
    };
    let dep = if auto { ", dep + 2u * m" } else { "" };
    let e = swap(
        entry,
        "ctl + m * PC_CTL, steps, max_steps, ns);\n",
        &format!("ctl + m * PC_CTL, steps, max_steps, ns,\n              {base}{dep});\n"),
        1,
    );
    let e = match (e.contains("[[buffer(11)]]"), auto) {
        (true, false) => swap(
            &e,
            "constant pc_ulong &pstride [[buffer(11)]],\n",
            "constant pc_ulong &pstride [[buffer(11)]],\n                   PC_GLOBAL pc_real *PC_RESTRICT stk [[buffer(12)]],\n",
            1,
        ),
        (true, true) => swap(
            &e,
            "constant pc_ulong &pstride [[buffer(11)]],\n",
            "constant pc_ulong &pstride [[buffer(11)]],\n                   PC_GLOBAL pc_real *PC_RESTRICT stk [[buffer(12)]],\n                   PC_GLOBAL pc_uint *PC_RESTRICT dep [[buffer(13)]],\n",
            1,
        ),
        (false, false) => swap(
            &e,
            "pc_ulong pstride)\n",
            "pc_ulong pstride,\n            PC_GLOBAL pc_real *PC_RESTRICT stk)\n",
            1,
        ),
        (false, true) => swap(
            &e,
            "pc_ulong pstride)\n",
            "pc_ulong pstride,\n            PC_GLOBAL pc_real *PC_RESTRICT stk, PC_GLOBAL pc_uint *PC_RESTRICT dep)\n",
            1,
        ),
    };
    b.push_str(&e);
    if auto && e.contains("__global__") {
        b.push_str(KEYS_CUDA);
    }
    b
}

/// The key kernel of the CUDA counting variant (SPEC section 8af-5): thread `i` evaluates the
/// right-hand side of member `idx[i]` once at `(x0, t0)` and writes the peak and the calls of
/// that evaluation to `dep[2i]` and `dep[2i + 1]`. The stack slots follow the warp layout over
/// `i`, so `stk` holds whole groups of 32 of the `count` threads.
const KEYS_CUDA: &str = r#"
extern "C" __global__ void pc_keys(PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
                                   PC_IN pc_ulong *PC_RESTRICT idx,
                                   PC_GLOBAL pc_real *PC_RESTRICT stk,
                                   PC_GLOBAL pc_uint *PC_RESTRICT dep, pc_ulong count,
                                   pc_ulong pstride, pc_real t0)
{
    pc_ulong i = (pc_ulong)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    pc_ulong m = idx[i];
    pc_real y[PC_N];
    pc_real k[PC_N];
    pc_uint j;
    for (j = 0u; j < PC_N; j++) {
        y[j] = x0[m * PC_N + j];
    }
    pc_uint cerr_w[3] = {0u, 0u, 0u};
    pc_rhs(t0, y, par + m * pstride, k, stk + (i / 32u) * (32u * (pc_ulong)PC_STK) + (i % 32u),
           cerr_w);
    dep[2u * i] = cerr_w[1];
    dep[2u * i + 1u] = cerr_w[2];
}
"#;

/// The line of a template that the declarations of the scratch replace.
const ARRAYS: &str = "    /*PC_ARRAYS*/\n";

/// The declarations of `y`, `y1`, and for the integrators `yt` and the stage vectors `k`. The
/// global ones point into `w` behind the `PC_PERSIST` words, one array after the other.
fn arrays(integrator: Integrator, layout: Layout) -> &'static str {
    match (layout, integrator) {
        (Layout::Private, Integrator::Rk4) => {
            "    pc_real y[PC_N];\n    pc_real y1[PC_N];\n    pc_real yt[PC_N];\n    pc_real k[4u * PC_N];\n"
        }
        (Layout::Private, Integrator::Dp5) => {
            "    pc_real y[PC_N];\n    pc_real y1[PC_N];\n    pc_real yt[PC_N];\n    pc_real k[7u * PC_N];\n"
        }
        (Layout::Private, Integrator::Iter) => "    pc_real y[PC_N];\n    pc_real y1[PC_N];\n",
        (Layout::Global, Integrator::Iter) => GLOBAL_ARRAYS_ITER,
        (Layout::Global, _) => GLOBAL_ARRAYS,
    }
}

const GLOBAL_ARRAYS: &str = "    PC_GLOBAL pc_real *y = w + PC_PERSIST;\n    PC_GLOBAL pc_real *y1 = y + PC_N;\n    PC_GLOBAL pc_real *yt = y1 + PC_N;\n    PC_GLOBAL pc_real *k = yt + PC_N;\n";

const GLOBAL_ARRAYS_ITER: &str =
    "    PC_GLOBAL pc_real *y = w + PC_PERSIST;\n    PC_GLOBAL pc_real *y1 = y + PC_N;\n";

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
        // The iteration of a map uses no constant.
        Integrator::Iter => {}
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
///
/// `lim` is the bound capped at the largest finite value (see `abi.rs`), so `|y| <= lim` is
/// false for NaN and infinity as well as for a state past the bound. One pass without an
/// early exit ORs that test over the state. Only when it fails does the slow path look again,
/// so that NONFINITE still comes before BOUNDS.
const BAD_STATE: &str = r#"
PC_FN pc_uint pc_bad_state(PC_SCR pc_real *y1, pc_real lim)
{
    pc_uint i;
    pc_uint bad = 0u;
    for (i = 0u; i < PC_N; i++) {
        bad |= (pc_fabs(y1[i]) <= lim) ? 0u : 1u;
    }
    if (bad == 0u) {
        return PC_RUNNING;
    }
    for (i = 0u; i < PC_N; i++) {
        if (!pc_isfinite(y1[i])) {
            return PC_NONFINITE;
        }
    }
    for (i = 0u; i < PC_N; i++) {
        if (pc_fabs(y1[i]) > lim) {
            return PC_BOUNDS;
        }
    }
    return PC_RUNNING;
}

"#;

const RK4: &str = r#"
/* The constants of segment `s`. They change only when a member moves to the next segment,
   so a member loads them there and keeps them in registers for the steps in between. */
#define PC_SEG_LOAD(s) \
    do { \
        m = segm[s]; \
        sa = seg_a[s]; \
        sb = seg_b[s]; \
        h = seg_h[s]; \
        hh = h * PC_HALF; \
        h6 = h / PC_SIX; \
    } while (0)

PC_FN void pc_member(PC_IN pc_real *PC_RESTRICT rc, PC_IN pc_ulong *PC_RESTRICT segm,
                     PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
                     PC_GLOBAL pc_real *PC_RESTRICT w, PC_GLOBAL pc_real *PC_RESTRICT out,
                     PC_GLOBAL pc_uint *PC_RESTRICT c, pc_ulong budget, pc_uint max_steps,
                     pc_uint ns)
{
    /*PC_ARRAYS*/
    PC_IN pc_real *seg_a = rc + 5u + ns;
    PC_IN pc_real *seg_b = rc + 5u + 2u * ns;
    PC_IN pc_real *seg_h = rc + 5u + 3u * ns;
    pc_real lim = rc[1];
    pc_uint st = c[0];
    pc_uint accepted = c[2];
    pc_uint si = c[4];
    pc_uint ii = c[5];
    pc_ulong m = 0u;
    pc_real sa = PC_ZERO;
    pc_real sb = PC_ZERO;
    pc_real h = PC_ZERO;
    pc_real hh = PC_ZERO;
    pc_real h6 = PC_ZERO;
    pc_real t;
    pc_uint i;

    if (st != PC_RUNNING) {
        return;
    }
    PC_SEG_LOAD(si);
    if (c[1] == 0u) {
        c[1] = 1u;
        t = rc[0];
        for (i = 0u; i < PC_N; i++) {
            y[i] = x0[i];
        }
    } else {
        t = w[0];
        for (i = 0u; i < PC_N; i++) {
            y[i] = w[1u + i];
        }
    }
    while (budget > 0u && st == PC_RUNNING) {
        pc_uint last = ((pc_ulong)ii + 1u == m) ? 1u : 0u;
        pc_real ts = sa + (pc_real)ii * h;
        pc_real t_next = (last != 0u) ? sb : sa + (pc_real)(ii + 1u) * h;
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
        bad = pc_bad_state(y1, lim);
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
                } else {
                    PC_SEG_LOAD(si);
                }
            }
            if (st == PC_RUNNING && accepted >= max_steps) {
                st = PC_MAX_STEPS;
            }
        }
    }
    w[0] = t;
    for (i = 0u; i < PC_N; i++) {
        w[1u + i] = y[i];
    }
    c[0] = st;
    c[2] = accepted;
    c[4] = si;
    c[5] = ii;
}

"#;

const ITER: &str = r#"
/* The iteration of a map (SPEC section 8aa): y1 = F(y, p, ts) with ts = a + i for the
   iterate i of the segment, then the state check of RK4. A member loads the segment
   constants only when it moves to the next segment. */
#define PC_SEG_LOAD(s) \
    do { \
        m = segm[s]; \
        sa = seg_a[s]; \
        sb = seg_b[s]; \
    } while (0)

PC_FN void pc_member(PC_IN pc_real *PC_RESTRICT rc, PC_IN pc_ulong *PC_RESTRICT segm,
                     PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
                     PC_GLOBAL pc_real *PC_RESTRICT w, PC_GLOBAL pc_real *PC_RESTRICT out,
                     PC_GLOBAL pc_uint *PC_RESTRICT c, pc_ulong budget, pc_uint max_steps,
                     pc_uint ns)
{
    /*PC_ARRAYS*/
    PC_IN pc_real *seg_a = rc + 5u + ns;
    PC_IN pc_real *seg_b = rc + 5u + 2u * ns;
    pc_real lim = rc[1];
    pc_uint st = c[0];
    pc_uint accepted = c[2];
    pc_uint si = c[4];
    pc_uint ii = c[5];
    pc_ulong m = 0u;
    pc_real sa = PC_ZERO;
    pc_real sb = PC_ZERO;
    pc_real t;
    pc_uint i;

    if (st != PC_RUNNING) {
        return;
    }
    PC_SEG_LOAD(si);
    if (c[1] == 0u) {
        c[1] = 1u;
        t = rc[0];
        for (i = 0u; i < PC_N; i++) {
            y[i] = x0[i];
        }
    } else {
        t = w[0];
        for (i = 0u; i < PC_N; i++) {
            y[i] = w[1u + i];
        }
    }
    while (budget > 0u && st == PC_RUNNING) {
        pc_uint last = ((pc_ulong)ii + 1u == m) ? 1u : 0u;
        pc_real ts = sa + (pc_real)ii;
        pc_real t_next = (last != 0u) ? sb : sa + (pc_real)(ii + 1u);
        pc_uint bad;
        budget -= 1u;
        pc_rhs(ts, y, par, y1);
        bad = pc_bad_state(y1, lim);
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
                } else {
                    PC_SEG_LOAD(si);
                }
            }
            if (st == PC_RUNNING && accepted >= max_steps) {
                st = PC_MAX_STEPS;
            }
        }
    }
    w[0] = t;
    for (i = 0u; i < PC_N; i++) {
        w[1u + i] = y[i];
    }
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
                       PC_SCR pc_real *y, PC_SCR pc_real *yt, PC_SCR pc_real *k,
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

PC_FN pc_real pc_dp5_step(PC_SCR pc_real *y, PC_SCR pc_real *y1, PC_SCR pc_real *yt,
                          PC_SCR pc_real *k, PC_IN pc_real *par, pc_real t, pc_real h,
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
PC_FN void pc_member(PC_IN pc_real *PC_RESTRICT rc, PC_IN pc_ulong *PC_RESTRICT segm,
                     PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
                     PC_GLOBAL pc_real *PC_RESTRICT w, PC_GLOBAL pc_real *PC_RESTRICT out,
                     PC_GLOBAL pc_uint *PC_RESTRICT c, pc_ulong budget, pc_uint max_steps,
                     pc_uint ns)
{
    /*PC_ARRAYS*/
    PC_IN pc_real *times = rc + 5u;
    pc_real lim = rc[1];
    pc_real rtol = rc[2];
    pc_real atol = rc[3];
    pc_real hmax = rc[4];
    pc_uint st = c[0];
    pc_uint accepted = c[2];
    pc_uint rejected = c[3];
    pc_uint si = c[4];
    pc_uint reject = c[6];
    pc_uint nan_err = c[7];
    pc_real t;
    pc_real h;
    pc_real facold;
    pc_uint i;

    (void)segm;
    if (st != PC_RUNNING) {
        return;
    }
    if (c[1] != 0u) {
        t = w[0];
        h = w[1];
        facold = w[2];
        for (i = 0u; i < PC_N; i++) {
            y[i] = w[3u + i];
        }
        for (i = 0u; i < PC_N; i++) {
            k[i] = w[3u + PC_N + i];
        }
    } else {
        c[1] = 1u;
        t = rc[0];
        h = PC_ZERO;
        facold = PC_ZERO;
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
            if (err <= PC_ONE) {
                st = pc_bad_state(y1, lim);
                if (st == PC_RUNNING) {
                    fac = pc_clamp(fac11 / pc_pow(facold, PC_BETA) / PC_SAFE, PC_INV_FAC2, PC_INV_FAC1);
                    hnew = h / fac;
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
    w[0] = t;
    w[1] = h;
    w[2] = facold;
    for (i = 0u; i < PC_N; i++) {
        w[3u + i] = y[i];
    }
    for (i = 0u; i < PC_N; i++) {
        w[3u + PC_N + i] = k[i];
    }
    c[0] = st;
    c[2] = accepted;
    c[3] = rejected;
    c[4] = si;
    c[6] = reject;
    c[7] = nan_err;
}

"#;

const ENTRY_C: &str = r#"
void pc_ens(PC_IN pc_real *PC_RESTRICT rc, PC_IN pc_ulong *PC_RESTRICT segm,
            PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
            PC_GLOBAL pc_real *PC_RESTRICT work, PC_GLOBAL pc_real *PC_RESTRICT out,
            PC_GLOBAL pc_uint *PC_RESTRICT ctl, pc_ulong members, pc_ulong steps, pc_uint max_steps, pc_uint ns, pc_ulong pstride)
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
extern "C" __global__ void pc_ens(PC_IN pc_real *PC_RESTRICT rc,
                                  PC_IN pc_ulong *PC_RESTRICT segm,
                                  PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
                                  PC_GLOBAL pc_real *PC_RESTRICT work,
                                  PC_GLOBAL pc_real *PC_RESTRICT out,
                                  PC_GLOBAL pc_uint *PC_RESTRICT ctl, pc_ulong members, pc_ulong steps, pc_uint max_steps,
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

/// One work item per member, as in the CUDA entry. The global size is rounded up to a whole
/// work group, so the guard comes first.
const ENTRY_OPENCL: &str = r#"
__kernel void pc_ens(PC_IN pc_real *PC_RESTRICT rc, PC_IN pc_ulong *PC_RESTRICT segm,
                     PC_IN pc_real *PC_RESTRICT x0, PC_IN pc_real *PC_RESTRICT par,
                     PC_GLOBAL pc_real *PC_RESTRICT work, PC_GLOBAL pc_real *PC_RESTRICT out,
                     PC_GLOBAL pc_uint *PC_RESTRICT ctl, pc_ulong members, pc_ulong steps, pc_uint max_steps,
                     pc_uint ns, pc_ulong pstride)
{
    pc_ulong m = (pc_ulong)get_global_id(0);
    if (m >= members) {
        return;
    }
    pc_member(rc, segm, x0 + m * PC_N, par + m * pstride, work + m * PC_WL,
              out + m * (pc_ulong)ns * PC_N, ctl + m * PC_CTL, steps, max_steps, ns);
}
"#;

/// One thread per member. Metal passes every buffer by index and every scalar as a one-value
/// buffer, so the host binds them in this order: the seven buffers at 0 to 6, then `members`,
/// `steps`, `max_steps`, `ns` and `pstride` at 7 to 11. The grid is rounded up to a whole
/// threadgroup, so the guard comes first.
const ENTRY_MSL: &str = r#"
kernel void pc_ens(PC_IN pc_real *PC_RESTRICT rc [[buffer(0)]],
                   PC_IN pc_ulong *PC_RESTRICT segm [[buffer(1)]],
                   PC_IN pc_real *PC_RESTRICT x0 [[buffer(2)]],
                   PC_IN pc_real *PC_RESTRICT par [[buffer(3)]],
                   PC_GLOBAL pc_real *PC_RESTRICT work [[buffer(4)]],
                   PC_GLOBAL pc_real *PC_RESTRICT out [[buffer(5)]],
                   PC_GLOBAL pc_uint *PC_RESTRICT ctl [[buffer(6)]],
                   constant pc_ulong &members [[buffer(7)]],
                   constant pc_ulong &steps [[buffer(8)]],
                   constant pc_uint &max_steps [[buffer(9)]],
                   constant pc_uint &ns [[buffer(10)]],
                   constant pc_ulong &pstride [[buffer(11)]],
                   uint gid [[thread_position_in_grid]])
{
    pc_ulong m = (pc_ulong)gid;
    if (m >= members) {
        return;
    }
    pc_member(rc, segm, x0 + m * PC_N, par + m * pstride, work + m * PC_WL,
              out + m * (pc_ulong)ns * PC_N, ctl + m * PC_CTL, steps, max_steps, ns);
}
"#;
