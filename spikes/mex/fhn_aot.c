/* FitzHugh-Nagumo in C, compiled into the gateway by mex (MSVC), for the AOT C
 * column of spike S2. Same formulas and operation order as the Rust and JIT
 * versions in spikes/pckern/src/s2_jit.rs.
 *
 *   x = [v, w], p = [eps, a, b, I]
 *   v' = v - v^3/3 - w + I
 *   w' = eps (v + a - b w)
 */

void fhn_aot_rhs(const double *x, const double *p, double *out)
{
    double v = x[0], w = x[1];
    out[0] = v - v * v * v / 3.0 - w + p[3];
    out[1] = p[0] * (v + p[1] - p[2] * w);
}

/* Column-major 2x2 Jacobian: [df1/dv, df2/dv, df1/dw, df2/dw]. */
void fhn_aot_jac(const double *x, const double *p, double *out)
{
    double v = x[0];
    out[0] = 1.0 - v * v;
    out[1] = p[0];
    out[2] = -1.0;
    out[3] = -(p[0] * p[2]);
}
