function r = s2_baseline(calls, reps)
% MATLAB baseline for spike S2: the FitzHugh-Nagumo right-hand side and
% Jacobian as anonymous functions, written the way a user would pass them
% to ode45. The loops sit in this function file, because loops in scripts
% run slower. One warm-up run, then `reps` timed runs with tic/toc. Returns
% ns per call for each run and the median extrapolated to 1e7 calls.
if nargin < 1
    calls = 1e5;
end
if nargin < 2
    reps = 5;
end
p = [0.08; 0.7; 0.8; 0.5];
rhs = @(x, p) [x(1) - x(1)^3/3 - x(2) + p(4); p(1)*(x(1) + p(2) - p(3)*x(2))];
jac = @(x, p) [1 - x(1)^2, -1; p(1), -p(1)*p(3)];
r.calls = calls;
r.reps = reps;
r.rhs_ns = time_loop(rhs, p, calls, reps);
r.jac_ns = time_loop(jac, p, calls, reps);
r.rhs_1e7_s = median(r.rhs_ns)*1e7/1e9;
r.jac_1e7_s = median(r.jac_ns)*1e7/1e9;
end

function ns = time_loop(f, p, calls, reps)
loop(f, p, calls);
ns = zeros(1, reps);
for k = 1:reps
    t0 = tic;
    loop(f, p, calls);
    ns(k) = toc(t0)*1e9/calls;
end
end

function s = loop(f, p, calls)
% x sweeps the same inputs as the Rust loops in s2_jit.rs.
step = 1/calls;
s = 0;
for k = 0:calls-1
    t = k*step;
    y = f([-2 + 4*t; 1 - 2*t], p);
    s = s + y(1);
end
end
