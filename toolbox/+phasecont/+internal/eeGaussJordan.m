function [M, steps] = eeGaussJordan(M, n)
%EEGAUSSJORDAN Gauss-Jordan reduction of [A | B] with recorded steps.
%   [M, STEPS] = phasecont.internal.eeGaussJordan(M, N) reduces the first N
%   columns of M to the identity (SPEC section 8ah). The pivot of column k
%   is the entry of largest magnitude at or below row k, the smallest row
%   index on a tie. A step is recorded only when it changes M; each holds
%   Op ("swap", "scale" or "add"), Rows, Factor and Matrix (M after the
%   step). The caller has checked that A is nonsingular and that M is
%   finite. After each step the Factor and the changed row are checked,
%   and phasecont:ee:outOfRange is raised at the first Inf or NaN, which
%   is the same as checking every Matrix (SPEC 8ah, check 5).
cap = n * n + n - 1;   % at most n - 1 swaps, n scales and n(n - 1) adds
op = cell(cap, 1);
rows = cell(cap, 1);
fac = cell(cap, 1);
mat = cell(cap, 1);
q = 0;
for k = 1:n
    [~, rel] = max(abs(M(k:n, k)));
    p = k + rel - 1;
    if p ~= k
        M([k p], :) = M([p k], :);
        q = q + 1;
        op{q} = "swap";
        rows{q} = [k p];
        fac{q} = 1;
        mat{q} = M;
    end
    if M(k, k) ~= 1
        f = 1 / M(k, k);
        M(k, :) = f * M(k, :);
        M(k, k) = 1;
        if ~(isfinite(f) && all(isfinite(M(k, :))))
            phasecont.internal.eeBackstop([], f, M(k, :));
        end
        q = q + 1;
        op{q} = "scale";
        rows{q} = [k k];
        fac{q} = f;
        mat{q} = M;
    end
    for i = [1:k-1, k+1:n]
        if M(i, k) ~= 0
            c = -M(i, k);
            M(i, :) = M(i, :) + c * M(k, :);
            M(i, k) = 0;
            if ~all(isfinite(M(i, :)))
                phasecont.internal.eeBackstop([], M(i, :));
            end
            q = q + 1;
            op{q} = "add";
            rows{q} = [i k];
            fac{q} = c;
            mat{q} = M;
        end
    end
end
steps = struct("Op", op(1:q), "Rows", rows(1:q), "Factor", fac(1:q), ...
    "Matrix", mat(1:q));
end
