function [M, steps] = eeGaussJordan(M, n)
%EEGAUSSJORDAN Gauss-Jordan reduction of [A | B] with recorded steps.
%   [M, STEPS] = phasecont.internal.eeGaussJordan(M, N) reduces the first N
%   columns of M to the identity (SPEC section 8ah). The pivot of column k
%   is the entry of largest magnitude at or below row k, the smallest row
%   index on a tie. A step is recorded only when it changes M; each holds
%   Op ("swap", "scale" or "add"), Rows, Factor and Matrix (M after the
%   step). The caller has checked that A is nonsingular.
steps = reshape(struct("Op", {}, "Rows", {}, "Factor", {}, "Matrix", {}), 0, 1);
for k = 1:n
    [~, rel] = max(abs(M(k:n, k)));
    p = k + rel - 1;
    if p ~= k
        M([k p], :) = M([p k], :);
        steps(end + 1, 1) = step("swap", [k p], 1, M); %#ok<AGROW>
    end
    if M(k, k) ~= 1
        f = 1 / M(k, k);
        M(k, :) = f * M(k, :);
        M(k, k) = 1;
        steps(end + 1, 1) = step("scale", [k k], f, M); %#ok<AGROW>
    end
    for i = [1:k-1, k+1:n]
        if M(i, k) ~= 0
            c = -M(i, k);
            M(i, :) = M(i, :) + c * M(k, :);
            M(i, k) = 0;
            steps(end + 1, 1) = step("add", [i k], c, M); %#ok<AGROW>
        end
    end
end
end

function s = step(op, rows, factor, M)
s = struct("Op", op, "Rows", rows, "Factor", factor, "Matrix", M);
end
