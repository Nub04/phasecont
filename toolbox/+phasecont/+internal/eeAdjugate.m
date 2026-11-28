function adj = eeAdjugate(A)
%EEADJUGATE Adjugate of a square matrix from its cofactors (SPEC 8ah).
%   ADJ = phasecont.internal.eeAdjugate(A) has the j,i cofactor of A in
%   row i, column j: the plain transpose (.') of the cofactor matrix, never
%   the conjugate transpose. The i,j cofactor is (-1)^(i+j) times det of A
%   with row i and column j deleted; det of the empty minor of a 1 by 1 is 1.
n = size(A, 1);
keep = cell(n, 1);                 % index vector with entry k deleted
for k = 1:n
    keep{k} = [1:k-1, k+1:n];
end
C = zeros(n);
for i = 1:n
    Ai = A(keep{i}, :);
    for j = 1:n
        m = det(Ai(:, keep{j}));
        if mod(i + j, 2)
            m = -m;
        end
        C(i, j) = m;
    end
end
adj = C.';
end
