function adj = eeAdjugate(A)
%EEADJUGATE Adjugate of a square matrix from its cofactors (SPEC 8ah).
%   ADJ = phasecont.internal.eeAdjugate(A) has the j,i cofactor of A in
%   row i, column j: the plain transpose (.') of the cofactor matrix, never
%   the conjugate transpose. The i,j cofactor is (-1)^(i+j) times det of A
%   with row i and column j deleted; det of the empty minor of a 1 by 1 is 1.
n = size(A, 1);
C = zeros(n);
for i = 1:n
    for j = 1:n
        C(i, j) = (-1)^(i + j) * det(A([1:i-1, i+1:n], [1:j-1, j+1:n]));
    end
end
adj = C.';
end
