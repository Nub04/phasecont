function c = classifyJacobian(J, isMap, tol)
%CLASSIFYJACOBIAN The type of a fixed point from its Jacobian.
%   C = phasecont.internal.classifyJacobian(J, ISMAP, TOL) classifies the
%   equilibrium (ISMAP false) or the fixed point of a map (ISMAP true) whose
%   Jacobian is the n-by-n matrix J, with the rules and the relative
%   tolerance TOL of SPEC section 8ac. C has the fields of Point.classify.
%   Point.classify and Branch.classify both call it. TOL is checked by the
%   callers.
arguments
    J (:,:) double
    isMap (1,1) logical
    tol (1,1) double
end
n = size(J, 1);
s = norm(J, "fro");
% One bit per tolerance test; Borderline is read back from it.
names = ["zeroD" "zeroT" "zeroDisc" "zeroRe" "unitCircle"];
bits = uint8([1 2 4 8 16]);
flags = uint8(0);
c = struct("Type", "", "Stable", false(0, 1), "Hyperbolic", true, ...
    "T", NaN, "D", NaN, "Disc", NaN, "Rotation", string.empty, ...
    "NStable", 0, "NUnstable", 0, "NCenter", 0, ...
    "Borderline", strings(1, 0), "Tol", tol, "Jacobian", J);
if n == 2
    c.T = J(1,1) + J(2,2);
    c.D = J(1,1)*J(2,2) - J(1,2)*J(2,1);
    c.Disc = c.T^2 - 4*c.D;
end
lam = eig(J);
if isMap
    r = abs(lam);
    on = abs(r - 1) <= tol;
    if any(on)
        flags = bitor(flags, bits(5));
    end
    c.NStable = nnz(r < 1 & ~on);
    c.NUnstable = nnz(r > 1 & ~on);
    c.NCenter = nnz(on);
    if c.NStable == n
        c.Type = sprintf("stable (%d inside the unit circle)", c.NStable);
        c.Stable = true;
    elseif c.NCenter == 0
        c.Type = sprintf("unstable (%d inside, %d outside the unit circle)", ...
            c.NStable, c.NUnstable);
        c.Stable = false;
    else
        c.Type = sprintf("non-hyperbolic (%d inside, %d outside, %d on the unit circle)", ...
            c.NStable, c.NUnstable, c.NCenter);
        c.Hyperbolic = false;
    end
elseif n == 1
    if J < 0
        c.Type = "stable";
        c.Stable = true;
        c.NStable = 1;
    elseif J > 0
        c.Type = "unstable";
        c.Stable = false;
        c.NUnstable = 1;
    else
        c.Type = "zero eigenvalue";
        c.Hyperbolic = false;
        c.NCenter = 1;
    end
elseif n == 2
    T = c.T;
    if abs(c.D) <= tol*s^2
        flags = bitor(flags, bits(1));
    end
    if abs(T) <= tol*s
        flags = bitor(flags, bits(2));
    end
    if abs(c.Disc) <= tol*s^2
        flags = bitor(flags, bits(3));
    end
    if T < 0
        sgn = "stable ";
    else
        sgn = "unstable ";
    end
    if bitand(flags, bits(1))
        c.Type = "zero eigenvalue";
        c.Hyperbolic = false;
        % One eigenvalue is zero within the band; the other is T.
        if bitand(flags, bits(2))
            c.NCenter = 2;
        else
            c.NCenter = 1;
            c.NStable = double(T < 0);
            c.NUnstable = double(T > 0);
        end
    elseif c.D < 0
        c.Type = "saddle";
        c.Stable = false;
        c.NStable = 1;
        c.NUnstable = 1;
    elseif bitand(flags, bits(2)) && c.Disc < 0
        c.Type = "center";
        c.Hyperbolic = false;
        c.NCenter = 2;
    else
        if bitand(flags, bits(3))
            if norm(J - (T/2)*eye(2), "fro") <= sqrt(tol)*s
                c.Type = sgn + "star";
            else
                c.Type = sgn + "degenerate node";
            end
        elseif c.Disc > 0
            c.Type = sgn + "node";
        else
            c.Type = sgn + "spiral";
        end
        c.Stable = T < 0;
        c.NStable = 2*double(T < 0);
        c.NUnstable = 2*double(T > 0);
    end
    if c.Type == "center" || endsWith(c.Type, "spiral")
        if J(2,1) > 0
            c.Rotation = "counterclockwise";
        elseif J(2,1) < 0
            c.Rotation = "clockwise";
        end
    end
else
    on = abs(real(lam)) <= tol*s;
    if any(on)
        flags = bitor(flags, bits(4));
    end
    c.NStable = nnz(real(lam) < 0 & ~on);
    c.NUnstable = nnz(real(lam) > 0 & ~on);
    c.NCenter = nnz(on);
    counts = sprintf("%d stable, %d unstable", c.NStable, c.NUnstable);
    if c.NCenter > 0
        c.Type = sprintf("non-hyperbolic (%s, %d on the imaginary axis)", ...
            counts, c.NCenter);
        c.Hyperbolic = false;
    else
        c.Stable = c.NStable == n;
        if n == 3
            focus = any(abs(imag(lam)) > (sqrt(tol)/2)*s);
            if c.NStable == 3 || c.NUnstable == 3
                if c.NStable == 3
                    name = "stable ";
                else
                    name = "unstable ";
                end
                if focus
                    name = name + "focus-node";
                else
                    name = name + "node";
                end
            elseif focus
                name = "saddle-focus";
            else
                name = "saddle";
            end
        else
            name = "hyperbolic";
        end
        c.Type = sprintf("%s (%s)", name, counts);
    end
end
c.Type = string(c.Type);
c.Borderline = names(bitand(flags, bits) ~= 0);
end
