classdef Point
    % A point of a model: an equilibrium or a special point on a branch.
    %
    %   p = phasecont.Point(U = u, P = pars, Label = "EP")
    %
    % The properties are immutable. Label-specific values (coefficients,
    % frequency, normalization, refined) go in Data.
    %
    % Stable is computed from Eigenvalues when it is read. For a Point with
    % Label "EP", a Model and at least one eigenvalue it is true when every
    % eigenvalue has negative real part (ODE) or every multiplier has
    % modulus below 1 (map, Model.IsMap). It is empty for every other Point:
    % at a special point such as LP, BP, H, PD or NS an eigenvalue or a
    % multiplier lies on the stability boundary, and a Point without a
    % Model or without eigenvalues has nothing to test.
    %
    % Jacobian and Type come from classify with the default Tol. They are
    % empty for a Point that classify refuses (see classify).

    properties (SetAccess = immutable)
        U (:,1) double = zeros(0,1)
        P (:,1) double = zeros(0,1)
        Eigenvalues (:,1) double = zeros(0,1)
        Label (1,1) string = "EP"
        Index (1,1) double = NaN
        Data (1,1) struct = struct()
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
    end

    properties (Dependent)
        % True or false for an "EP" Point of an ODE or a map, empty otherwise.
        Stable
        % The Jacobian of the model at U and P (of F^Iterate on a map),
        % n-by-n, or empty when classify refuses the Point.
        Jacobian
        % The type of the point from classify, or empty when classify
        % refuses the Point.
        Type
    end

    properties (SetAccess = immutable, Hidden)
        % The run that produced the point; empty for a point read from a branch.
        Record phasecont.internal.RunRecord {mustBeScalarOrEmpty} = phasecont.internal.RunRecord.empty
        % The iterate k of a point of a map: U is one point of a k-cycle and
        % Eigenvalues are the multipliers of the k-th iterate. 1 otherwise.
        Iterate (1,1) double = 1
    end

    methods
        function obj = Point(options)
            arguments
                options.U (:,1) double = zeros(0,1)
                options.P (:,1) double = zeros(0,1)
                options.Eigenvalues (:,1) double = zeros(0,1)
                options.Label (1,1) string = "EP"
                options.Index (1,1) double = NaN
                options.Data (1,1) struct = struct()
                options.Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
                options.Record phasecont.internal.RunRecord {mustBeScalarOrEmpty} = phasecont.internal.RunRecord.empty
                options.Iterate (1,1) double {mustBeInteger, mustBePositive} = 1
            end
            obj.U = options.U;
            obj.P = options.P;
            obj.Eigenvalues = options.Eigenvalues;
            obj.Label = options.Label;
            obj.Index = options.Index;
            obj.Data = options.Data;
            obj.Model = options.Model;
            obj.Record = options.Record;
            obj.Iterate = options.Iterate;
        end

        function s = get.Stable(obj)
            % The rules of Branch.Stable for an equilibrium or a fixed point.
            ev = obj.Eigenvalues;
            m = obj.Model;
            if obj.Label ~= "EP" || isempty(m) || isempty(ev) || m.IsDelay
                s = false(0, 1);
            elseif m.IsMap
                s = all(abs(ev) < 1);
            else
                s = all(real(ev) < 0);
            end
        end

        function J = get.Jacobian(obj)
            J = zeros(0, 0);
            try
                c = classify(obj);
                J = c.Jacobian;
            catch err
                if err.identifier ~= "phasecont:point:noJacobian"
                    rethrow(err);
                end
            end
        end

        function t = get.Type(obj)
            t = string.empty;
            try
                c = classify(obj);
                t = c.Type;
            catch err
                if err.identifier ~= "phasecont:point:noJacobian"
                    rethrow(err);
                end
            end
        end

        function c = classify(obj, options)
            %CLASSIFY The Jacobian and the type of an equilibrium or a fixed point.
            %   C = classify(PT) evaluates the Jacobian J of the model at PT.U
            %   and PT.P (of F^K with K = PT.Iterate on a map) and classifies
            %   the point from eig(J). C is a struct with the fields
            %     Type        the type, for example "saddle" or "stable spiral"
            %     Stable      true or false, or empty when the linearization
            %                 does not decide
            %     Hyperbolic  false when an eigenvalue is on the imaginary
            %                 axis (ODE) or a multiplier on the unit circle
            %     T, D, Disc  trace, determinant and T^2 - 4 D (NaN unless n = 2)
            %     Rotation    "counterclockwise" or "clockwise" for a spiral or
            %                 a center of a 2D ODE (u1 horizontal), else empty
            %     NStable, NUnstable, NCenter
            %                 counts of eigenvalues with negative real part,
            %                 positive real part and on the axis (on a map:
            %                 inside, outside and on the unit circle)
            %     Borderline  names of the tolerance tests that fired:
            %                 zeroD, zeroT, zeroDisc, zeroRe, unitCircle
            %     Tol         the tolerance used
            %     Jacobian    J
            %
            %   C = classify(PT, Tol = TOL) sets the relative tolerance of the
            %   tests (default 1e-9), a finite real number with 0 <= TOL < 1.
            %   With s the Frobenius norm of J, zeroD is |D| <= TOL s^2, zeroT
            %   is |T| <= TOL s, zeroDisc is |Disc| <= TOL s^2, zeroRe is
            %   |Re lambda| <= TOL s and unitCircle is ||mu| - 1| <= TOL.
            %
            %   classify raises phasecont:point:noJacobian for a Point whose
            %   Label is not "EP", with no Model or a deleted one, of a delay or
            %   non-autonomous model, with a U whose length is not the number
            %   of variables, or whose Jacobian has NaN or Inf entries.
            arguments
                obj (1,1) phasecont.Point
                options.Tol (1,1) double {mustBeReal} = 1e-9
            end
            tol = options.Tol;
            if ~(isfinite(tol) && tol >= 0 && tol < 1)
                error("phasecont:point:badOption", ...
                    "Tol must be a finite real number with 0 <= Tol < 1.");
            end
            J = jacobianAt(obj);
            m = obj.Model;
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
            if m.IsMap
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
    end

    methods (Access = private)
        function J = jacobianAt(obj)
            % The Jacobian at U and P, or phasecont:point:noJacobian.
            m = obj.Model;
            why = "";
            if obj.Label ~= "EP"
                why = "the Point has Label """ + obj.Label + """, not ""EP""";
            elseif isempty(m) || ~isvalid(m) || m.Id == 0
                why = "the Point has no model, or its model was deleted";
            elseif m.IsDelay
                why = "the model has delay terms";
            elseif ~m.Autonomous
                why = "the model uses t";
            elseif numel(obj.U) ~= numel(m.Variables)
                why = sprintf("U has %d entries and the model has %d variables", ...
                    numel(obj.U), numel(m.Variables));
            end
            if why == ""
                opts = '';
                if m.IsMap
                    opts = char("{""iterate"":" + ...
                        phasecont.internal.num2json(obj.Iterate) + "}");
                end
                try
                    J = pckern("model_jacobian", m.Id, obj.U, obj.P, opts);
                catch err
                    if ~startsWith(err.identifier, "phasecont:engine:")
                        rethrow(err);
                    end
                    why = string(err.message);
                end
            end
            if why == "" && ~all(isfinite(J), "all")
                why = "the Jacobian has NaN or Inf entries";
            end
            if why ~= ""
                error("phasecont:point:noJacobian", ...
                    "No Jacobian for this Point: %s.", why);
            end
        end
    end
end
