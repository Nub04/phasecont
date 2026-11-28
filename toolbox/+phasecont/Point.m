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
            %   tests (default 1e-9), a finite real double with 0 <= TOL < 1.
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
                options.Tol (1,1) {mustBeA(options.Tol, "double"), mustBeReal} = 1e-9
            end
            tol = options.Tol;
            if ~(isfinite(tol) && tol >= 0 && tol < 1)
                error("phasecont:point:badOption", ...
                    "Tol must be a finite real number with 0 <= Tol < 1.");
            end
            J = jacobianAt(obj);
            c = phasecont.internal.classifyJacobian(J, obj.Model.IsMap, tol);
        end
    end

    methods (Access = private)
        function J = jacobianAt(obj)
            % The Jacobian at U and P, or phasecont:point:noJacobian.
            m = obj.Model;
            J = zeros(0, 0);
            if obj.Label ~= "EP"
                why = "the Point has Label """ + obj.Label + """, not ""EP""";
            elseif isempty(m) || ~isvalid(m) || m.Id == 0
                why = "the Point has no model, or its model was deleted";
            else
                [J, why] = phasecont.internal.modelJacobian(m, obj.U, obj.P, ...
                    obj.Iterate);
            end
            if why ~= ""
                error("phasecont:point:noJacobian", ...
                    "No Jacobian for this Point: %s.", why);
            end
        end
    end
end
