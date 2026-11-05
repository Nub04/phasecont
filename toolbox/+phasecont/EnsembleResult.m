classdef EnsembleResult < handle
    % The result of an ensemble run: many trajectories of one model.
    %
    %   r = phasecont.ensemble(m, X0, P, [0 50], Precision = "single", ...
    %       Rtol = 1e-4, Atol = 1e-6)
    %   r.Final           % n-by-M states at t1
    %   r.Label           % which backend, device and precision made them
    %   f64 = r.refine(3) % member 3 again, in double precision on the CPU
    %
    % An EnsembleResult holds copied data and no engine handle. It is not
    % saved in a session; the model source is.
    %
    % A result computed in single precision has Label.SinglePrecision =
    % true. It is for screening: look at it, pick members, and call refine
    % before the states feed continuation, a normal form or Floquet
    % multipliers. Those need double precision.
    %
    % Properties
    %   Model       the phasecont.Model the run used
    %   X0          start states, n-by-M
    %   P           parameters, np-by-M (a shared set is repeated)
    %   Times       output times, S-by-1 (just t1 when no sample times were given)
    %   States      n-by-S-by-M; States(:, s, k) is member k at Times(s)
    %   Status      M-by-1 string: "ok", "bounds", "stepTooSmall",
    %               "maxSteps" or "notFinite". For a member that did not end
    %               "ok", States holds its last state and TReached the time
    %               it got to
    %   StatusCode  the same as numbers 0 to 4
    %   TReached    time each member reached, M-by-1
    %   Accepted    accepted steps of each member (fixed steps for RK4)
    %   Rejected    rejected steps of each member (0 for RK4)
    %   Label       struct with Backend, Device, Driver, Precision and
    %               SinglePrecision
    %   Job         struct with the settings of the run: TSpan, Method,
    %               Dt, Rtol, Atol, Hmax, MaxSteps, Bounds
    %   Final       n-by-M states at the last output time
    %   Members     number of trajectories
    %
    % Methods
    %   refine

    properties (SetAccess = private)
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
        X0 (:,:) double = zeros(0,0)
        P (:,:) double = zeros(0,0)
        Times (:,1) double = zeros(0,1)
        States (:,:,:) double = zeros(0,0,0)
        Status (:,1) string = strings(0,1)
        StatusCode (:,1) uint8 = zeros(0,1,"uint8")
        TReached (:,1) double = zeros(0,1)
        Accepted (:,1) double = zeros(0,1)
        Rejected (:,1) double = zeros(0,1)
        Label (1,1) struct = struct()
        Job (1,1) struct = struct()
    end

    properties (Dependent)
        Final
        Members
    end

    methods
        function obj = EnsembleResult(values)
            % Internal use: values is a struct with the property names as fields.
            arguments
                values (1,1) struct = struct()
            end
            for name = string(fieldnames(values))'
                obj.(name) = values.(name);
            end
        end

        function f = get.Final(obj)
            f = reshape(obj.States(:, end, :), size(obj.States, 1), []);
        end

        function k = get.Members(obj)
            k = size(obj.States, 3);
        end

        function r = refine(obj, idx)
            %REFINE Run members again in double precision on the CPU.
            %   R = REFINE(OBJ, IDX) integrates the members listed in IDX
            %   from their original start states and parameters with the
            %   same method, tolerances, limits and output times, in double
            %   precision on the CPU. It returns a phasecont.EnsembleResult
            %   whose Label says "cpu" and "double". Use R.Final(:, k) as a
            %   start point for continuation.
            arguments
                obj (1,1) phasecont.EnsembleResult
                idx (1,:) double
            end
            if isempty(idx) || ~all(isfinite(idx)) || any(idx ~= fix(idx)) ...
                    || any(idx < 1) || any(idx > obj.Members)
                error("phasecont:ensemble:badIndex", ...
                    "IDX must list members from 1 to %d.", obj.Members);
            end
            j = obj.Job;
            args = {"Backend", "cpu", "Precision", "double", ...
                "MaxSteps", j.MaxSteps, "Bounds", j.Bounds};
            if j.Method == "rk4"
                args = [args, {"Dt", j.Dt}];
            else
                args = [args, {"Rtol", j.Rtol, "Atol", j.Atol, "Hmax", j.Hmax}];
            end
            r = phasecont.ensemble(obj.Model, obj.X0(:, idx), ...
                obj.P(:, idx), j.TSpan, args{:});
        end
    end
end
