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
    end
end
