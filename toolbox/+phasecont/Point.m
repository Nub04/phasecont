classdef Point
    % A point of a model: an equilibrium or a special point on a branch.
    %
    %   p = phasecont.Point(U = u, P = pars, Label = "EP")
    %
    % The properties are immutable. Label-specific values (coefficients,
    % frequency, normalization, refined) go in Data.

    properties (SetAccess = immutable)
        U (:,1) double = zeros(0,1)
        P (:,1) double = zeros(0,1)
        Eigenvalues (:,1) double = zeros(0,1)
        Label (1,1) string = "EP"
        Index (1,1) double = NaN
        Data (1,1) struct = struct()
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
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
            end
            obj.U = options.U;
            obj.P = options.P;
            obj.Eigenvalues = options.Eigenvalues;
            obj.Label = options.Label;
            obj.Index = options.Index;
            obj.Data = options.Data;
            obj.Model = options.Model;
        end
    end
end
