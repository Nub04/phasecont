classdef Branch < handle
    % A computed branch of equilibria with its special points.
    %
    %   br = phasecont.continuation(m, eq, Free = "r")
    %   h  = br.points("H")     % special points with label H
    %   all = br.points()       % every special point
    %
    % A Branch holds copied data and no engine handle, so it stays valid
    % after the Model is deleted. Rows are points: N points give N-by-1 or
    % N-by-n arrays. Call points on one Branch at a time, not on an array.
    %
    % Properties
    %   Free         name of the free parameter
    %   FreeIndex    its index in Model.Parameters
    %   P            parameter vector the run started from
    %   Alpha        free parameter value at each point
    %   U            states, N-by-n
    %   Eigenvalues  N-by-n complex
    %   Stable       true where all eigenvalues have negative real part
    %   Tests        [psi_LP psi_BP psi_H] at each point, N-by-3
    %   StepSize     step length used to reach each point
    %   NewtonIters  Newton iterations used at each point
    %   Specials     column of phasecont.Point
    %   Stop         why the run stopped ("" when unknown)
    %   Warnings     warning messages from the run
    %   Model        the phasecont.Model the run used

    properties (SetAccess = private)
        Free (1,1) string = ""
        FreeIndex (1,1) double = NaN
        P (:,1) double = zeros(0,1)
        Alpha (:,1) double = zeros(0,1)
        U (:,:) double = zeros(0,0)
        Eigenvalues (:,:) double = zeros(0,0)
        Stable (:,1) logical = false(0,1)
        Tests (:,:) double = zeros(0,3)
        StepSize (:,1) double = zeros(0,1)
        NewtonIters (:,1) double = zeros(0,1)
        Specials (:,1) phasecont.Point = phasecont.Point.empty(0,1)
        Stop (1,1) string = ""
        Warnings (:,1) string = strings(0,1)
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
    end

    methods
        function obj = Branch(values)
            % Internal use: values is a struct with the property names as fields.
            arguments
                values (1,1) struct = struct()
            end
            for name = string(fieldnames(values))'
                obj.(name) = values.(name);
            end
        end

        function pts = points(br, label)
            % PTS = BR.POINTS() returns every special point, BR.POINTS(LABEL)
            % those with that label. An unknown label gives an empty array.
            if ~isscalar(br)
                error("phasecont:model:badBranch", ...
                    "points works on one Branch at a time; call it per element.");
            end
            pts = br.Specials;
            if nargin < 2 || isempty(pts)
                return
            end
            label = string(label);
            if ~isscalar(label)
                error("phasecont:model:badBranch", "The label must be one string.");
            end
            labels = strings(numel(pts), 1);
            for k = 1:numel(pts)
                labels(k) = pts(k).Label;
            end
            pts = pts(labels == label);
            if isempty(pts)
                pts = phasecont.Point.empty(0, 1);
            end
        end
    end
end
