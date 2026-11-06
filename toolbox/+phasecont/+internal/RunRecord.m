classdef RunRecord < handle
    % The inputs and the raw output of one engine run. Internal.
    %
    %   rec = phasecont.internal.RunRecord(fields)
    %
    % A session file stores one record per run, and a replay reruns the run
    % from the record. The sibling Branches of one run share one record, so
    % the record is a handle. It holds copied data and no engine handle.
    %
    % Properties
    %   Kind       "continuation", "simulate" or "equilibrium"
    %   Problem    the exact problem JSON string given to run_start; for an
    %              equilibrium, the exact options JSON string given to
    %              model_equilibrium
    %   U0         start state buffer, or the Newton guess of an equilibrium
    %   P          parameter vector of the run
    %   ModelSha   SHA-256 of the model source, 64 hex digits
    %   Type       "EP", "LP", "H", "LC", "LPC", "PD", "NS" or "HOM"; "IVP"
    %              for a trajectory; "" for an equilibrium
    %   Free       free parameter name or names, as passed to rows2branch
    %   Grid       [NTST NCOL] of a run of cycles or a HOM curve, else []
    %   N          number of states
    %   Json       the final status JSON string, as the engine wrote it
    %   Points     1-by-nb cell, one stride-by-k matrix per branch
    %   Specials   1-by-nb cell, the special-point rows of each branch
    %   Stride     values per row
    %   Result     equilibrium only: struct with U (n-by-1) and Eigenvalues
    %              (n-by-1 complex)

    properties (SetAccess = private)
        Kind (1,1) string = "continuation"
        Problem (1,1) string = ""
        U0 (:,1) double = zeros(0,1)
        P (:,1) double = zeros(0,1)
        ModelSha (1,1) string = ""
        Type (1,1) string = ""
        Free (1,:) string = strings(1,0)
        Grid (1,:) double = zeros(1,0)
        N (1,1) double = 0
        Json (1,1) string = ""
        Points (1,:) cell = cell(1,0)
        Specials (1,:) cell = cell(1,0)
        Stride (1,1) double = 0
        Result (1,1) struct = struct()
    end

    methods
        function obj = RunRecord(fields)
            % FIELDS is a struct whose field names are property names.
            arguments
                fields (1,1) struct
            end
            allowed = string(properties(obj));
            for name = string(fieldnames(fields))'
                if ~any(name == allowed)
                    error("phasecont:session:badRecord", ...
                        "RunRecord has no property ""%s"".", name);
                end
                obj.(name) = fields.(name);
            end
            if ~any(obj.Kind == ["continuation" "simulate" "equilibrium"])
                error("phasecont:session:badRecord", ...
                    "Kind must be ""continuation"", ""simulate"" or ""equilibrium"".");
            end
            if numel(obj.Specials) ~= numel(obj.Points)
                error("phasecont:session:badRecord", ...
                    "Points and Specials need one entry per branch.");
            end
        end

        function n = branchCount(obj)
            % Number of branches the run produced (0 for an equilibrium).
            n = numel(obj.Points);
        end
    end
end
