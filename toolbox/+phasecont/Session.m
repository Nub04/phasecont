classdef Session
    % A model and the results of its runs, saved to and loaded from a file.
    %
    %   s = phasecont.Session(m, {br, lc, tr, eq})
    %   save(s, "diagram.pcsession.json")
    %   s = phasecont.Session.load("diagram.pcsession.json")
    %   s.Items{2}                 % the loaded LC Branch
    %   r = phasecont.session.replay("diagram.pcsession.json")
    %
    % The items are Branch, Trajectory or Point objects that came from
    % continuation, simulate or equilibrium on the model M. A file holds the
    % model source and its SHA-256, the engine name and version, the
    % platform, the MATLAB release, the exact problem of every run, and the
    % rows the engine returned, with the numbers stored as 64-bit floats.
    %
    % Load builds the model again from the stored source, checks its hash
    % and rebuilds the items from the stored rows. It runs nothing. Use
    % phasecont.session.replay to run every run again and compare.
    %
    % Items that came from the same run, such as the branches of one
    % branch switch, are stored once. A Point that was taken from a Branch
    % has no run of its own and cannot be saved.
    %
    % Properties
    %   Model  the phasecont.Model of the session
    %   Items  1-by-k cell of Branch, Trajectory and Point objects
    %
    % Methods
    %   save, load (static)

    properties (SetAccess = private)
        Model phasecont.Model {mustBeScalarOrEmpty} = phasecont.Model.empty
        Items (1,:) cell = cell(1, 0)
    end

    methods
        function obj = Session(m, items)
            % S = phasecont.Session(M, ITEMS) holds the items of model M.
            % ITEMS is a cell array, or one Branch, Trajectory or Point, or
            % an array of Branch objects.
            arguments
                m (1,1) phasecont.Model
                items = {}
            end
            if ~iscell(items)
                items = num2cell(items);
            end
            items = reshape(items, 1, []);
            for k = 1:numel(items)
                rec = phasecont.Session.recordOf(items{k}, k);
                if rec.ModelSha ~= m.SourceSha256
                    error("phasecont:session:modelMismatch", ...
                        "Item %d came from a model with another source (SHA-256 %s), not from %s.", ...
                        k, rec.ModelSha, m.File);
                end
            end
            obj.Model = m;
            obj.Items = items;
        end

        function save(obj, file)
            % SAVE(S, FILE) writes the session to FILE as JSON. The usual
            % extension is .pcsession.json.
            arguments
                obj (1,1) phasecont.Session
                file (1,1) string
            end
            m = obj.Model;
            records = {};
            items = zeros(numel(obj.Items), 2);
            for k = 1:numel(obj.Items)
                it = obj.Items{k};
                rec = phasecont.Session.recordOf(it, k);
                at = find(cellfun(@(r) r == rec, records), 1);
                if isempty(at)
                    records{end+1} = rec; %#ok<AGROW>
                    at = numel(records);
                end
                items(k, :) = [at, phasecont.Session.branchOf(it)];
            end
            [~, base, ext] = fileparts(m.File);
            name = base + ext;
            if name == ""
                name = "model.ode";
            end
            v = split(string(pckern("version")));
            header = struct( ...
                "Created", string(datetime("now", TimeZone = "UTC", ...
                    Format = "yyyy-MM-dd'T'HH:mm:ss'Z'")), ...
                "EngineName", v(1), "EngineVersion", v(2), ...
                "Platform", string(computer("arch")), ...
                "Matlab", extractAfter(string(matlabRelease.Release), 1));
            model = struct("Name", name, "Source", m.Source, ...
                "Sha256", m.SourceSha256, "Evaluator", m.Evaluator);
            phasecont.internal.sessionCodec.writeFile(file, header, model, ...
                records, items);
        end
    end

    methods (Static)
        function obj = load(file, options)
            % S = phasecont.Session.load(FILE) reads a session file and
            % returns the Session. It builds the model from the stored
            % source and raises phasecont:session:hashMismatch when the
            % source does not match the stored SHA-256. Evaluator = "jit",
            % "fallback" or "aot" builds the model with that evaluator
            % instead of the recorded one.
            arguments
                file (1,1) string
                options.Evaluator (1,1) string {mustBeMember(options.Evaluator, ["" "jit" "fallback" "aot"])} = ""
            end
            S = phasecont.internal.sessionCodec.readFile(file);
            m = phasecont.internal.sessionCodec.buildModel(S, options.Evaluator);
            items = cell(1, size(S.Items, 1));
            for k = 1:numel(items)
                try
                    items{k} = phasecont.Session.rebuild(m, ...
                        S.Records{S.Items(k, 1)}, S.Items(k, 2));
                catch err
                    if startsWith(err.identifier, "phasecont:session:")
                        rethrow(err);
                    end
                    error("phasecont:session:badFile", ...
                        "Item %d cannot be rebuilt from its stored run: %s", ...
                        k, err.message);
                end
            end
            obj = phasecont.Session(m, items);
        end
    end

    methods (Static, Access = private)
        function rec = recordOf(item, k)
            % The run record of an item, or an error that names the item.
            if ~(isa(item, "phasecont.Branch") || isa(item, "phasecont.Trajectory") ...
                    || isa(item, "phasecont.Point")) || ~isscalar(item)
                error("phasecont:session:badRecord", ...
                    "Item %d is not a Branch, Trajectory or Point.", k);
            end
            rec = item.Record;
            if isempty(rec)
                error("phasecont:session:noRecord", ...
                    "Item %d has no run record. A Point taken from a Branch cannot be saved; save the Branch.", k);
            end
        end

        function b = branchOf(item)
            % The branch number of an item inside its run.
            if isa(item, "phasecont.Branch")
                b = item.RunBranch;
            elseif isa(item, "phasecont.Trajectory")
                b = 1;
            else
                b = 0;
            end
        end

        function item = rebuild(m, rec, b)
            % The item that run REC produced as its branch B (0 for an
            % equilibrium), built from the stored rows.
            switch rec.Kind
                case "equilibrium"
                    item = phasecont.Point(U = rec.Result.U, P = rec.P, ...
                        Eigenvalues = rec.Result.Eigenvalues, Label = "EP", ...
                        Model = m, Record = rec);
                case "simulate"
                    st = phasecont.internal.Run.decode(char(rec.Json));
                    rows = rec.Points{1};
                    problem = jsondecode(rec.Problem);
                    stop = "";
                    warn = strings(0, 1);
                    if isfield(st.branches, "stop") && ~isempty(st.branches(1).stop)
                        stop = string(st.branches(1).stop);
                    end
                    if isfield(st.branches, "warnings") && ~isempty(st.branches(1).warnings)
                        warn = string(st.branches(1).warnings);
                    end
                    item = phasecont.Trajectory(struct( ...
                        "T", rows(1, :).', "X", rows(2:end, :).', "X0", rec.U0, ...
                        "TSpan", reshape(problem.tspan, 1, 2), "P", rec.P, ...
                        "Stop", stop, "Warnings", warn(:), "Model", m));
                    item.attachRecord(rec);
                otherwise
                    st = phasecont.internal.Run.decode(char(rec.Json));
                    item = phasecont.internal.rows2branch(m, rec.Free, rec.P, ...
                        rec.N, rec.Points{b}, rec.Specials{b}, st.branches(b), ...
                        rec.Type, rec.Grid);
                    item.attachRecord(rec, b);
            end
        end
    end
end
