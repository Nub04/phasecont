classdef sessionCodec
    % Array blocks of the session file. Internal.
    %
    %   s = phasecont.internal.sessionCodec.encode(a)   % struct for jsonencode
    %   a = phasecont.internal.sessionCodec.decode(s)   % back to a matrix
    %   c = phasecont.internal.sessionCodec.elements(v) % jsondecode list to cell
    %
    % A block is {"shape":[r,c],"f64le":"<base64>"}: the r-by-c matrix as
    % row-major little-endian IEEE 754 binary64, base64 encoded (RFC 4648,
    % with padding). The bytes keep NaN payloads, infinities, -0 and
    % subnormals, and the Rust reader sees the same bits. MATLAB runs
    % little-endian on every platform the toolbox supports, so typecast needs
    % no byte swap.

    methods (Static)
        function s = encode(a)
            arguments
                a (:,:) double
            end
            [r, c] = size(a);
            bytes = typecast(reshape(a.', 1, []), "uint8");
            if isempty(bytes)
                text = "";
            else
                text = string(matlab.net.base64encode(bytes));
            end
            s = struct("shape", [r c], "f64le", text);
        end

        function a = decode(s)
            if ~(isstruct(s) && isscalar(s) && isfield(s, "shape") && isfield(s, "f64le"))
                error("phasecont:session:badFile", ...
                    "An array block needs ""shape"" and ""f64le"".");
            end
            shape = double(s.shape(:)).';
            if numel(shape) ~= 2 || ~all(isfinite(shape)) || any(shape < 0) ...
                    || any(shape ~= fix(shape))
                error("phasecont:session:badFile", ...
                    "An array block needs a shape of two counts, [rows,cols].");
            end
            if ~(ischar(s.f64le) || (isstring(s.f64le) && isscalar(s.f64le)))
                error("phasecont:session:badFile", ...
                    "The ""f64le"" of an array block must be a string.");
            end
            text = char(s.f64le);
            if isempty(text)
                bytes = zeros(1, 0, "uint8");
            else
                try
                    bytes = matlab.net.base64decode(text);
                catch
                    error("phasecont:session:badFile", ...
                        "The ""f64le"" of an array block is not valid base64.");
                end
            end
            if numel(bytes) ~= 8*prod(shape)
                error("phasecont:session:badFile", ...
                    "An array block of shape [%d,%d] needs %d bytes, not %d.", ...
                    shape(1), shape(2), 8*prod(shape), numel(bytes));
            end
            if isempty(bytes)
                a = zeros(shape);
            else
                a = reshape(typecast(bytes(:).', "double"), shape(2), shape(1)).';
            end
        end

        function c = elements(v)
            % A JSON array of objects as jsondecode returns it: a struct
            % array, a cell (the objects have different fields) or [].
            if isempty(v)
                c = cell(1, 0);
            elseif iscell(v)
                c = v(:).';
            elseif isstruct(v)
                c = num2cell(v(:).');
            else
                error("phasecont:session:badFile", ...
                    "Expected a list of objects in the session file.");
            end
        end

        function s = runStruct(rec)
            % The JSON struct of one RunRecord. Lists are cell arrays so that
            % jsonencode writes arrays for one entry and for none.
            enc = @phasecont.internal.sessionCodec.encode;
            s = struct("kind", rec.Kind, "problem", rec.Problem, ...
                "type", rec.Type, "free", {cellstr(rec.Free)}, ...
                "grid", {num2cell(rec.Grid)}, "n", rec.N, ...
                "u0", enc(rec.U0.'), "p", enc(rec.P.'));
            if rec.Kind == "equilibrium"
                ev = rec.Result.Eigenvalues(:).';
                s.result = struct("u", enc(rec.Result.U(:).'), ...
                    "eig_re", enc(real(ev)), "eig_im", enc(imag(ev)));
                return
            end
            nb = rec.branchCount();
            rows = cell(1, nb);
            sp = cell(1, nb);
            for b = 1:nb
                % One row per point: the engine matrices are stride-by-count.
                pts = rec.Points{b};
                if isempty(pts)
                    pts = zeros(rec.Stride, 0);
                end
                spc = rec.Specials{b};
                if isempty(spc)
                    spc = zeros(rec.Stride, 0);
                end
                rows{b} = enc(pts.');
                sp{b} = enc(spc.');
            end
            s.status = rec.Json;
            s.rows = rows;
            s.specials = sp;
        end

        function writeFile(file, header, model, records, items)
            % Write a version 1 session file.
            %   header  struct: Created, EngineName, EngineVersion, Platform, Matlab
            %   model   struct: Name, Source, Sha256, Evaluator
            %   records cell of RunRecord
            %   items   k-by-2 [run branch], counted from 1; an equilibrium has branch 0
            runs = cell(1, numel(records));
            for k = 1:numel(records)
                runs{k} = phasecont.internal.sessionCodec.runStruct(records{k});
            end
            list = cell(1, size(items, 1));
            for k = 1:size(items, 1)
                list{k} = struct("run", items(k, 1), "branch", items(k, 2));
            end
            s = struct("format", "phasecont-session", "version", 1, ...
                "created", header.Created, ...
                "engine", struct("name", header.EngineName, "version", header.EngineVersion), ...
                "platform", header.Platform, "matlab", header.Matlab, ...
                "precision", "f64", ...
                "model", struct("name", model.Name, "source", model.Source, ...
                    "sha256", model.Sha256, "evaluator", model.Evaluator), ...
                "runs", {runs}, "items", {list});
            text = string(jsonencode(s, PrettyPrint = true)) + newline;
            bytes = unicode2native(char(text), "UTF-8");
            fid = fopen(file, "w");
            if fid < 0
                error("phasecont:session:badFile", ...
                    "Cannot open ""%s"" for writing.", file);
            end
            closer = onCleanup(@() fclose(fid));
            fwrite(fid, bytes, "uint8");
        end

        function S = readFile(file)
            % Read and check a version 1 session file. S has the fields
            % Created, EngineName, EngineVersion, Platform, Matlab, Model
            % (Name, Source, Sha256, Evaluator), Records (cell of RunRecord)
            % and Items (k-by-2). Every error is phasecont:session:badFile.
            file = string(file);
            if ~isfile(file)
                error("phasecont:session:badFile", ...
                    "Cannot read ""%s"": no such file.", file);
            end
            try
                text = fileread(file, Encoding = "UTF-8");
            catch err
                error("phasecont:session:badFile", ...
                    "Cannot read ""%s"": %s", file, err.message);
            end
            try
                v = jsondecode(text);
            catch err
                error("phasecont:session:badFile", ...
                    "Not valid JSON: %s", err.message);
            end
            S = phasecont.internal.sessionCodec.parse(v);
        end

        function S = parse(v)
            % Check the decoded JSON of a session file. See readFile.
            import phasecont.internal.sessionCodec
            if ~(isstruct(v) && isscalar(v))
                sessionCodec.fail("A session file holds one JSON object.");
            end
            if ~(isfield(v, "format") && sessionCodec.isText(v.format) ...
                    && string(v.format) == "phasecont-session")
                sessionCodec.fail("""format"" must be ""phasecont-session"".");
            end
            if ~isfield(v, "version") || ~(isnumeric(v.version) && isscalar(v.version))
                sessionCodec.fail("version: expected a count.");
            end
            if v.version ~= 1
                sessionCodec.fail("unsupported session version %g; this build reads version 1", ...
                    v.version);
            end
            if sessionCodec.str(v, "precision", "session") ~= "f64"
                sessionCodec.fail("precision: only ""f64"" is supported.");
            end
            engine = sessionCodec.obj(v, "engine", "session");
            model = sessionCodec.obj(v, "model", "session");
            S = struct();
            S.Created = sessionCodec.str(v, "created", "session");
            S.EngineName = sessionCodec.str(engine, "name", "engine");
            S.EngineVersion = sessionCodec.str(engine, "version", "engine");
            S.Platform = sessionCodec.str(v, "platform", "session");
            S.Matlab = sessionCodec.str(v, "matlab", "session");
            S.Model = struct( ...
                "Name", sessionCodec.str(model, "name", "model"), ...
                "Source", sessionCodec.str(model, "source", "model"), ...
                "Sha256", sessionCodec.str(model, "sha256", "model"), ...
                "Evaluator", sessionCodec.str(model, "evaluator", "model"));
            runs = sessionCodec.list(v, "runs", "session");
            S.Records = cell(1, numel(runs));
            for k = 1:numel(runs)
                S.Records{k} = sessionCodec.parseRun(runs{k}, ...
                    sprintf("runs[%d]", k - 1), S.Model.Sha256);
            end
            items = sessionCodec.list(v, "items", "session");
            S.Items = zeros(numel(items), 2);
            for k = 1:numel(items)
                ctx = sprintf("items[%d]", k - 1);
                it = items{k};
                if ~(isstruct(it) && isscalar(it))
                    sessionCodec.fail("%s: expected an object.", ctx);
                end
                S.Items(k, :) = [sessionCodec.count(it, "run", ctx), ...
                    sessionCodec.count(it, "branch", ctx)];
                r = S.Items(k, 1);
                b = S.Items(k, 2);
                if r < 1 || r > numel(S.Records)
                    sessionCodec.fail("%s: run %d does not exist.", ctx, r);
                end
                rec = S.Records{r};
                if rec.Kind == "equilibrium"
                    ok = b == 0;
                else
                    ok = b >= 1 && b <= rec.branchCount();
                end
                if ~ok
                    sessionCodec.fail("%s: branch %d does not exist in run %d.", ctx, b, r);
                end
            end
        end

        function rec = parseRun(r, ctx, sha)
            % One run object of the file as a RunRecord.
            import phasecont.internal.sessionCodec
            if ~(isstruct(r) && isscalar(r))
                sessionCodec.fail("%s: expected an object.", ctx);
            end
            kind = sessionCodec.str(r, "kind", ctx);
            if ~any(kind == ["continuation" "simulate" "equilibrium"])
                sessionCodec.fail("%s.kind: ""%s"" is not continuation, simulate or equilibrium.", ...
                    ctx, kind);
            end
            f = struct("Kind", kind, "Problem", sessionCodec.str(r, "problem", ctx), ...
                "Type", sessionCodec.str(r, "type", ctx), ...
                "Free", sessionCodec.strings(r, "free", ctx), ...
                "Grid", sessionCodec.counts(r, "grid", ctx), ...
                "N", sessionCodec.count(r, "n", ctx), "ModelSha", sha);
            f.U0 = sessionCodec.block(r, "u0", ctx);
            f.P = sessionCodec.block(r, "p", ctx);
            f.U0 = f.U0(:);
            f.P = f.P(:);
            if kind == "equilibrium"
                rctx = ctx + ".result";
                if ~isfield(r, "result") || ~(isstruct(r.result) && isscalar(r.result))
                    sessionCodec.fail("%s: expected an object.", rctx);
                end
                u = sessionCodec.block(r.result, "u", rctx);
                re = sessionCodec.block(r.result, "eig_re", rctx);
                im = sessionCodec.block(r.result, "eig_im", rctx);
                if numel(re) ~= numel(im)
                    sessionCodec.fail("%s: eig_re and eig_im differ in length.", rctx);
                end
                f.Result = struct("U", u(:), "Eigenvalues", complex(re(:), im(:)));
                rec = phasecont.internal.RunRecord(f);
                return
            end
            f.Json = sessionCodec.str(r, "status", ctx);
            try
                st = phasecont.internal.Run.decode(char(f.Json));
            catch
                sessionCodec.fail("%s.status: not a JSON object.", ctx);
            end
            need = ["stride" "kind" "n" "state" "branches"];
            if ~all(isfield(st, need)) || ~isnumeric(st.stride) || ~isscalar(st.stride) ...
                    || (~isempty(st.branches) && ~all(isfield(st.branches, ["points" "specials"])))
                sessionCodec.fail("%s.status: missing stride, kind, n, state or branch fields.", ctx);
            end
            rows = sessionCodec.blocks(r, "rows", ctx);
            sp = sessionCodec.blocks(r, "specials", ctx);
            if numel(rows) ~= numel(sp)
                sessionCodec.fail("%s: %d row blocks but %d special blocks.", ...
                    ctx, numel(rows), numel(sp));
            end
            if numel(st.branches) ~= numel(rows)
                sessionCodec.fail("%s: the status lists %d branches but the file has %d row blocks.", ...
                    ctx, numel(st.branches), numel(rows));
            end
            for b = 1:numel(rows)
                if size(rows{b}, 2) ~= st.stride || size(sp{b}, 2) ~= st.stride ...
                        || size(rows{b}, 1) ~= st.branches(b).points ...
                        || size(sp{b}, 1) ~= numel(st.branches(b).specials)
                    sessionCodec.fail("%s: the blocks of branch %d do not match the status.", ...
                        ctx, b);
                end
                rows{b} = rows{b}.';
                sp{b} = sp{b}.';
            end
            f.Points = rows;
            f.Specials = sp;
            f.Stride = st.stride;
            rec = phasecont.internal.RunRecord(f);
        end

        function r = compareArrays(a, b)
            % Compare a new array A with a stored array B. Same needs equal
            % sizes, equal NaN positions and identical infinities. MaxRelDiff
            % is max |a-b|/max(1,|b|) over the finite entries. Identical is
            % bit equality.
            r = struct("Same", false, "MaxRelDiff", Inf, "Identical", false);
            if ~isequal(size(a), size(b))
                return
            end
            if isempty(a)
                r = struct("Same", true, "MaxRelDiff", 0, "Identical", true);
                return
            end
            if ~isequal(isnan(a), isnan(b))
                return
            end
            ia = isinf(a);
            if ~isequal(ia, isinf(b)) || ~isequal(a(ia), b(ia))
                return
            end
            fin = isfinite(a);
            md = 0;
            if any(fin(:))
                md = max(abs(a(fin) - b(fin)) ./ max(1, abs(b(fin))));
            end
            same = isequal(typecast(a(:), "uint64"), typecast(b(:), "uint64"));
            r = struct("Same", true, "MaxRelDiff", md, "Identical", same);
        end

        function c = compareStatus(newJson, storedJson)
            % Compare two final status strings: kind, n, stride, state, the
            % branch count, the points of each branch (Structure); the stop
            % reasons (Stops) and the special points' label and index
            % (Labels).
            import phasecont.internal.sessionCodec
            ns = phasecont.internal.Run.decode(char(newJson));
            ss = phasecont.internal.Run.decode(char(storedJson));
            same = @(k) isequal(sessionCodec.pick(ns, k), sessionCodec.pick(ss, k));
            c = struct("Structure", same("kind") && same("n") && same("stride") ...
                && numel(ns.branches) == numel(ss.branches), ...
                "Labels", true, "Stops", same("state"), "Notes", strings(0, 1));
            for b = 1:min(numel(ns.branches), numel(ss.branches))
                nb = ns.branches(b);
                sb = ss.branches(b);
                if ~isequal(sessionCodec.pick(nb, "points"), sessionCodec.pick(sb, "points"))
                    c.Structure = false;
                    c.Notes(end+1, 1) = sprintf("branch %d: point counts differ", b);
                end
                if ~isequal(sessionCodec.pick(nb, "stop"), sessionCodec.pick(sb, "stop"))
                    c.Stops = false;
                    c.Notes(end+1, 1) = sprintf("branch %d: stop reasons differ", b);
                end
                if ~isequal(sessionCodec.keys(nb), sessionCodec.keys(sb))
                    c.Labels = false;
                    c.Notes(end+1, 1) = sprintf("branch %d: special points differ", b);
                end
            end
        end

        function c = compareRun(rec, points, specials, json)
            % Compare a stored run with a rerun. POINTS and SPECIALS are
            % 1-by-nb cells of stride-by-k matrices and JSON is the final
            % status of the rerun. C has Same, MaxRelDiff and Identical over
            % all blocks, Structure, Labels and Stops from compareStatus,
            % the point counts Stored and New per branch, and Notes.
            import phasecont.internal.sessionCodec
            nb = rec.branchCount();
            c = struct("Same", true, "MaxRelDiff", 0, "Identical", true, ...
                "Structure", true, "Labels", true, "Stops", true, ...
                "Stored", cellfun(@(a) size(a, 2), rec.Points), ...
                "New", cellfun(@(a) size(a, 2), points), ...
                "Notes", strings(0, 1));
            if numel(points) ~= nb || numel(specials) ~= nb
                c.Same = false;
                c.MaxRelDiff = Inf;
                c.Identical = false;
                c.Structure = false;
                c.Notes(end+1, 1) = sprintf("the rerun has %d branches, the file %d", ...
                    numel(points), nb);
                return
            end
            for b = 1:nb
                pairs = {points{b}, rec.Points{b}, "points"; ...
                    specials{b}, rec.Specials{b}, "special points"};
                for k = 1:2
                    r = sessionCodec.compareArrays(pairs{k, 1}, pairs{k, 2});
                    if ~r.Same
                        c.Same = false;
                        c.Notes(end+1, 1) = sprintf("branch %d: the %s differ in size, NaN or Inf", ...
                            b, pairs{k, 3});
                    end
                    c.MaxRelDiff = max(c.MaxRelDiff, r.MaxRelDiff);
                    c.Identical = c.Identical && r.Identical;
                end
            end
            s = sessionCodec.compareStatus(json, rec.Json);
            c.Structure = s.Structure;
            c.Labels = s.Labels;
            c.Stops = s.Stops;
            c.Notes = [c.Notes; s.Notes];
            c.Identical = c.Identical && s.Structure && s.Labels && s.Stops;
        end

        function m = buildModel(S, evaluator)
            % Build the model a session file holds and check its hash.
            % EVALUATOR "" uses the one the file records.
            ev = S.Model.Evaluator;
            if evaluator ~= ""
                ev = evaluator;
            end
            m = phasecont.Model.fromSource(S.Model.Source, Name = S.Model.Name, ...
                Evaluator = ev);
            if m.SourceSha256 ~= S.Model.Sha256
                error("phasecont:session:hashMismatch", ...
                    "The model source in the file hashes to %s but the file records %s.", ...
                    m.SourceSha256, S.Model.Sha256);
            end
        end
    end

    methods (Static, Access = private)
        function fail(fmt, varargin)
            error("phasecont:session:badFile", fmt, varargin{:});
        end

        function tf = isText(v)
            tf = ischar(v) || (isstring(v) && isscalar(v));
        end

        function v = pick(s, key)
            % A field of a decoded status, or [] when it is absent or null.
            if isfield(s, key)
                v = s.(key);
            else
                v = [];
            end
        end

        function k = keys(branchStatus)
            % The (label, index) pairs of the special points of a branch.
            sp = phasecont.internal.sessionCodec.pick(branchStatus, "specials");
            k = cell(numel(sp), 2);
            for i = 1:numel(sp)
                k{i, 1} = phasecont.internal.sessionCodec.pick(sp(i), "label");
                k{i, 2} = phasecont.internal.sessionCodec.pick(sp(i), "index");
            end
        end

        function x = field(s, key, ctx)
            if ~isfield(s, key)
                phasecont.internal.sessionCodec.fail("%s: missing ""%s"".", ctx, key);
            end
            x = s.(key);
        end

        function t = str(s, key, ctx)
            x = phasecont.internal.sessionCodec.field(s, key, ctx);
            if ~phasecont.internal.sessionCodec.isText(x)
                phasecont.internal.sessionCodec.fail("%s.%s: expected a string.", ctx, key);
            end
            t = string(x);
        end

        function n = count(s, key, ctx)
            x = phasecont.internal.sessionCodec.field(s, key, ctx);
            if ~(isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0 && x == fix(x))
                phasecont.internal.sessionCodec.fail("%s.%s: expected a count.", ctx, key);
            end
            n = double(x);
        end

        function o = obj(s, key, ctx)
            o = phasecont.internal.sessionCodec.field(s, key, ctx);
            if ~(isstruct(o) && isscalar(o))
                phasecont.internal.sessionCodec.fail("%s: expected an object.", key);
            end
        end

        function c = list(s, key, ctx)
            x = phasecont.internal.sessionCodec.field(s, key, ctx);
            if ~(isstruct(x) || iscell(x) || (isnumeric(x) && isempty(x)))
                phasecont.internal.sessionCodec.fail("%s: expected a list.", key);
            end
            c = phasecont.internal.sessionCodec.elements(x);
        end

        function t = strings(s, key, ctx)
            x = phasecont.internal.sessionCodec.field(s, key, ctx);
            if isnumeric(x) && isempty(x)
                t = strings(1, 0);
            elseif iscell(x) && all(cellfun(@phasecont.internal.sessionCodec.isText, x(:)))
                t = reshape(string(x), 1, []);
            else
                phasecont.internal.sessionCodec.fail( ...
                    "%s.%s: expected a list of strings.", ctx, key);
            end
        end

        function g = counts(s, key, ctx)
            x = phasecont.internal.sessionCodec.field(s, key, ctx);
            if isnumeric(x) && (isempty(x) || (isvector(x) && all(isfinite(x)) ...
                    && all(x >= 0) && all(x == fix(x))))
                g = reshape(double(x), 1, []);
            else
                phasecont.internal.sessionCodec.fail( ...
                    "%s.%s: expected a list of counts.", ctx, key);
            end
        end

        function a = block(s, key, ctx)
            x = phasecont.internal.sessionCodec.field(s, key, ctx);
            try
                a = phasecont.internal.sessionCodec.decode(x);
            catch err
                phasecont.internal.sessionCodec.fail("%s.%s: %s", ctx, key, err.message);
            end
        end

        function c = blocks(s, key, ctx)
            e = phasecont.internal.sessionCodec.list(s, key, ctx);
            c = cell(1, numel(e));
            for k = 1:numel(e)
                try
                    c{k} = phasecont.internal.sessionCodec.decode(e{k});
                catch err
                    phasecont.internal.sessionCodec.fail("%s.%s[%d]: %s", ...
                        ctx, key, k - 1, err.message);
                end
            end
        end
    end
end
