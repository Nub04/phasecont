classdef SessionFormatTest < matlab.unittest.TestCase
    % Tests the array blocks of the session file. No test here calls pckern,
    % so the suite runs without the MEX binary.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function specialValuesKeepTheirBits(testCase)
            a = [NaN, Inf, -Inf, -0, 0, 5e-324, 2.2250738585072009e-308, realmax, ...
                -realmin, pi];
            a(1) = typecast(bitshift(uint64(hex2dec('7FF80000')), 32) + uint64(1), 'double'); % NaN payload
            s = phasecont.internal.sessionCodec.encode(a);
            b = phasecont.internal.sessionCodec.decode(s);
            testCase.verifyEqual(typecast(b, 'uint64'), typecast(a, 'uint64'));
            testCase.verifyEqual(size(b), size(a));
        end

        function bytesAreRowMajor(testCase)
            a = [1 2 3; 4 5 6];
            s = phasecont.internal.sessionCodec.encode(a);
            testCase.verifyEqual(s.shape, [2 3]);
            raw = typecast(matlab.net.base64decode(char(s.f64le)), 'double');
            testCase.verifyEqual(raw, [1 2 3 4 5 6]);
        end

        function nonSquareBlockRoundTrips(testCase)
            a = reshape(1:15, 5, 3).' * 0.1;     % 3-by-5
            b = phasecont.internal.sessionCodec.decode( ...
                phasecont.internal.sessionCodec.encode(a));
            testCase.verifyEqual(b, a);
        end

        function emptyBlocksKeepTheirShape(testCase)
            for sz = {[0 4], [3 0], [0 0]}
                a = zeros(sz{1});
                s = phasecont.internal.sessionCodec.encode(a);
                testCase.verifyEqual(s.f64le, "");
                b = phasecont.internal.sessionCodec.decode(s);
                testCase.verifyEqual(size(b), sz{1});
            end
        end

        function survivesAJsonRoundTrip(testCase)
            a = [1.5 NaN -0; Inf 2 3];
            s = phasecont.internal.sessionCodec.encode(a);
            back = jsondecode(jsonencode(s));
            b = phasecont.internal.sessionCodec.decode(back);
            testCase.verifyEqual(typecast(b(:), 'uint64'), typecast(a(:), 'uint64'));
        end

        function aColumnShapeFromJsondecodeIsAccepted(testCase)
            s = struct("shape", [2; 1], "f64le", ...
                string(matlab.net.base64encode(typecast([7 8], 'uint8'))));
            testCase.verifyEqual(phasecont.internal.sessionCodec.decode(s), [7; 8]);
        end

        function rejectsABlockWithMissingKeys(testCase)
            testCase.verifyError( ...
                @() phasecont.internal.sessionCodec.decode(struct("shape", [1 1])), ...
                "phasecont:session:badFile");
            testCase.verifyError( ...
                @() phasecont.internal.sessionCodec.decode(42), ...
                "phasecont:session:badFile");
        end

        function rejectsABadShape(testCase)
            for shape = {[1 2 3], [-1 2], [1.5 2], [NaN 2]}
                s = struct("shape", shape{1}, "f64le", "");
                testCase.verifyError(@() phasecont.internal.sessionCodec.decode(s), ...
                    "phasecont:session:badFile");
            end
        end

        function rejectsTheWrongByteCount(testCase)
            s = phasecont.internal.sessionCodec.encode([1 2 3]);
            s.shape = [1 4];
            testCase.verifyError(@() phasecont.internal.sessionCodec.decode(s), ...
                "phasecont:session:badFile");
        end

        function rejectsInvalidBase64(testCase)
            s = struct("shape", [1 1], "f64le", "!!!!not base64!!!!");
            testCase.verifyError(@() phasecont.internal.sessionCodec.decode(s), ...
                "phasecont:session:badFile");
        end

        function elementsHandlesTheThreeShapesOfJsondecode(testCase)
            c = phasecont.internal.sessionCodec.elements(jsondecode('[{"a":1},{"a":2}]'));
            testCase.verifyEqual(numel(c), 2);
            testCase.verifyEqual(c{2}.a, 2);
            c = phasecont.internal.sessionCodec.elements(jsondecode('[{"a":1},{"b":2}]'));
            testCase.verifyEqual(numel(c), 2);
            testCase.verifyEqual(c{2}.b, 2);
            c = phasecont.internal.sessionCodec.elements(jsondecode('[]'));
            testCase.verifyEqual(size(c), [1 0]);
            testCase.verifyError( ...
                @() phasecont.internal.sessionCodec.elements(3), ...
                "phasecont:session:badFile");
        end

        function decodesTheRustFixture(testCase)
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            testCase.verifyEqual(S.EngineName, "pckern");
            testCase.verifyEqual(S.Model.Name, "morris_lecar.ode");
            testCase.verifyEqual(S.Model.Evaluator, "jit");
            testCase.verifyEqual(strlength(S.Model.Sha256), 64);
            testCase.verifyEqual(S.Items, [1 0; 2 1; 3 1]);
            testCase.verifyEqual(cellfun(@(r) r.Kind, S.Records), ...
                ["equilibrium" "continuation" "simulate"]);
            eq = S.Records{1};
            testCase.verifyEqual(size(eq.Result.U), [2 1]);
            testCase.verifyEqual(size(eq.Result.Eigenvalues), [2 1]);
            testCase.verifyFalse(isreal(eq.Result.Eigenvalues));
            testCase.verifyEqual(eq.Free, strings(1, 0));
            ep = S.Records{2};
            testCase.verifyEqual(ep.Type, "EP");
            testCase.verifyEqual(ep.Free, "I");
            testCase.verifyEqual(ep.Stride, 12);
            testCase.verifyEqual(size(ep.Points{1}), [12 25]);
            testCase.verifyEqual(size(ep.Specials{1}), [12 1]);
            tr = S.Records{3};
            testCase.verifyEqual(tr.Type, "IVP");
            testCase.verifyEqual(size(tr.Points{1}), [3 21]);
            testCase.verifyEqual(size(tr.Specials{1}), [3 0]);
            testCase.verifyEqual(tr.Points{1}(1, 1), 0);
        end

        function writeFileAndReadFileRoundTrip(testCase)
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            file = fullfile(testCase.tempFolder(), "again.pcsession.json");
            testCase.write(file, S);
            T = phasecont.internal.sessionCodec.readFile(file);
            testCase.verifySameSession(T, S);
            text = fileread(file, Encoding = "UTF-8");
            testCase.verifySubstring(text, """format"": ""phasecont-session""");
            testCase.verifyEqual(text(end), newline);
        end

        function theMatlabWriterOutputIsReadByRust(testCase)
            % Writes the Rust fixture again with the MATLAB writer. The
            % ignored Rust test the_matlab_writer_matches_the_fixture reads
            % the file named by PC_REENCODED_SESSION and compares the bits.
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            file = string(getenv("PC_REENCODED_SESSION"));
            if file == ""
                file = fullfile(testCase.tempFolder(), "reencoded.pcsession.json");
            end
            testCase.write(file, S);
            T = phasecont.internal.sessionCodec.readFile(file);
            testCase.verifySameSession(T, S);
        end

        function aSourceWithNonAsciiTextSurvives(testCase)
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            S.Model.Source = S.Model.Source + "# mu µ, u ü, arrow →, quote "" and \ and tab " + sprintf("\t") + newline;
            file = fullfile(testCase.tempFolder(), "unicode.pcsession.json");
            testCase.write(file, S);
            T = phasecont.internal.sessionCodec.readFile(file);
            testCase.verifyEqual(T.Model.Source, S.Model.Source);
        end

        function freeAndGridListsKeepTheirLength(testCase)
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            base = S.Records{2};
            cases = {struct("Free", ["I" "phi"], "Grid", [10 4]), ...
                struct("Free", "I", "Grid", 8), ...
                struct("Free", strings(1, 0), "Grid", zeros(1, 0))};
            for k = 1:numel(cases)
                S.Records{2} = testCase.changed(base, cases{k});
                file = fullfile(testCase.tempFolder(), sprintf("lists%d.json", k));
                testCase.write(file, S);
                T = phasecont.internal.sessionCodec.readFile(file);
                testCase.verifyEqual(T.Records{2}.Free, cases{k}.Free);
                testCase.verifyEqual(T.Records{2}.Grid, cases{k}.Grid);
            end
        end

        function aWrongVersionNamesTheVersion(testCase)
            v = testCase.fixtureStruct();
            v.version = 2;
            try
                phasecont.internal.sessionCodec.parse(v);
                testCase.verifyFail("parse accepted version 2");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:session:badFile");
                testCase.verifySubstring(err.message, "unsupported session version 2");
            end
        end

        function aBrokenFileIsBadFile(testCase)
            mutations = {
                @(v) setfield(v, "format", "other"), ...
                @(v) setfield(v, "precision", "f32"), ...
                @(v) setfield(v, "version", "1"), ...
                @(v) rmfield(v, "model"), ...
                @(v) rmfield(v, "runs"), ...
                @(v) rmfield(v, "engine"), ...
                @(v) setfield(v, "items", {struct("run", 9, "branch", 1)}), ...
                @(v) setfield(v, "items", {struct("run", 2, "branch", 5)}), ...
                @(v) setfield(v, "items", {struct("run", 2, "branch", 0)}), ...
                @(v) setfield(v, "items", {struct("run", 1, "branch", 1)}), ...
                @(v) setfield(v, "items", {struct("run", 1.5, "branch", 0)}), ...
                @(v) setfield(v, "items", {struct("run", 1)}), ...
                @(v) editRun(v, 2, "kind", "bogus"), ...
                @(v) editRun(v, 2, "status", jsonencode(5)), ...
                @(v) editRun(v, 2, "status", "not json"), ...
                @(v) editRun(v, 2, "status", v.runs{3}.status), ...
                @(v) editRun(v, 2, "grid", {"x"}), ...
                @(v) editRun(v, 2, "free", {1, 2}), ...
                @(v) editRun(v, 2, "n", -1), ...
                @(v) editRun(v, 1, "result", 3), ...
                @(v) editRun(v, 2, "u0", struct("shape", [1 2])) ...
                };
            for k = 1:numel(mutations)
                v = mutations{k}(testCase.fixtureStruct());
                testCase.verifyError(@() phasecont.internal.sessionCodec.parse(v), ...
                    "phasecont:session:badFile", sprintf("mutation %d", k));
            end

            function v = editRun(v, k, key, value)
                v.runs{k}.(key) = value;
            end
        end

        function aBlockWithTheWrongRowsIsBadFile(testCase)
            v = testCase.fixtureStruct();
            rows = phasecont.internal.sessionCodec.elements(v.runs{2}.rows);
            rows{1} = phasecont.internal.sessionCodec.encode(zeros(24, 12));
            v.runs{2}.rows = rows;
            testCase.verifyError(@() phasecont.internal.sessionCodec.parse(v), ...
                "phasecont:session:badFile");
            rows{1} = phasecont.internal.sessionCodec.encode(zeros(25, 11));
            v.runs{2}.rows = rows;
            testCase.verifyError(@() phasecont.internal.sessionCodec.parse(v), ...
                "phasecont:session:badFile");
        end

        function aMissingOrCorruptFileIsBadFile(testCase)
            tmp = testCase.tempFolder();
            testCase.verifyError(@() phasecont.internal.sessionCodec.readFile( ...
                fullfile(tmp, "absent.pcsession.json")), "phasecont:session:badFile");
            bad = fullfile(tmp, "corrupt.pcsession.json");
            fid = fopen(bad, "w");
            fwrite(fid, "{not json", "char");
            fclose(fid);
            testCase.verifyError(@() phasecont.internal.sessionCodec.readFile(bad), ...
                "phasecont:session:badFile");
            testCase.verifyError(@() phasecont.Session.load(bad), ...
                "phasecont:session:badFile");
            testCase.verifyError(@() phasecont.session.replay(bad), ...
                "phasecont:session:badFile");
        end

        function compareArraysReportsSizeNanInfAndBits(testCase)
            ca = @phasecont.internal.sessionCodec.compareArrays;
            a = [1 2 3; 4 5 6];
            r = ca(a, a);
            testCase.verifyTrue(r.Same && r.Identical);
            testCase.verifyEqual(r.MaxRelDiff, 0);
            % relative to max(1, |b|)
            r = ca([100 + 1e-8, 2], [100, 2]);
            testCase.verifyTrue(r.Same);
            testCase.verifyFalse(r.Identical);
            testCase.verifyEqual(r.MaxRelDiff, 1e-10, "RelTol", 1e-3);
            % the floor of 1 for small values
            r = ca(1e-12, 0);
            testCase.verifyEqual(r.MaxRelDiff, 1e-12, "RelTol", 1e-9);
            % -0 equals 0 in value but not in bits
            r = ca(-0, 0);
            testCase.verifyTrue(r.Same);
            testCase.verifyFalse(r.Identical);
            testCase.verifyEqual(r.MaxRelDiff, 0);
            testCase.verifyFalse(ca(a, a(:, 1:2)).Same);
            testCase.verifyFalse(ca([1 NaN], [NaN 1]).Same);
            testCase.verifyFalse(ca([1 NaN], [1 2]).Same);
            testCase.verifyTrue(ca([1 NaN], [1 NaN]).Same);
            testCase.verifyFalse(ca([Inf 1], [-Inf 1]).Same);
            testCase.verifyFalse(ca([Inf 1], [1 1]).Same);
            testCase.verifyTrue(ca([Inf -Inf], [Inf -Inf]).Identical);
            r = ca(zeros(3, 0), zeros(3, 0));
            testCase.verifyTrue(r.Same && r.Identical);
            testCase.verifyFalse(ca(zeros(3, 0), zeros(0, 3)).Same);
        end

        function compareStatusFindsEachKindOfChange(testCase)
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            json = S.Records{2}.Json;
            cs = @phasecont.internal.sessionCodec.compareStatus;
            c = cs(json, json);
            testCase.verifyTrue(c.Structure && c.Labels && c.Stops);
            testCase.verifyEmpty(c.Notes);
            c = cs(strrep(json, """label"":""H""", """label"":""LP"""), json);
            testCase.verifyFalse(c.Labels);
            testCase.verifyTrue(c.Structure && c.Stops);
            c = cs(strrep(json, """index"":21", """index"":22"), json);
            testCase.verifyFalse(c.Labels);
            c = cs(strrep(json, """stop"":""Bounds""", """stop"":""MaxSteps"""), json);
            testCase.verifyFalse(c.Stops);
            testCase.verifyTrue(c.Structure && c.Labels);
            c = cs(strrep(json, """points"":25", """points"":24"), json);
            testCase.verifyFalse(c.Structure);
            c = cs(S.Records{3}.Json, json);
            testCase.verifyFalse(c.Structure);
            testCase.verifyNotEmpty(cs(strrep(json, """points"":25", """points"":24"), json).Notes);
        end

        function compareRunSeparatesDriftFromChange(testCase)
            S = phasecont.internal.sessionCodec.readFile(testCase.fixture());
            rec = S.Records{2};
            cr = @phasecont.internal.sessionCodec.compareRun;
            c = cr(rec, rec.Points, rec.Specials, rec.Json);
            testCase.verifyTrue(c.Same && c.Identical && c.Structure && c.Labels && c.Stops);
            testCase.verifyEqual(c.MaxRelDiff, 0);
            testCase.verifyEqual(c.Stored, 25);
            testCase.verifyEqual(c.New, 25);
            % a last-digit drift is Same but not Identical
            pts = rec.Points;
            pts{1}(2, 3) = pts{1}(2, 3) * (1 + 1e-12);
            c = cr(rec, pts, rec.Specials, rec.Json);
            testCase.verifyTrue(c.Same && c.Structure);
            testCase.verifyFalse(c.Identical);
            testCase.verifyGreaterThan(c.MaxRelDiff, 0);
            testCase.verifyLessThan(c.MaxRelDiff, 1e-8);
            % a NaN where the file has a number is not Same
            pts = rec.Points;
            pts{1}(2, 2) = NaN;
            testCase.verifyFalse(cr(rec, pts, rec.Specials, rec.Json).Same);
            % one point fewer
            pts = rec.Points;
            pts{1}(:, end) = [];
            c = cr(rec, pts, rec.Specials, rec.Json);
            testCase.verifyFalse(c.Same);
            testCase.verifyEqual(c.New, 24);
            % a changed stop reason or label
            c = cr(rec, rec.Points, rec.Specials, ...
                strrep(rec.Json, """stop"":""Bounds""", """stop"":""MaxSteps"""));
            testCase.verifyFalse(c.Stops);
            testCase.verifyFalse(c.Identical);
            c = cr(rec, rec.Points, rec.Specials, ...
                strrep(rec.Json, """label"":""H""", """label"":""LP"""));
            testCase.verifyFalse(c.Labels);
            % a different number of branches
            c = cr(rec, {}, {}, rec.Json);
            testCase.verifyFalse(c.Same || c.Structure || c.Identical);
        end

        function theSessionNamesResolve(testCase)
            % +phasecont/Session.m is a class and +phasecont/+session/ is a
            % package; both names must resolve on a case-insensitive file system.
            testCase.verifyClass(meta.class.fromName("phasecont.Session"), "meta.class");
            testCase.verifyTrue(isa(meta.class.fromName("phasecont.Session"), "meta.class"));
            testCase.verifyGreaterThanOrEqual(nargin("phasecont.session.replay"), 1);
            testCase.verifyNotEmpty(which("phasecont.session.replay"));
            testCase.verifyEmpty(meta.package.fromName("phasecont.session").ClassList);
            testCase.verifyEqual(string({meta.package.fromName("phasecont.session").FunctionList.Name}), ...
                "replay");
        end
    end

    methods (Access = private)
        function f = fixture(~)
            root = fileparts(fileparts(mfilename("fullpath")));
            f = fullfile(root, "tests", "+fixtures", "ml_small.pcsession.json");
        end

        function folder = tempFolder(testCase)
            folder = string(testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
        end

        function v = fixtureStruct(testCase)
            % The decoded fixture with the run and item lists as cells, so
            % a test can edit one run.
            v = jsondecode(fileread(testCase.fixture(), Encoding = "UTF-8"));
            v.runs = phasecont.internal.sessionCodec.elements(v.runs);
            v.items = phasecont.internal.sessionCodec.elements(v.items);
        end

        function write(~, file, S)
            header = struct("Created", S.Created, "EngineName", S.EngineName, ...
                "EngineVersion", S.EngineVersion, "Platform", S.Platform, ...
                "Matlab", S.Matlab);
            phasecont.internal.sessionCodec.writeFile(file, header, S.Model, ...
                S.Records, S.Items);
        end

        function rec = changed(~, rec, change)
            % A copy of a RunRecord with some properties replaced.
            f = struct();
            for name = string(properties(rec)).'
                f.(name) = rec.(name);
            end
            for name = string(fieldnames(change)).'
                f.(name) = change.(name);
            end
            rec = phasecont.internal.RunRecord(f);
        end

        function verifySameSession(testCase, T, S)
            testCase.verifyEqual(T.Created, S.Created);
            testCase.verifyEqual(T.EngineName, S.EngineName);
            testCase.verifyEqual(T.EngineVersion, S.EngineVersion);
            testCase.verifyEqual(T.Platform, S.Platform);
            testCase.verifyEqual(T.Matlab, S.Matlab);
            testCase.verifyEqual(T.Model, S.Model);
            testCase.verifyEqual(T.Items, S.Items);
            testCase.verifyEqual(numel(T.Records), numel(S.Records));
            bits = @(a) typecast(a(:), "uint64");
            for k = 1:numel(S.Records)
                a = T.Records{k};
                b = S.Records{k};
                for name = ["Kind" "Problem" "Type" "Free" "Grid" "N" "ModelSha" ...
                        "Json" "Stride"]
                    testCase.verifyEqual(a.(name), b.(name), sprintf("run %d %s", k, name));
                end
                testCase.verifyEqual(bits(a.U0), bits(b.U0));
                testCase.verifyEqual(bits(a.P), bits(b.P));
                testCase.verifyEqual(numel(a.Points), numel(b.Points));
                for j = 1:numel(b.Points)
                    testCase.verifyEqual(size(a.Points{j}), size(b.Points{j}));
                    testCase.verifyEqual(bits(a.Points{j}), bits(b.Points{j}));
                    testCase.verifyEqual(size(a.Specials{j}), size(b.Specials{j}));
                    testCase.verifyEqual(bits(a.Specials{j}), bits(b.Specials{j}));
                end
                if b.Kind == "equilibrium"
                    testCase.verifyEqual(bits(real(a.Result.Eigenvalues)), ...
                        bits(real(b.Result.Eigenvalues)));
                    testCase.verifyEqual(bits(imag(a.Result.Eigenvalues)), ...
                        bits(imag(b.Result.Eigenvalues)));
                    testCase.verifyEqual(bits(a.Result.U), bits(b.Result.U));
                end
            end
        end
    end
end
