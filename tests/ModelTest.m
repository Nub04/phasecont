classdef ModelTest < matlab.unittest.TestCase
    % Tests phasecont.Model: loading, rhs, errors, save and load.

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
    end

    properties
        Dir
    end

    properties (TestParameter)
        Evaluator = {"jit", "fallback"}
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.assumeTrue( ...
                isfile(fullfile(f.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
            t = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Dir = t.Folder;
        end
    end

    methods
        function f = write(testCase, name, text)
            f = fullfile(testCase.Dir, name);
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
        end
    end

    methods (Test)
        function lorenzProperties(testCase, Evaluator)
            f = testCase.write("lorenz.ode", testCase.Lorenz);
            m = phasecont.Model(f, Evaluator = Evaluator);
            testCase.verifyEqual(m.Variables, ["x" "y" "z"]);
            testCase.verifyEqual(m.Parameters, ["s" "r" "b"]);
            testCase.verifyEqual(m.ParameterValues, [10; 28; 2.6666666666666665]);
            testCase.verifyEqual(m.Init, [1; 1; 1]);
            testCase.verifyEqual(m.File, string(f));
            testCase.verifyEqual(m.Source, testCase.Lorenz);
            testCase.verifyEqual(m.Evaluator, string(Evaluator));
            testCase.verifyNotEqual(m.Id, uint64(0));
            testCase.verifySize(m.Diagnostics, [0 1]);
            testCase.verifyEqual(string(fieldnames(m.Diagnostics))', ...
                ["code" "severity" "file" "line" "col" "end_line" ...
                 "end_col" "message" "fix_it" "notes"]);
        end

        function paramVectorChecks(testCase)
            f = testCase.write("lorenz.ode", testCase.Lorenz);
            m = phasecont.Model(f);
            bad = "phasecont:model:badParameter";
            testCase.verifyError(@() m.paramVector(struct(r = NaN)), bad);
            testCase.verifyError(@() m.paramVector(struct(r = Inf)), bad);
            testCase.verifyError(@() m.paramVector([1 2]), bad);
            testCase.verifyError(@() m.paramVector([]), bad);
            testCase.verifyError(@() m.paramVector([1 2 NaN]), bad);
            testCase.verifyError(@() m.paramVector([1 2 Inf]), bad);
            testCase.verifyError(@() m.paramVector([1 2 3i]), bad);
            testCase.verifyError(@() m.paramVector(ones(3)), bad);
            testCase.verifyError(@() m.paramVector("abc"), bad);
            p = m.paramVector([1 2 3]);
            testCase.verifyEqual(p, [1; 2; 3]);
            testCase.verifyEqual(m.paramVector(int8([1 2 3])), [1; 2; 3]);
            testCase.verifyEqual(m.paramVector(struct(r = 5)), [10; 5; 8/3]);
        end

        function relativeFileIsResolved(testCase)
            testCase.write("rel.ode", testCase.Lorenz);
            old = cd(testCase.Dir);
            restore = onCleanup(@() cd(old));
            m = phasecont.Model("rel.ode");
            clear restore
            testCase.verifyTrue(endsWith(m.File, "rel.ode"));
            testCase.verifyTrue(strlength(m.File) > strlength("rel.ode"));
        end

        function rhsMatchesFormula(testCase, Evaluator)
            f = testCase.write("lorenz.ode", testCase.Lorenz);
            m = phasecont.Model(f, Evaluator = Evaluator);
            s = 10; r = 28; b = 2.6666666666666665;
            x = [1 2 3];
            expected = [s*(x(2)-x(1)); x(1)*(r-x(3))-x(2); x(1)*x(2)-b*x(3)];
            testCase.verifyEqual(m.rhs(x), expected);
            testCase.verifyEqual(m.rhs(x(:)), expected);
            testCase.verifySize(m.rhs(x), [3 1]);
            p = [5 20 1];
            e2 = [p(1)*(x(2)-x(1)); x(1)*(p(2)-x(3))-x(2); x(1)*x(2)-p(3)*x(3)];
            testCase.verifyEqual(m.rhs(x, p), e2);
            r2 = 20;
            e3 = [s*(x(2)-x(1)); x(1)*(r2-x(3))-x(2); x(1)*x(2)-b*x(3)];
            testCase.verifyEqual(m.rhs(x, struct(r = 20)), e3);
        end

        function argumentErrors(testCase)
            f = testCase.write("lorenz.ode", testCase.Lorenz);
            m = phasecont.Model(f);
            testCase.verifyError(@() m.rhs([1 2]), "phasecont:engine:badArg");
            testCase.verifyError(@() m.rhs([1 2 3], [1 2]), ...
                "phasecont:model:badParameter");
            testCase.verifyError(@() m.rhs([1 2 3], struct(q = 1)), ...
                "phasecont:model:unknownParameter");
            testCase.verifyError(@() m.rhs([1 2 3], struct(r = "a")), ...
                "phasecont:model:badParameter");
        end

        function syntaxErrorNamesTheFile(testCase)
            f = testCase.write("bad.ode", "x'=-x" + newline + "foo(" + newline + "done");
            try
                phasecont.Model(f);
                testCase.verifyFail("no error raised");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:engine:model");
                testCase.verifyTrue(startsWith(string(err.message), string(f) + ":2:"));
                testCase.verifyFalse(contains(err.message, "model.ode"));
            end
        end

        function loweringErrorKeepsTheLibraryMessage(testCase)
            f = testCase.write("noeq.ode", "par a=1" + newline + "done" + newline);
            try
                phasecont.Model(f);
                testCase.verifyFail("no error raised");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:engine:model");
                testCase.verifyTrue(startsWith(string(err.message), string(f) + ": "));
                testCase.verifyGreaterThan(strlength(err.message), strlength(f) + 3);
            end
        end

        function percentInPathIsNotAFormat(testCase)
            d = fullfile(testCase.Dir, "p%d");
            mkdir(d);
            f = fullfile(d, "bad.ode");
            fid = fopen(f, "w");
            fprintf(fid, "x'=-x\nfoo(\ndone\n");
            fclose(fid);
            try
                phasecont.Model(f);
                testCase.verifyFail("no error raised");
            catch err
                testCase.verifySubstring(err.message, "p%d");
            end
        end

        function missingFile(testCase)
            f = fullfile(testCase.Dir, "nope.ode");
            testCase.verifyError(@() phasecont.Model(f), ...
                "phasecont:model:fileNotFound");
        end

        function parseWarningIsIssued(testCase)
            f = testCase.write("warn.ode", "@ nosuchoption=1" + newline + ...
                "init x=1" + newline + "x'=-x" + newline + "done" + newline);
            testCase.verifyWarning(@() phasecont.Model(f), ...
                "phasecont:model:parseWarning");
            w = warning("off", "phasecont:model:parseWarning");
            restore = onCleanup(@() warning(w));
            m = phasecont.Model(f);
            clear restore
            testCase.verifyGreaterThanOrEqual(numel(m.Diagnostics), 1);
            testCase.verifyEqual(string(m.Diagnostics(1).severity), "warning");
        end

        function nonAutonomousModel(testCase)
            f = testCase.write("na.ode", "par a=1" + newline + "init x=1" + ...
                newline + "x'=-a*x+sin(t)" + newline + "done" + newline);
            testCase.verifyError(@() phasecont.Model(f), ...
                "phasecont:engine:nonAutonomous");
        end

        function largeFrameJitMatchesFallback(testCase)
            n = 1500; chunk = 100;
            t = "";
            for i = 1:n
                t = t + sprintf("par c%d=%s\n", i, num2str(0.001*i, 17));
            end
            t = t + "init x=0.5" + newline;
            for i = 1:n
                t = t + sprintf("p%d=x*c%d\n", i, i);
            end
            fwd = chunks(1:n, "s");
            rev = chunks(n:-1:1, "r");
            t = t + "x'=x+(" + strjoin(fwd, "+") + ")-(" + strjoin(rev, "+") + ")" + ...
                newline + "done" + newline;
            f = testCase.write("big.ode", t);
            mj = phasecont.Model(f, Evaluator = "jit");
            mf = phasecont.Model(f, Evaluator = "fallback");
            info = engineInfo(mj);
            testCase.verifyGreaterThan(info.jit_max_frame, 4096);
            a = mj.rhs(0.5);
            b = mf.rhs(0.5);
            testCase.verifyTrue(isfinite(a));
            testCase.verifyLessThanOrEqual(abs(a - b), 1e-12 * max(abs(b), 1));

            function names = chunks(idx, tag)
                names = strings(1, 0);
                k = 0;
                for s = 1:chunk:numel(idx)
                    ch = idx(s:min(s+chunk-1, numel(idx)));
                    name = tag + k;
                    t = t + name + "=" + strjoin("p" + ch, "+") + newline;
                    names(end+1) = name; %#ok<AGROW>
                    k = k + 1;
                end
            end
        end

        function saveAndLoad(testCase)
            f = testCase.write("sl.ode", testCase.Lorenz);
            m = phasecont.Model(f);
            holder = struct(model = m); %#ok<NASGU>
            a = fullfile(testCase.Dir, "a.mat");
            b = fullfile(testCase.Dir, "b.mat");
            save(a, "m");
            save(b, "m");
            s = fullfile(testCase.Dir, "s.mat");
            save(s, "holder");
            delete(f);
            la = load(a);
            lb = load(b);
            ls = load(s);
            for loaded = {la.m, lb.m, ls.holder.model}
                n = loaded{1};
                testCase.verifyNotEqual(n.Id, uint64(0));
                testCase.verifyNotEqual(n.Id, m.Id);
                testCase.verifyEqual(n.Variables, m.Variables);
                testCase.verifyEqual(n.Parameters, m.Parameters);
                testCase.verifyEqual(n.ParameterValues, m.ParameterValues);
                testCase.verifyEqual(n.Init, m.Init);
                testCase.verifyEqual(n.File, m.File);
                testCase.verifyEqual(n.Source, m.Source);
                testCase.verifyEqual(n.Evaluator, m.Evaluator);
                testCase.verifyEqual(n.rhs([1 2 3]), m.rhs([1 2 3]));
            end
            testCase.verifyNotEqual(la.m.Id, lb.m.Id);
        end
    end
end
