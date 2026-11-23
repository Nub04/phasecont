classdef ContinuationBatchTest < matlab.unittest.TestCase
    % Tests phasecont.continuationBatch: the same bits as solo calls, input
    % order, the cap on live runs, errors, interruption and validation.

    properties (Constant)
        Fold = "par a=-1" + newline + "x'=a+x^2" + newline + ...
            "init x=1" + newline + "done" + newline
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=r*x-y-x*z" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        Brusselator = "par a=1,b=1.5" + newline + ...
            "x'=a-(b+1)*x+x^2*y" + newline + "y'=b*x-x^2*y" + newline + ...
            "init x=1,y=1" + newline + "done" + newline
        Cusp = "par a=0,b=1" + newline + "x'=a+b*x-x^3" + newline + ...
            "init x=0" + newline + "done" + newline
        OneState = "par r=1" + newline + "x'=r-x" + newline + ...
            "init x=1" + newline + "done" + newline
    end

    properties
        Dir
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

    methods (TestMethodTeardown)
        function clearHook(~)
            phasecont.internal.pollHook([]);
        end
    end

    methods
        function m = model(testCase, text)
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = "jit");
        end

        function verifySameBits(testCase, got, want, label)
            % Two continuation results hold the same raw engine rows, bit
            % for bit, and the same decoded fields.
            testCase.assertClass(got, "phasecont.Branch", label);
            testCase.assertEqual(size(got), size(want), label);
            for b = 1:numel(want)
                g = got(b);
                w = want(b);
                testCase.verifyEqual(g.Record.Problem, w.Record.Problem, label);
                testCase.verifyEqual(g.Record.Json, w.Record.Json, label);
                testCase.verifyEqual(bits(g.Record.Points), bits(w.Record.Points), label);
                testCase.verifyEqual(bits(g.Record.Specials), bits(w.Record.Specials), label);
                testCase.verifyEqual(bits({g.U, g.Alpha, g.P}), ...
                    bits({w.U, w.Alpha, w.P}), label);
                testCase.verifyEqual(g.Stop, w.Stop, label);
                testCase.verifyEqual(g.Warnings, w.Warnings, label);
                testCase.verifyEqual([g.Specials.Label], [w.Specials.Label], label);
            end
        end
    end

    methods (Test)
        function aBatchGivesTheBitsOfSoloCalls(testCase)
            % EP both ways, an LC from a Hopf point, an LP curve and a
            % second model, with fewer live runs than jobs.
            mb = testCase.model(testCase.Brusselator);
            eq = phasecont.equilibrium(mb, [1 1.2], Params = struct(a = 1, b = 1.2));
            ep = phasecont.continuation(mb, eq, Free = "b", Bounds = [0.5 4]);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            mc = testCase.model(testCase.Cusp);
            eqc = phasecont.equilibrium(mc, 1, Params = struct(a = 0, b = 1));
            epc = phasecont.continuation(mc, eqc, Free = "a", Bounds = [-2 2], ...
                Direction = -1);
            lp = epc.points("LP");
            testCase.assertNotEmpty(lp);
            jobs = {
                {mb, eq, "Free", "b", "Bounds", [0.5 4], "Direction", 1}
                {mb, eq, "Free", "b", "Bounds", [0.5 4], "Direction", -1}
                {mb, h(1), "Type", "LC", "Free", "b", "NTST", 20, "NCOL", 4, ...
                    "Bounds", [1.5 3], "MaxSteps", 30}
                {mc, lp(1), "Type", "LP", "Free", ["a" "b"], "MaxSteps", 40, ...
                    "Bounds", [-3 3]}
                {mc, eqc, "Free", "a", "Bounds", [-2 2], "Direction", -1}
                }';
            got = phasecont.continuationBatch(jobs, MaxConcurrent = 2);
            testCase.assertClass(got, "cell");
            testCase.assertSize(got, [1 numel(jobs)]);
            for k = 1:numel(jobs)
                want = phasecont.continuation(jobs{k}{:});
                testCase.verifySameBits(got{k}, want, sprintf("job %d", k));
            end
        end

        function resultsKeepTheInputOrder(testCase)
            % The long job is first and ends last; its result stays first.
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            jobs = {
                {m, eq, "Free", "a", "Direction", 1, "MaxSteps", 3000, ...
                    "H0", 1e-3, "Hmax", 1e-3, "Bounds", [-5 5]}
                {m, eq, "Free", "a", "Direction", 1, "MaxSteps", 3}
                }';
            got = phasecont.continuationBatch(jobs);
            testCase.verifyEqual(got{1}.Stop, "MaxSteps");
            testCase.verifyEqual(numel(got{1}.Alpha), 3001);
            testCase.verifyEqual(numel(got{2}.Alpha), 4);
        end

        function noMoreThanMaxConcurrentRunsAreLive(testCase)
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            % Long enough that each run is still going at its first poll.
            job = {m, eq, "Free", "a", "Direction", 1, "MaxSteps", 40000, ...
                "H0", 1e-4, "Hmax", 1e-4, "Bounds", [-1000 1000]};
            live = zeros(1, 0, "uint64");
            peak = 0;
            seen = zeros(1, 0, "uint64");
            phasecont.internal.pollHook(@hook);
            got = phasecont.continuationBatch(repmat({job}, 1, 7), MaxConcurrent = 3);
            phasecont.internal.pollHook([]);
            testCase.verifyEqual(numel(got), 7);
            testCase.verifyEqual(numel(unique(seen)), 7);
            testCase.verifyLessThanOrEqual(peak, 3);
            testCase.verifyGreaterThan(peak, 1);
            testCase.verifyEmpty(live);

            function hook(run, st)
                seen(end+1) = run.Id;
                if st.state == "running"
                    live = union(live, run.Id);
                else
                    live = setdiff(live, run.Id);
                end
                peak = max(peak, numel(live));
            end
        end

        function anOptionErrorNamesTheJob(testCase)
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            jobs = {{m, eq, "Free", "a"}, {m, eq, "Free", "nope"}};
            solo = [];
            try
                phasecont.continuation(jobs{2}{:});
            catch solo
            end
            testCase.assertNotEmpty(solo, "the solo call must fail");
            try
                phasecont.continuationBatch(jobs);
                testCase.verifyFail("expected an error in job 2");
            catch err
                testCase.verifyEqual(err.identifier, solo.identifier);
                testCase.verifySubstring(err.message, "job 2");
                testCase.verifySubstring(err.message, solo.message);
            end
        end

        function anEngineErrorCancelsAndFreesTheOtherRuns(testCase)
            % Job 2 fails in the engine (a 1-state model has no Hopf pair).
            % Every run that the hook saw has been freed when the error
            % arrives.
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            bad = testCase.model(testCase.OneState);
            long = {m, eq, "Free", "a", "Direction", 1, "MaxSteps", 1000000, ...
                "H0", 1e-4, "Hmax", 1e-4, "Bounds", [-1000 1000]};
            jobs = {long, {bad, 1, "Type", "LC", "Free", "r", "NTST", 10, ...
                "NCOL", 3}, long};
            ids = zeros(1, 0, "uint64");
            phasecont.internal.pollHook(@(run, ~) note(run.Id));
            try
                phasecont.continuationBatch(jobs);
                testCase.verifyFail("expected an engine error");
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:engine:badProblem");
                testCase.verifySubstring(err.message, "job 2");
            end
            % The engine may refuse job 2 at its start, before any poll, so
            % the hook may have seen no run at all.
            for id = unique(ids)
                testCase.verifyError(@() pckern("run_poll", id), ...
                    "phasecont:engine:badHandle");
            end

            function note(v)
                ids(end+1) = v;
            end
        end

        function anInterruptFreesEveryLiveRun(testCase)
            % An error from the poll hook stands in for Ctrl+C.
            m = testCase.model(testCase.Lorenz);
            eq = phasecont.equilibrium(m, [0 0 0]);
            job = {m, eq, "Free", "r", "H0", 0.001, "Hmax", 0.001, ...
                "MaxSteps", 1000000, "Bounds", [-1000 1000]};
            ids = zeros(1, 0, "uint64");
            calls = 0;
            phasecont.internal.pollHook(@hook);
            testCase.verifyError(@() phasecont.continuationBatch( ...
                repmat({job}, 1, 4), MaxConcurrent = 3), "test:interrupt");
            testCase.verifyEqual(numel(unique(ids)), 3);
            for id = unique(ids)
                testCase.verifyError(@() pckern("run_poll", id), ...
                    "phasecont:engine:badHandle");
            end

            function hook(run, ~)
                calls = calls + 1;
                ids(end+1) = run.Id;
                if calls == 7
                    error("test:interrupt", "interrupted");
                end
            end
        end

        function plotIsRefused(testCase)
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            testCase.verifyError(@() phasecont.continuationBatch( ...
                {{m, eq, "Free", "a", "Plot", true}}), "phasecont:model:badOption");
        end

        function badArgumentsAreRefused(testCase)
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            job = {m, eq, "Free", "a", "MaxSteps", 3};
            testCase.verifyEqual(phasecont.continuationBatch({}), cell(1, 0));
            for bad = {0, -1, 1.5, NaN, Inf, [1 2]}
                testCase.verifyError(@() phasecont.continuationBatch({job}, ...
                    MaxConcurrent = bad{1}), ?MException);
            end
            testCase.verifyError(@() phasecont.continuationBatch({1}), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuationBatch({{m}}), ...
                "phasecont:model:badOption");
            testCase.verifyError(@() phasecont.continuationBatch(job), ...
                "phasecont:model:badOption");
        end

        function aFailedBatchLeavesContinuationWorking(testCase)
            % The batch puts continuation into a prepare-only mode while it
            % collects jobs. An error must not leave that mode on.
            m = testCase.model(testCase.Fold);
            eq = phasecont.equilibrium(m, -1);
            testCase.verifyError(@() phasecont.continuationBatch( ...
                {{m, eq, "Free", "a"}, {m, eq, "Free", "nope"}}), ?MException);
            br = phasecont.continuation(m, eq, Free = "a", MaxSteps = 3);
            testCase.verifyClass(br, "phasecont.Branch");
        end
    end
end

function b = bits(c)
% The bit patterns of every value in a cell of double matrices, so NaN rows
% compare equal.
if ~iscell(c)
    c = {c};
end
b = cellfun(@(x) {size(x), typecast(double(x(:)), "uint64")}, c, ...
    UniformOutput = false);
end
