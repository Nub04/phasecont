classdef HandleTest < matlab.unittest.TestCase
    % Handle lifetime tests. Model-level cases; run cases are added later.

    properties (Constant)
        Text = "par a=2" + newline + "init x=1" + newline + ...
            "x'=-a*x" + newline + "done" + newline
    end

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        LongProblem = ['{"type":"EP","free":"r","options":{"h0":0.001,', ...
            '"hmax":0.001,"max_steps":1000000,"bounds":[-1000,1000]}}']
    end

    properties
        File
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
            testCase.File = fullfile(t.Folder, "h.ode");
            fid = fopen(testCase.File, "w");
            fwrite(fid, char(testCase.Text), "char");
            fclose(fid);
        end
    end

    methods (Test)
        function freedHandleIsBadHandle(testCase)
            m = phasecont.Model(testCase.File);
            pckern("free", m.Id);
            testCase.verifyError(@() m.rhs(1), "phasecont:engine:badHandle");
            testCase.verifyWarningFree(@() delete(m));
        end

        function deleteTwiceIsFine(testCase)
            m = phasecont.Model(testCase.File);
            delete(m);
            testCase.verifyWarningFree(@() delete(m));
        end

        function badHandleKinds(testCase)
            testCase.verifyError( ...
                @() pckern("model_rhs", 12345678901234, 1, 2), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_rhs", uint64(0), 1, 2), ...
                "phasecont:engine:badHandle");
            testCase.verifyError( ...
                @() pckern("model_rhs", uint64(12345678901234), 1, 2), ...
                "phasecont:engine:badHandle");
        end

        function modelsFromOneFileAreIndependent(testCase)
            a = phasecont.Model(testCase.File);
            b = phasecont.Model(testCase.File);
            testCase.verifyNotEqual(a.Id, b.Id);
            delete(a);
            testCase.verifyEqual(b.rhs(3), -6);
        end

        function clearFreesTheHandle(testCase)
            m = phasecont.Model(testCase.File);
            id = m.Id;
            clear m
            testCase.verifyError(@() pckern("model_info", id), ...
                "phasecont:engine:badHandle");
        end

        function freedAndUnknownRunIdsAreBadHandle(testCase)
            m = phasecont.Model(testCase.File);
            r = pckern("run_start", m.Id, ...
                '{"type":"EP","free":"a","options":{"max_steps":3}}', 0, 2);
            pckern("free", r);
            testCase.verifyError(@() pckern("run_poll", r), ...
                "phasecont:engine:badHandle");
            testCase.verifyError(@() pckern("run_poll", uint64(987654321)), ...
                "phasecont:engine:badHandle");
            testCase.verifyError(@() pckern("run_cancel", r), ...
                "phasecont:engine:badHandle");
        end

        function deletingARunFreesIt(testCase)
            m = phasecont.Model(testCase.File);
            run = phasecont.internal.Run(m, ...
                '{"type":"EP","free":"a","options":{"max_steps":3}}', 0, 2);
            id = run.Id;
            testCase.verifyNotEqual(id, uint64(0));
            delete(run);
            testCase.verifyError(@() pckern("run_poll", id), ...
                "phasecont:engine:badHandle");
            testCase.verifyWarningFree(@() delete(run));
        end

        function runOutlivesItsModel(testCase)
            f = fullfile(testCase.Dir, "lorenz.ode");
            fid = fopen(f, "w");
            fwrite(fid, char(testCase.Lorenz), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = "fallback");
            eqm = phasecont.Model(testCase.File);
            eq = phasecont.equilibrium(eqm, 0.5);
            br = phasecont.continuation(eqm, eq, Free = "a", MaxSteps = 40, ...
                Bounds = [-5 5]);
            alpha = br.Alpha;
            u = br.U;
            nsp = numel(br.Specials);
            run = phasecont.internal.Run(m, testCase.LongProblem, ...
                [0 0 0], m.ParameterValues);
            delete(m);
            delete(eqm);
            st = run.poll();
            testCase.verifyEqual(st.state, "running");
            run.cancel();
            t0 = tic;
            while st.state == "running" && toc(t0) < 20
                pause(0.01);
                st = run.poll();
            end
            testCase.verifyEqual(st.state, "cancelled");
            testCase.verifyEqual(br.Alpha, alpha);
            testCase.verifyEqual(br.U, u);
            testCase.verifyEqual(numel(br.Specials), nsp);
            delete(run);
        end
    end
end
