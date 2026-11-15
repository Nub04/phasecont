classdef MapRefusalTest < matlab.unittest.TestCase
    % Tests slice 0 of map support (SPEC section 8t, test 8): every run on a
    % map is refused with phasecont:engine:unsupportedModel, PhasePlane
    % refuses a map with phasecont:app:mapModel, and the right-hand side of
    % a map still evaluates. A map is written as x(t+1)=... or as x'=...
    % with @ meth=discrete. Every model is written here.

    properties (Constant)
        Logistic = "par r=2" + newline + "init x=0.2" + newline + ...
            "x(t+1)=r*x*(1-x)" + newline + "done" + newline
        % The Henon map in the form of crates/pc-ode/tests/corpus/henon_map.ode.
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
        LogisticDiscrete = "par r=2" + newline + "init x=0.2" + newline + ...
            "x'=r*x*(1-x)" + newline + "@ meth=discrete" + newline + ...
            "done" + newline
        HenonDiscrete = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x'=1-a*x^2+y" + newline + "y'=b*x" + newline + ...
            "@ meth=discrete" + newline + "done" + newline
        % An ODE with a meth value that does not start with d stays an ODE.
        Decay = "par a=2" + newline + "init x=3" + newline + ...
            "x'=-a*x" + newline + "@ meth=rungekutta" + newline + ...
            "done" + newline
        Refusal = "is a map"
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

    methods (Access = private)
        function m = model(testCase, text)
            m = phasecont.Model.fromSource(text, Evaluator = "jit");
            testCase.addTeardown(@() delete(m));
        end

        function verifyErrorText(testCase, fn, id, text)
            % FN raises the error ID, and its message contains TEXT.
            try
                fn();
            catch err
                testCase.verifyEqual(string(err.identifier), string(id));
                testCase.verifySubstring(string(err.message), text);
                return
            end
            testCase.verifyFail("no error was raised, expected " + id);
        end

        function simulateRefused(testCase, m)
            % simulate on the map M raises the engine refusal.
            try
                phasecont.simulate(m, 0.5, [0 1]);
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:engine:unsupportedModel");
                return
            end
            testCase.verifyFail("simulate ran on a map");
        end

        function verifyRunsRefused(testCase, m, x0, free)
            id = "phasecont:engine:unsupportedModel";
            testCase.verifyErrorText(@() phasecont.simulate(m, x0, [0 1]), ...
                id, testCase.Refusal);
            testCase.verifyErrorText(@() phasecont.equilibrium(m, x0), ...
                id, testCase.Refusal);
            testCase.verifyErrorText( ...
                @() phasecont.continuation(m, x0, Free = free), ...
                id, testCase.Refusal);
        end
    end

    methods (Test)
        function modelInfoReportsTheMap(testCase)
            for text = [testCase.Logistic testCase.Henon ...
                    testCase.LogisticDiscrete testCase.HenonDiscrete]
                info = testCase.model(text).engineInfo();
                testCase.verifyTrue(info.map, text);
            end
            info = testCase.model(testCase.Decay).engineInfo();
            testCase.verifyFalse(info.map);
        end

        function everyRunOnTheLogisticMapIsRefused(testCase)
            testCase.verifyRunsRefused(testCase.model(testCase.Logistic), 0.5, "r");
        end

        function everyRunOnTheHenonMapIsRefused(testCase)
            testCase.verifyRunsRefused(testCase.model(testCase.Henon), [0.5 0.1], "a");
        end

        function everyRunWithMethDiscreteIsRefused(testCase)
            testCase.verifyRunsRefused( ...
                testCase.model(testCase.LogisticDiscrete), 0.5, "r");
            testCase.verifyRunsRefused( ...
                testCase.model(testCase.HenonDiscrete), [0.5 0.1], "a");
        end

        function aRefusedMapGetsNoIgnoredOptionWarning(testCase)
            % simulate warns about ignored @ options only after a run, so the
            % meth=discrete file gets the refusal and no warning.
            m = testCase.model(testCase.LogisticDiscrete);
            testCase.verifyWarningFree(@() testCase.simulateRefused(m));
        end

        function theRightHandSideOfAMapStillEvaluates(testCase)
            % F(x) of the logistic map at x = 0.2, r = 2 is 2*0.2*0.8 = 0.32,
            % and F of the Henon map at (0.5, 0.1) is (1 - 1.4*0.25 + 0.1,
            % 0.3*0.5) = (0.75, 0.15).
            for text = [testCase.Logistic testCase.LogisticDiscrete]
                f = testCase.model(text).rhs(0.2);
                testCase.verifyEqual(f, 0.32, AbsTol = 4 * eps);
            end
            for text = [testCase.Henon testCase.HenonDiscrete]
                f = testCase.model(text).rhs([0.5; 0.1]);
                testCase.verifyEqual(f, [0.75; 0.15], AbsTol = 4 * eps);
            end
        end

        function phasePlaneRefusesAMap(testCase)
            for text = [testCase.Henon testCase.HenonDiscrete]
                m = testCase.model(text);
                testCase.verifyError( ...
                    @() phasecont.app.PhasePlane(m, Visible = "off"), ...
                    "phasecont:app:mapModel");
                % The caller's model is not deleted.
                testCase.verifyTrue(isvalid(m));
            end
        end

        function phasePlaneRefusesAMapFromASessionFile(testCase)
            m = testCase.model(testCase.Henon);
            file = fullfile(testCase.Dir, "henon.pcsession.json");
            save(phasecont.Session(m, {}), file);
            testCase.verifyError( ...
                @() phasecont.app.PhasePlane(file, Visible = "off"), ...
                "phasecont:app:mapModel");
            % PhasePlane frees the model it built from the file. Handle ids
            % count up in steps of 2, so every id between m and a model made
            % after the refusal must be unknown to the engine.
            after = testCase.model(testCase.Henon);
            ids = m.Id + uint64(2):uint64(2):after.Id - uint64(2);
            testCase.verifyNotEmpty(ids, "the session load made no handle");
            for id = ids
                testCase.verifyError(@() pckern("model_info", id), ...
                    "phasecont:engine:badHandle", ...
                    sprintf("handle %d is still alive", id));
            end
        end

        function replayOfAMapSessionFailsEveryRun(testCase)
            % A 1.0.0 session can hold runs on a map. Here the runs come
            % from the same right-hand side written as an ODE, and the file
            % then gets the map source and its hash, as such a session had.
            ode = testCase.model("par r=2" + newline + "init x=0.2" + newline + ...
                "x'=r*x*(1-x)" + newline + "done" + newline);
            eq = phasecont.equilibrium(ode, 0.9);
            tr = phasecont.simulate(ode, 0.2, [0 5]);
            file = fullfile(testCase.Dir, "logistic_ode.pcsession.json");
            save(phasecont.Session(ode, {eq, tr}), file);
            S = phasecont.internal.sessionCodec.readFile(file);
            header = struct("Created", "2026-11-14T00:00:00Z", ...
                "EngineName", "pckern", "EngineVersion", "1.0.0", ...
                "Platform", string(computer("arch")), "Matlab", "2026b");
            testCase.applyFixture(matlab.unittest.fixtures.SuppressedWarningsFixture( ...
                "phasecont:session:runFailed"));
            for text = [testCase.Logistic testCase.LogisticDiscrete]
                map = testCase.model(text);
                model = struct("Name", "logistic.ode", "Source", map.Source, ...
                    "Sha256", map.SourceSha256, "Evaluator", "jit");
                changed = fullfile(testCase.Dir, "logistic_map.pcsession.json");
                phasecont.internal.sessionCodec.writeFile(changed, header, ...
                    model, S.Records, S.Items);
                lastwarn("");
                r = phasecont.session.replay(changed);
                [msg, id] = lastwarn;
                testCase.verifyEqual(string(id), "phasecont:session:runFailed");
                testCase.verifySubstring(string(msg), testCase.Refusal);
                testCase.verifySize(r, [1 2]);
                testCase.verifyEqual([r.Kind], ["equilibrium" "simulate"]);
                testCase.verifyFalse(any([r.Pass]), text);
            end
        end

        function anOdeWithAnotherMethStillSimulates(testCase)
            m = testCase.model(testCase.Decay);
            tr = testCase.verifyWarning(@() phasecont.simulate(m, 3, [0 1]), ...
                "phasecont:simulate:ignoredOption");
            testCase.verifyEqual(tr.X(end), 3 * exp(-2), AbsTol = 1e-6);
        end
    end
end
