classdef OnlyOutputTest < matlab.unittest.TestCase
    % Tests the Output table of a Trajectory (SPEC section 11, "Output
    % columns"): the columns of the model's only list in order, aux values
    % from the aux columns of the run rows (SPEC section 10), the empty
    % table without a list, a session round trip, the noAuxColumns error
    % and the continuation warning.

    properties (Constant)
        % w is listed and v is not, so the run carries aux columns and the
        % store keeps w only.
        WithAux = "par a=2" + newline + "x'=-x" + newline + "y'=1" + newline + ...
            "aux w=a*x+y" + newline + "aux v=x^2" + newline + ...
            "only y,t,w" + newline + "init x=1,y=0" + newline + "done" + newline
        NoOnly = "par a=2" + newline + "x'=-x" + newline + "y'=1" + newline + ...
            "aux w=a*x+y" + newline + "init x=1,y=0" + newline + "done" + newline
        StatesOnly = "par a=2" + newline + "x'=-x" + newline + "y'=1" + newline + ...
            "aux w=a*x+y" + newline + "only x t" + newline + ...
            "init x=1,y=0" + newline + "done" + newline
        % a is a parameter: the file loads with ODE0077 and the list keeps x.
        WithParam = "par a=2" + newline + "x'=-x" + newline + ...
            "only x,a" + newline + "init x=1" + newline + "done" + newline
        % For ITER the aux value uses the iterate number as t.
        Map = "x(t+1)=0.5*x" + newline + "aux s=x+t" + newline + ...
            "only t,s,x" + newline + "init x=1" + newline + "done" + newline
        % An equilibrium at x = a, for the continuation warning.
        Steady = "par a=1" + newline + "x'=a-x" + newline + ...
            "aux w=2*x" + newline + "only x,w" + newline + ...
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

    methods
        function m = model(testCase, text)
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = "jit");
        end
    end

    methods (Test)
        function theListedColumnsComeInOrder(testCase)
            m = testCase.model(testCase.WithAux);
            testCase.verifyEqual(m.Only, ["y" "t" "w"]);
            tr = phasecont.simulate(m, [1 0], [0 2]);
            out = tr.Output;
            testCase.verifyClass(out, "table");
            testCase.verifyEqual(string(out.Properties.VariableNames), ["y" "t" "w"]);
            testCase.verifyEqual(height(out), numel(tr.T));
            testCase.verifyEqual(out.t, tr.T);
            testCase.verifyEqual(out.y, tr.X(:, 2));
            % X still holds every state and nothing else.
            testCase.verifySize(tr.X, [numel(tr.T) 2]);
        end

        function anAuxColumnEqualsItsFormula(testCase)
            m = testCase.model(testCase.WithAux);
            tr = phasecont.simulate(m, [1 0], [0 2]);
            testCase.verifySubstring(tr.Record.Problem, """aux"":true");
            w = 2*tr.X(:, 1) + tr.X(:, 2);
            testCase.verifyEqual(tr.Output.w, w, AbsTol = 1e-14);
        end

        function aMapAuxUsesTheIterateNumber(testCase)
            m = testCase.model(testCase.Map);
            tr = phasecont.simulate(m, 1, [3 8]);
            out = tr.Output;
            testCase.verifyEqual(string(out.Properties.VariableNames), ["t" "s" "x"]);
            testCase.verifyEqual(out.t, (3:8).');
            testCase.verifyEqual(out.x, tr.X);
            testCase.verifyEqual(out.s, tr.X + tr.T, AbsTol = 1e-14);
        end

        function noListGivesAnEmptyTable(testCase)
            m = testCase.model(testCase.NoOnly);
            testCase.verifyEqual(m.Only, strings(1, 0));
            tr = phasecont.simulate(m, [1 0], [0 2]);
            testCase.verifyEqual(width(tr.Output), 0);
            testCase.verifyEqual(height(tr.Output), numel(tr.T));
            % No aux key goes to the engine without a list.
            testCase.verifyFalse(contains(tr.Record.Problem, """aux"""));
        end

        function aListOfStatesSendsNoAuxKey(testCase)
            m = testCase.model(testCase.StatesOnly);
            tr = phasecont.simulate(m, [1 0], [0 2]);
            testCase.verifyFalse(contains(tr.Record.Problem, """aux"""));
            testCase.verifyEqual(tr.Record.Stride, 3);
            testCase.verifyEqual(string(tr.Output.Properties.VariableNames), ["x" "t"]);
            testCase.verifyEqual(tr.Output.x, tr.X(:, 1));
        end

        function aParameterInTheListIsDropped(testCase)
            m = testCase.verifyWarning(@() testCase.model(testCase.WithParam), ...
                "phasecont:model:parseWarning");
            testCase.verifyTrue(any(string({m.Diagnostics.code}) == "ODE0077"));
            testCase.verifyEqual(m.Only, "x");
            tr = phasecont.simulate(m, 1, [0 1]);
            testCase.verifyEqual(string(tr.Output.Properties.VariableNames), "x");
        end

        function aSessionKeepsTheOutput(testCase)
            m = testCase.model(testCase.WithAux);
            tr = phasecont.simulate(m, [1 0], [0 2]);
            file = fullfile(testCase.Dir, "only.pcsession.json");
            save(phasecont.Session(m, {tr}), file);
            r = phasecont.session.replay(file);
            testCase.verifyTrue(all([r.Pass]), "the run did not reproduce");
            back = phasecont.Session.load(file);
            testCase.verifyEqual(back.Items{1}.Output, tr.Output);
            testCase.verifyEqual(back.Items{1}.X, tr.X);
        end

        function aTrajectoryWithoutAuxColumnsRefusesOutput(testCase)
            % A session from before round E stores rows of stride n + 1, so
            % its Trajectory has an empty aux store.
            m = testCase.model(testCase.WithAux);
            tr = phasecont.Trajectory(struct("T", [0; 1], "X", [1 0; 0.5 1], ...
                "Model", m));
            testCase.verifyError(@() tr.Output, "phasecont:simulate:noAuxColumns");
            % Without an aux name in the list Output needs no store.
            s = testCase.model(testCase.StatesOnly);
            tr = phasecont.Trajectory(struct("T", [0; 1], "X", [1 0; 0.5 1], ...
                "Model", s));
            testCase.verifyEqual(tr.Output.x, [1; 0.5]);
        end

        function continuationWarnsThatItIgnoresTheList(testCase)
            m = testCase.model(testCase.Steady);
            eq = phasecont.equilibrium(m, 1);
            br = testCase.verifyWarning(@() phasecont.continuation(m, eq, ...
                Free = "a", MaxSteps = 3), "phasecont:continuation:onlyIgnored");
            testCase.verifyGreaterThan(size(br.U, 1), 1);
            % A model without a list gets no such warning.
            n = testCase.model(erase(testCase.Steady, "only x,w" + newline));
            eq = phasecont.equilibrium(n, 1);
            lastwarn("");
            phasecont.continuation(n, eq, Free = "a", MaxSteps = 3);
            [~, id] = lastwarn();
            testCase.verifyNotEqual(string(id), "phasecont:continuation:onlyIgnored");
        end
    end
end
