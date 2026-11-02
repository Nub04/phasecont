classdef SessionTest < matlab.unittest.TestCase
    % Tests phasecont.Session and phasecont.session.replay on the
    % Morris-Lecar diagram of the P7b acceptance test: a branch of
    % equilibria in I, the cycles from its Hopf point, the LPC and H curves
    % in (I, phi), a trajectory and a Newton solve. Every test loads pckern,
    % so the suite needs the MEX binary. The file format itself is tested
    % without it in SessionFormatTest.

    properties (Constant)
        MorrisLecar = "par I=100,phi=0.04" + newline + ...
            "par C=20,gL=2,gK=8,gCa=4.4" + newline + ...
            "par vL=-60,vK=-84,vCa=120" + newline + ...
            "par v1=-1.2,v2=18,v3=2,v4=30" + newline + ...
            "minf=0.5*(1+tanh((v-v1)/v2))" + newline + ...
            "winf=0.5*(1+tanh((v-v3)/v4))" + newline + ...
            "tauw=1/cosh((v-v3)/(2*v4))" + newline + ...
            "v'=(I-gL*(v-vL)-gK*w*(v-vK)-gCa*minf*(v-vCa))/C" + newline + ...
            "w'=phi*(winf-w)/tauw" + newline + ...
            "init v=-20,w=0.1" + newline + "done" + newline
        Decay = "par a=1" + newline + "x'=-a*x" + newline + ...
            "init x=1" + newline + "done" + newline
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
        function m = model(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            m = phasecont.Model.fromSource(text, Evaluator = ev);
            testCase.addTeardown(@() delete(m));
        end

        function [m, items] = diagram(testCase, ev)
            % The Morris-Lecar diagram: Newton solve, EP branch in I, LC
            % branch from the first Hopf point, LPC and H curves in
            % (I, phi) and one trajectory.
            m = testCase.model(testCase.MorrisLecar, ev);
            eq = phasecont.equilibrium(m, [-20 0.1], ...
                Params = struct(I = 80, phi = 0.04));
            rest = phasecont.equilibrium(m, [-60 0], ...
                Params = struct(I = 0, phi = 0.04));
            ep = phasecont.continuation(m, rest, Free = "I", Bounds = [0 150], ...
                MaxSteps = 400, Hmax = 2, H0 = 0.5);
            h = ep.points("H");
            testCase.assertNotEmpty(h, "the EP branch has no Hopf point");
            lc = phasecont.continuation(m, h(1), Type = "LC", Free = "I", ...
                NTST = 40, NCOL = 4, Bounds = [80 120], Hmax = 0.5, ...
                MaxNewton = 15, MaxSteps = 300);
            fold = lc.points("LPC");
            testCase.assertNotEmpty(fold, "the LC branch has no fold of cycles");
            lpc = phasecont.continuation(m, fold(1), Type = "LPC", ...
                Free = ["I" "phi"], Bounds = [0.02 0.06], Hmax = 0.5, ...
                MaxNewton = 15, MaxSteps = 100);
            hopf = phasecont.continuation(m, h(1), Type = "H", ...
                Free = ["I" "phi"], Bounds = [0.02 0.06], Hmax = 0.5, ...
                MaxNewton = 15, MaxSteps = 200);
            tr = phasecont.simulate(m, [-20 0.1], [0 300], ...
                Params = struct(I = 100, phi = 0.04));
            items = {eq, ep, lc, lpc, hopf, tr};
        end

        function verifySameItem(testCase, a, b)
            % The loaded item B has the numbers of the original A.
            testCase.verifyClass(b, class(a));
            switch class(a)
                case "phasecont.Branch"
                    testCase.verifyEqual(b.Type, a.Type);
                    testCase.verifyEqual(b.Free, a.Free);
                    testCase.verifyEqual(b.Stop, a.Stop);
                    testCase.verifyEqual(b.NTST, a.NTST);
                    testCase.verifyEqual(b.NCOL, a.NCOL);
                    for name = ["Alpha" "U" "Eigenvalues" "P" "Period" ...
                            "Mesh" "Tests" "Codim2"]
                        testCase.verifyEqual(b.(name), a.(name), name);
                    end
                    testCase.verifyEqual(b.Stable, a.Stable);
                    testCase.verifyEqual(b.Codim2Labels, a.Codim2Labels);
                    testCase.verifyEqual([b.Specials.Label], [a.Specials.Label]);
                    testCase.verifyEqual([b.Specials.Index], [a.Specials.Index]);
                    for k = 1:numel(a.Specials)
                        testCase.verifyEqual(b.Specials(k).U, a.Specials(k).U);
                    end
                case "phasecont.Trajectory"
                    testCase.verifyEqual(b.T, a.T);
                    testCase.verifyEqual(b.X, a.X);
                    testCase.verifyEqual(b.X0, a.X0);
                    testCase.verifyEqual(b.TSpan, a.TSpan);
                    testCase.verifyEqual(b.P, a.P);
                    testCase.verifyEqual(b.Stop, a.Stop);
                case "phasecont.Point"
                    testCase.verifyEqual(b.U, a.U);
                    testCase.verifyEqual(b.P, a.P);
                    testCase.verifyEqual(b.Eigenvalues, a.Eigenvalues);
                    testCase.verifyEqual(b.Label, a.Label);
            end
        end
    end

    methods (Test)
        function theMorrisLecarDiagramSurvivesARoundTrip(testCase, Evaluator)
            [m, items] = testCase.diagram(Evaluator);
            s = phasecont.Session(m, items);
            file = fullfile(testCase.Dir, "ml_" + Evaluator + ".pcsession.json");
            save(s, file);
            testCase.verifyTrue(isfile(file));

            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.verifyEqual(back.Model.SourceSha256, m.SourceSha256);
            testCase.verifyEqual(back.Model.Evaluator, m.Evaluator);
            testCase.verifyNumElements(back.Items, numel(items));
            for k = 1:numel(items)
                testCase.verifySameItem(items{k}, back.Items{k});
            end

            % The loaded items have run records again, so a second save
            % writes the same runs.
            file2 = fullfile(testCase.Dir, "ml_again_" + Evaluator + ".pcsession.json");
            save(back, file2);
            a = phasecont.internal.sessionCodec.readFile(file);
            b = phasecont.internal.sessionCodec.readFile(file2);
            testCase.verifyEqual(numel(b.Records), numel(a.Records));
            testCase.verifyEqual(b.Items, a.Items);
            for k = 1:numel(a.Records)
                testCase.verifyEqual(b.Records{k}.Problem, a.Records{k}.Problem);
                testCase.verifyEqual(b.Records{k}.Json, a.Records{k}.Json);
            end
        end

        function replayReproducesEveryRun(testCase, Evaluator)
            [m, items] = testCase.diagram(Evaluator);
            file = fullfile(testCase.Dir, "replay_" + Evaluator + ".pcsession.json");
            save(phasecont.Session(m, items), file);
            r = phasecont.session.replay(file);
            testCase.verifyNumElements(r, 6);
            testCase.verifyEqual([r.Kind], ...
                ["equilibrium" "continuation" "continuation" ...
                "continuation" "continuation" "simulate"]);
            testCase.verifyEqual([r.Type], ["" "EP" "LC" "LPC" "H" "IVP"]);
            testCase.verifyTrue(all([r.Pass]), "a run did not reproduce");
            testCase.verifyTrue(all([r.LabelsMatch]));
            testCase.verifyTrue(all([r.StopMatch]));
            testCase.verifyLessThanOrEqual(max([r.MaxRelDiff]), 1e-8);
            for k = 1:numel(r)
                testCase.verifyEqual(r(k).Rows.New, r(k).Rows.Stored);
            end
            % The same engine on the same machine repeats bit for bit.
            testCase.verifyTrue(all([r.Identical]), ...
                "a replay on the same machine should be bit-exact");
        end

        function replayReportsAChangedRow(testCase)
            m = testCase.model(testCase.Decay);
            tr = phasecont.simulate(m, 1, [0 2]);
            file = fullfile(testCase.Dir, "decay.pcsession.json");
            save(phasecont.Session(m, {tr}), file);
            S = phasecont.internal.sessionCodec.readFile(file);
            rec = S.Records{1};
            f = struct("Kind", rec.Kind, "Problem", rec.Problem, ...
                "U0", rec.U0, "P", rec.P, "ModelSha", rec.ModelSha, ...
                "Type", rec.Type, "Free", rec.Free, "Grid", rec.Grid, ...
                "N", rec.N, "Json", rec.Json, "Points", {rec.Points}, ...
                "Specials", {rec.Specials}, "Stride", rec.Stride);
            f.Points{1}(end, end) = f.Points{1}(end, end) + 1e-3;
            S.Records{1} = phasecont.internal.RunRecord(f);

            % Write the changed rows through the codec and replay them.
            changed = fullfile(testCase.Dir, "decay_changed.pcsession.json");
            header = struct("Created", "2026-11-01T00:00:00Z", ...
                "EngineName", "pckern", "EngineVersion", "x", ...
                "Platform", string(computer("arch")), "Matlab", "2026b");
            model = struct("Name", "decay.ode", "Source", m.Source, ...
                "Sha256", m.SourceSha256, "Evaluator", m.Evaluator);
            phasecont.internal.sessionCodec.writeFile(changed, header, model, ...
                S.Records, S.Items);
            r = phasecont.session.replay(changed);
            testCase.verifyFalse(r.Pass);
            testCase.verifyFalse(r.Identical);
            testCase.verifyGreaterThan(r.MaxRelDiff, 1e-6);
            loose = phasecont.session.replay(changed, Tol = 1);
            testCase.verifyTrue(loose.Pass);
            testCase.verifyFalse(loose.Identical);
        end

        function aChangedSourceIsAHashMismatch(testCase)
            m = testCase.model(testCase.Decay);
            eq = phasecont.equilibrium(m, 0.5);
            file = fullfile(testCase.Dir, "hash.pcsession.json");
            save(phasecont.Session(m, {eq}), file);
            text = fileread(file);
            bad = replace(text, "x'=-a*x", "x'=-2*a*x");
            testCase.assertNotEqual(bad, text, "the source text was not found");
            changed = fullfile(testCase.Dir, "hash_changed.pcsession.json");
            fid = fopen(changed, "w", "n", "UTF-8");
            fwrite(fid, char(bad), "char");
            fclose(fid);
            testCase.verifyError(@() phasecont.Session.load(changed), ...
                "phasecont:session:hashMismatch");
            testCase.verifyError(@() phasecont.session.replay(changed), ...
                "phasecont:session:hashMismatch");
        end

        function aPointFromABranchHasNoRecord(testCase)
            m = testCase.model(testCase.MorrisLecar);
            rest = phasecont.equilibrium(m, [-60 0], ...
                Params = struct(I = 0, phi = 0.04));
            ep = phasecont.continuation(m, rest, Free = "I", Bounds = [0 150], ...
                MaxSteps = 400, Hmax = 2, H0 = 0.5);
            h = ep.points("H");
            testCase.assertNotEmpty(h);
            testCase.verifyError(@() phasecont.Session(m, {h(1)}), ...
                "phasecont:session:noRecord");
            testCase.verifyError(@() phasecont.Session(m, {42}), ...
                "phasecont:session:badRecord");
            % The branch itself saves.
            s = phasecont.Session(m, {ep});
            testCase.verifyNumElements(s.Items, 1);
        end

        function anItemOfAnotherModelIsRejected(testCase)
            m1 = testCase.model(testCase.MorrisLecar);
            m2 = testCase.model(testCase.Decay);
            eq = phasecont.equilibrium(m2, 0.5);
            testCase.verifyError(@() phasecont.Session(m1, {eq}), ...
                "phasecont:session:modelMismatch");
        end

        function branchesOfOneRunAreStoredOnce(testCase)
            m = testCase.model(testCase.MorrisLecar);
            rest = phasecont.equilibrium(m, [-60 0], ...
                Params = struct(I = 0, phi = 0.04));
            ep = phasecont.continuation(m, rest, Free = "I", Bounds = [0 150], ...
                MaxSteps = 400, Hmax = 2, H0 = 0.5);
            eq = phasecont.equilibrium(m, [-20 0.1], ...
                Params = struct(I = 80, phi = 0.04));
            % The same item twice is one run with two entries.
            s = phasecont.Session(m, {ep, eq, ep});
            file = fullfile(testCase.Dir, "once.pcsession.json");
            save(s, file);
            S = phasecont.internal.sessionCodec.readFile(file);
            testCase.verifyNumElements(S.Records, 2);
            testCase.verifyEqual(S.Items, [1 1; 2 0; 1 1]);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.verifyNumElements(back.Items, 3);
        end

        function aLoadedSessionBuildsWithAnotherEvaluator(testCase)
            m = testCase.model(testCase.Decay, "jit");
            tr = phasecont.simulate(m, 1, [0 2]);
            file = fullfile(testCase.Dir, "ev.pcsession.json");
            save(phasecont.Session(m, {tr}), file);
            back = phasecont.Session.load(file, Evaluator = "fallback");
            testCase.addTeardown(@() delete(back.Model));
            testCase.verifyEqual(back.Model.Evaluator, "fallback");
            testCase.verifyEqual(back.Items{1}.X, tr.X);
            r = phasecont.session.replay(file, Evaluator = "fallback");
            testCase.verifyTrue(r.Pass);
        end
    end
end
