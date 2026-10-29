classdef PlotTest < matlab.unittest.TestCase
    % Tests Branch.plot: stability line styles, labeled special points,
    % hold behavior, Var selection, arrays and the end of continuation(Plot=true).

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=r*x-y-x*z" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        Fold = "par a=-1" + newline + "x'=a+x^2" + newline + ...
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
            f = fullfile(testCase.Dir, "m" + string(randi(1e9)) + ".ode");
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
            m = phasecont.Model(f, Evaluator = ev);
        end

        function br = foldBranch(testCase, ev)
            m = testCase.model(testCase.Fold, ev);
            eq = phasecont.equilibrium(m, -1);
            br = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                MaxSteps = 100, Bounds = [-5 5]);
        end

        function brs = lorenzBranches(testCase, ev)
            m = testCase.model(testCase.Lorenz, ev);
            eq = phasecont.equilibrium(m, [0 0 0], ...
                Params = struct(s = 10, r = 0.5, b = 8/3));
            brs = phasecont.continuation(m, eq, Free = "r", Direction = 1, ...
                Hmax = 0.5, MaxSteps = 1000, Bounds = [0 30], Switch = "BP1");
        end

        function ax = newAxes(testCase)
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            ax = axes(fig);
        end
    end

    methods (Test)
        function test1LineStyles(testCase, Evaluator)
            br = testCase.foldBranch(Evaluator);
            ax = testCase.newAxes();
            plot(br, Parent = ax);
            st = findobj(ax, "Tag", "pc-stable");
            un = findobj(ax, "Tag", "pc-unstable");
            testCase.assertNotEmpty(st);
            testCase.assertNotEmpty(un);
            covered = false(numel(br.Alpha), 1);
            for kind = ["stable" "unstable"]
                if kind == "stable"
                    lns = st;
                    style = "-";
                    want = true;
                else
                    lns = un;
                    style = "--";
                    want = false;
                end
                for ln = lns'
                    testCase.verifyEqual(string(ln.LineStyle), style);
                    for j = 1:numel(ln.XData)
                        hit = find(br.Alpha == ln.XData(j) ...
                            & br.U(:, 1) == ln.YData(j));
                        testCase.assertNotEmpty(hit);
                        % The last point of a run is shared with the next run.
                        if j < numel(ln.XData)
                            testCase.verifyTrue(any(br.Stable(hit) == want));
                        end
                        covered(hit) = true;
                    end
                end
            end
            testCase.verifyTrue(all(covered));
            colors = vertcat(st.Color, un.Color);
            testCase.verifyEqual(size(unique(colors, "rows"), 1), 1);
        end

        function test2Specials(testCase, Evaluator)
            br = testCase.foldBranch(Evaluator);
            ax = testCase.newAxes();
            plot(br, Parent = ax);
            sp = br.Specials;
            testCase.assertNotEmpty(sp);
            mk = flipud(findobj(ax, "Tag", "pc-special"));
            tx = flipud(findobj(ax, "Tag", "pc-label"));
            testCase.assertSize(mk, [numel(sp) 1]);
            testCase.assertSize(tx, [numel(sp) 1]);
            for k = 1:numel(sp)
                testCase.verifyEqual(string(tx(k).String), sp(k).Label);
                testCase.verifyEqual(mk(k).XData, sp(k).P(br.FreeIndex));
                testCase.verifyEqual(mk(k).YData, sp(k).U(1));
            end
        end

        function test3HoldOnIsRespected(testCase)
            br = testCase.foldBranch("jit");
            ax = testCase.newAxes();
            old = line(ax, [0 1], [0 1]);
            hold(ax, "on");
            plot(br, Parent = ax);
            testCase.verifyTrue(isvalid(old));
            testCase.verifyTrue(any(ax.Children == old));

            fig = ancestor(ax, "figure");
            clf(fig);
            plot(br);
            ax2 = gca;
            sentinel = line(ax2, [0 1], [0 1]);
            plot(br);
            testCase.verifyFalse(isvalid(sentinel));
            hold(ax2, "on");
            keep = line(ax2, [0 1], [0 1]);
            plot(br);
            testCase.verifyTrue(isvalid(keep));
        end

        function test4VarSelection(testCase)
            brs = testCase.lorenzBranches("jit");
            br = brs(1);
            for v = {"y", 2}
                ax = testCase.newAxes();
                h = plot(br, Var = v{1}, Parent = ax);
                testCase.verifyNotEmpty(h);
                ln = [findobj(ax, "Tag", "pc-stable"); ...
                    findobj(ax, "Tag", "pc-unstable")];
                y = [];
                for k = 1:numel(ln)
                    y = [y, ln(k).YData]; %#ok<AGROW>
                end
                testCase.verifyEqual(sort(unique(y)), sort(unique(br.U(:, 2)'))); %#ok<TRSRT>
                testCase.verifyEqual(string(ax.YLabel.String), "y");
                testCase.verifyEqual(string(ax.XLabel.String), "r");
            end
            ax = testCase.newAxes();
            testCase.verifyError(@() plot(br, Var = "q", Parent = ax), ...
                "phasecont:model:unknownVariable");
            testCase.verifyError(@() plot(br, Var = 0, Parent = ax), ...
                "phasecont:model:unknownVariable");
            testCase.verifyError(@() plot(br, Var = 4, Parent = ax), ...
                "phasecont:model:unknownVariable");
            bare = phasecont.Branch();
            testCase.verifyError(@() plot(bare, Var = "x", Parent = ax), ...
                "phasecont:model:unknownVariable");
        end

        function test5Array(testCase, Evaluator)
            brs = testCase.lorenzBranches(Evaluator);
            testCase.assertSize(brs, [1 2]);
            ax = testCase.newAxes();
            plot(brs, Parent = ax);
            cols = zeros(0, 3);
            for ln = findobj(ax, "Type", "line")'
                if ismember(string(ln.Tag), ["pc-stable" "pc-unstable"])
                    cols(end+1, :) = ln.Color; %#ok<AGROW>
                end
            end
            testCase.verifyEqual(size(unique(cols, "rows"), 1), 2);
        end

        function test6ContinuationPlotEndsStyled(testCase, Evaluator)
            m = testCase.model(testCase.Fold, Evaluator);
            eq = phasecont.equilibrium(m, -1);
            fig = figure(Visible = "off");
            testCase.addTeardown(@() close(fig, "force"));
            br = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                MaxSteps = 100, Bounds = [-5 5], Plot = true);
            ax = gca;
            lines = findobj(ax, "Type", "line");
            tags = string({lines.Tag});
            testCase.verifyTrue(all(ismember(tags, ...
                ["pc-stable" "pc-unstable" "pc-special"])));
            testCase.verifyTrue(any(tags == "pc-stable"));
            testCase.verifyTrue(any(tags == "pc-unstable"));
            testCase.verifyEqual(nnz(tags == "pc-special"), numel(br.Specials));
        end

        function test7EmptyBranch(testCase)
            ax = testCase.newAxes();
            h = plot(phasecont.Branch(), Parent = ax);
            testCase.verifyEmpty(h);
            testCase.verifyEmpty(ax.Children);
        end
    end
end
