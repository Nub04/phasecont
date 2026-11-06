classdef LatexTest < matlab.unittest.TestCase
    % Tests phasecont.latex and phasecont.internal.latex*.
    %   - JSON building, table data, figure data and file writing need no
    %     binary and run on hand-made Branch objects.
    %   - Rendering needs the pckern gateway and is filtered when it has not
    %     been built.
    %   - Compiling needs latexmk and is filtered when it is not installed.

    properties
        HasGateway (1,1) logical = false
        HasLatexmk (1,1) logical = false
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
            bin = fixtures.binFolder(root);
            if isfile(fullfile(bin, "pckern." + mexext))
                testCase.applyFixture( ...
                    matlab.unittest.fixtures.PathFixture(char(bin)));
                testCase.HasGateway = true;
            end
            [status, ~] = system("latexmk -v");
            testCase.HasLatexmk = status == 0;
        end
    end

    methods (Test)
        % ---- latexJson ----------------------------------------------------

        function jsonNeedsDigits(testCase)
            spec = testCase.tableSpec();
            testCase.verifyError( ...
                @() phasecont.internal.latexJson("table", spec, struct("Standalone", false)), ...
                "phasecont:latex:missingDigits");
            testCase.verifyError( ...
                @() phasecont.internal.latexJson("table", spec, ...
                    struct("Digits", NaN, "Standalone", false)), ...
                "phasecont:latex:missingDigits");
        end

        function jsonKeepsOneElementVectorsAsArrays(testCase)
            fig = testCase.figureSpec([1; 2], [3; 4]);
            fig.Series(1).X = 1;
            fig.Series(1).Y = 2;
            json = testCase.figureJson(fig);
            testCase.verifySubstring(json, '"x":[1]');
            testCase.verifySubstring(json, '"y":[2]');
            tab = phasecont.internal.latexJson("table", ...
                struct("Columns", "a", "Rows", struct("Label", "EP1", ...
                    "Type", "Equilibrium", "Values", 7)), ...
                struct("Digits", 5, "Standalone", false));
            testCase.verifySubstring(tab, '"columns":["a"]');
            testCase.verifySubstring(tab, '"values":[7]');
            testCase.verifySubstring(tab, '"rows":[{');
        end

        function jsonKeepsEveryDigit(testCase)
            fig = testCase.figureSpec([0.1 + 0.2; 1], [1; 2]);
            json = testCase.figureJson(fig);
            testCase.verifySubstring(json, "0.30000000000000004");
        end

        function jsonWritesNonFiniteAsNull(testCase)
            fig = testCase.figureSpec([1; 2], [1; NaN]);
            fig.Series(1).X(1) = Inf;
            json = testCase.figureJson(fig);
            testCase.verifySubstring(json, '"x":[null,2]');
            testCase.verifySubstring(json, '"y":[1,null]');
        end

        function jsonOptionalKeysAppearOnlyWhenSet(testCase)
            fig = testCase.figureSpec([1; 2], [3; 4]);
            plain = testCase.figureJson(fig);
            testCase.verifyFalse(contains(plain, ["standalone" "basename" "data" "title" "marks"]));
            fig.Title = "T";
            fig.Marks = struct("Label", "LP", "X", 1, "Y", 3);
            opts = struct("Digits", 6, "Standalone", true, ...
                "Data", "external", "Basename", "fig1");
            json = phasecont.internal.latexJson("pgfplots", fig, opts);
            d = jsondecode(json);
            testCase.verifyEqual(d.title, 'T');
            testCase.verifyEqual(d.data, 'external');
            testCase.verifyEqual(d.basename, 'fig1');
            testCase.verifyTrue(d.standalone);
            testCase.verifyEqual(d.marks.label, 'LP');
            testCase.verifyEqual(d.digits, 6);
        end

        function jsonRejectsWrongShapes(testCase)
            opts = struct("Digits", 6, "Standalone", false, "Data", "inline", "Basename", "f");
            fig = testCase.figureSpec([1; 2], [3; 4]);
            bad = rmfield(fig, "XLabel");
            testCase.verifyError(@() phasecont.internal.latexJson("pgfplots", bad, opts), ...
                "phasecont:latex:badInput");
            bad = fig;
            bad.Series = rmfield(bad.Series, "Class");
            testCase.verifyError(@() phasecont.internal.latexJson("pgfplots", bad, opts), ...
                "phasecont:latex:badInput");
            bad = fig;
            bad.Series(1).X = "text";
            testCase.verifyError(@() phasecont.internal.latexJson("pgfplots", bad, opts), ...
                "phasecont:latex:badInput");
            bad = fig;
            bad.Marks = struct("Label", "LP", "X", [1 2], "Y", 3);
            testCase.verifyError(@() phasecont.internal.latexJson("pgfplots", bad, opts), ...
                "phasecont:latex:badInput");
            bad = fig;
            bad.XLabel = ["a" "b"];
            testCase.verifyError(@() phasecont.internal.latexJson("pgfplots", bad, opts), ...
                "phasecont:latex:badInput");
        end

        function jsonPassesEmptyListsOn(testCase)
            % An empty series list is for the renderer to report.
            fig = testCase.figureSpec([1; 2], [3; 4]);
            fig.Series = fig.Series([]);
            json = testCase.figureJson(fig);
            testCase.verifySubstring(json, '"series":[]');
        end

        % ---- latexTable ---------------------------------------------------

        function tableOfEquilibriumBranch(testCase)
            br = testCase.epBranch();
            spec = phasecont.internal.latexTable(br);
            testCase.verifyEqual(spec.Columns, ["a" "u1" "u2"]);
            testCase.verifySize(spec.Rows, [3 1]);
            testCase.verifyEqual(spec.Rows(1).Label, "LP1");
            testCase.verifyEqual(spec.Rows(1).Type, "Limit point");
            testCase.verifyEqual(spec.Rows(1).Values, [5 1 2]);
            testCase.verifyEqual(spec.Rows(2).Label, "H1");
            testCase.verifyEqual(spec.Rows(2).Type, "Hopf");
            testCase.verifyEqual(spec.Rows(2).Values, [6 3 4]);
            % A repeated label counts up; an unknown label is its own type.
            testCase.verifyEqual(spec.Rows(3).Label, "XYZ1");
            testCase.verifyEqual(spec.Rows(3).Type, "XYZ");
        end

        function tableRepeatedLabelsCountUp(testCase)
            p = @(l, a) phasecont.Point(U = 1, P = [0; a], Label = l);
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 2, ...
                "U", [1; 1], "Specials", [p("H", 1); p("LP", 2); p("H", 3)]));
            spec = phasecont.internal.latexTable(br);
            testCase.verifyEqual([spec.Rows.Label], ["H1" "LP1" "H2"]);
        end

        function tableOfCycleBranchHasNoStateColumns(testCase)
            p = phasecont.Point(U = [1; 2], P = [0; 4], Label = "LPC");
            br = phasecont.Branch(struct("Type", "LC", "Free", "a", "FreeIndex", 2, ...
                "U", [1 2], "NTST", 1, "NCOL", 1, "Specials", p));
            spec = phasecont.internal.latexTable(br);
            testCase.verifyEqual(spec.Columns, "a");
            testCase.verifyEqual(spec.Rows(1).Values, 4);
            testCase.verifyEqual(spec.Rows(1).Type, "Limit point of cycles");
        end

        function tableOfCurveHasBothParameters(testCase)
            p = phasecont.Point(U = 7, P = [0; 4; 5], Label = "BT");
            br = phasecont.Branch(struct("Type", "H", "Free", ["a" "b"], ...
                "FreeIndex", [2 3], "U", 7, "Specials", p));
            spec = phasecont.internal.latexTable(br);
            testCase.verifyEqual(spec.Columns, ["a" "b" "u1"]);
            testCase.verifyEqual(spec.Rows(1).Values, [4 5 7]);
            testCase.verifyEqual(spec.Rows(1).Type, "Bogdanov-Takens");
        end

        function tableWithoutSpecialsHasNoRows(testCase)
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 2, ...
                "U", [1; 2]));
            spec = phasecont.internal.latexTable(br);
            testCase.verifyEmpty(spec.Rows);
        end

        % ---- figureFromBranch ---------------------------------------------

        function figureSplitsRunsAtStabilityChanges(testCase)
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 2, ...
                "Alpha", (0:3).', "U", (1:4).', ...
                "Stable", [true; true; false; false], ...
                "Specials", phasecont.Point(U = 3, P = [0; 2], Label = "LP")));
            fig = phasecont.latex.figureFromBranch(br);
            testCase.verifyEqual(fig.XLabel, "a");
            testCase.verifyEqual(fig.YLabel, "u1");
            testCase.verifySize(fig.Series, [2 1]);
            % The runs share their boundary point, so the line has no gap.
            testCase.verifyEqual(fig.Series(1).X, [0; 1; 2]);
            testCase.verifyEqual(fig.Series(1).Class, "stable");
            testCase.verifyEqual(fig.Series(2).X, [2; 3]);
            testCase.verifyEqual(fig.Series(2).Y, [3; 4]);
            testCase.verifyEqual(fig.Series(2).Class, "unstable");
            testCase.verifyEqual(fig.Series(1).Color, "#000000");
            testCase.verifySize(fig.Marks, [1 1]);
            testCase.verifyEqual(fig.Marks(1).Label, "LP");
            testCase.verifyEqual(fig.Marks(1).X, 2);
            testCase.verifyEqual(fig.Marks(1).Y, 3);
        end

        function figureTakesColorTitleAndVariable(testCase)
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 1, ...
                "Alpha", [0; 1], "U", [1 10; 2 20], "Stable", [true; true]));
            fig = phasecont.latex.figureFromBranch(br, Var = 2, ...
                Color = "#FF8800", Title = "Branch");
            testCase.verifyEqual(fig.YLabel, "u2");
            testCase.verifyEqual(fig.Series(1).Y, [10; 20]);
            testCase.verifyEqual(fig.Series(1).Color, "#FF8800");
            testCase.verifyEqual(fig.Title, "Branch");
            testCase.verifyEmpty(fig.Marks);
        end

        function figureRejectsUnknownVariable(testCase)
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 1, ...
                "Alpha", [0; 1], "U", [1; 2], "Stable", [true; true]));
            testCase.verifyError(@() phasecont.latex.figureFromBranch(br, Var = 2), ...
                "phasecont:model:unknownVariable");
            testCase.verifyError(@() phasecont.latex.figureFromBranch(br, Var = "q"), ...
                "phasecont:model:unknownVariable");
        end

        function figureOfCycleBranchHasMaxAndMin(testCase)
            % One state, two base points per orbit.
            br = phasecont.Branch(struct("Type", "LC", "Free", "a", "FreeIndex", 1, ...
                "Alpha", [0; 1], "U", [1 3; 2 5], "NTST", 1, "NCOL", 1, ...
                "Stable", [true; true]));
            fig = phasecont.latex.figureFromBranch(br);
            testCase.verifySize(fig.Series, [2 1]);
            testCase.verifyEqual(fig.Series(1).Y, [3; 5]);
            testCase.verifyEqual(fig.Series(2).Y, [1; 2]);
            testCase.verifyEqual(fig.YLabel, "max, min u1");
        end

        function figureOfCurveUsesBothParameters(testCase)
            sp = phasecont.Point(U = 1, P = [0; 1; 2], Label = "BT");
            br = phasecont.Branch(struct("Type", "H", "Free", ["a" "b"], ...
                "FreeIndex", [2 3], "Alpha", [0 1; 1 2; 2 3], "U", [1; 1; 1], ...
                "Specials", sp));
            fig = phasecont.latex.figureFromBranch(br);
            testCase.verifyEqual(fig.XLabel, "a");
            testCase.verifyEqual(fig.YLabel, "b");
            testCase.verifySize(fig.Series, [1 1]);
            testCase.verifyEqual(fig.Series(1).X, [0; 1; 2]);
            testCase.verifyEqual(fig.Series(1).Y, [1; 2; 3]);
            testCase.verifyEqual(fig.Series(1).Class, "stable");
            testCase.verifyEqual([fig.Marks.X fig.Marks.Y], [1 2]);
            testCase.verifyError(@() phasecont.latex.figureFromBranch(br, Var = 1), ...
                "phasecont:model:badOption");
        end

        % ---- latexWrite ---------------------------------------------------

        function writeCreatesFolderAndFiles(testCase)
            tmp = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            folder = fullfile(tmp.Folder, "out", "deeper");
            files = struct("Name", {"fig-s1.dat"; "fig-s2.dat"}, ...
                "Content", {"x y" + newline + "1 2" + newline; "x y" + newline});
            paths = phasecont.internal.latexWrite(folder, "fig", "caf" + char(233) + newline, files);
            testCase.verifySize(paths, [3 1]);
            testCase.verifyEqual(paths(1), fullfile(folder, "fig.tex"));
            for p = paths.'
                testCase.verifyTrue(isfile(p));
            end
            bytes = fileread(paths(1), Encoding = "UTF-8");
            testCase.verifyEqual(string(bytes), "caf" + char(233) + newline);
            raw = fopen(paths(1), "r");
            testCase.addTeardown(@() fclose(raw));
            testCase.verifyEqual(fread(raw, Inf, "*uint8").', ...
                uint8([99 97 102 195 169 10]));
            testCase.verifyEqual(string(fileread(paths(2))), "x y" + newline + "1 2" + newline);
        end

        function writeRejectsBadBasename(testCase)
            tmp = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            none = repmat(struct("Name", "", "Content", ""), 0, 1);
            for name = ["" "a b" "../x" "a.tex" "a/b" string(repmat('a', 1, 101))]
                testCase.verifyError( ...
                    @() phasecont.internal.latexWrite(tmp.Folder, name, "x", none), ...
                    "phasecont:latex:badName");
            end
        end

        % ---- build (no TeX needed) ----------------------------------------

        function buildRejectsUnsafeExecutable(testCase)
            for exe = ["latexmk; rm -rf x" "a&b" "a|b" "a`b" "a$b" "a""b" "a%b"]
                testCase.verifyError( ...
                    @() phasecont.latex.build("x", Executable = exe), ...
                    "phasecont:latex:badEngine");
            end
        end

        function buildNamesAMissingProgram(testCase)
            testCase.verifyError( ...
                @() phasecont.latex.build("x", Executable = "no-such-latexmk-program"), ...
                "phasecont:latex:engineNotFound");
        end

        function buildRejectsANonPositiveTimeout(testCase)
            for t = [0 -1 NaN]
                testCase.verifyError( ...
                    @() phasecont.latex.build("x", Timeout = t), ?MException);
            end
        end

        function buildOnlyKnowsLatexmk(testCase)
            testCase.verifyError(@() phasecont.latex.build("x", Engine = "tectonic"), ...
                "MATLAB:validators:mustBeMember");
        end

        % ---- pgfplots and specialTable ------------------------------------

        function pgfplotsNeedsDigits(testCase)
            fig = testCase.figureSpec([1; 2], [3; 4]);
            testCase.verifyError(@() phasecont.latex.pgfplots(fig), ...
                "phasecont:latex:missingDigits");
            testCase.verifyError(@() phasecont.latex.specialTable(testCase.tableSpec()), ...
                "phasecont:latex:missingDigits");
        end

        function pgfplotsRendersStableAndUnstableSeries(testCase)
            testCase.assumeGateway();
            fig = testCase.figureSpec([1; 2], [3; 4]);
            fig.Series(1).Color = "#000000";
            fig.Series(2) = struct("X", [2; 3], "Y", [4; 5], "Class", "unstable", ...
                "Color", "#000000");
            fig.Marks = struct("Label", "LP", "X", 2, "Y", 4);
            r = phasecont.latex.pgfplots(fig, Digits = 6);
            testCase.verifyEqual(r.Data, "inline");
            testCase.verifyEmpty(r.Files);
            testCase.verifyEmpty(r.Paths);
            testCase.verifySubstring(r.Tex, "\begin{axis}");
            testCase.verifySubstring(r.Tex, "dashed");
            testCase.verifySubstring(r.Tex, "LP");
            testCase.verifyFalse(contains(r.Tex, "\documentclass"));
            s = phasecont.latex.pgfplots(fig, Digits = 6, Standalone = true);
            testCase.verifySubstring(s.Tex, "\documentclass");
        end

        function pgfplotsWritesExternalData(testCase)
            testCase.assumeGateway();
            tmp = testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            fig = testCase.figureSpec([1; 2], [3; 4]);
            r = phasecont.latex.pgfplots(fig, Digits = 6, Data = "external", ...
                Basename = "fig1", Folder = tmp.Folder);
            testCase.verifyEqual(r.Data, "external");
            testCase.verifySize(r.Files, [1 1]);
            testCase.verifyEqual(r.Files(1).Name, "fig1-s1.dat");
            testCase.verifySubstring(r.Tex, "fig1-s1.dat");
            testCase.verifySize(r.Paths, [2 1]);
            testCase.verifyTrue(isfile(fullfile(tmp.Folder, "fig1.tex")));
            testCase.verifyTrue(isfile(fullfile(tmp.Folder, "fig1-s1.dat")));
            % Without a folder the data comes back and nothing is written.
            r2 = phasecont.latex.pgfplots(fig, Digits = 6, Data = "external");
            testCase.verifyEqual(r2.Files(1).Name, "figure-s1.dat");
            testCase.verifyEmpty(r2.Paths);
        end

        function pgfplotsReportsTheRendererErrors(testCase)
            testCase.assumeGateway();
            fig = testCase.figureSpec([1; 2], [1; NaN]);
            try
                phasecont.latex.pgfplots(fig, Digits = 6);
                testCase.verifyFail("A NaN in the data must raise an error.");
            catch e
                testCase.verifyEqual(string(e.identifier), "phasecont:latex:nonFinite");
                testCase.verifySubstring(string(e.message), "series 1");
            end
            empty = testCase.figureSpec([1; 2], [3; 4]);
            empty.Series = empty.Series([]);
            testCase.verifyError(@() phasecont.latex.pgfplots(empty, Digits = 6), ...
                "phasecont:latex:emptySeries");
            clean = testCase.figureSpec([1; 2], [3; 4]);
            testCase.verifyError(@() phasecont.latex.pgfplots(clean, Digits = 0), ...
                "phasecont:latex:badDigits");
            testCase.verifyError(@() phasecont.latex.pgfplots(clean, Digits = 18), ...
                "phasecont:latex:badDigits");
            short = testCase.figureSpec([1; 2], [3; 4]);
            short.Series(1).Y = 3;
            testCase.verifyError(@() phasecont.latex.pgfplots(short, Digits = 6), ...
                "phasecont:latex:lengthMismatch");
        end

        function specialTableRendersABranch(testCase)
            testCase.assumeGateway();
            r = phasecont.latex.specialTable(testCase.epBranch(), Digits = 4);
            testCase.verifySubstring(r.Tex, "\toprule");
            testCase.verifySubstring(r.Tex, "LP1");
            testCase.verifySubstring(r.Tex, "Hopf");
            testCase.verifyEmpty(r.Files);
            s = phasecont.latex.specialTable(testCase.epBranch(), Digits = 4, ...
                Standalone = true);
            testCase.verifySubstring(s.Tex, "\documentclass");
        end

        function specialTableReportsAnEmptyBranch(testCase)
            testCase.assumeGateway();
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 2, ...
                "U", [1; 2]));
            testCase.verifyError( ...
                @() phasecont.latex.specialTable(br, Digits = 4), ...
                "phasecont:latex:emptyTable");
        end

        function specialTableRejectsOtherSources(testCase)
            testCase.verifyError(@() phasecont.latex.specialTable(42, Digits = 4), ...
                "phasecont:latex:badInput");
        end

        % ---- compiling ----------------------------------------------------

        function builtFigureAndTableMakeAPdf(testCase)
            testCase.assumeGateway();
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
            fig = testCase.figureSpec([1; 2; 3], [3; 4; 2]);
            f = phasecont.latex.pgfplots(fig, Digits = 6, Standalone = true);
            rf = phasecont.latex.build(f.Tex);
            testCase.verifyTrue(rf.Ok, rf.Log);
            testCase.verifyTrue(isfile(rf.Pdf));
            testCase.verifyEmpty(rf.Errors);
            t = phasecont.latex.specialTable(testCase.epBranch(), Digits = 4, ...
                Standalone = true);
            rt = phasecont.latex.build(t.Tex);
            testCase.verifyTrue(rt.Ok, rt.Log);
        end

        function builtExternalDataFigureMakesAPdf(testCase)
            testCase.assumeGateway();
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
            fig = testCase.figureSpec([1; 2; 3], [3; 4; 2]);
            f = phasecont.latex.pgfplots(fig, Digits = 6, Data = "external", ...
                Basename = "main", Standalone = true);
            r = phasecont.latex.build(f.Tex, Files = f.Files);
            testCase.verifyTrue(r.Ok, r.Log);
        end

        function buildTimesOut(testCase)
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
            % \def\a{\a}\a calls itself without end and without using memory.
            tex = "\documentclass{article}" + newline + "\begin{document}" + newline + ...
                "\def\a{\a}\a" + newline + "\end{document}" + newline;
            clock = tic;
            r = phasecont.latex.build(tex, Timeout = 3);
            testCase.verifyLessThan(toc(clock), 60);
            testCase.verifyTrue(r.TimedOut);
            testCase.verifyFalse(r.Ok);
            testCase.verifyEqual(r.Pdf, "");
            if isunix && ~ismac
                % No process may keep the build folder as its working directory.
                pause(1);
                [~, n] = system("ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -c " + ...
                    """" + r.Folder + """");
                testCase.verifyEqual(str2double(n), 0, ...
                    "latexmk or a TeX program is still running");
            end
        end

        function buildWithoutALimitStillWorks(testCase)
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
            tex = "\documentclass{article}" + newline + "\begin{document}" + newline + ...
                "x" + newline + "\end{document}" + newline;
            r = phasecont.latex.build(tex, Timeout = Inf);
            testCase.verifyTrue(r.Ok, r.Log);
            testCase.verifyFalse(r.TimedOut);
            testCase.verifyTrue(isfile(r.Pdf));
        end

        function buildWithinTheLimitIsNotTimedOut(testCase)
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
            tex = "\documentclass{article}" + newline + "\begin{document}" + newline + ...
                "x" + newline + "\end{document}" + newline;
            r = phasecont.latex.build(tex, Timeout = 120);
            testCase.verifyTrue(r.Ok, r.Log);
            testCase.verifyFalse(r.TimedOut);
        end

        function buildReportsATexErrorWithItsLine(testCase)
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
            tex = "\documentclass{article}" + newline + "\begin{document}" + newline + ...
                "\badmacro" + newline + "\end{document}" + newline;
            r = phasecont.latex.build(tex);
            testCase.verifyFalse(r.Ok);
            testCase.verifyEqual(r.Pdf, "");
            testCase.verifyNotEmpty(r.Errors);
            testCase.verifyEqual(r.Errors(1).File, "main.tex");
            testCase.verifyEqual(r.Errors(1).Line, 3);
            testCase.verifySubstring(r.Errors(1).Message, "Undefined control sequence");
        end
    end

    methods (Access = private)
        function assumeGateway(testCase)
            testCase.assumeTrue(testCase.HasGateway, ...
                "pckern has not been built (run buildtool mex)");
        end

        function json = figureJson(~, fig)
            opts = struct("Digits", 6, "Standalone", false, ...
                "Data", "inline", "Basename", "f");
            json = char(phasecont.internal.latexJson("pgfplots", fig, opts));
        end

        function fig = figureSpec(~, x, y)
            s = struct("X", x, "Y", y, "Class", "stable");
            fig = struct("XLabel", "a", "YLabel", "u", "Series", s);
        end

        function spec = tableSpec(~)
            spec = struct("Columns", ["a" "b"], "Rows", struct("Label", "EP1", ...
                "Type", "Equilibrium", "Values", [1 2]));
        end

        function br = epBranch(~)
            p = @(label, a, u) phasecont.Point(U = u, P = [0; a], Label = label);
            sp = [p("LP", 5, [1; 2]); p("H", 6, [3; 4]); p("XYZ", 7, [5; 6])];
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 2, ...
                "Alpha", [5; 6; 7], "U", [1 2; 3 4; 5 6], ...
                "Stable", [true; true; true], "Specials", sp));
        end
    end
end
