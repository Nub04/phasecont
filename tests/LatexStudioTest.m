classdef LatexStudioTest < matlab.unittest.TestCase
    % Tests phasecont.app.LatexStudio through its methods, with the figure
    % hidden.
    %   - Editing, Digits, saving and opening need no binary.
    %   - Insert figure and Insert table need the pckern gateway and are
    %     filtered when it has not been built.
    %   - Compile needs latexmk and is filtered when it is not installed.
    % The Branch menu and the toolbar callbacks are not driven here, since
    % they need a clicked window.

    properties
        HasGateway (1,1) logical = false
        HasLatexmk (1,1) logical = false
        Dir
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
            t = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Dir = t.Folder;
        end
    end

    methods (Test)
        % ---- editing ------------------------------------------------------

        function startsWithAnArticle(testCase)
            app = testCase.makeApp();
            testCase.verifySubstring(app.Source, "\usepackage{pgfplots}");
            testCase.verifySubstring(app.Source, "\usepackage{booktabs}");
            testCase.verifySubstring(app.Source, "\begin{document}");
            testCase.verifyTrue(endsWith(app.Source, "\end{document}"));
            testCase.verifyEqual(app.File, "");
        end

        function setSourceKeepsLines(testCase)
            app = testCase.makeApp();
            app.setSource(["first"; "second"]);
            testCase.verifyEqual(app.Source, "first" + newline + "second");
            app.setSource("one" + newline + "two" + newline + "three");
            testCase.verifyEqual(app.Source, "one" + newline + "two" + newline + "three");
        end

        function newDocumentResetsText(testCase)
            app = testCase.makeApp();
            app.setSource("something else");
            app.newDocument();
            testCase.verifySubstring(app.Source, "\documentclass{article}");
        end

        function saveAndOpenRoundTrip(testCase)
            app = testCase.makeApp();
            text = "\documentclass{article}" + newline + "caf" + char(233) + newline + "\end{document}";
            app.setSource(text);
            file = string(fullfile(testCase.Dir, "round.tex"));
            app.saveFile(file);
            testCase.verifyEqual(app.File, file);
            app.setSource("replaced");
            app.openFile(file);
            testCase.verifyEqual(app.Source, text);
            testCase.verifyEqual(app.File, file);
        end

        function openMissingFileFails(testCase)
            app = testCase.makeApp();
            testCase.verifyError( ...
                @() app.openFile(fullfile(testCase.Dir, "absent.tex")), ...
                "phasecont:app:openFailed");
        end

        function saveToBadFolderFails(testCase)
            app = testCase.makeApp();
            testCase.verifyError( ...
                @() app.saveFile(fullfile(testCase.Dir, "no", "such", "a.tex")), ...
                "phasecont:app:saveFailed");
        end

        function openerMustBeAHandle(testCase)
            testCase.verifyError( ...
                @() phasecont.app.LatexStudio(Visible = "off", Opener = 3), ...
                "phasecont:app:badOpener");
        end

        % ---- digits and insert --------------------------------------------

        function digitsHasNoDefault(testCase)
            app = testCase.makeApp();
            testCase.verifyTrue(isnan(app.Digits));
            app.setDigits(6);
            testCase.verifyEqual(app.Digits, 6);
        end

        function insertNeedsDigits(testCase)
            app = testCase.makeApp();
            before = app.Source;
            testCase.verifyError(@() app.insertFigure(testCase.epBranch()), ...
                "phasecont:latex:missingDigits");
            testCase.verifyError(@() app.insertTable(testCase.epBranch()), ...
                "phasecont:latex:missingDigits");
            testCase.verifyEqual(app.Source, before);
        end

        function insertFigureGoesBeforeEnd(testCase)
            testCase.assumeGateway();
            app = testCase.makeApp();
            app.setDigits(6);
            app.insertFigure(testCase.epBranch(), Title = "Branch");
            src = app.Source;
            testCase.verifySubstring(src, "\begin{axis}");
            testCase.verifySubstring(src, "\addplot");
            testCase.verifyTrue(endsWith(src, "\end{document}"));
            testCase.verifyLessThan(strfind(src, "\begin{tikzpicture}"), ...
                strfind(src, "\end{document}"));
            testCase.verifyEqual(count(src, "\end{document}"), 1);
        end

        function insertTableGoesBeforeEnd(testCase)
            testCase.assumeGateway();
            app = testCase.makeApp();
            app.setDigits(4);
            app.insertTable(testCase.epBranch());
            src = app.Source;
            testCase.verifySubstring(src, "\toprule");
            testCase.verifySubstring(src, "LP1");
            testCase.verifyTrue(endsWith(src, "\end{document}"));
        end

        function insertWithoutEndDocumentAppends(testCase)
            testCase.assumeGateway();
            app = testCase.makeApp();
            app.setSource("before");
            app.setDigits(4);
            app.insertTable(testCase.epBranch());
            testCase.verifyTrue(startsWith(app.Source, "before"));
            testCase.verifySubstring(app.Source, "\toprule");
            testCase.verifyTrue(contains(app.Source, "\bottomrule"));
        end

        function insertTakesTheLastEndDocument(testCase)
            testCase.assumeGateway();
            app = testCase.makeApp();
            app.setSource(["\begin{document}"; "\end{document}"; "% notes"; "\end{document}"]);
            app.setDigits(4);
            app.insertTable(testCase.epBranch());
            src = app.Source;
            ends = strfind(src, "\end{document}");
            testCase.verifyEqual(numel(ends), 2);
            testCase.verifyGreaterThan(strfind(src, "\toprule"), ends(1));
            testCase.verifyLessThan(strfind(src, "\toprule"), ends(2));
        end

        % ---- compile ------------------------------------------------------

        function compileNeedsText(testCase)
            app = testCase.makeApp();
            app.setSource("");
            testCase.verifyError(@() app.compile(), "phasecont:app:empty");
        end

        function missingLatexmkRaises(testCase)
            app = phasecont.app.LatexStudio(Visible = "off", ...
                Executable = "phasecont-no-such-latexmk");
            testCase.addTeardown(@() delete(app));
            testCase.verifyError(@() app.compile(), "phasecont:latex:engineNotFound");
        end

        function timeoutMustBePositive(testCase)
            testCase.verifyError(@() phasecont.app.LatexStudio( ...
                Visible = "off", Timeout = 0), ?MException);
        end

        function compileShowsATimeout(testCase)
            testCase.assumeLatexmk();
            app = phasecont.app.LatexStudio(Visible = "off", Timeout = 3);
            testCase.addTeardown(@() delete(app));
            app.setSource(["\documentclass{article}"; "\begin{document}"; ...
                "\def\a{\a}\a"; "\end{document}"]);
            r = app.compile();
            testCase.verifyTrue(r.TimedOut);
            testCase.verifyFalse(r.Ok);
            testCase.verifySubstring(app.Status, "stopped after 3 s");
            testCase.verifyError(@() app.openPdf(), "phasecont:app:noPdf");
        end

        function noPdfBeforeCompile(testCase)
            app = testCase.makeApp();
            testCase.verifyError(@() app.openPdf(), "phasecont:app:noPdf");
        end

        function noErrorsBeforeCompile(testCase)
            app = testCase.makeApp();
            testCase.verifyError(@() app.selectError(1), "phasecont:app:noSuchError");
        end

        function compileMakesAPdf(testCase)
            testCase.assumeLatexmk();
            rec = containers.Map();
            app = testCase.makeApp(Opener = @(p) record(rec, p));
            app.setSource(["\documentclass{article}"; "\begin{document}"; ...
                "Hello"; "\end{document}"]);
            r = app.compile();
            testCase.verifyTrue(r.Ok);
            testCase.verifyTrue(isfile(r.Pdf));
            testCase.verifyEqual(app.Result.Pdf, r.Pdf);
            testCase.verifySubstring(app.Status, "Compiled");
            app.openPdf();
            testCase.verifyEqual(string(rec("path")), r.Pdf);
        end

        function compileShowsTheErrorLine(testCase)
            testCase.assumeLatexmk();
            app = testCase.makeApp();
            app.setSource(["\documentclass{article}"; "\begin{document}"; ...
                "\phasecontBadCommand"; "\end{document}"]);
            r = app.compile();
            testCase.verifyFalse(r.Ok);
            testCase.verifyGreaterThanOrEqual(numel(r.Errors), 1);
            testCase.verifyEqual(r.Errors(1).Line, 3);
            lines = app.selectError(1);
            marked = lines(startsWith(lines, ">"));
            testCase.verifyEqual(numel(marked), 1);
            testCase.verifySubstring(marked, "\phasecontBadCommand");
            testCase.verifySubstring(app.Status, "failed");
            testCase.verifyError(@() app.openPdf(), "phasecont:app:noPdf");
        end

        function insertedFigureAndTableCompile(testCase)
            testCase.assumeGateway();
            testCase.assumeLatexmk();
            app = testCase.makeApp();
            app.setDigits(6);
            app.insertFigure(testCase.epBranch());
            app.insertTable(testCase.epBranch());
            r = app.compile();
            testCase.verifyTrue(r.Ok, r.Log);
            testCase.verifyTrue(isfile(r.Pdf));
        end
    end

    methods (Access = private)
        function app = makeApp(testCase, options)
            arguments
                testCase
                options.Opener = []
            end
            app = phasecont.app.LatexStudio(Visible = "off", Opener = options.Opener);
            testCase.addTeardown(@() delete(app));
        end

        function assumeGateway(testCase)
            testCase.assumeTrue(testCase.HasGateway, ...
                "pckern has not been built (run buildtool mex)");
        end

        function assumeLatexmk(testCase)
            testCase.assumeTrue(testCase.HasLatexmk, "latexmk is not installed");
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

function record(map, path)
% Stands in for the system viewer: remember the path it was given.
map("path") = path;
end
