classdef ModelEditorTest < matlab.unittest.TestCase
    % Tests phasecont.app.ModelEditor through its methods, with the figure
    % hidden: the three input tabs, diagnostics and where they point, that
    % a failed Check keeps the last good model, file opening for .ode, SBML
    % and .m files, saving, and the hand-off to the phase plane. The test
    % models are our own. Every test loads pckern, so the suite needs the
    % MEX binary.

    properties (Constant)
        OscFile = "par w=2, c=0.1" + newline + ...
            "x' = y" + newline + ...
            "y' = -w^2*x - c*y" + newline + ...
            "init x=1, y=0" + newline + "done" + newline
        Equations = ["x' = y"; "dy/dt = -w^2*x - c*y"]
        Parameters = "w = 2" + newline + "c = 0.1"
        Initial = "x = 1" + newline + "y = 0"
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
        function app = makeApp(testCase)
            app = phasecont.app.ModelEditor(Visible = "off");
            testCase.addTeardown(@() delete(app));
        end

        function app = filled(testCase)
            % An editor whose Equations tab holds the damped oscillator.
            app = testCase.makeApp();
            app.setEquations(testCase.Equations);
            app.setParameters(testCase.Parameters);
            app.setInitial(testCase.Initial);
        end

        function f = write(testCase, name, text)
            f = fullfile(testCase.Dir, name);
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
        end

        function ids = idsOf(~, app)
            ids = string({app.Diagnostics.Id});
        end

        function items = listItems(~, app)
            box = findobj(app.Figure, "Type", "uilistbox");
            items = string(box.Items);
        end

        function states = oscStates(~)
            states = table(["x"; "y"], ["y"; "-w^2*x - c*y"], [1; 0], ...
                VariableNames = ["Name" "Rhs" "Init"]);
        end

        function params = oscParams(~)
            params = table(["w"; "c"], [2; 0.1], VariableNames = ["Name" "Value"]);
        end

        function body = decaySbml(~)
            mml = "http://www.w3.org/1998/Math/MathML";
            body = "<?xml version=""1.0"" encoding=""UTF-8""?>" + ...
                "<sbml xmlns=""http://www.sbml.org/sbml/level3/version1/core"" " + ...
                "level=""3"" version=""1""><model id=""m"">" + ...
                "<listOfCompartments><compartment id=""C"" size=""1"" " + ...
                "constant=""true""/></listOfCompartments>" + ...
                "<listOfSpecies><species id=""S"" compartment=""C"" " + ...
                "initialConcentration=""4""/></listOfSpecies>" + ...
                "<listOfParameters><parameter id=""k"" value=""0.5"" " + ...
                "constant=""true""/></listOfParameters>" + ...
                "<listOfReactions><reaction id=""R"" reversible=""false"">" + ...
                "<listOfReactants><speciesReference species=""S"" " + ...
                "stoichiometry=""1""/></listOfReactants><kineticLaw>" + ...
                "<math xmlns=""" + mml + """><apply><times/><ci>k</ci>" + ...
                "<ci>S</ci></apply></math></kineticLaw></reaction>" + ...
                "</listOfReactions></model></sbml>";
        end
    end

    methods (Test)
        function startsEmpty(testCase)
            app = testCase.makeApp();
            testCase.verifyEmpty(app.Model);
            testCase.verifyEqual(app.Source, "");
            testCase.verifyEmpty(app.Diagnostics);
            testCase.verifyEqual(app.Tab, "Equations");
            testCase.verifyTrue(isvalid(app.Figure));
        end

        function equationsTabBuildsAModel(testCase)
            app = testCase.filled();
            testCase.verifyTrue(app.check());
            m = app.Model;
            testCase.verifyEqual(m.Variables, ["x" "y"]);
            testCase.verifyEqual(m.Init, [1; 0]);
            testCase.verifyEqual(sort(m.Parameters), sort(["w" "c"]));
            testCase.verifyEmpty(app.Diagnostics);
            testCase.verifyTrue(contains(app.Status, "OK"));
            testCase.verifyTrue(contains(app.Source, "dy/dt"));
            ref = phasecont.Model.fromSource(testCase.OscFile);
            testCase.addTeardown(@() delete(ref));
            testCase.verifyEqual(m.rhs([0.3; -0.7]), ref.rhs([0.3; -0.7]), ...
                RelTol = 1e-13);
        end

        function commentsAndCommasInTheValueBoxes(testCase)
            app = testCase.makeApp();
            app.setEquations(testCase.Equations);
            app.setParameters("# damping" + newline + "w = 2, c = 0.1 % light");
            app.setInitial("x = 1; y = 0");
            testCase.verifyTrue(app.check());
            testCase.verifyEqual(app.Model.Init, [1; 0]);
        end

        function aMissingInitialValueIsADiagnosticNotAZero(testCase)
            app = testCase.makeApp();
            app.setEquations(testCase.Equations);
            app.setParameters(testCase.Parameters);
            app.setInitial("x = 1");
            testCase.verifyFalse(app.check());
            testCase.verifyEmpty(app.Model);
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:model:missingInit");
            testCase.verifyTrue(contains(app.Diagnostics(1).Message, "y"));
            testCase.verifyTrue(isnan(app.Diagnostics(1).Line));
            testCase.verifyEqual(numel(testCase.listItems(app)), 1);
        end

        function aMissingParameterValueIsADiagnostic(testCase)
            app = testCase.makeApp();
            app.setEquations(testCase.Equations);
            app.setParameters("w = 2");
            app.setInitial(testCase.Initial);
            testCase.verifyFalse(app.check());
            testCase.verifyEmpty(app.Model);
            testCase.verifyTrue(startsWith(testCase.idsOf(app), "phasecont:"));
        end

        function anUndeclaredSymbolPointsAtItsLine(testCase)
            app = testCase.makeApp();
            app.setEquations(["x' = y"; "y' = -q*x"]);
            app.setParameters("");
            app.setInitial(testCase.Initial);
            testCase.verifyFalse(app.check());
            d = app.Diagnostics(1);
            testCase.verifyEqual(d.Id, "phasecont:engine:model");
            testCase.verifyEqual(d.Line, 2);
            testCase.verifyTrue(d.Column >= 1);
            testCase.verifyTrue(contains(d.Message, "q"));
            testCase.verifyTrue(startsWith(testCase.listItems(app), "line 2"));
        end

        function aBadValueLineIsADiagnostic(testCase)
            app = testCase.filled();
            app.setParameters("w = two");
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:app:badAssignment");
            testCase.verifyTrue(isnan(app.Diagnostics(1).Line));
            app.setParameters("w = Inf");
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:app:badAssignment");
            app.setParameters("w = 2" + newline + "w = 3");
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:model:duplicateName");
        end

        function emptyInputIsReportedNotBuilt(testCase)
            app = testCase.makeApp();
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:app:empty");
            app.selectTab("Table");
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:app:empty");
            app.selectTab("Source");
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:app:empty");
            testCase.verifyEmpty(app.Model);
        end

        function aFailedCheckKeepsTheLastGoodModel(testCase)
            app = testCase.filled();
            testCase.verifyTrue(app.check());
            good = app.Model;
            source = app.Source;
            app.setEquations(["x' = y"; "y' = -q*x"]);
            testCase.verifyFalse(app.check());
            testCase.verifySameHandle(app.Model, good);
            testCase.verifyEqual(app.Source, source);
            testCase.verifyNotEmpty(app.Diagnostics);
            testCase.verifyTrue(contains(app.Status, "last good"));
            app.setEquations(testCase.Equations);
            testCase.verifyTrue(app.check());
            testCase.verifyEmpty(app.Diagnostics);
            testCase.verifyEmpty(testCase.listItems(app));
        end

        function theTableTabBuildsTheSameModel(testCase)
            app = testCase.makeApp();
            app.selectTab("Table");
            testCase.verifyEqual(app.Tab, "Table");
            app.setTable(States = testCase.oscStates(), Params = testCase.oscParams());
            testCase.verifyTrue(app.check());
            m = app.Model;
            testCase.verifyEqual(m.Variables, ["x" "y"]);
            testCase.verifyEqual(m.Init, [1; 0]);
            ref = phasecont.Model.fromSource(testCase.OscFile);
            testCase.addTeardown(@() delete(ref));
            p = [2; 0.1];
            ip = arrayfun(@(n) find(m.Parameters == n), ["w" "c"]);
            pm = zeros(2, 1);
            pm(ip) = p;
            testCase.verifyEqual(m.rhs([0.3; -0.7], pm), ...
                ref.rhs([0.3; -0.7]), RelTol = 1e-13);
        end

        function theTableTabKeepsAuxAndDropsBlankRows(testCase)
            app = testCase.makeApp();
            app.selectTab("Table");
            states = [testCase.oscStates(); ...
                table("", "", NaN, VariableNames = ["Name" "Rhs" "Init"])];
            aux = table("e", "x^2 + y^2", VariableNames = ["Name" "Expr"]);
            app.setTable(States = states, Params = testCase.oscParams(), Aux = aux);
            testCase.verifyTrue(app.check());
            testCase.verifyEqual(app.Model.Variables, ["x" "y"]);
            testCase.verifyTrue(contains(app.Source, "aux e"));
        end

        function aTableWithoutAnInitialValueIsADiagnostic(testCase)
            app = testCase.makeApp();
            app.selectTab("Table");
            states = testCase.oscStates();
            states.Init(2) = NaN;
            app.setTable(States = states, Params = testCase.oscParams());
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(testCase.idsOf(app), "phasecont:model:missingInit");
            testCase.verifyEmpty(app.Model);
        end

        function aTableDiagnosticNamesTheRow(testCase)
            app = testCase.makeApp();
            app.selectTab("Table");
            states = table(["x"; "y"], ["y"; "-q*x"], [1; 0], ...
                VariableNames = ["Name" "Rhs" "Init"]);
            app.setTable(States = states, ...
                Params = table(strings(0, 1), zeros(0, 1), ...
                VariableNames = ["Name" "Value"]));
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(app.Diagnostics(1).Line, 2);
            testCase.verifyTrue(startsWith(testCase.listItems(app), "States row 2"));
        end

        function setTableChecksItsColumns(testCase)
            app = testCase.makeApp();
            bad = table("x", VariableNames = "Name");
            testCase.verifyError(@() app.setTable(States = bad), ...
                "phasecont:model:badTable");
        end

        function theSourceTabBuildsFromOdeText(testCase)
            app = testCase.makeApp();
            app.setSource(testCase.OscFile);
            app.selectTab("Source");
            testCase.verifyTrue(app.check());
            testCase.verifyEqual(app.Model.Variables, ["x" "y"]);
            ref = phasecont.Model.fromSource(testCase.OscFile);
            testCase.addTeardown(@() delete(ref));
            testCase.verifyEqual(app.Model.SourceSha256, ref.SourceSha256);
        end

        function theSourceTabAddsAFinalNewlineOnlyWhenMissing(testCase)
            app = testCase.makeApp();
            app.setSource(["x' = y"; "y' = -x"; "init x=1, y=0"]);
            app.selectTab("Source");
            testCase.verifyTrue(app.check());
            ref = phasecont.Model.fromSource( ...
                join(["x' = y" "y' = -x" "init x=1, y=0"], newline) + newline);
            testCase.addTeardown(@() delete(ref));
            testCase.verifyEqual(app.Model.SourceSha256, ref.SourceSha256);
        end

        function theSourceTabPointsAtTheErrorLine(testCase)
            app = testCase.makeApp();
            app.setSource(["par w=2"; "x' = y"; "y' = -q*x - w*y"; "init x=1,y=0"]);
            app.selectTab("Source");
            testCase.verifyFalse(app.check());
            testCase.verifyEqual(app.Diagnostics(1).Line, 3);
        end

        function openFileReadsAnOdeFile(testCase)
            f = testCase.write("osc.ode", testCase.OscFile);
            app = testCase.makeApp();
            app.openFile(f);
            testCase.verifyEqual(app.Tab, "Source");
            testCase.verifyNotEmpty(app.Model);
            testCase.verifyEqual(app.Model.Variables, ["x" "y"]);
            ref = phasecont.Model(f);
            testCase.addTeardown(@() delete(ref));
            testCase.verifyEqual(app.Model.SourceSha256, ref.SourceSha256);
        end

        function openFileNamesAnOdeErrorByFile(testCase)
            f = testCase.write("bad.ode", "x' = -q*x" + newline);
            app = testCase.makeApp();
            app.openFile(f);
            testCase.verifyEmpty(app.Model);
            testCase.verifyEqual(app.Diagnostics(1).Line, 1);
            testCase.verifyTrue(contains(app.Diagnostics(1).Message, "q"));
        end

        function openFileReadsSbml(testCase)
            f = testCase.write("decay.xml", testCase.decaySbml());
            app = testCase.makeApp();
            app.openFile(f);
            testCase.verifyEqual(app.Tab, "Source");
            testCase.verifyEqual(app.Model.Variables, "S");
            testCase.verifyEqual(app.Model.Init, 4);
            tr = phasecont.simulate(app.Model, app.Model.Init, [0 2], ...
                RelTol = 1e-10, AbsTol = 1e-12);
            testCase.verifyEqual(tr.X(end, 1), 4*exp(-1), RelTol = 1e-7);
            testCase.verifyTrue(contains(app.Source, "S"));
        end

        function openFileRefusesAnSbmlEventByName(testCase)
            body = replace(testCase.decaySbml(), "</model>", ...
                "<listOfEvents><event id=""e1""/></listOfEvents></model>");
            f = testCase.write("events.xml", body);
            app = testCase.makeApp();
            testCase.verifyError(@() app.openFile(f), "phasecont:import:unsupported");
            testCase.verifyEmpty(app.Model);
        end

        function openFileReadsAFunctionFile(testCase)
            f = testCase.write("pend_rhs.m", ...
                "function dydt = pend_rhs(t, y, p)" + newline + ...
                "th = y(1);" + newline + "om = y(2);" + newline + ...
                "dydt = zeros(2, 1);" + newline + ...
                "dydt(1) = om;" + newline + ...
                "dydt(2) = -sin(th) - p(1)*om;" + newline + "end" + newline);
            app = testCase.makeApp();
            app.openFile(f, Init = [0.1; 0], Params = struct(damp = 0.2), ...
                StateNames = ["th"; "om"]);
            testCase.verifyEqual(app.Model.Variables, ["th" "om"]);
            x = [0.4; 0.9];
            testCase.verifyEqual(app.Model.rhs(x, 0.2), ...
                [x(2); -sin(x(1)) - 0.2*x(2)], RelTol = 1e-13);
        end

        function openFileNeedsInitialValuesForAFunctionFile(testCase)
            f = testCase.write("decay_rhs.m", ...
                "function dydt = decay_rhs(t, y)" + newline + ...
                "dydt = -y(1);" + newline + "end" + newline);
            app = testCase.makeApp();
            testCase.verifyError(@() app.openFile(f, StateNames = "y"), ...
                "phasecont:import:missingValue");
        end

        function openFileChecksItsOptionsAndExtension(testCase)
            bad = "phasecont:model:badOption";
            app = testCase.makeApp();
            ode = testCase.write("osc.ode", testCase.OscFile);
            testCase.verifyError(@() app.openFile(ode, Init = [1; 0]), bad);
            txt = testCase.write("model.txt", "x' = -x");
            testCase.verifyError(@() app.openFile(txt), bad);
            testCase.verifyError(@() app.openFile(fullfile(testCase.Dir, "gone.ode")), ...
                "phasecont:import:file");
        end

        function saveOdeWritesTheLastGoodSource(testCase)
            app = testCase.filled();
            f = fullfile(testCase.Dir, "saved.ode");
            testCase.verifyError(@() testCase.makeApp().saveOde(f), ...
                "phasecont:app:noModel");
            testCase.verifyTrue(app.check());
            app.saveOde(f);
            testCase.verifyTrue(isfile(f));
            back = phasecont.Model(f);
            testCase.addTeardown(@() delete(back));
            testCase.verifyEqual(back.SourceSha256, app.Model.SourceSha256);
            testCase.verifyEqual(back.Variables, app.Model.Variables);
        end

        function saveOdeKeepsTheLastGoodModelAfterAFailedCheck(testCase)
            app = testCase.filled();
            testCase.verifyTrue(app.check());
            source = app.Source;
            app.setEquations(["x' = y"; "y' = -q*x"]);
            testCase.verifyFalse(app.check());
            f = fullfile(testCase.Dir, "good.ode");
            app.saveOde(f);
            testCase.verifyEqual(string(fileread(f, Encoding = "UTF-8")), source);
        end

        function openPhasePlaneUsesTheLastGoodModel(testCase)
            app = testCase.filled();
            testCase.verifyError(@() testCase.makeApp().openPhasePlane(Visible = "off"), ...
                "phasecont:app:noModel");
            testCase.verifyTrue(app.check());
            pp = app.openPhasePlane(Visible = "off");
            testCase.addTeardown(@() delete(pp));
            testCase.verifyClass(pp, "phasecont.app.PhasePlane");
            testCase.verifySameHandle(pp.Model, app.Model);
        end

        function aLaterCheckDoesNotChangeAnOpenPhasePlane(testCase)
            app = testCase.filled();
            testCase.verifyTrue(app.check());
            first = app.Model;
            pp = app.openPhasePlane(Visible = "off");
            testCase.addTeardown(@() delete(pp));
            app.setParameters("w = 3" + newline + "c = 0.1");
            testCase.verifyTrue(app.check());
            testCase.verifyNotSameHandle(app.Model, first);
            testCase.verifySameHandle(pp.Model, first);
        end

        function aOneStateModelIsRefusedByThePhasePlane(testCase)
            app = testCase.makeApp();
            app.setEquations("x' = -x");
            app.setInitial("x = 1");
            testCase.verifyTrue(app.check());
            testCase.verifyError(@() app.openPhasePlane(Visible = "off"), ...
                "phasecont:app:needTwoStates");
        end

        function deleteClosesTheFigure(testCase)
            app = phasecont.app.ModelEditor(Visible = "off");
            fig = app.Figure;
            testCase.verifyTrue(isvalid(fig));
            delete(app);
            testCase.verifyFalse(isvalid(fig));
        end

        function selectTabChecksItsName(testCase)
            app = testCase.makeApp();
            testCase.verifyError(@() app.selectTab("Graph"), ...
                "MATLAB:validators:mustBeMember");
        end
    end
end
