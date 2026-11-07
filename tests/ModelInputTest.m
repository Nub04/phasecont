classdef ModelInputTest < matlab.unittest.TestCase
    % Tests the model-input routes of phasecont.Model: fromEquations,
    % fromTable, fromFunction and fromFile (.ode, SBML and .m). Every route
    % ends as .ode text, so each test builds the same model two ways and
    % compares. The test models are our own. The suite needs the MEX binary;
    % the engine-free helpers are tested in ModelInputHelperTest.

    properties (Constant)
        % A damped oscillator in the three spellings the routes accept.
        OscFile = "par w=2, c=0.1" + newline + ...
            "x' = y" + newline + ...
            "y' = -w^2*x - c*y" + newline + ...
            "init x=1, y=0" + newline + "done" + newline
        OscEquations = ["x' = y"; "dy/dt = -w^2*x - c*y"]
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
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

        function m = track(testCase, m)
            testCase.addTeardown(@() delete(m));
        end

        function m = fileModel(testCase, text, ev)
            if nargin < 3
                ev = "jit";
            end
            f = testCase.write("ref_" + string(randi(1e9)) + ".ode", text);
            m = testCase.track(phasecont.Model(f, Evaluator = ev));
        end

        function id = idOf(~, fcn)
            % The identifier of the error fcn raises, or "" when it does not.
            id = "";
            try
                fcn();
            catch e
                id = string(e.identifier);
            end
        end

        function msg = messageOf(~, fcn)
            msg = "";
            try
                fcn();
            catch e
                msg = string(e.message);
            end
        end

        function verifySameBehavior(testCase, a, b, x, p, tol)
            % Equal right sides at (x, p) and equal end points of a run.
            testCase.verifyEqual(a.Variables, b.Variables);
            testCase.verifyEqual(a.Parameters, b.Parameters);
            testCase.verifyEqual(a.Init, b.Init);
            testCase.verifyEqual(a.rhs(x, p), b.rhs(x, p), ...
                RelTol = tol, AbsTol = tol);
            ta = phasecont.simulate(a, a.Init, [0 5]);
            tb = phasecont.simulate(b, b.Init, [0 5]);
            testCase.verifyEqual(ta.X(end, :), tb.X(end, :), ...
                RelTol = max(tol, 1e-9), AbsTol = max(tol, 1e-9));
        end

        function body = sbml(~, level3Body)
            body = "<?xml version=""1.0"" encoding=""UTF-8""?>" + ...
                "<sbml xmlns=""http://www.sbml.org/sbml/level3/version1/core"" " + ...
                "level=""3"" version=""1""><model id=""m"">" + level3Body + ...
                "</model></sbml>";
        end

        function body = decaySbml(testCase, extra)
            mml = "http://www.w3.org/1998/Math/MathML";
            body = testCase.sbml( ...
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
                "</listOfReactions>" + extra);
        end
    end

    methods (Test)
        % Route A: equation text

        function equationsMatchTheFileModel(testCase, Evaluator)
            ref = testCase.fileModel(testCase.OscFile, Evaluator);
            m = testCase.track(phasecont.Model.fromEquations( ...
                testCase.OscEquations, Params = struct(w = 2, c = 0.1), ...
                Init = struct(x = 1, y = 0), Evaluator = Evaluator));
            testCase.verifySameBehavior(ref, m, [0.3; -0.7], [2; 0.1], 0);
        end

        function equationsAcceptOneStringWithNewlines(testCase)
            m = testCase.track(phasecont.Model.fromEquations( ...
                join(testCase.OscEquations, newline), ...
                Params = struct(w = 2, c = 0.1), Init = struct(x = 1, y = 0)));
            testCase.verifyEqual(m.Variables, ["x" "y"]);
            testCase.verifyEqual(m.ParameterValues, [2; 0.1]);
        end

        function bothDerivativeSpellingsAreRead(testCase)
            m = testCase.track(phasecont.Model.fromEquations( ...
                ["dx/dt = -a*x"; "y' = a*x"], Params = struct(a = 3), ...
                Init = struct(x = 1, y = 0)));
            testCase.verifyEqual(m.rhs([2; 0]), [-6; 6]);
        end

        function aDoneLineStillEndsTheText(testCase)
            m = testCase.track(phasecont.Model.fromEquations( ...
                ["x' = -a*x"; "done"], Params = struct(a = 3), ...
                Init = struct(x = 1)));
            testCase.verifyEqual(m.rhs(2), -6);
        end

        function initialValuesAreOnTheModel(testCase)
            m = testCase.track(phasecont.Model.fromEquations( ...
                testCase.OscEquations, Params = struct(w = 2, c = 0.1), ...
                Init = struct(y = 5, x = -1)));
            % The model orders them as the states, not as the struct.
            testCase.verifyEqual(m.Init, [-1; 5]);
        end

        function aSessionOfAnEquationModelRoundTrips(testCase)
            m = testCase.track(phasecont.Model.fromEquations( ...
                testCase.OscEquations, Params = struct(w = 2, c = 0.1), ...
                Init = struct(x = 1, y = 0)));
            tr = phasecont.simulate(m, m.Init, [0 5]);
            file = fullfile(testCase.Dir, "osc.pcsession.json");
            save(phasecont.Session(m, {tr}), file);
            back = phasecont.Session.load(file);
            testCase.addTeardown(@() delete(back.Model));
            testCase.verifyEqual(back.Model.SourceSha256, m.SourceSha256);
            testCase.verifyEqual(back.Model.Variables, m.Variables);
            testCase.verifyNumElements(back.Items, 1);
            testCase.verifyEqual(back.Items{1}.X(end, :), tr.X(end, :), ...
                RelTol = 1e-8, AbsTol = 1e-8);
        end

        function theSameEquationsGiveTheSameHash(testCase)
            args = {"Params", struct("w", 2, "c", 0.1), "Init", struct("x", 1, "y", 0)};
            a = testCase.track(phasecont.Model.fromEquations( ...
                testCase.OscEquations, args{:}));
            b = testCase.track(phasecont.Model.fromEquations( ...
                testCase.OscEquations, args{:}));
            testCase.verifyEqual(a.SourceSha256, b.SourceSha256);
        end

        function anUndeclaredSymbolIsAParseErrorWithAPosition(testCase)
            fcn = @() phasecont.Model.fromEquations("x' = -k*x", ...
                Init = struct(x = 1), Name = "mine.ode");
            testCase.verifyEqual(testCase.idOf(fcn), "phasecont:engine:model");
            msg = testCase.messageOf(fcn);
            testCase.verifyTrue(contains(msg, "mine.ode:1:"), msg);
            testCase.verifyTrue(contains(msg, "k"), msg);
        end

        function aMissingParameterIsNotDefaultedToZero(testCase)
            % a is used and not given. It must not become 0.
            fcn = @() phasecont.Model.fromEquations(["x' = -a*x"; "y' = x"], ...
                Params = struct(b = 1), Init = struct(x = 1, y = 0));
            testCase.verifyEqual(testCase.idOf(fcn), "phasecont:engine:model");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "a"));
        end

        function aMissingInitialValueIsNamed(testCase)
            fcn = @() phasecont.Model.fromEquations(["x' = y"; "y' = -x"], ...
                Init = struct(x = 1));
            testCase.verifyEqual(testCase.idOf(fcn), "phasecont:model:missingInit");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "y"));
            none = @() phasecont.Model.fromEquations(["x' = y"; "y' = -x"]);
            testCase.verifyEqual(testCase.idOf(none), "phasecont:model:missingInit");
        end

        function anInitNameThatIsNotAStateIsNamed(testCase)
            fcn = @() phasecont.Model.fromEquations("x' = -x", ...
                Init = struct(x = 1, z = 3));
            testCase.verifyEqual(testCase.idOf(fcn), "phasecont:model:unknownInit");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "z"));
        end

        function anInitLineInTheTextIsRefused(testCase)
            for line = ["init x=1" "i x=1" "ic x=1" "  INIT x=1"]
                fcn = @() phasecont.Model.fromEquations( ...
                    ["x' = -x"; line], Init = struct(x = 1));
                testCase.verifyEqual(testCase.idOf(fcn), ...
                    "phasecont:model:initInText", "line: " + line);
            end
        end

        function aStateNamedIIsNotAnInitLine(testCase)
            m = testCase.track(phasecont.Model.fromEquations( ...
                "i' = -i", Init = struct(i = 2)));
            testCase.verifyEqual(m.Variables, "i");
        end

        function badParamsAndInitValuesAreRefused(testCase)
            eq = "x' = -a*x";
            badValue = "phasecont:model:badValue";
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromEquations(eq, Params = struct(a = NaN), ...
                Init = struct(x = 1))), badValue);
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromEquations(eq, Params = struct(a = 1), ...
                Init = struct(x = Inf))), badValue);
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromEquations(eq, Params = struct(a = [1 2]), ...
                Init = struct(x = 1))), badValue);
        end

        % Route B: table or struct

        function aTableModelMatchesTheFileModel(testCase, Evaluator)
            ref = testCase.fileModel(testCase.OscFile, Evaluator);
            states = table(["x"; "y"], ["y"; "-w^2*x - c*y"], [1; 0], ...
                VariableNames = ["Name" "Rhs" "Init"]);
            params = table(["w"; "c"], [2; 0.1], ...
                VariableNames = ["Name" "Value"]);
            m = testCase.track(phasecont.Model.fromTable(States = states, ...
                Params = params, Evaluator = Evaluator));
            testCase.verifySameBehavior(ref, m, [0.3; -0.7], [2; 0.1], 0);
        end

        function aStructModelMatchesTheFileModel(testCase)
            ref = testCase.fileModel(testCase.OscFile);
            m = testCase.track(phasecont.Model.fromTable( ...
                States = struct(x = "y", y = "-w^2*x - c*y"), ...
                Params = struct(w = 2, c = 0.1), Init = struct(x = 1, y = 0)));
            testCase.verifySameBehavior(ref, m, [0.3; -0.7], [2; 0.1], 0);
        end

        function auxRowsAreAccepted(testCase)
            m = testCase.track(phasecont.Model.fromTable( ...
                States = struct(x = "y", y = "-x"), ...
                Aux = struct(energy = "0.5*(x^2 + y^2)"), ...
                Init = struct(x = 1, y = 0)));
            testCase.verifyEqual(m.Variables, ["x" "y"]);
            testCase.verifyEqual(m.rhs([2; 3]), [3; -2]);
        end

        function aConstantRightSideIsWrittenAsANumber(testCase)
            m = testCase.track(phasecont.Model.fromTable( ...
                States = struct(x = 0.25), Init = struct(x = 0)));
            testCase.verifyEqual(m.rhs(7), 0.25);
        end

        function tableErrorsKeepTheirIds(testCase)
            testCase.verifyEqual(testCase.idOf(@() phasecont.Model.fromTable()), ...
                "phasecont:model:badTable");
            states = table("x", "-x", 1, VariableNames = ["Name" "Rhs" "Init"]);
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromTable(States = states, ...
                Init = struct(x = 2))), "phasecont:model:ambiguousInit");
            gap = table(["x"; "y"], ["y"; "-x"], [1; NaN], ...
                VariableNames = ["Name" "Rhs" "Init"]);
            fcn = @() phasecont.Model.fromTable(States = gap);
            testCase.verifyEqual(testCase.idOf(fcn), "phasecont:model:missingInit");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "y"));
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromTable(States = struct(x = "-k*x"), ...
                Init = struct(x = 1))), "phasecont:engine:model");
        end

        % Route C: function handles

        function theOscillatorHandleMatchesTheTextModel(testCase, Evaluator)
            ref = testCase.fileModel(testCase.OscFile, Evaluator);
            h = @(t, x, p) [x(2); -p(1)^2*x(1) - p(2)*x(2)];
            m = testCase.track(phasecont.Model.fromFunction(h, ...
                Init = [1; 0], Params = struct(w = 2, c = 0.1), ...
                StateNames = ["x"; "y"], Evaluator = Evaluator));
            testCase.verifySameBehavior(ref, m, [0.3; -0.7], [2; 0.1], 1e-13);
        end

        function rowVectorsWorkForInitAndStateNames(testCase)
            h = @(t, x) [x(2); -x(1)];
            m = testCase.track(phasecont.Model.fromFunction(h, ...
                Init = [1 0], StateNames = ["u" "v"]));
            testCase.verifyEqual(m.Variables, ["u" "v"]);
            testCase.verifyEqual(m.Init, [1; 0]);
        end

        function stateNamesDefaultToTheStateArgument(testCase)
            m = testCase.track(phasecont.Model.fromFunction( ...
                @(t, x) [x(2); -x(1)], Init = [1; 0]));
            testCase.verifyEqual(m.Variables, ["x1" "x2"]);
        end

        function theMorrisLecarHandleMatchesTheTextModel(testCase)
            ref = testCase.fileModel(testCase.MorrisLecar);
            C = 20; gL = 2; gK = 8; gCa = 4.4; %#ok<NASGU>
            vL = -60; vK = -84; vCa = 120; %#ok<NASGU>
            v1 = -1.2; v2 = 18; v3 = 2; v4 = 30; %#ok<NASGU>
            h = @(t, x, p) [ ...
                (p(1) - gL*(x(1)-vL) - gK*x(2)*(x(1)-vK) ...
                - gCa*(0.5*(1+tanh((x(1)-v1)/v2)))*(x(1)-vCa))/C; ...
                p(2)*((0.5*(1+tanh((x(1)-v3)/v4))) - x(2)) ...
                /(1/cosh((x(1)-v3)/(2*v4)))];
            m = testCase.track(phasecont.Model.fromFunction(h, ...
                Init = [-20; 0.1], Params = struct(I = 100, phi = 0.04), ...
                StateNames = ["v"; "w"]));
            x = [-15; 0.2];
            testCase.verifyEqual(m.rhs(x, [100; 0.04]), ...
                ref.rhs(x, ref.ParameterValues), RelTol = 1e-12, AbsTol = 1e-12);
            x = [10; 0.5];
            testCase.verifyEqual(m.rhs(x, [90; 0.05]), ...
                ref.rhs(x, [90; 0.05; ref.ParameterValues(3:end)]), ...
                RelTol = 1e-12, AbsTol = 1e-12);
        end

        function theLorenzHandleWithCapturedConstantsMatches(testCase)
            ref = testCase.fileModel(testCase.Lorenz);
            sigma = 10; rho = 28; beta = 8/3; %#ok<NASGU>
            h = @(t, x) [sigma*(x(2)-x(1)); x(1)*(rho-x(3))-x(2); ...
                x(1)*x(2)-beta*x(3)];
            m = testCase.track(phasecont.Model.fromFunction(h, ...
                Init = [1; 1; 1], StateNames = ["x"; "y"; "z"]));
            testCase.verifyEmpty(m.Parameters);
            x = [1.2; -0.4; 22];
            testCase.verifyEqual(m.rhs(x), ref.rhs(x), RelTol = 1e-13, AbsTol = 1e-13);
        end

        function aCapturedVectorIsIndexed(testCase)
            g = [1 2 3]; %#ok<NASGU>
            m = testCase.track(phasecont.Model.fromFunction( ...
                @(t, x) -g(2)*x(1), Init = 1, StateNames = "x"));
            testCase.verifyEqual(m.rhs(1.5), -3);
        end

        function timeReachesTheRightSide(testCase)
            m = testCase.track(phasecont.Model.fromFunction( ...
                @(t, x) -x(1) + t, Init = 0, StateNames = "x"));
            testCase.verifyEqual(m.rhs(2, [], 5), 3);
        end

        function aHandleWithoutInitIsRefusedByName(testCase)
            fcn = @() phasecont.Model.fromFunction(@(t, x) -x(1), ...
                StateNames = "x");
            testCase.verifyEqual(testCase.idOf(fcn), ...
                "phasecont:import:missingValue");
        end

        function anUnknownFunctionIsRefusedByName(testCase)
            fcn = @() phasecont.Model.fromFunction( ...
                @(t, x) frobnicate(x(1)), Init = 1, StateNames = "x");
            testCase.verifyEqual(testCase.idOf(fcn), ...
                "phasecont:import:unsupported");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "frobnicate"));
        end

        function aCapturedMatrixIsNamedInTheError(testCase)
            A = [1 2; 3 4]; %#ok<NASGU>
            fcn = @() phasecont.Model.fromFunction(@(t, x) A*x, ...
                Init = [1; 1], StateNames = ["a"; "b"]);
            testCase.verifyEqual(testCase.idOf(fcn), ...
                "phasecont:import:undeclared");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "A"));
        end

        function aHandleToAFunctionWithoutAFileIsRefused(testCase)
            fcn = @() phasecont.Model.fromFunction(@sin, Init = 1);
            testCase.verifyEqual(testCase.idOf(fcn), ...
                "phasecont:import:unsupported");
        end

        function aParameterWithoutAValueIsRefused(testCase)
            fcn = @() phasecont.Model.fromFunction( ...
                @(t, x, p) -p(1)*x(1), Init = 1, StateNames = "x");
            id = testCase.idOf(fcn);
            testCase.verifyTrue(startsWith(id, "phasecont:import:"), id);
        end

        function aBadParamValueIsRefused(testCase)
            fcn = @() phasecont.Model.fromFunction(@(t, x, p) -p(1)*x(1), ...
                Init = 1, StateNames = "x", Params = struct(k = NaN));
            testCase.verifyEqual(testCase.idOf(fcn), "phasecont:model:badValue");
        end

        % Route D: files

        function aFunctionFileWithLocalsAndIndexedAssignment(testCase)
            f = testCase.write("pend_rhs.m", ...
                "function dydt = pend_rhs(t, y, p)" + newline + ...
                "% damped pendulum" + newline + ...
                "th = y(1);" + newline + "om = y(2);" + newline + ...
                "dydt = zeros(2, 1);" + newline + ...
                "dydt(1) = om;" + newline + ...
                "dydt(2) = -sin(th) - p(1)*om;" + newline + "end" + newline);
            m = testCase.track(phasecont.Model.fromFile(f, ...
                Init = [0.1; 0], Params = struct(damp = 0.2), ...
                StateNames = ["th"; "om"]));
            testCase.verifyEqual(m.Variables, ["th" "om"]);
            x = [0.4; 0.9];
            testCase.verifyEqual(m.rhs(x, 0.2), ...
                [x(2); -sin(x(1)) - 0.2*x(2)], RelTol = 1e-13);
        end

        function aFunctionFileWithALoopIsRefusedWithAPosition(testCase)
            f = testCase.write("loopy.m", ...
                "function dydt = loopy(t, y)" + newline + ...
                "dydt = zeros(2, 1);" + newline + ...
                "for k = 1:2" + newline + ...
                "    dydt(k) = -y(k);" + newline + "end" + newline + ...
                "end" + newline);
            fcn = @() phasecont.Model.fromFile(f, Init = [1; 1], ...
                StateNames = ["a"; "b"]);
            testCase.verifyEqual(testCase.idOf(fcn), ...
                "phasecont:import:unsupported");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "line"));
        end

        function anOdeFileReadsAsPhasecontModel(testCase)
            f = testCase.write("osc.ode", testCase.OscFile);
            a = testCase.track(phasecont.Model.fromFile(f));
            b = testCase.track(phasecont.Model(f));
            testCase.verifyEqual(a.SourceSha256, b.SourceSha256);
        end

        function sbmlDecayMatchesTheClosedForm(testCase, Evaluator)
            for ext = [".xml" ".sbml"]
                f = testCase.write("decay" + ext, testCase.decaySbml(""));
                m = testCase.track(phasecont.Model.fromFile(f, Evaluator = Evaluator));
                testCase.verifyEqual(m.Variables, "S");
                testCase.verifyEqual(m.Init, 4);
                tr = phasecont.simulate(m, m.Init, [0 2], ...
                    RelTol = 1e-10, AbsTol = 1e-12);
                testCase.verifyEqual(tr.X(end, 1), 4*exp(-1), RelTol = 1e-7);
            end
        end

        function sbmlMichaelisMentenKeepsMassAndTheImplicitSolution(testCase)
            mml = "http://www.w3.org/1998/Math/MathML";
            body = testCase.sbml( ...
                "<listOfCompartments><compartment id=""V"" size=""1""/>" + ...
                "</listOfCompartments><listOfSpecies>" + ...
                "<species id=""S"" compartment=""V"" initialConcentration=""5""/>" + ...
                "<species id=""P"" compartment=""V"" initialConcentration=""0""/>" + ...
                "</listOfSpecies><listOfParameters>" + ...
                "<parameter id=""vmax"" value=""2""/>" + ...
                "<parameter id=""km"" value=""0.7""/></listOfParameters>" + ...
                "<listOfReactions><reaction id=""conv"">" + ...
                "<listOfReactants><speciesReference species=""S"" " + ...
                "stoichiometry=""1""/></listOfReactants><listOfProducts>" + ...
                "<speciesReference species=""P"" stoichiometry=""1""/>" + ...
                "</listOfProducts><kineticLaw><math xmlns=""" + mml + """>" + ...
                "<apply><divide/><apply><times/><ci>vmax</ci><ci>S</ci></apply>" + ...
                "<apply><plus/><ci>km</ci><ci>S</ci></apply></apply>" + ...
                "</math></kineticLaw></reaction></listOfReactions>");
            f = testCase.write("mm.xml", body);
            m = testCase.track(phasecont.Model.fromFile(f));
            iS = find(m.Variables == "S");
            iP = find(m.Variables == "P");
            testCase.verifyNotEmpty(iS);
            testCase.verifyNotEmpty(iP);
            tr = phasecont.simulate(m, m.Init, [0 1], RelTol = 1e-10, AbsTol = 1e-12);
            S = tr.X(end, iS);
            P = tr.X(end, iP);
            testCase.verifyEqual(S + P, 5, AbsTol = 1e-9);
            % Integrated rate law: km*log(S0/S) + (S0 - S) = vmax*t.
            testCase.verifyEqual(0.7*log(5/S) + (5 - S), 2*1, AbsTol = 1e-6);
        end

        function sbmlEventsAreRefusedByName(testCase)
            body = testCase.decaySbml( ...
                "<listOfEvents><event id=""e1""/></listOfEvents>");
            f = testCase.write("events.xml", body);
            fcn = @() phasecont.Model.fromFile(f);
            testCase.verifyEqual(testCase.idOf(fcn), ...
                "phasecont:import:unsupported");
            testCase.verifyTrue(contains(testCase.messageOf(fcn), "event"));
        end

        function sbmlNotXmlIsAnXmlError(testCase)
            f = testCase.write("broken.xml", "<sbml><model");
            testCase.verifyEqual(testCase.idOf(@() phasecont.Model.fromFile(f)), ...
                "phasecont:import:xml");
        end

        function fromFileChecksItsOptions(testCase)
            bad = "phasecont:model:badOption";
            ode = testCase.write("osc.ode", testCase.OscFile);
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromFile(ode, Init = [1; 0])), bad);
            xml = testCase.write("decay.xml", testCase.decaySbml(""));
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromFile(xml, Params = struct(k = 1))), bad);
            txt = testCase.write("model.txt", "x' = -x");
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromFile(txt)), bad);
        end

        function aMissingFileIsNamed(testCase)
            gone = fullfile(testCase.Dir, "gone");
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromFile(gone + ".m", Init = 1)), ...
                "phasecont:import:file");
            testCase.verifyEqual(testCase.idOf(@() ...
                phasecont.Model.fromFile(gone + ".xml")), ...
                "phasecont:import:file");
        end
    end
end
