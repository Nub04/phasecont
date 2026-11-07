classdef ModelInputHelperTest < matlab.unittest.TestCase
    % Tests the model-input helpers that need no engine: odeText, num2ode,
    % tableToEquations and readSbml. These run without pckern.

    properties
        Dir
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
            t = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Dir = t.Folder;
        end
    end

    methods
        function f = writeBytes(testCase, name, bytes)
            f = fullfile(testCase.Dir, name);
            fid = fopen(f, "w");
            fwrite(fid, uint8(bytes), "uint8");
            fclose(fid);
        end
    end

    methods (Test)
        function num2odeRoundTrips(testCase)
            values = [0 1 -1 0.1 1/3 2/3 1e-5 1e300 -2.5e-300 pi ...
                2.6666666666666665 realmin 5e-324 realmax];
            for v = values
                s = phasecont.internal.num2ode(v);
                testCase.verifyEqual(str2double(s), v, ...
                    "value " + sprintf("%.17g", v), AbsTol = 0);
            end
        end

        function num2odeUsesFewDigits(testCase)
            testCase.verifyEqual(phasecont.internal.num2ode(0.1), "0.1");
            testCase.verifyEqual(phasecont.internal.num2ode(28), "28");
            testCase.verifyEqual(phasecont.internal.num2ode(-1.5), "-1.5");
        end

        function num2odeRefusesNonFinite(testCase)
            bad = "phasecont:model:badValue";
            testCase.verifyError(@() phasecont.internal.num2ode(NaN), bad);
            testCase.verifyError(@() phasecont.internal.num2ode(Inf), bad);
            testCase.verifyError(@() phasecont.internal.num2ode(-Inf), bad);
        end

        function odeTextAppendsParAndInitAfterTheEquations(testCase)
            t = phasecont.internal.odeText("x' = -a*x", struct(a = 2), ...
                struct(x = 1));
            testCase.verifyEqual(t, "x' = -a*x" + newline + "par a=2" + ...
                newline + "init x=1" + newline);
        end

        function odeTextKeepsTheEquationLineNumbers(testCase)
            eq = "x' = y" + newline + "y' = -x";
            t = phasecont.internal.odeText(eq, struct(), struct(x = 1, y = 0));
            lines = splitlines(t);
            testCase.verifyEqual(lines(1), "x' = y");
            testCase.verifyEqual(lines(2), "y' = -x");
        end

        function odeTextPutsTheBlocksBeforeDone(testCase)
            eq = "x' = -a*x" + newline + "done" + newline;
            t = phasecont.internal.odeText(eq, struct(a = 3), struct(x = 1));
            lines = splitlines(t);
            lines(end) = [];
            testCase.verifyEqual(lines(end), "done");
            testCase.verifyEqual(lines(end-1), "init x=1");
            testCase.verifyEqual(lines(end-2), "par a=3");
        end

        function odeTextBreaksLongBlocksIntoLinesOfEight(testCase)
            p = struct();
            for k = 1:10
                p.("p" + k) = k;
            end
            t = phasecont.internal.odeText("x' = p1*x", p, struct(x = 1));
            lines = splitlines(t);
            parLines = lines(startsWith(lines, "par "));
            testCase.verifyNumElements(parLines, 2);
            testCase.verifyEqual(count(parLines(1), "="), 8);
            testCase.verifyEqual(count(parLines(2), "="), 2);
        end

        function odeTextWithNoStructsAddsNothing(testCase)
            t = phasecont.internal.odeText("x' = -x", struct(), struct());
            testCase.verifyEqual(t, "x' = -x" + newline);
        end

        function odeTextHandlesWindowsLineEndings(testCase)
            eq = "x' = y" + sprintf("\r\n") + "y' = -x" + sprintf("\r\n");
            t = phasecont.internal.odeText(eq, struct(), struct(x = 1, y = 0));
            testCase.verifyFalse(contains(t, sprintf("\r")));
            testCase.verifyEqual(extractBefore(t, "init"), ...
                "x' = y" + newline + "y' = -x" + newline);
        end

        function odeTextChecksValues(testCase)
            badValue = "phasecont:model:badValue";
            f = @(p, x) phasecont.internal.odeText("x' = -x", p, x);
            testCase.verifyError(@() f(struct(a = NaN), struct(x = 1)), badValue);
            testCase.verifyError(@() f(struct(a = Inf), struct(x = 1)), badValue);
            testCase.verifyError(@() f(struct(a = [1 2]), struct(x = 1)), badValue);
            testCase.verifyError(@() f(struct(a = "2"), struct(x = 1)), badValue);
            testCase.verifyError(@() f(struct(a = 1i), struct(x = 1)), badValue);
            testCase.verifyError(@() f(struct(), struct(x = NaN)), badValue);
        end

        function odeTextKeepsUnderscoresInNames(testCase)
            p = struct();
            p.a_b = 1;
            t = phasecont.internal.odeText("x' = -a_b*x", p, struct(x = 1));
            testCase.verifyTrue(contains(t, "par a_b=1"));
        end

        function tableToEquationsFromAStruct(testCase)
            [eq, p, x0] = phasecont.internal.tableToEquations( ...
                struct(x = "y", y = "-x - c*y"), struct(c = 0.1), ...
                struct(x = 1, y = 0));
            testCase.verifyEqual(eq, "x' = y" + newline + "y' = -x - c*y");
            testCase.verifyEqual(p, struct(c = 0.1));
            testCase.verifyEqual(x0, struct(x = 1, y = 0));
        end

        function tableToEquationsFromTables(testCase)
            states = table(["x"; "y"], ["y"; "-x"], [1; 0], ...
                VariableNames = ["Name" "Rhs" "Init"]);
            params = table("c", 0.5, VariableNames = ["Name" "Value"]);
            aux = table("e", "x^2 + y^2", VariableNames = ["Name" "Expr"]);
            [eq, p, x0] = phasecont.internal.tableToEquations(states, params, ...
                struct(), aux);
            testCase.verifyEqual(eq, "x' = y" + newline + "y' = -x" + ...
                newline + "aux e=x^2 + y^2");
            testCase.verifyEqual(p, struct(c = 0.5));
            testCase.verifyEqual(x0, struct(x = 1, y = 0));
        end

        function tableToEquationsReadsNumbersAsConstants(testCase)
            eq = phasecont.internal.tableToEquations(struct(x = 0.25));
            testCase.verifyEqual(eq, "x' = 0.25");
        end

        function aNanInitCellMeansNoValue(testCase)
            states = table(["x"; "y"], ["y"; "-x"], [1; NaN], ...
                VariableNames = ["Name" "Rhs" "Init"]);
            [~, ~, x0] = phasecont.internal.tableToEquations(states);
            testCase.verifyEqual(x0, struct(x = 1));
        end

        function initGivenTwiceIsAmbiguous(testCase)
            states = table("x", "-x", 1, VariableNames = ["Name" "Rhs" "Init"]);
            testCase.verifyError(@() phasecont.internal.tableToEquations( ...
                states, struct(), struct(x = 2)), ...
                "phasecont:model:ambiguousInit");
        end

        function tableToEquationsChecksItsInput(testCase)
            f = @phasecont.internal.tableToEquations;
            testCase.verifyError(@() f(struct()), "phasecont:model:badTable");
            testCase.verifyError(@() f(42), "phasecont:model:badTable");
            testCase.verifyError(@() f(table("x", VariableNames = "Name")), ...
                "phasecont:model:badTable");
            badName = table("1x", "1", VariableNames = ["Name" "Rhs"]);
            testCase.verifyError(@() f(badName), "phasecont:model:badName");
            spaced = table("a b", "1", VariableNames = ["Name" "Rhs"]);
            testCase.verifyError(@() f(spaced), "phasecont:model:badName");
            dup = table(["x"; "x"], ["1"; "2"], VariableNames = ["Name" "Rhs"]);
            testCase.verifyError(@() f(dup), "phasecont:model:duplicateName");
            testCase.verifyError(@() f(struct(x = "")), "phasecont:model:badValue");
            testCase.verifyError(@() f(struct(x = NaN)), "phasecont:model:badValue");
            testCase.verifyError(@() f(struct(x = "a" + newline + "b")), ...
                "phasecont:model:badValue");
            testCase.verifyError(@() f(struct(x = "-x"), table("c", NaN, ...
                VariableNames = ["Name" "Value"])), "phasecont:model:missingValue");
        end

        function readSbmlReadsUtf8(testCase)
            f = testCase.writeBytes("a.xml", uint8('<?xml version="1.0"?><a>b</a>'));
            testCase.verifyEqual(phasecont.internal.readSbml(f), ...
                "<?xml version=""1.0""?><a>b</a>");
        end

        function readSbmlStripsAUtf8Mark(testCase)
            % U+00E9 is the two bytes 195 169 in UTF-8.
            bytes = [239 187 191 uint8('<a>') 195 169 uint8('</a>')];
            f = testCase.writeBytes("bom.xml", bytes);
            testCase.verifyEqual(phasecont.internal.readSbml(f), ...
                "<a>" + string(char(233)) + "</a>");
        end

        function readSbmlReadsUtf16(testCase)
            text = '<?xml version="1.0" encoding="UTF-16"?><a>x</a>';
            u = double(text);
            le = reshape([mod(u, 256); floor(u / 256)], 1, []);
            be = reshape([floor(u / 256); mod(u, 256)], 1, []);
            f1 = testCase.writeBytes("le.xml", [255 254 le]);
            f2 = testCase.writeBytes("be.xml", [254 255 be]);
            testCase.verifyEqual(phasecont.internal.readSbml(f1), string(text));
            testCase.verifyEqual(phasecont.internal.readSbml(f2), string(text));
        end

        function readSbmlReadsADeclaredLatin1File(testCase)
            bytes = [uint8('<?xml version="1.0" encoding="ISO-8859-1"?><a>') ...
                233 uint8('</a>')];
            f = testCase.writeBytes("latin.xml", bytes);
            t = phasecont.internal.readSbml(f);
            testCase.verifyTrue(endsWith(t, "<a>" + string(char(233)) + "</a>"));
        end

        function readSbmlRefusesBadBytes(testCase)
            f = testCase.writeBytes("bad.xml", [uint8('<a>') 255 254 253 uint8('</a>')]);
            testCase.verifyError(@() phasecont.internal.readSbml(f), ...
                "phasecont:import:xml");
            odd = testCase.writeBytes("odd.xml", [255 254 60 0 97]);
            testCase.verifyError(@() phasecont.internal.readSbml(odd), ...
                "phasecont:import:xml");
        end

        function readSbmlReportsAMissingFile(testCase)
            testCase.verifyError(@() phasecont.internal.readSbml( ...
                fullfile(testCase.Dir, "absent.xml")), "phasecont:import:file");
        end
    end
end
