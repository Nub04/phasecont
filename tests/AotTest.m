classdef AotTest < matlab.unittest.TestCase
    % Tests the AOT route: phasecont.config, Evaluator = "aot", AotInfo,
    % the library cache and the fallback to the JIT.

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        Fhn = "par a=0.7,b=0.8,e=0.08,drive=0" + newline + ...
            "init v=-1.2,w=-0.6" + newline + ...
            "v'=v-v^3/3-w+drive" + newline + ...
            "w'=e*(v+a-b*w)" + newline + "done" + newline
        Henon = "par a=1.4, b=0.3" + newline + "init x=0, y=0" + newline + ...
            "x(t+1)=1-a*x^2+y" + newline + "y(t+1)=b*x" + newline + ...
            "done" + newline
    end

    properties
        Dir
        Compiler
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

    methods (TestMethodSetup)
        function resetState(testCase)
            old = phasecont.config("Codegen");
            testCase.addTeardown(@() phasecont.config("Codegen", old));
            testCase.addTeardown(@() phasecont.internal.aotHook([]));
            phasecont.config("Codegen", "jit");
            phasecont.internal.aotHook([]);
        end
    end

    methods
        function f = write(testCase, name, text)
            f = fullfile(testCase.Dir, name);
            fid = fopen(f, "w", "n", "UTF-8");
            fwrite(fid, char(text), "char");
            fclose(fid);
        end

        function needCompiler(testCase)
            testCase.assumeNotEmpty( ...
                mex.getCompilerConfigurations("C", "Installed"), ...
                "no mex C compiler is installed");
            testCase.assumeNotEmpty( ...
                mex.getCompilerConfigurations("C", "Selected"), ...
                "no mex C compiler is selected (run mex -setup C)");
        end

        function verifyUlp(testCase, actual, expected, maxUlp)
            testCase.verifySize(actual, size(expected));
            nanA = isnan(actual);
            testCase.verifyEqual(nanA, isnan(expected));
            d = abs(actual(~nanA) - expected(~nanA));
            tol = maxUlp * eps(abs(expected(~nanA)));
            testCase.verifyTrue(all(d <= tol), ...
                sprintf("difference exceeds %d ulp", maxUlp));
        end
    end

    methods (Test)
        function configDefaultsAndSets(testCase)
            testCase.verifyEqual(phasecont.config("Codegen"), "jit");
            phasecont.config("Codegen", "aot");
            testCase.verifyEqual(phasecont.config("Codegen"), "aot");
            phasecont.config("Codegen", "jit");
            testCase.verifyEqual(phasecont.config("Codegen"), "jit");
        end

        function configRejectsBadNameAndValue(testCase)
            testCase.verifyError(@() phasecont.config("Nope"), ...
                "phasecont:config:badName");
            testCase.verifyError(@() phasecont.config("Nope", "jit"), ...
                "phasecont:config:badName");
            testCase.verifyError(@() phasecont.config("Codegen", "gpu"), ...
                "phasecont:config:badValue");
            testCase.verifyError(@() phasecont.config("Codegen", 3), ...
                "phasecont:config:badValue");
            testCase.verifyEqual(phasecont.config("Codegen"), "jit");
        end

        function jitModelHasEmptyAotInfo(testCase)
            m = phasecont.Model(testCase.write("l.ode", testCase.Lorenz));
            testCase.verifyEqual(m.Evaluator, "jit");
            testCase.verifyEqual(string(fieldnames(m.AotInfo))', ...
                ["Compiler" "Version" "Flags" "Library"]);
            testCase.verifyEqual(m.AotInfo.Compiler, "");
            testCase.verifyEqual(m.AotInfo.Version, "");
            testCase.verifyEqual(m.AotInfo.Flags, "");
            testCase.verifyEqual(m.AotInfo.Library, "");
        end

        function aotRhsMatchesJitWithin4Ulp(testCase)
            testCase.needCompiler();
            xs = {[1.3 -0.7 2.1], [-4.2 0.5 17.25], [0 0 0], [1e-3 -2e2 3.5]};
            fl = testCase.write("l.ode", testCase.Lorenz);
            j = phasecont.Model(fl);
            a = phasecont.Model(fl, Evaluator = "aot");
            testCase.assertEqual(a.Evaluator, "aot");
            for k = 1:numel(xs)
                testCase.verifyUlp(a.rhs(xs{k}), j.rhs(xs{k}), 4);
            end
            ff = testCase.write("f.ode", testCase.Fhn);
            j = phasecont.Model(ff);
            a = phasecont.Model(ff, Evaluator = "aot");
            testCase.assertEqual(a.Evaluator, "aot");
            for x = {[-1.2 -0.6], [0.3 0.9], [2.5 -1.1]}
                testCase.verifyUlp(a.rhs(x{1}), j.rhs(x{1}), 4);
                testCase.verifyUlp(a.rhs(x{1}, [0.7 0.8 0.08 0.4]), ...
                    j.rhs(x{1}, [0.7 0.8 0.08 0.4]), 4);
            end
        end

        function aotClassifyMatchesJit(testCase)
            % Point.classify through model_jacobian on an AOT model (SPEC 8ac).
            testCase.needCompiler();
            fl = testCase.write("lc.ode", testCase.Lorenz);
            j = phasecont.Model(fl);
            a = phasecont.Model(fl, Evaluator = "aot");
            testCase.assertEqual(a.Evaluator, "aot");
            u = [0.5; -0.25; 2];
            cj = classify(phasecont.Point(U = u, P = j.ParameterValues, Label = "EP", Model = j));
            ca = classify(phasecont.Point(U = u, P = a.ParameterValues, Label = "EP", Model = a));
            testCase.verifyUlp(ca.Jacobian(:).', cj.Jacobian(:).', 4);
            testCase.verifyEqual(ca.Type, cj.Type);
        end

        function aotBranchMatchesJit(testCase)
            testCase.needCompiler();
            ff = testCase.write("f.ode", testCase.Fhn);
            br = cell(1, 2);
            kinds = ["jit" "aot"];
            for k = 1:2
                m = phasecont.Model(ff, Evaluator = kinds(k));
                eq = phasecont.equilibrium(m, [-1.2 -0.6]);
                br{k} = phasecont.continuation(m, eq, Free = "drive", ...
                    Direction = 1, MaxSteps = 60, Bounds = [-1 2]);
            end
            testCase.assertEqual(br{2}.Model.Evaluator, "aot");
            testCase.assertEqual(numel(br{2}.Alpha), numel(br{1}.Alpha));
            testCase.verifyEqual(br{2}.Alpha, br{1}.Alpha, RelTol = 1e-9, AbsTol = 1e-12);
            testCase.verifyEqual(br{2}.U, br{1}.U, RelTol = 1e-9, AbsTol = 1e-12);
            testCase.verifyEqual(br{2}.Stable, br{1}.Stable);
            testCase.verifyEqual(numel(br{2}.points()), numel(br{1}.points()));
        end

        function aotInfoHasCompilerVersionAndFlags(testCase)
            testCase.needCompiler();
            m = phasecont.Model(testCase.write("f.ode", testCase.Fhn), ...
                Evaluator = "aot");
            i = m.AotInfo;
            testCase.verifyNotEqual(strlength(i.Compiler), 0);
            testCase.verifyNotEqual(strlength(i.Version), 0);
            testCase.verifyNotEqual(strlength(i.Flags), 0);
            testCase.verifyTrue(isfile(i.Library));
            if contains(i.Compiler, "Microsoft", IgnoreCase = true)
                testCase.verifySubstring(i.Flags, "/fp:strict");
            else
                testCase.verifySubstring(i.Flags, "-ffp-contract=off");
            end
        end

        function secondModelReusesTheLibrary(testCase)
            testCase.needCompiler();
            ff = testCase.write("f.ode", testCase.Fhn);
            a = phasecont.Model(ff, Evaluator = "aot");
            lib = a.AotInfo.Library;
            d1 = dir(lib);
            pause(1.1);
            b = phasecont.Model(ff, Evaluator = "aot");
            d2 = dir(b.AotInfo.Library);
            testCase.verifyEqual(b.AotInfo.Library, lib);
            testCase.verifyEqual(d2.datenum, d1.datenum);
            testCase.verifyEqual(d2.bytes, d1.bytes);
            testCase.verifyEqual(b.rhs([0.3 0.9]), a.rhs([0.3 0.9]));
        end

        function libraryFolderIsNamedFromTheCFileHash(testCase)
            % Without a hook the C text is the file the engine wrote, whose
            % name carries a hash of that text, so the MATLAB side reads the
            % hash from the name instead of hashing the text again.
            testCase.needCompiler();
            m = phasecont.Model(testCase.write("f.ode", testCase.Fhn));
            lib = phasecont.internal.aotBuild(m.Id);
            cfile = string(jsondecode(pckern("model_info", m.Id)).c_file);
            hash = extractBetween(cfile, "pc_model_", ".c");
            testCase.assertNotEmpty(hash);
            testCase.verifySubstring(lib, hash);
            testCase.verifyTrue(isfile(lib));
        end

        function mapCurveAtIterate2RunsOnAot(testCase)
            % The k-th iterate of a map asks for the directional derivative
            % of the whole map, so the PD curve of the Henon 2-cycle needs
            % that entry in the library. It runs on AOT and agrees with the
            % JIT.
            testCase.needCompiler();
            fh = testCase.write("h.ode", testCase.Henon);
            runs = cell(1, 2);
            evs = ["aot" "jit"];
            for k = 1:2
                m = phasecont.Model(fh, Evaluator = evs(k));
                testCase.assertEqual(m.Evaluator, evs(k));
                b = 0.3;
                x = (-(1 - b) + sqrt((1 - b)^2 + 0.8)) / 0.4;
                eq = phasecont.equilibrium(m, [x b*x], Params = struct(a = 0.2, b = b));
                brs = phasecont.continuation(m, eq, Free = "a", Direction = 1, ...
                    Hmax = 0.05, MaxSteps = 1000, Bounds = [0.2 1.2], Switch = "PD1");
                testCase.assertSize(brs, [1 2]);
                pd2 = brs(2).points("PD");
                testCase.assertSize(pd2, [1 1]);
                runs{k} = testCase.verifyWarningFree(@() phasecont.continuation( ...
                    m, pd2, Type = "PD", Free = ["a" "b"], Bounds = [0.1 0.6]));
            end
            [a, j] = runs{:};
            testCase.verifyEqual(a.Iterate, 2);
            testCase.verifyGreaterThan(size(a.Alpha, 1), 5);
            testCase.assertSize(a.Alpha, size(j.Alpha));
            testCase.verifyEqual(a.Alpha, j.Alpha, AbsTol = 1e-9);
            testCase.verifyEqual(a.U, j.U, AbsTol = 1e-9);
        end

        function hookedTextGetsItsOwnLibrary(testCase)
            % The hook changes the text that is compiled, so the library must
            % not be the one built for the unchanged text.
            testCase.needCompiler();
            ff = testCase.write("f.ode", testCase.Fhn);
            plain = phasecont.Model(ff, Evaluator = "aot");
            phasecont.internal.aotHook(@(text) text + newline + "/* hooked */" + newline);
            hooked = phasecont.Model(ff, Evaluator = "aot");
            testCase.assertEqual(hooked.Evaluator, "aot");
            testCase.verifyNotEqual(hooked.AotInfo.Library, plain.AotInfo.Library);
            testCase.verifyEqual(hooked.rhs([0.3 0.9]), plain.rhs([0.3 0.9]));
        end

        function configAotSelectsAotWithoutEvaluator(testCase)
            testCase.needCompiler();
            phasecont.config("Codegen", "aot");
            m = phasecont.Model(testCase.write("f.ode", testCase.Fhn));
            testCase.verifyEqual(m.Evaluator, "aot");
            m2 = phasecont.Model(testCase.write("f.ode", testCase.Fhn), ...
                Evaluator = "jit");
            testCase.verifyEqual(m2.Evaluator, "jit");
        end

        function brokenCompileWarnsAndFallsBackToJit(testCase)
            testCase.needCompiler();
            phasecont.internal.aotHook(@(text) text + newline + "#error broken" + newline);
            ff = testCase.write("f.ode", testCase.Fhn);
            m = testCase.verifyWarning( ...
                @() phasecont.Model(ff, Evaluator = "aot"), "phasecont:aot:compile");
            testCase.verifyEqual(m.Evaluator, "jit");
            testCase.verifyEqual(m.AotInfo.Compiler, "");
            testCase.verifyEqual(m.AotInfo.Flags, "");
            phasecont.internal.aotHook([]);
            j = phasecont.Model(ff);
            testCase.verifyEqual(m.rhs([0.3 0.9]), j.rhs([0.3 0.9]));
        end

        function damagedCachedLibraryIsRebuilt(testCase)
            % A cache hit trusts the file, so a library that a killed build
            % left half written would otherwise warn on every load. The
            % loader removes the cache folder and builds once more.
            testCase.needCompiler();
            ff = testCase.write("f.ode", testCase.Fhn);
            tag = "/* damaged-cache-" + string(randi(1e9)) + " */";
            phasecont.internal.aotHook(@(text) text + newline + tag + newline);
            j = phasecont.Model(ff);
            lib = phasecont.internal.aotBuild(j.Id);
            testCase.addTeardown(@() rmdir(fileparts(lib), "s"));
            fid = fopen(lib, "w");
            fwrite(fid, "not a library", "char");
            fclose(fid);
            lastwarn("");
            m = phasecont.Model(ff, Evaluator = "aot");
            [msg, id] = lastwarn;
            testCase.verifyEmpty(id, "unexpected warning: " + string(msg));
            testCase.assertEqual(m.Evaluator, "aot");
            testCase.verifyEqual(m.AotInfo.Library, lib);
            testCase.verifyGreaterThan(dir(lib).bytes, 100);
            testCase.verifyEqual(m.rhs([0.3 0.9]), j.rhs([0.3 0.9]));
        end

        function unloadableLibraryWarnsAndFallsBackToJit(testCase)
            testCase.needCompiler();
            % A text file named like a library: the build step is bypassed
            % through the hook, which replaces the C text by a valid file
            % that exports nothing the loader needs.
            phasecont.internal.aotHook(@(text) "int not_a_model(void){return 0;}" + newline);
            ff = testCase.write("f.ode", testCase.Fhn);
            m = testCase.verifyWarning( ...
                @() phasecont.Model(ff, Evaluator = "aot"), "phasecont:aot:load");
            testCase.verifyEqual(m.Evaluator, "jit");
            testCase.verifyEqual(m.AotInfo.Compiler, "");
            testCase.verifyEqual(m.rhs([0.3 0.9]), ...
                phasecont.Model(ff).rhs([0.3 0.9]));
        end
    end
end
