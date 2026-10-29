classdef GatewayTest < matlab.unittest.TestCase
    % Tests the pckern MEX gateway: loading, error mapping, models and runs.
    % The suite is filtered (not failed) when the gateway has not been built.

    properties (Constant)
        Lorenz = "par s=10, r=28, b=2.6666666666666665" + newline + ...
            "init x=1,y=1,z=1" + newline + ...
            "x'=s*(y-x)" + newline + ...
            "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "done" + newline
        P = [10 28 2.6666666666666665]
    end

    properties
        Bin
    end

    methods (TestClassSetup)
        function addGatewayToPath(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "tests")));
            testCase.Bin = fixtures.binFolder(root);
            testCase.assumeTrue( ...
                isfile(fullfile(testCase.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(char(testCase.Bin)));
        end
    end

    methods (Test)
        function versionNamesAbiAndLibrary(testCase)
            [v, lib] = pckern("version");
            testCase.verifyTrue(startsWith(v, "pckern "));
            testCase.verifySubstring(v, "abi 1");
            if ispc
                name = "pckern.dll";
            elseif ismac
                name = "libpckern.dylib";
            else
                name = "libpckern.so";
            end
            expected = fullfile(testCase.Bin, name);
            if ispc
                testCase.verifyEqual(lower(string(lib)), lower(string(expected)));
            else
                testCase.verifyEqual(string(lib), string(expected));
            end
            testCase.verifyTrue(isfile(lib));
        end

        function panicRaisesAndTheNextCallWorks(testCase)
            try
                pckern("panic");
                testCase.verifyFail("pckern(""panic"") did not raise");
            catch err
                testCase.verifyEqual(string(err.identifier), "phasecont:engine:panic");
                testCase.verifySubstring(err.message, ".rs:");
            end
            testCase.verifyTrue(startsWith(pckern("version"), "pckern "));
        end

        function panicInWorkerReachesPoll(testCase)
            r = pckern("panic_in_worker");
            testCase.verifyClass(r, "uint64");
            cleanup = onCleanup(@() pckern("free", r));
            t0 = tic;
            id = "";
            while toc(t0) < 10
                try
                    pckern("run_poll", r);
                catch err
                    id = string(err.identifier);
                    break
                end
                pause(0.01);
            end
            testCase.verifyEqual(id, "phasecont:engine:panic");
            clear cleanup
        end

        function handleChecks(testCase)
            testCase.verifyError(@() pckern("free", 5), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("free", uint64(0)), ...
                "phasecont:engine:badHandle");
            testCase.verifyError(@() pckern("free", uint64(987654321)), ...
                "phasecont:engine:badHandle");
            m = pckern("model_from_ode", testCase.Lorenz);
            pckern("free", m);
            testCase.verifyError(@() pckern("free", m), ...
                "phasecont:engine:badHandle");
        end

        function lorenzInfoAndRhs(testCase)
            [m, diag] = pckern("model_from_ode", testCase.Lorenz);
            cleanup = onCleanup(@() pckern("free", m));
            testCase.verifyClass(m, "uint64");
            testCase.verifyNotEqual(m, uint64(0));
            testCase.verifyClass(diag, "char");
            info = jsondecode(pckern("model_info", m));
            testCase.verifyEqual(info.n, 3);
            testCase.verifyEqual(string(info.parameters(:))', ["s" "r" "b"]);
            x = [1 2 3];
            p = testCase.P;
            expected = [p(1)*(x(2)-x(1)); x(1)*(p(2)-x(3))-x(2); x(1)*x(2)-p(3)*x(3)];
            for ev = ["fallback" "jit"]
                pckern("model_compile", m, "{""evaluator"":""" + ev + """}");
                f = pckern("model_rhs", m, x, p);
                testCase.verifyEqual(f, expected);
                testCase.verifyEqual(size(f), [3 1]);
                f = pckern("model_rhs", m, x(:), p(:));
                testCase.verifyEqual(f, expected);
            end
            testCase.verifyError(@() pckern("model_rhs", m, [1 2], p), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_rhs", m, ones(3), p), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_rhs", m, single(x), p), ...
                "phasecont:engine:badArg");
            clear cleanup
        end

        function equilibriumAtTheLorenzOrigin(testCase)
            m = pckern("model_from_ode", testCase.Lorenz);
            cleanup = onCleanup(@() pckern("free", m));
            pckern("model_compile", m);
            p = testCase.P;
            [u, ev] = pckern("model_equilibrium", m, [0.01 -0.01 0.02], p, "");
            testCase.verifyEqual(u, zeros(3, 1), AbsTol = 1e-10);
            J = [-p(1) p(1) 0; p(2) -1 0; 0 0 -p(3)];
            testCase.verifyEqual(size(ev), [3 1]);
            testCase.verifyEqual(sort(real(ev)), sort(eig(J)), RelTol = 1e-10);
            testCase.verifyEqual(imag(ev), zeros(3, 1), AbsTol = 1e-10);
            clear cleanup
        end

        function modelErrorsComeBackAsDiagnostics(testCase)
            [m, diag] = pckern("model_from_ode", "x'=-x" + newline + "foo(" + newline + "done");
            testCase.verifyEqual(m, uint64(0));
            d = jsondecode(diag);
            testCase.verifyGreaterThanOrEqual(numel(d.diagnostics), 1);
            testCase.verifyError(@() pckern("model_from_ode", ...
                "x'=-x" + string(char(0)) + newline + "done"), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_from_ode", ...
                ['x' char(0) 'y']), "phasecont:engine:badArg");
        end

        function nonAutonomousModelIsRejected(testCase)
            % model_compile rejects a model that uses t, so run_start never
            % sees a compiled one (the library keeps a second guard there).
            m = pckern("model_from_ode", ...
                "par a=1" + newline + "x'=-a*x+sin(t)" + newline + "done" + newline);
            cleanup = onCleanup(@() pckern("free", m));
            testCase.verifyError(@() pckern("model_compile", m, ...
                "{""evaluator"":""fallback""}"), "phasecont:engine:nonAutonomous");
            testCase.verifyError(@() pckern("run_start", m, ...
                '{"type":"EP","free":"a","options":{}}', 0, 1), ...
                "phasecont:engine:badArg");
            clear cleanup
        end

        function lorenzRunAndCopy(testCase)
            m = pckern("model_from_ode", testCase.Lorenz);
            cleanup = onCleanup(@() pckern("free", m));
            pckern("model_compile", m, "{""evaluator"":""fallback""}");
            problem = ['{"type":"EP","free":"r","options":{"h0":0.05,', ...
                '"hmax":0.5,"max_steps":300,"direction":-1,"bounds":[-5,60]}}'];
            r = pckern("run_start", m, problem, [0 0 0], testCase.P);
            freeRun = onCleanup(@() pckern("free", r));
            st = testCase.waitFor(r);
            testCase.verifyEqual(st.state, "done");
            stride = st.stride;
            testCase.verifyEqual(stride, 15);
            b1 = st.branches;
            if iscell(b1)
                b1 = b1{1};
            else
                b1 = b1(1);
            end
            testCase.verifyGreaterThan(b1.points, 1);
            rows = pckern("run_copy", r, 0, 0, 0, b1.points, stride);
            testCase.verifyEqual(size(rows), [stride b1.points]);
            testCase.verifyEqual(rows(1, 1), testCase.P(2));
            testCase.verifyEqual(rows(2:4, :), zeros(3, b1.points), AbsTol = 1e-8);
            % A window, and more than exist.
            part = pckern("run_copy", r, 0, 0, 2, 3, stride);
            testCase.verifyEqual(part, rows(:, 3:5));
            tail = pckern("run_copy", r, 0, 0, b1.points - 1, 10, stride);
            testCase.verifyEqual(size(tail), [stride 1]);
            % Specials.
            nsp = numel(b1.specials);
            sp = pckern("run_copy", r, 0, 1, 0, 100, stride);
            testCase.verifyEqual(size(sp), [stride nsp]);
            % count = 0
            none = pckern("run_copy", r, 0, 0, 0, 0, stride);
            testCase.verifyEqual(size(none), [stride 0]);
            % Bad arguments.
            testCase.verifyError(@() pckern("run_copy", r, 0, 0, 0, 1, 0), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("run_copy", r, -1, 0, 0, 1, stride), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("run_copy", r, 0, 0, 0, 2^60, stride), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("run_copy", r, 99, 0, 0, 1, stride), ...
                "phasecont:engine:badArg");
            clear freeRun cleanup
        end

        function cancelEndsALongRun(testCase)
            m = pckern("model_from_ode", testCase.Lorenz);
            cleanup = onCleanup(@() pckern("free", m));
            pckern("model_compile", m, "{""evaluator"":""fallback""}");
            problem = ['{"type":"EP","free":"r","options":{"h0":0.001,', ...
                '"hmax":0.001,"max_steps":1000000,"bounds":[-1000,1000]}}'];
            r = pckern("run_start", m, problem, [0 0 0], testCase.P);
            freeRun = onCleanup(@() pckern("free", r));
            st = jsondecode(pckern("run_poll", r));
            testCase.verifyEqual(string(st.state), "running");
            pckern("run_cancel", r);
            st = testCase.waitFor(r);
            testCase.verifyEqual(st.state, "cancelled");
            clear freeRun cleanup
        end

        function oneOutputIsEnoughForModelFromOde(testCase)
            m = pckern("model_from_ode", testCase.Lorenz);
            testCase.verifyNotEqual(m, uint64(0));
            pckern("free", m);
            bad = pckern("model_from_ode", "x'=-x" + newline + "foo(" + newline + "done");
            testCase.verifyEqual(bad, uint64(0));
        end

        function tooManyOutputsAreBadArg(testCase)
            testCase.verifyError(@() GatewayTest.threeOutputs("version"), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() GatewayTest.threeOutputs("model_from_ode", "x"), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() GatewayTest.oneOutput("panic"), ...
                "phasecont:engine:badArg");
        end

        function handleMustBeRealUint64(testCase)
            testCase.verifyError(@() pckern("free", complex(uint64(1), uint64(1))), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("free", uint32(1)), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("free", uint64([1 2])), ...
                "phasecont:engine:badArg");
        end

        function emptyVectorsMustMatchTheModel(testCase)
            m = pckern("model_from_ode", testCase.Lorenz);
            cleanup = onCleanup(@() pckern("free", m));
            pckern("model_compile", m, "{""evaluator"":""fallback""}");
            testCase.verifyError(@() pckern("model_rhs", m, [], testCase.P), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_equilibrium", m, zeros(1, 0), ...
                testCase.P, ""), "phasecont:engine:badArg");
            clear cleanup
        end

        function nonAsciiTextRoundTrips(testCase)
            e = char(233);
            omega = char(937);
            smile = char([55357 56832]);   % U+1F600 as a surrogate pair
            % Library to MATLAB: the diagnostics echo the stray character.
            [~, diag] = pckern("model_from_ode", "x'=-x" + newline + "y'=x" + e + newline + "done");
            d = jsondecode(diag);
            testCase.verifySubstring(string(d.diagnostics(1).message), "'" + e + "'");
            % MATLAB to library and back through an error message.
            m = pckern("model_from_ode", testCase.Lorenz);
            cleanup = onCleanup(@() pckern("free", m));
            key = string(e) + string(omega) + string(smile);
            try
                pckern("model_compile", m, "{""" + key + """:1}");
                testCase.verifyFail("model_compile accepted an unknown key");
            catch err
                testCase.verifySubstring(string(err.message), key);
            end
            clear cleanup
            % Text with a comment in non-ASCII still parses.
            m = pckern("model_from_ode", "# caf" + e + " " + omega + newline + testCase.Lorenz);
            testCase.verifyNotEqual(m, uint64(0));
            pckern("free", m);
        end

        function argumentErrors(testCase)
            testCase.verifyError(@() pckern(), "phasecont:engine:badArg");
            testCase.verifyError(@() pckern(1), "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("no_such_command"), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("version", 1), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("free"), "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_from_ode", 3), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("model_rhs", uint64(1), 1), ...
                "phasecont:engine:badArg");
            testCase.verifyError(@() pckern("panic"), "phasecont:engine:panic");
            try
                pckern("no_such_command");
            catch err
                testCase.verifySubstring(err.message, ...
                    "unknown pckern command ""no_such_command""");
            end
        end
    end

    methods (Static, Access = private)
        function threeOutputs(varargin)
            [~, ~, ~] = pckern(varargin{:});
        end

        function oneOutput(varargin)
            [~] = pckern(varargin{:});
        end

        function st = waitFor(r)
            % Polls until the state is not "running" (at most 20 s).
            t0 = tic;
            while true
                st = jsondecode(pckern("run_poll", r));
                st.state = string(st.state);
                if st.state ~= "running" || toc(t0) > 20
                    break
                end
                pause(0.005);
            end
        end
    end
end
