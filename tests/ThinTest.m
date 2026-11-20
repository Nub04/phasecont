classdef ThinTest < matlab.unittest.TestCase
    % Tests simulate(..., Thin = tol) (SPEC section 8af-2): every dropped row
    % is reproduced within tol times the span of each column by the linear
    % interpolation in t between its kept neighbours, every kept row is an
    % unthinned row bit for bit, the ends and the non-finite rows with their
    % neighbours are kept, the refusals, a session replay and the LC start
    % from a thinned trajectory.

    properties (Constant)
        % The Hopf normal form; its cycle at mu = 1 is the unit circle.
        Normal = "par mu=1" + newline + ...
            "x'=mu*x-y-x*(x^2+y^2)" + newline + ...
            "y'=x+mu*y-y*(x^2+y^2)" + newline + ...
            "init x=0.5,y=0" + newline + "done" + newline
        Lorenz = "par s=10,r=28,b=2.6666666666666665" + newline + ...
            "x'=s*(y-x)" + newline + "y'=x*(r-z)-y" + newline + ...
            "z'=x*y-b*z" + newline + "init x=1,y=1,z=1" + newline + ...
            "done" + newline
        % A stiff scalar model: x follows cos(t) on a time scale of 1e-3.
        Stiff = "par k=1000" + newline + "x'=-k*(x-cos(t))" + newline + ...
            "init x=0" + newline + "done" + newline
        % The Hopf normal form with two stored aux columns.
        WithAux = "par mu=1" + newline + ...
            "x'=mu*x-y-x*(x^2+y^2)" + newline + ...
            "y'=x+mu*y-y*(x^2+y^2)" + newline + ...
            "aux r2=x^2+y^2" + newline + "aux s=sin(3*t)*x" + newline + ...
            "only x,y,r2,s" + newline + ...
            "init x=0.5,y=0" + newline + "done" + newline
        % An aux column that is NaN after t = 1.
        WithNaN = "par a=1" + newline + "x'=-a*x" + newline + ...
            "aux q=sqrt(1-t)" + newline + "only x,q" + newline + ...
            "init x=1" + newline + "done" + newline
        Map = "par r=3.7" + newline + "x(t+1)=r*x*(1-x)" + newline + ...
            "init x=0.1" + newline + "done" + newline
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

        function kept = verifyGuarantee(testCase, full, thin, tol)
            % FULL and THIN are the rows [t x aux] of the same run without
            % and with Thin = tol. Returns how many rows were kept.
            n = size(full, 1);
            [found, loc] = ismember(thin(:, 1), full(:, 1));
            testCase.verifyTrue(all(found), "a kept row is not an unthinned row");
            loc = loc(found);
            testCase.verifyTrue(all(diff(loc) > 0), "the kept rows are out of order");
            testCase.verifyEqual(typecast(reshape(full(loc, :), [], 1), "uint64"), ...
                typecast(reshape(thin(found, :), [], 1), "uint64"), ...
                "a kept row differs from its unthinned row");
            testCase.verifyEqual(loc([1 end]).', [1 n], "an end row was dropped");
            keep = false(n, 1);
            keep(loc) = true;
            bad = find(any(~isfinite(full), 2));
            near = unique(max(1, min(n, [bad - 1; bad; bad + 1])));
            testCase.verifyTrue(all(keep(near)), ...
                "a non-finite row or its neighbour was dropped");
            % The span of each column over its finite values; 1 for a span
            % of 0 or a column with no finite value.
            span = ones(1, size(full, 2) - 1);
            for j = 2:size(full, 2)
                v = full(isfinite(full(:, j)), j);
                if ~isempty(v) && max(v) > min(v)
                    span(j - 1) = max(v) - min(v);
                end
            end
            drop = find(~keep);
            c = cumsum(keep);
            a = loc(c(drop));
            b = loc(c(drop) + 1);
            % The formula of SPEC 8af-2 in its order, one operation a line
            % so nothing is fused.
            w = (full(drop, 1) - full(a, 1)) ./ (full(b, 1) - full(a, 1));
            d = full(b, 2:end) - full(a, 2:end);
            q = w .* d;
            v = full(a, 2:end) + q;
            err = abs(v - full(drop, 2:end));
            testCase.verifyTrue(all(err <= tol .* span, "all"), ...
                sprintf("a dropped row misses the bound at tol %g", tol));
            kept = numel(loc);
        end

        function rows = rowsOf(~, tr)
            rows = [tr.T tr.X tr.AuxStore];
        end
    end

    methods (Test)
        function theGuaranteeHoldsOnThreeModels(testCase)
            cases = {
                testCase.Normal, [0.5 0], [0 40], {}
                testCase.Lorenz, [1 1 1], [0 20], {"RelTol", 1e-9, "AbsTol", 1e-12}
                testCase.Stiff, 0, [0 6], {}
                };
            for k = 1:size(cases, 1)
                m = testCase.model(cases{k, 1});
                opts = cases{k, 4};
                full = phasecont.simulate(m, cases{k, 2}, cases{k, 3}, opts{:});
                testCase.verifyEqual(full.Thin, []);
                for tol = [1e-2 1e-3 1e-6]
                    tr = phasecont.simulate(m, cases{k, 2}, cases{k, 3}, ...
                        opts{:}, Thin = tol);
                    testCase.verifyEqual(tr.Thin, tol);
                    kept = testCase.verifyGuarantee(testCase.rowsOf(full), ...
                        testCase.rowsOf(tr), tol);
                    testCase.verifyLessThanOrEqual(kept, numel(full.T));
                end
                % A coarse tolerance drops rows on every model.
                tr = phasecont.simulate(m, cases{k, 2}, cases{k, 3}, ...
                    opts{:}, Thin = 1e-2);
                testCase.verifyLessThan(numel(tr.T), numel(full.T));
            end
        end

        function theAuxColumnsMeetTheBoundToo(testCase)
            m = testCase.model(testCase.WithAux);
            full = phasecont.simulate(m, [0.5 0], [0 30]);
            testCase.verifySize(full.AuxStore, [numel(full.T) 2]);
            for tol = [1e-2 1e-3 1e-6]
                tr = phasecont.simulate(m, [0.5 0], [0 30], Thin = tol);
                testCase.verifySize(tr.AuxStore, [numel(tr.T) 2]);
                testCase.verifyGuarantee(testCase.rowsOf(full), ...
                    testCase.rowsOf(tr), tol);
            end
        end

        function nonFiniteRowsAndTheirNeighboursAreKept(testCase)
            m = testCase.model(testCase.WithNaN);
            full = phasecont.simulate(m, 1, [0 2]);
            testCase.assumeTrue(any(isnan(full.AuxStore)), "no NaN row");
            tr = phasecont.simulate(m, 1, [0 2], Thin = 1e-2);
            testCase.verifyGuarantee(testCase.rowsOf(full), testCase.rowsOf(tr), 1e-2);
            testCase.verifyEqual(nnz(isnan(tr.AuxStore)), nnz(isnan(full.AuxStore)));
        end

        function aMapRefusesThin(testCase)
            m = testCase.model(testCase.Map);
            for value = {0.5, -1}
                testCase.verifyError(@() phasecont.simulate(m, 0.1, [0 10], ...
                    Thin = value{1}), "phasecont:simulate:mapOption");
            end
        end

        function badValuesAreRefused(testCase)
            m = testCase.model(testCase.Normal);
            for value = {0, 1, -1, NaN, Inf, [1e-3 1e-3], "x", 1e-3i}
                testCase.verifyError(@() phasecont.simulate(m, [0.5 0], [0 1], ...
                    Thin = value{1}), "phasecont:model:badOption");
            end
        end

        function aSessionWithAThinnedRunReplays(testCase)
            m = testCase.model(testCase.Normal);
            tr = phasecont.simulate(m, [0.5 0], [0 20], Thin = 1e-3);
            file = fullfile(testCase.Dir, "thin.pcsession.json");
            save(phasecont.Session(m, {tr}), file);
            r = phasecont.session.replay(file);
            testCase.verifyTrue(all([r.Pass]), "the thinned run did not reproduce");
            back = phasecont.Session.load(file);
            testCase.verifyEqual(back.Items{1}.Thin, 1e-3);
            testCase.verifyEqual(back.Items{1}.T, tr.T);
            testCase.verifyEqual(back.Items{1}.X, tr.X);
        end

        function anLCStartFromAThinnedRunRunsItAgain(testCase)
            m = testCase.model(testCase.Normal);
            args = {[0.5 0], [0 60], "RelTol", 1e-10, "AbsTol", 1e-12};
            full = phasecont.simulate(m, args{:});
            tr = phasecont.simulate(m, args{:}, Thin = 1e-3);
            testCase.verifyLessThan(numel(tr.T), numel(full.T));
            lc = @(s) phasecont.continuation(m, s, "Type", "LC", "Free", "mu", ...
                "NTST", 20, "NCOL", 4, "MaxSteps", 2, "Period", 2*pi);
            want = lc(full);
            got = testCase.verifyWarning(@() lc(tr), ...
                "phasecont:continuation:thinRerun");
            testCase.verifyEqual(got.U(1, :), want.U(1, :));
        end

        function aThinnedTrajectoryWithoutARecordIsRefused(testCase)
            m = testCase.model(testCase.Normal);
            tr = phasecont.Trajectory(struct("T", (0:0.1:10).', ...
                "X", [cos(0:0.1:10); sin(0:0.1:10)].', "P", 1, ...
                "Model", m, "Thin", 1e-3));
            testCase.verifyError(@() phasecont.continuation(m, tr, "Type", "LC", ...
                "Free", "mu", "NTST", 20, "NCOL", 4, "Period", 2*pi), "phasecont:model:badStart");
            try
                phasecont.continuation(m, tr, "Type", "LC", "Free", "mu", ...
                    "NTST", 20, "NCOL", 4, "Period", 2*pi);
                testCase.verifyFail("the run was not refused");
            catch err
                testCase.verifySubstring(err.message, "Thin");
            end
        end
    end
end
