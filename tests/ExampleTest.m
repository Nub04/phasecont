classdef ExampleTest < matlab.unittest.TestCase
    % Runs the shipped examples and checks their results against
    % independent calculations.

    methods (TestClassSetup)
        function addToolbox(testCase)
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.assumeTrue( ...
                isfile(fullfile(f.Bin, "pckern." + mexext)), ...
                "pckern has not been built (run buildtool mex)");
            testCase.applyFixture(fixtures.ExamplesFixture);
        end
    end

    methods (Test)
        function fitzhughNagumoHopfPoints(testCase)
            out = evalc("br = fitzhugh_nagumo_example(Plot = false);");
            testCase.verifyNotEmpty(out);
            hp = [];
            for k = 1:numel(br)
                hp = [hp; br(k).points("H")]; %#ok<AGROW>
            end
            testCase.verifyNumElements(hp, 2, "expected two Hopf points");
            m = br(1).Model;
            idx = br(1).FreeIndex;
            for k = 1:numel(hp)
                found = hp(k).P(idx);
                % Independent root find: the trace of the Jacobian at the
                % equilibrium vanishes at a Hopf point of a planar system.
                ref = fzero(@(I) traceAt(m, I, hp(k).U), found);
                testCase.verifyEqual(found, ref, AbsTol = 1e-6);
                testCase.verifyGreaterThan(hp(k).Data.omega, 0);
            end
        end

        function morrisLecarSessionReplays(testCase)
            dir = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            file = fullfile(dir, "ml.pcsession.json");
            out = evalc("r = morris_lecar_session_example(File = file);");
            testCase.verifyNotEmpty(out);
            testCase.verifyTrue(isfile(file));
            testCase.verifyEqual([r.Type], ["EP" "LC" "LPC" "H" "IVP"]);
            testCase.verifyTrue(all([r.Pass]), "a run did not reproduce");
            testCase.verifyLessThanOrEqual(max([r.MaxRelDiff]), 1e-8);
        end
    end
end

function t = traceAt(m, I, u0)
% Trace of the Jacobian (central differences of Model.rhs) at the
% equilibrium of the model with drive I, found by Newton from u0.
p = m.ParameterValues;
p(m.Parameters == "I") = I;
u = u0(:);
for it = 1:50
    f = m.rhs(u, p);
    J = jac(m, u, p);
    du = J \ f;
    u = u - du;
    if norm(du) < 1e-14
        break
    end
end
t = trace(jac(m, u, p));
end

function J = jac(m, u, p)
n = numel(u);
J = zeros(n);
for j = 1:n
    h = 1e-6 * max(1, abs(u(j)));
    e = zeros(n, 1);
    e(j) = h;
    J(:, j) = (m.rhs(u + e, p) - m.rhs(u - e, p)) / (2 * h);
end
end
