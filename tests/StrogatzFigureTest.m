classdef StrogatzFigureTest < matlab.unittest.TestCase
    % Runs each done row of books/strogatz/manifest.json: the figure script
    % with Plot = false, then the row's check against an independent oracle.
    % Also checks the schema of manifest.json and gaps.json. The row tests
    % skip when pckern is not built; the schema tests always run.

    properties (TestParameter)
        Row
    end

    properties
        HasEngine (1,1) logical = false
    end

    methods (TestParameterDefinition, Static)
        function Row = initializeRow()
            Row = StrogatzFigureTest.rowsFrom(StrogatzFigureTest.readRows("manifest.json"));
        end
    end

    methods (Static)
        function rows = readRows(name)
            % The rows of books/strogatz/NAME as a cell array of structs.
            root = fileparts(fileparts(mfilename("fullpath")));
            rows = jsondecode(fileread(fullfile(root, "books", "strogatz", name)));
            if isstruct(rows)
                rows = num2cell(rows);
            end
        end

        function rows = rowsFrom(list)
            % The done rows of LIST as a struct with one field per row,
            % "f" plus the figure number with dots as underscores. With no
            % done row it returns one placeholder row, so the class still
            % loads and the row test reports it as incomplete.
            rows = struct();
            for k = 1:numel(list)
                r = list{k};
                if string(r.state) == "done"
                    rows.("f" + strrep(string(r.figure), ".", "_")) = r;
                end
            end
            if isempty(fieldnames(rows))
                rows.none = struct("figure", "none", "script", "", "state", "none");
            end
        end

        function [out, seconds] = runScript(name, plot)
            % Runs the figure script NAME. When the script errors, the
            % figures it opened are closed before the error goes on.
            before = findall(groot, "Type", "figure");
            clock = tic;
            try
                out = feval(char(name), "Plot", plot);
            catch err
                delete(StrogatzFigureTest.newFigures(before));
                rethrow(err);
            end
            seconds = toc(clock);
        end

        function f = newFigures(before)
            % The figures that are open now and were not open in BEFORE,
            % found by comparing the handles one at a time with ==.
            figs = findall(groot, "Type", "figure");
            keep = true(size(figs));
            for k = 1:numel(figs)
                keep(k) = ~any(before == figs(k));
            end
            f = figs(keep);
        end
    end

    methods (TestClassSetup)
        function setup(testCase)
            f = testCase.applyFixture(fixtures.BinFixture);
            testCase.HasEngine = isfile(fullfile(f.Bin, "pckern." + mexext));
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(root, "books", "strogatz", "figures")));
        end
    end

    methods (Test)
        function figureMatchesItsOracle(testCase, Row)
            testCase.assumeNotEqual(string(Row.state), "none", "The manifest has no done row");
            testCase.assumeTrue(testCase.HasEngine, "pckern is not built");
            before = findall(groot, "Type", "figure");
            testCase.addTeardown(@() delete(StrogatzFigureTest.newFigures(before)));
            [out, seconds] = StrogatzFigureTest.runScript(Row.script, false);
            fprintf("strogatz-time %s %.3f\n", Row.figure, seconds);
            if isfield(out, "App")
                testCase.addTeardown(@() delete(out.App));
            end
            opened = StrogatzFigureTest.newFigures(before);
            if isfield(out, "App")
                testCase.verifyTrue(all(arrayfun(@(f) f.Visible == "off", opened)), ...
                    "Plot = false must leave the app window hidden");
            else
                testCase.verifyEmpty(opened, "Plot = false must draw nothing");
            end
            testCase.verifyEqual(orderfields(jsondecode(jsonencode(out.Values))), ...
                orderfields(Row.values), ...
                "out.Values must equal the values in the manifest", RelTol = 1e-14);
            testCase.("check_" + string(Row.script))(out);
        end

        function plotTrueDraws(testCase, Row)
            testCase.assumeNotEqual(string(Row.state), "none", "The manifest has no done row");
            testCase.assumeTrue(testCase.HasEngine, "pckern is not built");
            testCase.assumeFalse(any(contains(string(Row.api), "PhasePlane")), ...
                "An app row with Plot = true opens a visible window");
            old = get(groot, "DefaultFigureVisible");
            testCase.addTeardown(@() set(groot, "DefaultFigureVisible", old));
            set(groot, "DefaultFigureVisible", "off");
            before = findall(groot, "Type", "figure");
            testCase.addTeardown(@() delete(StrogatzFigureTest.newFigures(before)));
            [~, seconds] = StrogatzFigureTest.runScript(Row.script, true);
            fprintf("strogatz-plot-time %s %.3f\n", Row.figure, seconds);
            opened = StrogatzFigureTest.newFigures(before);
            testCase.verifyNotEmpty(opened, "Plot = true must draw a figure");
            ax = findall(opened, "Type", "axes");
            testCase.verifyTrue(any(arrayfun(@(a) ~isempty(a.Children), ax)), ...
                "The figure must hold a non-empty axes");
        end

        function manifestRowsFollowTheSchema(testCase)
            rows = StrogatzFigureTest.readRows("manifest.json");
            testCase.verifyNotEmpty(rows);
            fields = ["figure" "edition" "model" "script" "api" "values" "source" "check" "state"];
            names = string({metaclass(testCase).MethodList.Name});
            root = fileparts(fileparts(mfilename("fullpath")));
            figures = strings(1, numel(rows));
            for k = 1:numel(rows)
                r = rows{k};
                id = "manifest row " + k;
                testCase.verifyEqual(sort(string(fieldnames(r)))', sort(fields), id);
                figures(k) = string(r.figure);
                testCase.verifyMatches(figures(k), "^\d+\.\d+\.\d+$", id);
                testCase.verifyEqual(r.edition, 3, id);
                testCase.verifyEqual(string(r.script), "fig_" + strrep(figures(k), ".", "_"), id);
                testCase.verifyTrue(iscellstr(r.api) || ischar(r.api), id);
                testCase.verifyNotEmpty(r.api, id);
                testCase.verifyClass(r.values, "struct", id);
                testCase.verifyClass(r.source, "struct", id);
                testCase.verifyEqual(sort(string(fieldnames(r.source))), ...
                    sort(string(fieldnames(r.values))), id);
                kinds = string(struct2cell(r.source));
                testCase.verifyTrue(all(ismember(kinds, ["stated" "chosen"])), id);
                testCase.verifyNotEmpty(r.check, id);
                testCase.verifyLessThanOrEqual(strlength(string(r.check)), 200, id);
                state = string(r.state);
                testCase.verifyTrue(state == "done" || startsWith(state, "blocked: "), id);
                if state == "done"
                    testCase.verifyTrue(isfile(fullfile(root, "books", "strogatz", "models", r.model)), id);
                    testCase.verifyTrue(isfile(fullfile(root, "books", "strogatz", "figures", ...
                        string(r.script) + ".m")), id);
                    testCase.verifyTrue(ismember("check_" + string(r.script), names), id);
                end
            end
            testCase.verifyEqual(numel(unique(figures)), numel(figures), "Figures must be unique");
        end

        function gapRowsFollowTheSchema(testCase)
            rows = StrogatzFigureTest.readRows("gaps.json");
            figures = strings(1, numel(rows));
            states = strings(1, numel(rows));
            for k = 1:numel(rows)
                r = rows{k};
                id = "gaps row " + k;
                testCase.verifyEqual(sort(string(fieldnames(r)))', ["figure" "reason" "state"], id);
                figures(k) = string(r.figure);
                states(k) = string(r.state);
                testCase.verifyMatches(figures(k), "^\d+\.\d+\.\d+$", id);
                testCase.verifyTrue(ismember(states(k), ["gap" "n/a"]), id);
                testCase.verifyNotEmpty(r.reason, id);
                testCase.verifyLessThanOrEqual(strlength(string(r.reason)), 120, id);
            end
            na = ["3.5.7" "4.2.1" "4.5.2" "6.7.6" "9.2.3" "9.6.2" "10.7.2" "10.7.3" ...
                "11.5.4" "12.2.1" "13.2.5" "13.5.3"];   % decision D3
            testCase.verifyEqual(sort(figures(states == "n/a")), sort(na));
            testCase.verifyGreaterThanOrEqual(nnz(states == "gap"), 64);
            testCase.verifyEqual(numel(unique(figures)), numel(figures), "Figures must be unique");
            manifest = string(cellfun(@(r) r.figure, ...
                StrogatzFigureTest.readRows("manifest.json"), UniformOutput = false));
            testCase.verifyEmpty(intersect(figures, manifest), ...
                "A figure is either covered or a gap, not both");
        end

        function noDoneRowsGivesAPlaceholder(testCase)
            rows = StrogatzFigureTest.rowsFrom({});
            testCase.verifyEqual(string(fieldnames(rows)), "none");
            rows = StrogatzFigureTest.rowsFrom({struct("figure", "1.1.1", "state", "blocked: x")});
            testCase.verifyEqual(string(fieldnames(rows)), "none");
        end

        function aFailingScriptLeavesNoFigure(testCase)
            folder = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            fid = fopen(fullfile(folder, "fig_fail_probe.m"), "w");
            fprintf(fid, "function out = fig_fail_probe(options)\n");
            fprintf(fid, "arguments\n    options.Plot (1,1) logical = true\nend\n");
            fprintf(fid, "figure(Visible = ""off"");\n");
            fprintf(fid, "error(""strogatz:probe:fail"", ""probe"");\nend\n");
            fclose(fid);
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(folder));
            before = findall(groot, "Type", "figure");
            testCase.verifyError(@() StrogatzFigureTest.runScript("fig_fail_probe", false), ...
                "strogatz:probe:fail");
            testCase.verifyEmpty(StrogatzFigureTest.newFigures(before));
        end
    end

    methods (Access = protected)
        function check_fig_2_2_6(testCase, out)
            % f(x) = x - cos x has f'(x) = 1 + sin x >= 0, zero only at
            % isolated points, so f is strictly increasing and has one root.
            % Oracle: fzero. The eigenvalue is f'(x*) = 1 + sin x* > 0, so
            % the point is unstable.
            pt = out.Point;
            xs = fzero(@(x) x - cos(x), out.Values.x0);
            testCase.verifyEqual(pt.U, xs, AbsTol = 1e-9);
            testCase.verifyEqual(pt.Eigenvalues, 1 + sin(xs), AbsTol = 1e-8);
            testCase.verifyFalse(pt.Stable);
        end
    end
end

function J = cdJacobian(m, u, varargin)
% The Jacobian of m.rhs at U by central differences with step 1e-6. VARARGIN
% is the parameter argument of m.rhs, left out for a model without
% parameters. It does not use the engine's symbolic Jacobian, so it is an
% independent oracle for eigenvalues.
u = u(:);
n = numel(u);
J = zeros(n);
h = 1e-6;
for k = 1:n
    e = zeros(n, 1);
    e(k) = h;
    J(:, k) = (m.rhs(u + e, varargin{:}) - m.rhs(u - e, varargin{:})) / (2*h);
end
end
