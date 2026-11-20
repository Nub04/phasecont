classdef DrawThinTest < matlab.unittest.TestCase
    % Tests the thinning of drawn lines (SPEC section 8af-3): every dropped
    % row lies within half a pixel of the last drawn row before it, at
    % several zoom levels and after a resize, in a figure and in a uiaxes in
    % a grid layout; the drawn vertices are data rows bit for bit and
    % UserData.PhaseContRows names them; the vertex bound on monotone data;
    % log axes, NaN breaks, short lines, 3-D views and the dots of a map.
    % No engine call: the Trajectory and Branch objects are built from
    % structs.

    properties
        Fig
    end

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (TestMethodTeardown)
        function closeFigure(testCase)
            if ~isempty(testCase.Fig) && isvalid(testCase.Fig)
                close(testCase.Fig);
            end
            testCase.Fig = [];
        end
    end

    methods
        function ax = figureAxes(testCase)
            testCase.Fig = figure(Visible = "off", Position = [100 100 700 500]);
            ax = axes(testCase.Fig);
        end

        function ax = gridAxes(testCase)
            testCase.Fig = uifigure(Visible = "off", Position = [100 100 700 500]);
            g = uigridlayout(testCase.Fig, [1 1]);
            ax = uiaxes(g);
        end

        function settle(~)
            for k = 1:4
                drawnow;
            end
        end

        function [px, py] = pixels(~, ax, x, y)
            % Pixel coordinates of the rows on ax; NaN where a log axis
            % cannot place a value.
            p = getpixelposition(ax);
            px = toPixel(x, ax.XLim, ax.XScale, p(3));
            py = toPixel(y, ax.YLim, ax.YScale, p(4));
        end

        function verifyDrawn(testCase, h, x, y)
            % The vertices of h are the rows PhaseContRows of x and y, bit
            % for bit, in order, with the first and last rows, and every
            % row a pixel coordinate cannot be found for is drawn.
            rows = h.UserData.PhaseContRows;
            testCase.verifyTrue(iscolumn(rows) && all(diff(rows) > 0), ...
                "PhaseContRows is not an increasing column");
            testCase.verifyEqual(rows([1 end]).', [1 numel(x)], ...
                "an end row was dropped");
            testCase.verifyEqual(typecast(h.XData(:), "uint64"), ...
                typecast(x(rows), "uint64"), "a drawn x is not a data x");
            testCase.verifyEqual(typecast(h.YData(:), "uint64"), ...
                typecast(y(rows), "uint64"), "a drawn y is not a data y");
        end

        function verifyHalfPixel(testCase, h, x, y, what)
            % Every dropped row lies within half a pixel of the last drawn
            % row before it.
            testCase.verifyDrawn(h, x, y);
            ax = ancestor(h, "axes");
            [px, py] = testCase.pixels(ax, x, y);
            keep = false(numel(x), 1);
            keep(h.UserData.PhaseContRows) = true;
            bad = ~isfinite(px) | ~isfinite(py);
            testCase.verifyTrue(all(keep(bad)), what + ": a non-finite row was dropped");
            c = cumsum(keep);
            rows = h.UserData.PhaseContRows;
            drop = find(~keep);
            prev = rows(c(drop));
            d = hypot(px(drop) - px(prev), py(drop) - py(prev));
            testCase.verifyLessThan(max([d; 0]), 0.5, ...
                what + ": a dropped row is half a pixel or more from the line");
        end

        function h = drawPath(~, ax, x, y)
            % A trajectory whose phase plane is (x, y), drawn on ax.
            tr = phasecont.Trajectory(struct("T", (1:numel(x)).', "X", [x y]));
            h = plot(tr, Parent = ax);
            h = h(1);
        end
    end

    methods (Test)
        function aMonotoneRunMeetsTheVertexBound(testCase)
            ax = testCase.figureAxes();
            t = linspace(0, 1, 1e6).';
            x = t;
            y = t.^2;
            h = testCase.drawPath(ax, x, y);
            testCase.settle();
            p = getpixelposition(ax);
            bound = 6 + (p(3) / 0.35 + 1) + (p(4) / 0.35 + 1);
            testCase.verifyLessThanOrEqual(numel(h.XData), bound);
            testCase.verifyHalfPixel(h, x, y, "monotone");
        end

        function theGuaranteeHoldsAtThreeZoomsAndAfterAResize(testCase)
            ax = testCase.figureAxes();
            t = linspace(0, 1, 1e6).';
            x = t;
            y = sin(40 * t) + 1e-3 * sin(1e4 * t);
            h = testCase.drawPath(ax, x, y);
            testCase.settle();
            testCase.verifyLessThan(numel(h.XData), numel(x), "nothing was thinned");
            testCase.verifyHalfPixel(h, x, y, "start");
            % Automatic limits are those of the full data.
            axis(ax, "tight");
            testCase.settle();
            testCase.verifyEqual(ax.XLim, [min(x) max(x)]);
            testCase.verifyEqual(ax.YLim, [min(y) max(y)]);
            for lim = {[0.3 0.6], [0.40 0.41], [0.4050 0.4051]}
                xlim(ax, lim{1});
                testCase.settle();
                testCase.verifyHalfPixel(h, x, y, "zoom " + mat2str(lim{1}));
            end
            xlim(ax, "auto");
            testCase.settle();
            before = numel(h.XData);
            testCase.Fig.Position(3:4) = 2 * testCase.Fig.Position(3:4);
            testCase.settle();
            testCase.verifyGreaterThan(numel(h.XData), before, ...
                "the line was not thinned again after the resize");
            testCase.verifyHalfPixel(h, x, y, "resized");
        end

        function theGuaranteeHoldsInAUIAxesInAGrid(testCase)
            ax = testCase.gridAxes();
            t = linspace(0, 1, 1e6).';
            x = cos(6 * pi * t) .* (1 - t);
            y = sin(6 * pi * t) .* (1 - t);
            h = testCase.drawPath(ax, x, y);
            testCase.settle();
            testCase.verifyLessThan(numel(h.XData), numel(x), "nothing was thinned");
            testCase.verifyHalfPixel(h, x, y, "uiaxes");
            zoom(ax, 3);
            testCase.settle();
            testCase.verifyHalfPixel(h, x, y, "uiaxes zoom");
            before = numel(h.XData);
            p0 = getpixelposition(ax);
            testCase.Fig.Position(3:4) = 2 * testCase.Fig.Position(3:4);
            % A grid layout sizes its children after the figure, not in
            % the same drawnow; wait up to 10 s for the new size.
            for k = 1:50
                drawnow;
                if ~isequal(getpixelposition(ax), p0)
                    break
                end
                pause(0.2);
            end
            testCase.assertNotEqual(getpixelposition(ax), p0, ...
                "the grid layout did not resize the axes");
            testCase.settle();
            testCase.verifyNotEqual(numel(h.XData), before, ...
                "the line was not thinned again after the resize");
            testCase.verifyHalfPixel(h, x, y, "uiaxes resized");
        end

        function aLogAxisDrawsTheRowsItCannotPlace(testCase)
            ax = testCase.figureAxes();
            t = linspace(0, 1, 4e5).';
            x = t;
            y = sin(20 * t);
            h = testCase.drawPath(ax, x, y);
            testCase.applyFixture(matlab.unittest.fixtures.SuppressedWarningsFixture( ...
                "MATLAB:Axes:NegativeDataInLogAxis"));
            ax.YScale = "log";
            testCase.settle();
            testCase.verifyLessThan(numel(h.XData), numel(x), "nothing was thinned");
            testCase.verifyHalfPixel(h, x, y, "log y");
            keep = false(numel(x), 1);
            keep(h.UserData.PhaseContRows) = true;
            testCase.verifyTrue(all(keep(y <= 0)), "a row <= 0 was dropped");
        end

        function aNaNBreakIsKept(testCase)
            ax = testCase.figureAxes();
            t = linspace(0, 1, 4e5).';
            x = t;
            y = cos(30 * t);
            y(1e5:1e5+10) = NaN;
            x(2e5) = Inf;
            h = testCase.drawPath(ax, x, y);
            testCase.settle();
            testCase.verifyLessThan(numel(h.XData), numel(x), "nothing was thinned");
            testCase.verifyHalfPixel(h, x, y, "NaN break");
            testCase.verifyEqual(nnz(isnan(h.YData)), 11);
            testCase.verifyEqual(nnz(isinf(h.XData)), 1);
        end

        function aShortLineIsDrawnExact(testCase)
            ax = testCase.figureAxes();
            p = getpixelposition(ax);
            n = floor(4 * p(3));
            t = linspace(0, 1, n).';
            h = testCase.drawPath(ax, t, sin(50 * t));
            testCase.settle();
            testCase.verifyEqual(h.XData(:), t);
            testCase.verifyEqual(h.UserData.PhaseContRows, (1:n).');
        end

        function aThreeDViewIsDrawnExact(testCase)
            ax = testCase.figureAxes();
            t = linspace(0, 1, 4e5).';
            y = sin(40 * t);
            h = testCase.drawPath(ax, t, y);
            testCase.settle();
            testCase.verifyLessThan(numel(h.XData), numel(t), "nothing was thinned");
            view(ax, 3);
            testCase.settle();
            testCase.verifyEqual(h.XData(:), t);
            testCase.verifyEqual(h.YData(:), y);
            testCase.verifyEqual(h.UserData.PhaseContRows, (1:numel(t)).');
            view(ax, 2);
            testCase.settle();
            testCase.verifyHalfPixel(h, t, y, "back to 2-D");
        end

        function theDotsOfAMapAreDrawnExact(testCase)
            ax = testCase.figureAxes();
            n = 1e5;
            x = mod((1:n).' * 0.6180339887, 1);
            tr = phasecont.Trajectory(struct("T", (0:n-1).', "X", x, "Map", true));
            h = plot(tr, Parent = ax);
            testCase.settle();
            testCase.verifyEqual(h(1).YData(:), x);
            testCase.verifyEqual(h(1).UserData.PhaseContRows, (1:n).');
        end

        function theRunsOfABranchCarryTheirRows(testCase)
            ax = testCase.figureAxes();
            n = 2e5;
            a = linspace(-1, 1, n).';
            u = a.^3 - 0.5 * a + 0.01 * sin(2000 * a);
            stable = a < 0;
            br = phasecont.Branch(struct("Type", "EP", "Free", "a", "FreeIndex", 2, ...
                "Alpha", a, "U", u, "Stable", stable));
            h = plot(br, Parent = ax);
            testCase.settle();
            runs = [findobj(h, "Tag", "pc-stable"); findobj(h, "Tag", "pc-unstable")];
            testCase.verifyNumElements(runs, 2);
            for k = 1:numel(runs)
                rows = runs(k).UserData.PhaseContRows;
                testCase.verifyLessThan(numel(rows), n / 2, "nothing was thinned");
                % Each run against its own rows of the branch.
                span = (rows(1):rows(end)).';
                sub = runs(k);
                testCase.verifyEqual(typecast(sub.XData(:), "uint64"), ...
                    typecast(a(rows), "uint64"));
                testCase.verifyEqual(typecast(sub.YData(:), "uint64"), ...
                    typecast(u(rows), "uint64"));
                [px, py] = testCase.pixels(ax, a(span), u(span));
                keep = false(numel(span), 1);
                keep(rows - span(1) + 1) = true;
                c = cumsum(keep);
                drawn = find(keep);
                drop = find(~keep);
                prev = drawn(c(drop));
                d = hypot(px(drop) - px(prev), py(drop) - py(prev));
                testCase.verifyLessThan(max([d; 0]), 0.5);
            end
            % The runs share their boundary row.
            r1 = runs(1).UserData.PhaseContRows;
            r2 = runs(2).UserData.PhaseContRows;
            testCase.verifyEqual(sort([r1([1 end]); r2([1 end])]).', ...
                [1 n/2+1 n/2+1 n]);
        end

        function aCurveCarriesItsRows(testCase)
            ax = testCase.figureAxes();
            n = 2e5;
            s = linspace(0, 2 * pi, n).';
            alpha = [cos(s) sin(3 * s)];
            br = phasecont.Branch(struct("Type", "H", "Free", ["a" "b"], ...
                "FreeIndex", [2 3], "Alpha", alpha, "U", zeros(n, 1)));
            h = plot(br, Parent = ax);
            testCase.settle();
            c = findobj(h, "Tag", "pc-curve");
            testCase.verifyLessThan(numel(c.XData), n, "nothing was thinned");
            testCase.verifyHalfPixel(c, alpha(:, 1), alpha(:, 2), "curve");
        end
    end
end

function p = toPixel(v, lim, scale, w)
% Pixel coordinate of v on an axis with limits lim and width w pixels.
if scale == "log"
    v(v <= 0) = NaN;
    p = (log10(v) - log10(lim(1))) / (log10(lim(2)) - log10(lim(1))) * w;
else
    p = (v - lim(1)) / (lim(2) - lim(1)) * w;
end
p(~isfinite(p)) = NaN;
end
