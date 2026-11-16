function [X, Y] = cobwebSteps(x)
%COBWEBSTEPS Vertices of the cobweb staircase of a list of iterates.
%   [X, Y] = phasecont.internal.cobwebSteps(X) takes the iterates x_0, ...,
%   x_m of a map with one variable and returns the 2m + 1 vertices (X_i, Y_i),
%   i = 0, ..., 2m, with X_i = x_floor(i/2) and Y_i = x_ceil(i/2): (x_0, x_0),
%   (x_0, x_1), (x_1, x_1), ..., (x_m, x_m). The vertices are copies of the
%   iterates. X and Y are column vectors.
arguments
    x (:,1) double
end
if isempty(x)
    X = zeros(0, 1);
    Y = zeros(0, 1);
    return
end
i = (0:2 * (numel(x) - 1)).';
X = x(floor(i / 2) + 1);
Y = x(ceil(i / 2) + 1);
end
