function [color, after] = nextColor(ax)
%NEXTCOLOR The next color of the color order of an axes.
%   [COLOR, AFTER] = phasecont.internal.nextColor(AX) returns the color at
%   AX.ColorOrderIndex and the index of the color after it, wrapping at the
%   end of AX.ColorOrder, and sets AX.ColorOrderIndex to AFTER. On held
%   axes plot moves the index on for every line it draws, even a line with
%   a given Color, so a caller that draws its lines with plot sets
%   AX.ColorOrderIndex = AFTER when it is done. The next plot then takes the
%   next color.
order = ax.ColorOrder;
idx = ax.ColorOrderIndex;
color = order(mod(idx - 1, size(order, 1)) + 1, :);
after = mod(idx, size(order, 1)) + 1;
ax.ColorOrderIndex = after;
end
