function s = jsonRows(v, names)
%JSONROWS A JSON array of objects as a column struct array. Internal.
%   S = phasecont.internal.jsonRows(V, NAMES) takes what jsondecode returned
%   for a JSON array of objects: a struct array, a cell of structs whose
%   fields differ, or [] for an empty array. It returns an n-by-1 struct array
%   that has every field in the string array NAMES, filled with [] where an
%   object has no such key. Keys that are not in NAMES are dropped.
arguments
    v
    names (1,:) string
end
blank = cell2struct(repmat({[]}, numel(names), 1), names, 1);
if isempty(v)
    s = repmat(blank, 0, 1);
    return
end
if isstruct(v)
    v = num2cell(v(:));
end
s = repmat(blank, numel(v), 1);
for k = 1:numel(v)
    for n = names
        if isfield(v{k}, n)
            s(k).(n) = v{k}.(n);
        end
    end
end
end
