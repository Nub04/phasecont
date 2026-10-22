function rows = s3_rows(v)
%S3_ROWS Turns a decoded JSON array of objects into a struct array.
%   jsondecode returns a struct array when the objects share their keys.
%   A cell array of structs is accepted too, and [] (an empty JSON array)
%   gives an empty struct array. The s4_* helpers use this function as well.
if isempty(v)
    rows = struct([]);
elseif isstruct(v)
    rows = v;
elseif iscell(v)
    rows = [v{:}];
else
    error('spikes:badjson', 'expected an array of objects, got %s', class(v));
end
end
