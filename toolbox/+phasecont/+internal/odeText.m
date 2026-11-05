function text = odeText(equations, params, init)
%ODETEXT Build .ode text from equation lines and two structs of numbers.
%   TEXT = phasecont.internal.odeText(EQUATIONS, PARAMS, INIT) appends one
%   `par` block (from PARAMS) and one `init` block (from INIT) to the
%   equation text. The equations keep their line numbers, so a parser
%   diagnostic points at the line the caller wrote. When the equations hold
%   a `done` line, the blocks go in front of it, because `done` ends the
%   file.
%
%   Field names must be plain identifiers. Values must be finite real
%   numeric scalars. Nothing is defaulted: a name that is missing here is
%   missing in the model.
%
%   Raises phasecont:model:badValue and phasecont:model:badName.
arguments
    equations (1,1) string {mustBeNonmissing}
    params (1,1) struct
    init (1,1) struct
end

par = declarations("par", params, "Params");
ini = declarations("init", init, "Init");

lines = splitlines(replace(replace(equations, sprintf("\r\n"), newline), ...
    sprintf("\r"), newline));
% A trailing newline gives an empty last entry. Drop it and add it back.
if lines(end) == ""
    lines(end) = [];
end
isDone = ~cellfun(@isempty, regexp(cellstr(lines), ...
    '^\s*d(one)?\s*(#.*)?$', 'once', 'ignorecase'));
at = find(isDone, 1);
if isempty(at)
    at = numel(lines) + 1;
end
text = join([lines(1:at-1); par; ini; lines(at:end)], newline) + newline;
end

function out = declarations(keyword, s, label)
% One `par a=1,b=2` line per eight entries.
names = string(fieldnames(s));
out = strings(0, 1);
entries = strings(numel(names), 1);
for k = 1:numel(names)
    name = names(k);
    if isempty(regexp(name, '^[A-Za-z_][A-Za-z0-9_]*$', 'once'))
        error("phasecont:model:badName", ...
            "%s: ""%s"" is not a valid name; use letters, digits and " + ...
            "underscores, starting with a letter or underscore.", ...
            label, name);
    end
    v = s.(name);
    if ~(isnumeric(v) || islogical(v)) || ~isreal(v) || ~isscalar(v) ...
            || ~isfinite(v)
        error("phasecont:model:badValue", ...
            "%s.%s must be a finite real numeric scalar.", label, name);
    end
    entries(k) = name + "=" + phasecont.internal.num2ode(double(v));
end
for first = 1:8:numel(entries)
    last = min(first + 7, numel(entries));
    out(end+1, 1) = keyword + " " + join(entries(first:last), ","); %#ok<AGROW>
end
end
