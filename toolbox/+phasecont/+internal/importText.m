function [ode, warnings] = importText(format, text, optsJson, skipped)
%IMPORTTEXT Run the engine importer and return .ode text.
%   [ODE, W] = phasecont.internal.importText(FORMAT, TEXT, OPTS, SKIPPED)
%   calls pckern("import", FORMAT, TEXT, OPTS) with FORMAT "sbml" or
%   "matlab". The engine reads TEXT and never runs it.
%
%   A refused import raises the importer's own error id,
%   phasecont:import:<kind>, with kind one of syntax, unsupported,
%   undeclared, missingValue, mismatch, nameCollision, tooLarge, noDynamics
%   and xml. The message carries line and column when the importer knows
%   them. SKIPPED names captured variables that were not passed on; an
%   `undeclared` error mentions them. Importer warnings are raised as
%   phasecont:import:warning, and returned in W.
arguments
    format (1,1) string {mustBeMember(format, ["sbml" "matlab"])}
    text (1,1) string {mustBeNonmissing}
    optsJson (1,1) string = ""
    skipped (:,1) string = strings(0, 1)
end

if optsJson == ""
    out = pckern("import", char(format), char(text));
else
    out = pckern("import", char(format), char(text), char(optsJson));
end
r = jsondecode(out);
if ~r.ok
    e = r.error;
    msg = string(e.message);
    if isfield(e, "line") && isfield(e, "col")
        msg = msg + sprintf(" (line %d, column %d)", e.line, e.col);
    end
    if string(e.kind) == "undeclared" && ~isempty(skipped)
        msg = msg + "; these captured variables could not be translated " + ...
            "because they are not numeric vectors: " + join(skipped, ", ");
    end
    error(string(e.id), "%s", msg);
end
ode = string(r.ode);
if isempty(r.warnings)
    warnings = strings(0, 1);
else
    warnings = string(r.warnings(:));
end
for k = 1:numel(warnings)
    warning("phasecont:import:warning", "%s", warnings(k));
end
end
