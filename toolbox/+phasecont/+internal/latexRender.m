function [tex, files] = latexRender(kind, json)
%LATEXRENDER Run the LaTeX renderer of the engine.
%   [TEX, FILES] = phasecont.internal.latexRender(KIND, JSON) calls
%   pckern("latex", KIND, JSON) with KIND "pgfplots" or "table". TEX is the
%   LaTeX text. FILES is a struct array (column) with fields Name and
%   Content, the data files that external data needs, empty otherwise.
%
%   A document the renderer rejects raises its own error id,
%   phasecont:latex:<kind>, with kind one of missingDigits, badDigits,
%   nonFinite, emptySeries, emptyTable, lengthMismatch, badName and
%   badInput. The message names the series, row or key at fault.
arguments
    kind (1,1) string {mustBeMember(kind, ["pgfplots" "table"])}
    json (1,1) string {mustBeNonmissing}
end

out = pckern("latex", char(kind), char(json));
r = jsondecode(out);
if ~r.ok
    e = r.error;
    error(string(e.id), "%s", string(e.message));
end
tex = string(r.tex);
n = numel(r.data_files);
files = repmat(struct("Name", "", "Content", ""), n, 1);
for k = 1:n
    files(k).Name = string(r.data_files(k).name);
    files(k).Content = string(r.data_files(k).content);
end
end
