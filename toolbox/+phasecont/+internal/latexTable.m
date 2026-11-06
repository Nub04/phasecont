function spec = latexTable(br)
%LATEXTABLE Table data for the special points of a Branch.
%   S = phasecont.internal.latexTable(BR) returns the struct that
%   phasecont.latex.specialTable renders: Columns (string row) and Rows
%   (struct array with Label, Type and Values).
%
%   The columns are the free parameters of BR. For a branch of
%   equilibria, and for an LP or H curve, the state variables at the
%   special point follow. A branch of cycles and a curve of cycles have no
%   single state at a point, so they get no state columns. A row label is
%   the point label with a running number per label, such as "H1" and
%   "H2". The type is the name of the point label, or the label itself when
%   this function has no name for it.
arguments
    br (1,1) phasecont.Branch
end

n = size(br.U, 2);
withStates = any(br.Type == ["EP" "LP" "H"]);
columns = br.Free;
if withStates
    if isempty(br.Model)
        columns = [columns, "u" + (1:n)];
    else
        columns = [columns, br.Model.Variables(:).'];
    end
end

sp = br.Specials;
rows = repmat(struct("Label", "", "Type", "", "Values", zeros(1, 0)), numel(sp), 1);
labels = strings(numel(sp), 1);
for k = 1:numel(sp)
    labels(k) = sp(k).Label;
    values = sp(k).P(br.FreeIndex).';
    if withStates
        values = [values, sp(k).U(:).']; %#ok<AGROW>
    end
    rows(k).Label = labels(k) + sum(labels(1:k) == labels(k));
    rows(k).Type = typeName(labels(k));
    rows(k).Values = values;
end
spec = struct("Columns", columns, "Rows", rows);
end

function name = typeName(label)
switch label
    case "EP", name = "Equilibrium";
    case "LP", name = "Limit point";
    case "BP", name = "Branch point";
    case "H", name = "Hopf";
    case "NSA", name = "Neutral saddle";
    case "LPC", name = "Limit point of cycles";
    case "PD", name = "Period doubling";
    case "NS", name = "Neimark-Sacker";
    case "CP", name = "Cusp";
    case "BT", name = "Bogdanov-Takens";
    case "ZH", name = "Zero-Hopf";
    case "GH", name = "Generalized Hopf";
    case "HH", name = "Hopf-Hopf";
    case "CPC", name = "Cusp of cycles";
    case "GPD", name = "Generalized period doubling";
    case "CH", name = "Chenciner";
    case "R1", name = "1:1 resonance";
    case "R2", name = "1:2 resonance";
    case "R3", name = "1:3 resonance";
    case "R4", name = "1:4 resonance";
    otherwise, name = label;
end
end
