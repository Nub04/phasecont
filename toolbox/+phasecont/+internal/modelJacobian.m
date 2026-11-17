function [J, why] = modelJacobian(m, u, p, iterate)
%MODELJACOBIAN The Jacobian of a model at a point, or why there is none.
%   [J, WHY] = phasecont.internal.modelJacobian(M, U, P, ITERATE) calls
%   model_jacobian (SPEC section 8ac) at the state U and the parameter
%   vector P, with ITERATE on a map. WHY is "" on success; otherwise J is
%   empty and WHY says why: no model or a deleted one, a delay or
%   non-autonomous model, a U of the wrong length, an engine refusal, or a
%   Jacobian with NaN or Inf entries. Point.classify and Branch.classify
%   turn WHY into their own error.
J = zeros(0, 0);
why = "";
if isempty(m) || ~isvalid(m) || m.Id == 0
    why = "there is no model, or the model was deleted";
elseif m.IsDelay
    why = "the model has delay terms";
elseif ~m.Autonomous
    why = "the model uses t";
elseif numel(u) ~= numel(m.Variables)
    why = sprintf("U has %d entries and the model has %d variables", ...
        numel(u), numel(m.Variables));
end
if why ~= ""
    return
end
opts = '';
if m.IsMap
    opts = char("{""iterate"":" + phasecont.internal.num2json(iterate) + "}");
end
try
    J = pckern("model_jacobian", m.Id, u(:), p(:), opts);
catch err
    if ~startsWith(err.identifier, "phasecont:engine:")
        rethrow(err);
    end
    J = zeros(0, 0);
    why = string(err.message);
    return
end
if ~all(isfinite(J), "all")
    J = zeros(0, 0);
    why = "the Jacobian has NaN or Inf entries";
end
end
