function pt = equilibrium(m, x0, options)
%EQUILIBRIUM Find an equilibrium of a model by Newton's method.
%   PT = phasecont.equilibrium(M, X0) solves f(x, p) = 0 from the guess X0
%   at the parameter values of the model and returns a phasecont.Point with
%   Label "EP", its eigenvalues and the parameters used.
%
%   Name-value options
%     Params   struct or numeric vector of parameter values; a struct changes
%              only the fields it names (default M.ParameterValues)
%     MaxIter  most Newton iterations (default 50)
%     Tol      tolerance on the step and the residual (default 1e-10)
%
%   If Newton's method does not converge, the call raises the engine error
%   phasecont:engine:numeric.
arguments
    m (1,1) phasecont.Model
    x0 {mustBeNumeric, mustBeReal, mustBeVector}
    options.Params = []
    options.MaxIter (1,1) {mustBeNumeric, mustBeInteger, mustBePositive} = 50
    options.Tol (1,1) {mustBeNumeric, mustBeReal} = 1e-10
end
if ~(isfinite(options.Tol) && options.Tol > 0)
    error("phasecont:model:badOption", "Tol must be a positive finite number.");
end
if isempty(options.Params) && ~isstruct(options.Params)
    p = m.ParameterValues;
else
    p = m.paramVector(options.Params);
end
opts = "{""max_newton"":" + phasecont.internal.num2json(options.MaxIter) + ...
    ",""tol_x"":" + phasecont.internal.num2json(options.Tol) + ...
    ",""tol_f"":" + phasecont.internal.num2json(options.Tol) + "}";
[u, ev] = pckern("model_equilibrium", m.Id, double(x0(:)), p, char(opts));
record = phasecont.internal.RunRecord(struct( ...
    "Kind", "equilibrium", "Problem", opts, "U0", double(x0(:)), "P", p(:), ...
    "ModelSha", m.SourceSha256, "N", numel(m.Variables), ...
    "Result", struct("U", u(:), "Eigenvalues", ev(:))));
pt = phasecont.Point(U = u, P = p, Eigenvalues = ev, Label = "EP", Model = m, ...
    Record = record);
end
