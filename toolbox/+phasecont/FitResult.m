classdef FitResult < handle
    % The result of phasecont.fit: parameters fitted to trajectory data.
    %
    %   r = phasecont.fit(m, x0, P, t, Y, Fit = "a", Observe = "x", ...)
    %   r.Estimate        % fitted values, in the order of Fit
    %   r.Parameters.a    % the same by name
    %   r.Residual        % model minus data, S-by-K
    %
    % A FitResult holds copied data. It is not saved in a session.
    %
    % Properties
    %   Fit         names of the fitted parameters, 1-by-|Fit| string
    %   Observe     names of the observed variables, 1-by-K string
    %   Times       data times, S-by-1
    %   Estimate    fitted values, |Fit|-by-1
    %   P           all parameters, np-by-1, the fitted values in place
    %   Parameters  struct of the fitted values by name
    %   Fitted      model values at Times, S-by-K, at the estimate
    %   Residual    Fitted minus the data, S-by-K
    %   Cost        F = 1/2 * sum(Residual.^2, "all") at the estimate
    %   GradNorm    largest absolute entry of J'*f at the estimate
    %   Jacobian    S*K-by-|Fit| forward-difference Jacobian at the estimate;
    %               rows follow Residual(:)
    %   Iterations  trial steps taken
    %   Converged   true when the gradient or the step test stopped the run
    %   Stop        "gradient", "step" or "maxIter", or "failed" for a start
    %               of a fit with Starts whose fit raised an error
    %   History     table with one row per trial step: Iter, Cost, GradNorm,
    %               StepNorm, Mu, Rho, Accepted. Cost and GradNorm are those
    %               of the iterate after the step (unchanged when the step is
    %               rejected). Rho is NaN when the trial point did not finish.
    %   Starts      1-by-K FitResult, one per column of the Starts option in
    %               input order; 1-by-0 for a fit without Starts and for
    %               each entry of Starts
    %   BestStart   index in Starts of this result; empty without Starts
    %   Error       the MException of a start with Stop "failed", the one
    %               a fit from that start alone raises; empty otherwise

    properties (SetAccess = private)
        Fit (1,:) string = strings(1,0)
        Observe (1,:) string = strings(1,0)
        Times (:,1) double = zeros(0,1)
        Estimate (:,1) double = zeros(0,1)
        P (:,1) double = zeros(0,1)
        Parameters (1,1) struct = struct()
        Fitted (:,:) double = zeros(0,0)
        Residual (:,:) double = zeros(0,0)
        Cost (1,1) double = NaN
        GradNorm (1,1) double = NaN
        Jacobian (:,:) double = zeros(0,0)
        Iterations (1,1) double = 0
        Converged (1,1) logical = false
        Stop (1,1) string = ""
        History table = table()
        Starts                        % 1-by-K FitResult, set in the constructor
        BestStart double = []
        Error = MException.empty
    end

    methods
        function obj = FitResult(values)
            % Internal use: values is a struct with the property names as fields.
            arguments
                values (1,1) struct = struct()
            end
            obj.Starts = phasecont.FitResult.empty(1, 0);
            for name = string(fieldnames(values))'
                obj.(name) = values.(name);
            end
        end
    end
end
