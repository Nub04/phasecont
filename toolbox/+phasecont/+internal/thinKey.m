function key = thinKey(tol)
%THINKEY The thin entry of an IVP problem JSON. Internal.
%   KEY = phasecont.internal.thinKey(TOL) is ,"thin":<TOL> as simulate
%   writes it at the end of the problem, or "" when TOL is empty.
%   continuation takes the entry out of a stored problem by matching this
%   exact text (SPEC section 8af-2, F-1).
key = "";
if ~isempty(tol)
    key = ",""thin"":" + phasecont.internal.num2json(double(tol));
end
end
