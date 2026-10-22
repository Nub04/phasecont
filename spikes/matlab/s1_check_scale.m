function ok = s1_check_scale(label, f, x)
%S1_CHECK_SCALE Checks that f(x) equals 2*x, with the size of x and class double.
%   f is the call under test, for example @(x) pckern("scale", x, 2). NaN values
%   compare equal, and the sign of zero must match too. An error raised by f
%   counts as a failure, not as an abort.
try
    y = f(x);
catch e
    ok = report(false, label, sprintf('%s | %s', e.identifier, e.message));
    return
end
ok = isequaln(y, 2 * x) && isequal(size(y), size(x)) && isa(y, 'double') && isequaln(1 ./ y, 1 ./ (2 * x));
ok = report(ok, label, sprintf('size %s, class %s', mat2str(size(y)), class(y)));
end
