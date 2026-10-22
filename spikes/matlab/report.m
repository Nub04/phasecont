function ok = report(ok, label, detail)
% Prints "PASS label: detail" or "FAIL label: detail" and returns ok.
if nargin < 3
    detail = '';
end
if ok
    tag = 'PASS';
else
    tag = 'FAIL';
end
fprintf('%s %s: %s\n', tag, label, detail);
end
