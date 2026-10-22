function s = s4_numstr(v, fmt)
%S4_NUMSTR Formats a decoded JSON number. A JSON null decodes to [] and prints as "null".
%   The s3_* helpers use this function too.
if isempty(v)
    s = 'null';
else
    s = sprintf(fmt, v);
end
end
