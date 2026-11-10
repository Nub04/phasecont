function ok = aotCacheOwnerOk(statText, uidText)
%AOTCACHEOWNEROK True if a cache folder belongs to the user and is private.
%   OK = phasecont.internal.aotCacheOwnerOk(STATTEXT, UIDTEXT) reads the
%   output of stat -c '%u %a' (owner id and octal mode) and the output of
%   id -u. It returns true when the owner is the current user and neither
%   the group nor others can write to the folder. Any text that does not
%   have that form gives false.
arguments
    statText (1,1) string
    uidText (1,1) string
end
ok = false;
uid = strtrim(uidText);
parts = split(strtrim(statText));
if numel(parts) ~= 2 || isempty(regexp(uid, "^[0-9]+$", "once")) ...
        || isempty(regexp(parts(1), "^[0-9]+$", "once")) ...
        || isempty(regexp(parts(2), "^[0-7]{3,4}$", "once"))
    return
end
digits = char(parts(2)) - '0';
groupWrite = bitand(digits(end - 1), 2) ~= 0;
otherWrite = bitand(digits(end), 2) ~= 0;
ok = parts(1) == uid && ~groupWrite && ~otherWrite;
end
