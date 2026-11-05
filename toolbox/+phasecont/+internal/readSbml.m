function text = readSbml(file)
%READSBML Read an SBML file as text, honoring its byte order mark and encoding.
%   TEXT = phasecont.internal.readSbml(FILE) decodes the file with the
%   encoding its byte order mark gives (UTF-8, UTF-16LE or UTF-16BE) or, with
%   no mark, the one in the XML declaration. A file with neither is read as
%   UTF-8 and must be valid UTF-8. The engine reads the decoded text, and
%   its XML reader ignores the encoding attribute.
%
%   Raises phasecont:import:file and phasecont:import:xml.
arguments
    file (1,1) string
end

fid = fopen(file, "r");
if fid < 0
    error("phasecont:import:file", "Cannot open the file %s.", file);
end
closer = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, "*uint8").';
clear closer

if numel(bytes) >= 3 && isequal(bytes(1:3), uint8([239 187 191]))
    text = utf8(bytes(4:end), file);
elseif numel(bytes) >= 2 && isequal(bytes(1:2), uint8([255 254]))
    text = utf16(bytes(3:end), "L", file);
elseif numel(bytes) >= 2 && isequal(bytes(1:2), uint8([254 255]))
    text = utf16(bytes(3:end), "B", file);
else
    head = char(bytes(1:min(end, 200)));
    tok = regexp(head, '^\s*<\?xml[^>]*encoding\s*=\s*["'']([^"'']+)["'']', ...
        "tokens", "once", "ignorecase");
    if isempty(tok) || any(strcmpi(tok{1}, ["UTF-8" "UTF8" "US-ASCII" "ASCII"]))
        text = utf8(bytes, file);
    else
        try
            text = string(native2unicode(bytes, tok{1}));
        catch
            error("phasecont:import:xml", ...
                "%s declares the encoding %s, which this MATLAB cannot read.", ...
                file, tok{1});
        end
    end
end
end

function text = utf8(bytes, file)
% native2unicode may replace invalid bytes without a message. Encode the
% result again and compare, so a damaged file is an error either way.
try
    chars = native2unicode(bytes, "UTF-8");
    same = isequal(unicode2native(chars, "UTF-8"), bytes);
catch
    same = false;
end
if ~same
    error("phasecont:import:xml", ...
        "%s is not valid UTF-8 and declares no other encoding.", file);
end
text = string(chars);
end

function text = utf16(bytes, order, file)
if mod(numel(bytes), 2) ~= 0
    error("phasecont:import:xml", ...
        "%s is UTF-16 text with an odd number of bytes.", file);
end
units = typecast(bytes, "uint16");
[~, ~, native] = computer;
if native ~= order
    units = swapbytes(units);
end
text = string(char(units));
end
