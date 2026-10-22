function tf = s4_path_in_folder(p, folder)
%S4_PATH_IN_FOLDER True when the file path p lies directly in folder.
%   Windows paths: case is ignored, / counts as \, a trailing \ on folder is ignored.
if isempty(p) || isempty(folder)
    tf = false;
    return
end
tf = strcmpi(parent_of(p), strip_end(folder));
end

function d = parent_of(p)
d = regexprep(strrep(char(p), '/', '\'), '\\[^\\]*$', '');
end

function d = strip_end(f)
d = regexprep(strrep(char(f), '/', '\'), '\\+$', '');
end
