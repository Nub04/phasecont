function m = strogatzModel(name)
%STROGATZMODEL Load a model of books/strogatz/models with the JIT evaluator.
%   M = strogatzModel(NAME) returns phasecont.Model for the file NAME in
%   books/strogatz/models.
arguments
    name (1,1) string
end
m = phasecont.Model(fullfile(fileparts(mfilename("fullpath")), "..", "models", name), ...
    Evaluator = "jit");
end
