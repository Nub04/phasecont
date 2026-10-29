classdef BaseMatlabTest < matlab.unittest.TestCase
    % Checks that the toolbox code needs only base MATLAB (CLAUDE.md rule 8).

    methods (TestClassSetup)
        function addToolboxToPath(testCase)
            % Imports in the toolbox code resolve only when the toolbox
            % root is on the path.
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function onlyBaseMatlabIsRequired(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            fileList = toolboxFiles(root);
            testCase.verifyNotEmpty(fileList, "no .m files under toolbox/");
            [~, products] = matlab.codetools.requiredFilesAndProducts(fileList);
            testCase.verifyNotEmpty(products, ...
                "requiredFilesAndProducts returned no products");
            names = unique(string({products.Name}));
            offending = names(names ~= "MATLAB");
            testCase.verifyEmpty(offending, ...
                "toolbox needs products other than base MATLAB: " + ...
                join(offending, ", "));
        end

        function fileListHasCoreFiles(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            fileList = toolboxFiles(root);
            testCase.verifyNotEmpty(fileList);
            required = fullfile(root, "toolbox", "+phasecont", ...
                ["Model.m" "Branch.m" "continuation.m"]);
            testCase.verifyTrue(all(ismember(required, string(fileList))), ...
                "missing from the file list: " + ...
                join(required(~ismember(required, string(fileList))), ", "));
        end
    end
end

function fileList = toolboxFiles(root)
% Returns the full paths of every .m file under toolbox/, recursively.
entries = dir(fullfile(root, "toolbox", "**", "*.m"));
fileList = string(fullfile({entries.folder}, {entries.name}))';
end
