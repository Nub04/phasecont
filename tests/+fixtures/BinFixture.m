classdef BinFixture < matlab.unittest.fixtures.Fixture
    % Puts <repo>/toolbox and the folder with the pckern binaries on the
    % path, and restores the path on teardown.

    properties (SetAccess = private)
        Bin (1,1) string
    end

    methods
        function setup(fixture)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            fixture.Bin = fixtures.binFolder(root);
            old = path;
            fixture.addTeardown(@path, old);
            addpath(fullfile(root, "toolbox"));
            if isfolder(fixture.Bin)
                addpath(char(fixture.Bin));
            end
        end
    end

    methods (Access = protected)
        function tf = isCompatible(~, other)
            tf = isa(other, "fixtures.BinFixture");
        end
    end
end
