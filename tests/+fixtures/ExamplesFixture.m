classdef ExamplesFixture < matlab.unittest.fixtures.Fixture
    % Puts <repo>/toolbox/examples on the path and restores the path on
    % teardown. Use it together with fixtures.BinFixture.

    methods
        function setup(fixture)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            old = path;
            fixture.addTeardown(@path, old);
            addpath(fullfile(root, "toolbox", "examples"));
        end
    end

    methods (Access = protected)
        function tf = isCompatible(~, other)
            tf = isa(other, "fixtures.ExamplesFixture");
        end
    end
end
