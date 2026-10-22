function plan = buildfile
% Build plan for PhaseCont.
%
% In MATLAB, from the repo root:  buildtool <task>
% From WSL:                       scripts/matlab-win "buildtool <task>"
%
% "buildtool -tasks" lists the tasks. The cargo, mex and package tasks stop
% with an error until the phases that implement them (P7a and P8a).

plan = buildplan(localfunctions);

plan("test") = matlab.buildtool.tasks.TestTask("tests", ...
    Description = "Run the matlab.unittest suites in tests/", ...
    TestResults = fullfile("build", "test-results", "junit.xml"));

plan("mex").Dependencies = "cargo";
plan("package").Dependencies = ["mex", "test"];
plan.DefaultTasks = "test";
end

function cargoTask(~)
% Build the Rust engine (pckern) in release mode for the current platform
error("phasecont:build:notImplemented", ...
    "The cargo task arrives in P7a. Until then, build from WSL with cargo.");
end

function mexTask(~)
% Build the MEX gateway from mex/pckern_gateway.c
% mex works when the current folder is \\wsl.localhost\... (spike S1 checked
% this), but Windows MATLAB should load binaries from C:, so this task writes
% the MEX file to the staging folder next to pckern.dll.
error("phasecont:build:notImplemented", "The mex task arrives in P7a.");
end

function packageTask(~)
% Package the toolbox as a .mltbx with matlab.addons.toolbox.ToolboxOptions
error("phasecont:build:notImplemented", ...
    "The package task arrives in P8a. Toolbox UUID: %s", toolboxUuid());
end

function id = toolboxUuid
% Fixed toolbox identifier, generated once with uuidgen. Every release must
% use the same value so that installing a new .mltbx replaces the old one.
id = "1abed604-32e4-4ec7-9cea-49389af763ff";
end
