classdef PackageTest < matlab.unittest.TestCase
    % Builds a .mltbx with scripts/packageToolbox into a temp folder and
    % inspects it. The binaries are dummy files: the test checks packaging.

    properties
        Root
        Temp
        BinFolder
        LicensesFile
        Stage
        Output
        Identifier = "5b1f7c2e-9d34-4a6b-8e0f-123456789abc"
        Version = "9.8.7"
    end

    methods (TestClassSetup)
        function addScriptsToPath(testCase)
            testCase.Root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture( ...
                fullfile(testCase.Root, "scripts")));
        end
    end

    methods (TestMethodSetup)
        function makeInputs(testCase)
            tmp = testCase.applyFixture( ...
                matlab.unittest.fixtures.TemporaryFolderFixture);
            testCase.Temp = string(tmp.Folder);
            testCase.BinFolder = fullfile(testCase.Temp, "bin");
            mkdir(testCase.BinFolder);
            writeBytes(fullfile(testCase.BinFolder, "pckern.mexw64"), uint8(1:200));
            writeBytes(fullfile(testCase.BinFolder, "pckern.dll"), uint8(255:-1:100));
            testCase.LicensesFile = fullfile(testCase.Temp, "licenses.txt");
            writelines(["Made-up license text"; "line two"], testCase.LicensesFile);
            testCase.Stage = fullfile(testCase.Temp, "package", "PhaseCont");
            testCase.Output = fullfile(testCase.Temp, "out", "Test-9.8.7.mltbx");
        end
    end

    methods (Test)
        function buildsPackageWithRequiredFiles(testCase)
            file = testCase.build();
            testCase.verifyEqual(string(file), string(testCase.Output));
            testCase.verifyTrue(isfile(testCase.Output));
            [names, ~] = testCase.unpack();
            required = ["+phasecont/Model.m", "bin/win64/pckern.mexw64", ...
                "bin/win64/pckern.dll", "examples/fitzhugh_nagumo_example.m", ...
                "examples/fitzhugh_nagumo.ode", "LICENSE", ...
                "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_LICENSES.txt"];
            for r = required
                testCase.verifyTrue(any(names == r), "missing from package: " + r);
            end
        end

        function excludesRepoFolders(testCase)
            testCase.build();
            names = testCase.unpack();
            for prefix = ["tests", ".git", "build", "target", "crates"]
                testCase.verifyFalse(any(startsWith(names, prefix)), ...
                    "package contains a path starting with " + prefix);
            end
        end

        function recordsIdentifierVersionAndPlatform(testCase)
            testCase.build();
            [~, folder] = testCase.unpack();
            addon = readMetadata(folder, "addonProperties.xml");
            testCase.verifyEqual(tagText(addon, "identifier"), testCase.Identifier);
            core = readMetadata(folder, "coreProperties.xml");
            testCase.verifyEqual(tagText(core, "cp:version"), testCase.Version);
            testCase.verifyEqual(tagText(core, "dc:title"), "PhaseCont");
            req = readMetadata(folder, "systemRequirements.xml");
            testCase.verifyEqual(attribute(req, "win"), "true");
            testCase.verifyEqual(attribute(req, "linux"), "false");
            testCase.verifyEqual(attribute(req, "mac"), "false");
            testCase.verifyEqual(attribute(req, "MATLABOnline"), "false");
            testCase.verifyEqual(attribute(req, "start"), "R2026b");
        end

        function putsRootExamplesAndBinOnThePath(testCase)
            testCase.build();
            [~, folder] = testCase.unpack();
            cfg = readMetadata(folder, "configuration.xml");
            tok = regexp(cfg, '<matlabPath>([^<]*)</matlabPath>', "tokens");
            paths = string(cellfun(@(c) c{1}, tok, UniformOutput = false));
            testCase.verifyEqual(sort(paths(:)), sort(["", "/bin/win64", "/examples"]'));
        end

        function copiesBinariesByteForByte(testCase)
            testCase.build();
            [~, folder] = testCase.unpack();
            for f = ["pckern.mexw64", "pckern.dll"]
                testCase.verifyEqual( ...
                    readBytes(fullfile(folder, "fsroot", "bin", "win64", f)), ...
                    readBytes(fullfile(testCase.BinFolder, f)));
            end
        end

        function replacesAnExistingStage(testCase)
            mkdir(testCase.Stage);
            writelines("stale", fullfile(testCase.Stage, "stale.m"));
            testCase.build();
            names = testCase.unpack();
            testCase.verifyFalse(any(names == "stale.m"));
        end

        function missingBinaryRaisesError(testCase)
            delete(fullfile(testCase.BinFolder, "pckern.dll"));
            testCase.verifyError(@() testCase.build(), ...
                "phasecont:package:missingBinary");
        end

        function emptyLicensesRaisesError(testCase)
            writelines(strings(0, 1), testCase.LicensesFile);
            testCase.verifyError(@() testCase.build(), "phasecont:package:licenses");
        end

        function missingLicensesRaisesError(testCase)
            delete(testCase.LicensesFile);
            testCase.verifyError(@() testCase.build(), "phasecont:package:licenses");
        end

        function badVersionRaisesError(testCase)
            for v = ["", "1.0", "v1.0.0", "1.0.0-beta", "a.b.c"]
                testCase.Version = v;
                testCase.verifyError(@() testCase.build(), ...
                    "phasecont:package:badVersion", "version: '" + v + "'");
            end
        end

        function stageOutsideAllowedFolderRaisesError(testCase)
            keep = fullfile(testCase.Temp, "important");
            mkdir(keep);
            writelines("keep me", fullfile(keep, "data.txt"));
            testCase.Stage = keep;
            testCase.verifyError(@() testCase.build(), "phasecont:package:badStage");
            testCase.verifyTrue(isfile(fullfile(keep, "data.txt")));
        end

        function buildfileDefinesVersionAndUuid(testCase)
            text = string(fileread(fullfile(testCase.Root, "buildfile.m")));
            testCase.verifyTrue(~isempty(regexp(text, ...
                'function\s+\w+\s*=\s*toolboxVersion[^\n]*\n(\s*%[^\n]*\n)*\s*\w+\s*=\s*"0\.3\.0"\s*;', ...
                "once")), "toolboxVersion must return ""0.3.0""");
            uuid = regexp(text, ...
                'function\s+\w+\s*=\s*toolboxUuid[^\n]*\n(?:\s*%[^\n]*\n)*\s*\w+\s*=\s*"([^"]*)"', ...
                "tokens", "once");
            testCase.verifyNotEmpty(uuid);
            testCase.verifyTrue(~isempty(regexp(uuid(1), ...
                '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', "once")));
        end
    end

    methods
        function file = build(testCase)
            file = packageToolbox(Root = testCase.Root, ...
                StageFolder = testCase.Stage, OutputFile = testCase.Output, ...
                Version = testCase.Version, Identifier = testCase.Identifier, ...
                BinFolder = testCase.BinFolder, ...
                LicensesFile = testCase.LicensesFile);
        end

        function [names, folder] = unpack(testCase)
            % Copy to .zip (unzip wants that extension), extract, list files.
            zipFile = fullfile(testCase.Temp, "copy.zip");
            copyfile(testCase.Output, zipFile);
            folder = fullfile(testCase.Temp, "unpacked");
            if isfolder(folder)
                rmdir(folder, "s");
            end
            unzip(zipFile, folder);
            listing = dir(fullfile(folder, "**", "*"));
            listing = listing(~[listing.isdir]);
            full = string(fullfile({listing.folder}, {listing.name}));
            rel = extractAfter(full, strlength(folder) + 1);
            rel = replace(rel, "\", "/");
            names = rel(startsWith(rel, "fsroot/"));
            names = extractAfter(names, "fsroot/");
            names = names(:);
        end
    end
end

function writeBytes(file, bytes)
fid = fopen(file, "w");
fwrite(fid, bytes, "uint8");
fclose(fid);
end

function bytes = readBytes(file)
fid = fopen(file, "r");
bytes = fread(fid, Inf, "uint8=>uint8");
fclose(fid);
end

function xml = readMetadata(folder, name)
xml = string(fileread(fullfile(folder, "metadata", name)));
end

function value = tagText(xml, tag)
% Text of the first <tag ...>text</tag> element
tok = regexp(xml, "<" + regexptranslate("escape", tag) + "[^>]*>([^<]*)</", ...
    "tokens", "once");
if isempty(tok)
    value = "<absent>";
else
    value = strtrim(string(tok{1}));
end
end

function value = attribute(xml, name)
tok = regexp(xml, " " + name + '="([^"]*)"', "tokens", "once");
if isempty(tok)
    value = "<absent>";
else
    value = string(tok{1});
end
end
