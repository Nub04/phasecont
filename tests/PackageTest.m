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
            testCase.verifyNotEmpty(names);
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
            for v = ["1.0", "v1.0.0", "1.0.0-beta", "a.b.c"]
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

        function stageWithDotDotRaisesErrorAndKeepsFolder(testCase)
            keep = fullfile(testCase.Temp, "keepme");
            mkdir(fullfile(keep, "package"));
            writelines("marker", fullfile(keep, "marker.txt"));
            stages = [fullfile(keep, "package", ".."), ...
                fullfile(keep, "package", "."), ...
                fullfile(keep, "package") + filesep, ...
                fullfile(keep, "package", "PhaseCont") + filesep, ...
                fullfile(keep, "package", "..", "package", "PhaseCont"), ...
                "package/PhaseCont"];
            for s = stages
                testCase.Stage = s;
                testCase.verifyError(@() testCase.build(), ...
                    "phasecont:package:badStage", "stage: " + s);
            end
            testCase.verifyTrue(isfile(fullfile(keep, "marker.txt")));
            testCase.verifyTrue(isfolder(fullfile(keep, "package")));
        end

        function missingInputsRaiseErrorAndKeepStage(testCase)
            mkdir(testCase.Stage);
            marker = fullfile(testCase.Stage, "marker.txt");
            writelines("marker", marker);
            try
                testCase.build(["Identifier", "OutputFile"]);
                testCase.verifyFail("expected an error");
            catch err
                testCase.verifyEqual(string(err.identifier), ...
                    "phasecont:package:missingInput");
                testCase.verifyTrue(contains(err.message, "Identifier") ...
                    && contains(err.message, "OutputFile"));
            end
            testCase.verifyTrue(isfile(marker));
            testCase.verifyError(@() testCase.build(strings(1, 0), ...
                struct(Root = "")), "phasecont:package:missingInput");
            testCase.verifyTrue(isfile(marker));
        end

        function invalidInputsRaiseErrorAndKeepStage(testCase)
            mkdir(testCase.Stage);
            marker = fullfile(testCase.Stage, "marker.txt");
            writelines("marker", marker);
            testCase.verifyError(@() testCase.build(strings(1, 0), ...
                struct(Identifier = "not-a-uuid")), "phasecont:package:badIdentifier");
            testCase.verifyError(@() testCase.build(strings(1, 0), ...
                struct(OutputFile = fullfile(testCase.Temp, "out.txt"))), ...
                "phasecont:package:badOutput");
            noToolbox = fullfile(testCase.Temp, "noroot");
            mkdir(noToolbox);
            writelines("x", fullfile(noToolbox, "LICENSE"));
            writelines("x", fullfile(noToolbox, "THIRD_PARTY_NOTICES.md"));
            testCase.verifyError(@() testCase.build(strings(1, 0), ...
                struct(Root = noToolbox)), "phasecont:package:missingFile");
            testCase.verifyTrue(isfile(marker));
        end

        function windowsAliasesOfStageSegmentsAreRejected(testCase)
            % Windows drops trailing dots and spaces, so ".. " means "..".
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            pkg = fullfile(work, "package");
            stages = [pkg + filesep + [".. ", "...", " ", ". ", " ..", ".. ."], ...
                string(work) + filesep + ["package ", "package."] + filesep + "PhaseCont"];
            for s = stages(:)'
                testCase.Stage = s;
                testCase.verifyError(@() testCase.build(), ...
                    "phasecont:package:badStage", "stage: '" + s + "'");
            end
            testCase.Stage = origStage;
            testCase.verifyIntact(work);
        end

        function reservedDeviceNamesAreRejected(testCase)
            % NUL itself is not tested here: mkdir on it ends the MATLAB
            % process if the check ever regresses. It was checked by hand.
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            pkg = fullfile(work, "package");
            for n = ["CON", "con", "CON.txt", "nul.txt", "COM1", "lpt9", "AUX", "PRN.x"]
                testCase.Stage = fullfile(pkg, n);
                testCase.verifyError(@() testCase.build(), ...
                    "phasecont:package:badStage", "stage leaf: " + n);
            end
            testCase.Stage = fullfile(work, "CON", "package", "PhaseCont");
            testCase.verifyError(@() testCase.build(), "phasecont:package:badStage");
            testCase.Stage = origStage;
            testCase.verifyIntact(work);
        end

        function inputsInsideTheStageAreRejected(testCase)
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            inStage = fullfile(testCase.Stage, "inner");
            mkdir(fullfile(inStage, "toolbox"));
            writelines("x", fullfile(inStage, "LICENSE"));
            writelines("x", fullfile(inStage, "THIRD_PARTY_NOTICES.md"));
            bin = fullfile(testCase.Stage, "bin");
            mkdir(bin);
            writeBytes(fullfile(bin, "pckern.mexw64"), uint8(1:9));
            writeBytes(fullfile(bin, "pckern.dll"), uint8(1:9));
            lic = fullfile(testCase.Stage, "lic.txt");
            writelines("license text", lic);
            cases = {struct(Root = inStage), struct(BinFolder = bin), ...
                struct(LicensesFile = lic), struct(Root = testCase.Stage), ...
                struct(Root = fullfile(testCase.Stage, "inner", ".."))};
            for k = 1:numel(cases)
                testCase.verifyError(@() testCase.build(strings(1, 0), cases{k}), ...
                    "phasecont:package:inputInsideStage");
            end
            testCase.verifyTrue(isfile(lic));
            testCase.verifyTrue(isfile(fullfile(bin, "pckern.dll")));
            testCase.verifyTrue(isfile(fullfile(inStage, "LICENSE")));
            testCase.verifyIntact(work);
        end

        function unusableOutputFilesAreRejectedBeforeDeleting(testCase)
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            asFile = fullfile(work, "afile.txt");
            writelines("x", asFile);
            readOnly = fullfile(work, "ro.mltbx");
            writelines("old", readOnly);
            fileattrib(readOnly, "-w");
            testCase.addTeardown(@() fileattrib(readOnly, "+w"));
            asFolder = fullfile(work, "dir.mltbx");
            mkdir(asFolder);
            outputs = [fullfile(asFile, "sub", "x.mltbx"), ...
                fullfile(work, "x<.mltbx"), fullfile(work, "a|b", "x.mltbx"), ...
                fullfile(work, ".mltbx"), readOnly, asFolder, ...
                fullfile(testCase.Stage, "out", "x.mltbx")];
            for o = outputs
                testCase.verifyError(@() testCase.build(strings(1, 0), ...
                    struct(OutputFile = o)), "phasecont:package:badOutput", ...
                    "output: " + o);
            end
            % An output path that equals the stage path.
            stage = fullfile(work, "package", "X.mltbx");
            mkdir(stage);
            writelines("m", fullfile(stage, "marker.txt"));
            testCase.Stage = stage;
            testCase.verifyError(@() testCase.build(strings(1, 0), ...
                struct(OutputFile = stage)), "phasecont:package:badOutput");
            testCase.verifyTrue(isfile(fullfile(stage, "marker.txt")));
            testCase.Stage = origStage;
            testCase.verifyTrue(isfolder(asFolder));
            testCase.verifyIntact(work);
        end

        function nulLeafIsRejectedBeforeAnythingIsCreated(testCase)
            % mkdir on NUL ends the MATLAB process. A LicensesFile that does
            % not exist makes a regression fail at the licenses check, before
            % any mkdir, so the test cannot take MATLAB down.
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            pkg = fullfile(work, "package");
            for n = ["NUL", "nul"]
                testCase.Stage = fullfile(pkg, n);
                testCase.verifyError(@() testCase.build(strings(1, 0), ...
                    struct(LicensesFile = fullfile(work, "no-such-licenses.txt"))), ...
                    "phasecont:package:badStage", "stage leaf: " + n);
            end
            testCase.Stage = origStage;
            testCase.verifyIntact(work);
        end

        function uncAndMixedSeparatorsMatchTheStage(testCase)
            % The same stage written as \\localhost\C$\... must still catch
            % an input that uses forward slashes or a different case.
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            drive = extractBefore(string(work), 2);
            share = "\\localhost\" + drive + "$" + extractAfter(string(work), 2);
            if ~isfolder(share)
                testCase.assumeFail("admin share not available");
            end
            testCase.Stage = fullfile(share, "package", "PhaseCont");
            inner = fullfile(origStage, "inner");
            mkdir(fullfile(inner, "toolbox"));
            writelines("x", fullfile(inner, "LICENSE"));
            writelines("x", fullfile(inner, "THIRD_PARTY_NOTICES.md"));
            forward = replace(share, "\", "/");
            roots = [forward + "/package/PhaseCont/inner", ...
                upper(forward) + "/package/PhaseCont/inner", ...
                share + "/package/PhaseCont\inner"];
            for r = roots
                testCase.verifyError(@() testCase.build(strings(1, 0), ...
                    struct(Root = r)), "phasecont:package:inputInsideStage", ...
                    "root: " + r);
            end
            testCase.Stage = origStage;
            testCase.verifyTrue(isfile(fullfile(inner, "LICENSE")));
            testCase.verifyIntact(work);
        end

        function pathsWithoutDriveOrUncPrefixAreRejected(testCase)
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            drive = extractBefore(string(work), 2);
            % A leading separator names the root of the current drive, and
            % "C:x" is relative to a per-drive folder: neither is accepted.
            stages = ["\phasecont_r3_none\package\PhaseCont", ...
                "/phasecont_r3_none/package/PhaseCont", ...
                drive + ":package\PhaseCont"];
            for s = stages
                testCase.Stage = s;
                testCase.verifyError(@() testCase.build(), ...
                    "phasecont:package:badStage", "stage: " + s);
            end
            testCase.Stage = origStage;
            outputs = ["\phasecont_r3_none\x.mltbx", "/phasecont_r3_none/x.mltbx", ...
                drive + ":x.mltbx"];
            for o = outputs
                testCase.verifyError(@() testCase.build(strings(1, 0), ...
                    struct(OutputFile = o)), "phasecont:package:badOutput", ...
                    "output: " + o);
            end
            for n = ["Root", "BinFolder", "LicensesFile"]
                for v = ["/tmp/phasecont_r3_none", "\tmp\phasecont_r3_none", ...
                        drive + ":tmp"]
                    o = struct();
                    o.(n) = v;
                    testCase.verifyError(@() testCase.build(strings(1, 0), o), ...
                        "phasecont:package:missingFile", n + ": " + v);
                end
            end
            testCase.verifyFalse(isfolder(fullfile(drive + ":\", "phasecont_r3_none")));
            testCase.verifyIntact(work);
        end

        function shellCharactersAndHiddenLeafInTheStageAreRejected(testCase)
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            pkg = fullfile(work, "package");
            for n = ["a&b", "a;b", ".hidden"]
                testCase.Stage = fullfile(pkg, n);
                testCase.verifyError(@() testCase.build(), ...
                    "phasecont:package:badStage", "stage leaf: " + n);
            end
            testCase.Stage = fullfile(work, "x&y", "package", "PhaseCont");
            testCase.verifyError(@() testCase.build(), "phasecont:package:badStage");
            testCase.Stage = origStage;
            testCase.verifyFalse(isfolder(fullfile(pkg, "a&b")));
            testCase.verifyFalse(isfolder(fullfile(pkg, ".hidden")));
            testCase.verifyIntact(work);
        end

        function currentFolderInsideTheStageIsRejected(testCase)
            work = testCase.prepareStage();
            sub = fullfile(testCase.Stage, "sub");
            mkdir(sub);
            old = cd(sub);
            testCase.addTeardown(@() cd(old));
            testCase.verifyError(@() testCase.build(), "phasecont:package:badStage");
            cd(testCase.Stage);
            testCase.verifyError(@() testCase.build(), "phasecont:package:badStage");
            cd(old);
            testCase.verifyTrue(isfolder(sub));
            testCase.verifyIntact(work);
        end

        function missingExamplesFolderIsCaughtBeforeDeleting(testCase)
            work = testCase.prepareStage();
            root = fullfile(testCase.Temp, "noexamples");
            mkdir(fullfile(root, "toolbox", "+phasecont"));
            writelines("x", fullfile(root, "toolbox", "+phasecont", "f.m"));
            writelines("x", fullfile(root, "LICENSE"));
            writelines("x", fullfile(root, "THIRD_PARTY_NOTICES.md"));
            testCase.verifyError(@() testCase.build(strings(1, 0), ...
                struct(Root = root)), "phasecont:package:missingFile");
            testCase.verifyIntact(work);
        end

        function returnsTheNormalizedOutputPath(testCase)
            file = testCase.build(strings(1, 0), ...
                struct(OutputFile = testCase.Output + filesep));
            testCase.verifyEqual(string(file), string(testCase.Output));
            testCase.verifyTrue(isfile(file));
        end

        function strictVersionsAreEnforced(testCase)
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            arabic = string(char([1632 46 1635 46 1632]));
            for v = ["00.03.000", "1.2.4294967296", arabic, "0.3.0" + newline, ...
                    "99999999999.0.0"]
                testCase.verifyError(@() testCase.build(strings(1, 0), ...
                    struct(Version = v)), "phasecont:package:badVersion");
            end
            testCase.verifyIntact(work);
        end

        function nonTextInputsAreTreatedAsMissing(testCase)
            work = testCase.prepareStage();
            origStage = testCase.Stage;
            names = ["Root", "StageFolder", "OutputFile", "Version", ...
                "Identifier", "BinFolder", "LicensesFile"];
            for n = names
                for v = {NaN, string(missing), 5, ["a" "b"], {}}
                    o = struct();
                    o.(n) = v{1};
                    testCase.verifyError(@() testCase.build(strings(1, 0), o), ...
                        "phasecont:package:missingInput", n + " as " + class(v{1}));
                end
            end
            testCase.verifyIntact(work);
        end

        function summaryAndDescriptionMakeOnlyTrueClaims(testCase)
            testCase.build();
            [~, folder] = testCase.unpack();
            addon = readMetadata(folder, "addonProperties.xml");
            core = readMetadata(folder, "coreProperties.xml");
            summary = tagText(core, "dc:description");
            description = tagText(addon, "description");
            testCase.verifyEqual(summary, ...
                "Equilibrium continuation and bifurcation detection for XPP .ode models.");
            for text = [summary, description]
                testCase.verifyNotEmpty(char(text));
                testCase.verifyEmpty(regexpi(text, ...
                    'phase[- ]plane|limit cycle|linux|macos', "once"));
                testCase.verifyTrue(all(double(char(text)) < 128), ...
                    "text must be ASCII (no em dashes): " + text);
            end
        end

        function packageHasNoDotfilesOrStrayBinaries(testCase)
            testCase.build();
            names = testCase.unpack();
            leafs = regexp(names, '[^/]*$', "match", "once");
            testCase.verifyFalse(any(startsWith(leafs, ".")), ...
                "package contains a dotfile");
            binary = endsWith(names, [".dll", ".mexw64", ".so", ".dylib"]);
            testCase.verifyTrue(all(startsWith(names(binary), "bin/win64/")));
        end

        function licenseFilesMatchTheirSources(testCase)
            testCase.build();
            [~, folder] = testCase.unpack();
            pairs = {"LICENSE", fullfile(testCase.Root, "LICENSE"); ...
                "THIRD_PARTY_NOTICES.md", ...
                fullfile(testCase.Root, "THIRD_PARTY_NOTICES.md"); ...
                "THIRD_PARTY_LICENSES.txt", testCase.LicensesFile};
            for k = 1:size(pairs, 1)
                testCase.verifyEqual( ...
                    readBytes(fullfile(folder, "fsroot", pairs{k, 1})), ...
                    readBytes(pairs{k, 2}), pairs{k, 1});
            end
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
            testCase.verifyEqual(uuid(1), "1abed604-32e4-4ec7-9cea-49389af763ff");
            testCase.verifyTrue(~isempty(regexp(uuid(1), ...
                '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', "once")));
        end
    end

    methods
        function work = prepareStage(testCase)
            % Stage with a marker inside it and a sibling next to package/.
            % Returns the folder that holds package/.
            mkdir(testCase.Stage);
            writelines("m", fullfile(testCase.Stage, "marker.txt"));
            work = fullfile(testCase.Temp);
            writelines("s", fullfile(work, "sibling.txt"));
            writelines("s", fullfile(work, "package", "sibling.txt"));
        end

        function verifyIntact(testCase, work)
            testCase.verifyTrue(isfile(fullfile(testCase.Stage, "marker.txt")), ...
                "stage marker was deleted");
            testCase.verifyTrue(isfile(fullfile(work, "sibling.txt")), ...
                "sibling of package/ was deleted");
            testCase.verifyTrue(isfile(fullfile(work, "package", "sibling.txt")), ...
                "sibling inside package/ was deleted");
            testCase.verifyTrue(isfile(testCase.BinFolder + "/pckern.dll"));
        end

        function file = build(testCase, omit, override)
            % Call packageToolbox with the test inputs. omit names inputs to
            % leave out; override is a struct of replacement values.
            arguments
                testCase
                omit (1,:) string = strings(1, 0)
                override (1,1) struct = struct()
            end
            args = struct(Root = testCase.Root, StageFolder = testCase.Stage, ...
                OutputFile = testCase.Output, Version = testCase.Version, ...
                Identifier = testCase.Identifier, BinFolder = testCase.BinFolder, ...
                LicensesFile = testCase.LicensesFile);
            for f = string(fieldnames(override))'
                args.(f) = override.(f);
            end
            args = rmfield(args, omit);
            nv = namedargs2cell(args);
            file = packageToolbox(nv{:});
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
