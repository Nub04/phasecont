classdef AotCacheRootTest < matlab.unittest.TestCase
    % The AOT cache holds libraries that MATLAB loads, so on Linux (a shared
    % /tmp) the cache folder must be owned by the current user and must not
    % be writable by anyone else.

    methods (TestClassSetup)
        function addToolbox(testCase)
            root = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture( ...
                matlab.unittest.fixtures.PathFixture(fullfile(root, "toolbox")));
        end
    end

    methods (Test)
        function acceptsAnOwnedFolderNobodyElseCanWrite(testCase)
            testCase.verifyTrue(phasecont.internal.aotCacheOwnerOk("1000 700", "1000"));
            testCase.verifyTrue(phasecont.internal.aotCacheOwnerOk("1000 755", "1000"));
            testCase.verifyTrue(phasecont.internal.aotCacheOwnerOk("1000 750" + newline, "1000" + newline));
        end

        function refusesAnotherOwner(testCase)
            testCase.verifyFalse(phasecont.internal.aotCacheOwnerOk("1001 700", "1000"));
            testCase.verifyFalse(phasecont.internal.aotCacheOwnerOk("0 755", "1000"));
        end

        function refusesGroupOrOtherWrite(testCase)
            for mode = ["775", "770", "777", "757", "702", "720", "1777"]
                testCase.verifyFalse( ...
                    phasecont.internal.aotCacheOwnerOk("1000 " + mode, "1000"), mode);
            end
        end

        function readsAModeWithASpecialBit(testCase)
            testCase.verifyTrue(phasecont.internal.aotCacheOwnerOk("1000 2755", "1000"));
            testCase.verifyFalse(phasecont.internal.aotCacheOwnerOk("1000 2775", "1000"));
        end

        function refusesOutputItCannotRead(testCase)
            for text = ["", "stat: cannot stat", "1000", "1000 9z9", "abc 700", "1000 700 extra"]
                testCase.verifyFalse( ...
                    phasecont.internal.aotCacheOwnerOk(text, "1000"), text);
            end
            testCase.verifyFalse(phasecont.internal.aotCacheOwnerOk("1000 700", ""));
            testCase.verifyFalse(phasecont.internal.aotCacheOwnerOk("1000 700", "me"));
        end

        function rootIsTheTempFolderOnWindowsAndMac(testCase)
            testCase.assumeFalse(isunix && ~ismac, "Linux checks the owner");
            testCase.verifyEqual(phasecont.internal.aotCacheRoot(), ...
                string(fullfile(tempdir, "phasecont-aot")));
            testCase.verifyTrue(isfolder(phasecont.internal.aotCacheRoot()));
        end

        function linuxRootIsOwnedByTheCurrentUserAndPrivate(testCase)
            testCase.assumeTrue(isunix && ~ismac, "Linux only");
            root = phasecont.internal.aotCacheRoot();
            [~, out] = system("stat -c '%u %a' '" + root + "'");
            [~, uid] = system("id -u");
            testCase.verifyTrue(phasecont.internal.aotCacheOwnerOk(out, uid));
        end
    end
end
