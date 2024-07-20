#!/usr/bin/env python3
#

import sys, optparse, subprocess, platform, glob

distDir = "/Users/erwin/coding/pyimfit/dist/"

WHEEL_NAME_TEMPLATE = "pyimfit-{0}-cp{1}-cp{1}-{2}.whl"
WHEEL_PREFIX_TEMPLATE = "pyimfit-{0}-"
# delocate-wheel -w fixed_wheels -v pyimfit-${VERSION_NUM}-cp312-cp312-${WHEEL_SUFFIX}.whl

def main( argv=None ):

    usageString = "%prog <version-number> [options]\n"
    parser = optparse.OptionParser(usage=usageString, version="%prog ")


    parser.add_option("--test-upload", action="store_true", dest="doTestUpload",
                                      default=False, help="do test upload (to TestPyPI)")
    parser.add_option("--upload", action="store_true", dest="doUpload",
                                      default=False, help="Upload to PyPI)")
    parser.add_option("--skip-build", action="store_false", dest="doBuild",
                                      default=True, help="Skip the build process")

    (options, args) = parser.parse_args(argv)
    # args[0] = name program was called with
    # args[1] = first actual argument, etc.
    if len(argv) < 2:
        print("You must supply a version number!\n")
        return None
    versionNum = args[1]
    # Figure out which type of macOS architecture we're running under
    proctype = platform.processor()
    if proctype == "arm":
        usingAppleSilicon = True
        prelimString = "export _PYTHON_HOST_PLATFORM='macosx-11.0-arm64' ; export ARCHFLAGS='-arch arm64' ; "
        wheelSuffix = "macosx_11_0_arm64"
        pythonVersionList = ["3.10", "3.11", "3.12"]
    else:
        usingAppleSilicon = False
        prelimString = "export _PYTHON_HOST_PLATFORM='macosx-10.9-x86_64' ; export ARCHFLAGS='-arch x86_64' ; "
        wheelSuffix = "macosx_10_9_x86_64"
        pythonVersionList = ["3.8", "3.9", "3.10", "3.11", "3.12"]

    if options.doBuild:
        # Make sdist (.tar.gz) and macOS binary wheels
        # Note that this particular formatting of cmdLine is necessary (trying to define the environment
        # variables separately and passing them to subprocess.run via its "env" keyword ends up producing
        # an error when wheel.py tries to parse the output of distutils.util.get_platform)
        for pythonVersion in pythonVersionList:
            cmdLine = prelimString + "python" + pythonVersion + " setup.py sdist bdist_wheel"
            print(cmdLine)
            result = subprocess.run([cmdLine], shell=True)

        # Copy shared libs into wheel using delocate
        print("\nRunning delocate...\n")
        for pythonVersion in pythonVersionList:
            vname = pythonVersion.replace(".", "")
            wheelname = WHEEL_NAME_TEMPLATE.format(versionNum, vname, wheelSuffix)
            cmdLine = "cd dist ; delocate-wheel -w fixed_wheels -v {0}".format(wheelname)
            print(cmdLine)
            result = subprocess.run([cmdLine], shell=True)
    else:
        print("Skipping build phase...")

    # Upload processed wheels

    if options.doUpload or options.doTestUpload:
        if options.doUpload:
            print("\nUploading to PyPi...")
            repoString = ""
        else:
            print("\nUploading to TestPyPi...")
            repoString = "-r testpypi"
        # source distribution:
        cmdLine = "python3 -m twine upload {0} dist/pyimfit-{1}.tar.gz".format(repoString, versionNum)
        print(cmdLine)
        result = subprocess.run([cmdLine], shell=True)
        # binary wheels
        wheel_prefix = WHEEL_PREFIX_TEMPLATE.format(versionNum)
        wheelList = glob.glob(distDir + "{0}*.whl".format(wheel_prefix))
        for wheelname in wheelList:
            cmdLine = "python3 -m twine upload {0} {1}".format(repoString, wheelname)
            print(cmdLine)
            result = subprocess.run([cmdLine], shell=True)

    print("\nDone!")





if __name__ == '__main__':

    main(sys.argv)
