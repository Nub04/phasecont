# Security policy

## Reporting a problem

Please do not open a public issue for a security problem. Use the "Report a
vulnerability" button on the repository's Security tab, which sends a private
report to the maintainer. Include the PhaseCont version, the operating system,
the MATLAB release and the smallest model, session or folder that shows the
problem.

You should get an answer within 7 days. A fix goes into the next release, and
the release notes name the problem once the fix is out.

## Supported versions

Only the latest release gets security fixes. Code on `main` that is not yet
released is fixed on `main`.

## What counts

PhaseCont reads `.ode` model files and session files, compiles models to native
code inside MATLAB and loads GPU libraries. A problem is a security problem
when opening such a file, or running PhaseCont from a folder someone else
prepared, can run code, write files outside the folders PhaseCont documents, or
send data off the machine. A crash or a wrong result from a hostile file is a
normal bug unless it corrupts memory.
