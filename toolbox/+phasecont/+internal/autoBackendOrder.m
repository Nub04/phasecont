function order = autoBackendOrder(onMac)
%AUTOBACKENDORDER Order in which Backend = "auto" tries the backends.
%   ORDER = phasecont.internal.autoBackendOrder() returns the order for the
%   platform MATLAB runs on. ORDER = phasecont.internal.autoBackendOrder(ONMAC)
%   returns the order for macOS when ONMAC is true and for Windows and Linux
%   otherwise, so tests can check both on any platform.
%
%   On macOS the order is Metal, OpenCL, CPU. Apple deprecated OpenCL in
%   favor of Metal, and CUDA does not run on macOS. Elsewhere the order is
%   CUDA, OpenCL, Metal, CPU.
arguments
    onMac (1,1) logical = ismac
end
if onMac
    order = ["metal" "opencl" "cpu"];
else
    order = ["cuda" "opencl" "metal" "cpu"];
end
end
