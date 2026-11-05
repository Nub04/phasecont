function [devices, backends] = gpuDevices()
%GPUDEVICES List the devices that can run an ensemble.
%   DEVICES = phasecont.gpuDevices() returns a table with one row per device
%   the engine found: the CPU, and every CUDA and OpenCL device. The columns
%   are
%     Backend    "cpu", "cuda" or "opencl"
%     Index      position among the devices of that backend, counting from 0.
%                This is the value of the Device option of phasecont.ensemble
%                and the --device value of pc-cli
%     Name       device name as the driver reports it
%     Driver     driver or runtime version string
%     Fp32       true when the device runs single precision
%     Fp64       true when the device runs double precision
%     MemoryMiB  device memory in MiB, NaN when the driver does not say
%
%   [DEVICES, BACKENDS] = phasecont.gpuDevices() also returns one row per
%   backend with its State: "available", "not_built" (this build does not
%   contain it) or "unavailable" (Why says what is missing, for example a
%   driver or NVRTC). A backend with no devices is not an error.
%
%   The first call loads the GPU libraries, which can take a moment. A
%   missing library only changes the State of its backend.
raw = jsondecode(string(pckern("ens_devices")));

d = phasecont.internal.jsonRows(raw.devices, ...
    ["backend" "index" "name" "driver" "fp32" "fp64" "memory_bytes"]);
bytes = nan(numel(d), 1);
for k = 1:numel(d)
    if ~isempty(d(k).memory_bytes)
        bytes(k) = d(k).memory_bytes / 2^20;
    end
end
devices = table(col(string({d.backend})), col(double([d.index])), ...
    col(string({d.name})), col(string({d.driver})), ...
    col(logical([d.fp32])), col(logical([d.fp64])), bytes, ...
    VariableNames = ["Backend" "Index" "Name" "Driver" "Fp32" "Fp64" ...
        "MemoryMiB"]);

if nargout > 1
    b = phasecont.internal.jsonRows(raw.backends, ["backend" "state" "why"]);
    why = strings(numel(b), 1);
    for k = 1:numel(b)
        if ~isempty(b(k).why)
            why(k) = string(b(k).why);
        end
    end
    backends = table(col(string({b.backend})), col(string({b.state})), why, ...
        VariableNames = ["Backend" "State" "Why"]);
end
end

function v = col(v)
% A column, 0-by-1 when empty.
v = reshape(v, [], 1);
end
