function names = s4_nvrtc_names()
%S4_NVRTC_NAMES Windows file names under which an NVRTC 13.x library can be loaded.
%   The name the CUDA 13 toolkit and MATLAB both ship comes first, then the
%   other Windows names from cudarc 0.19.10, src/lib.rs, get_lib_name_candidates("nvrtc")
%   for major 13, minor 0.
names = {'nvrtc64_130_0.dll', 'nvrtc.dll', 'nvrtc64.dll', 'nvrtc64_13.dll', 'nvrtc64_130.dll', ...
    'nvrtc64_10.dll', 'nvrtc64_11.dll', 'nvrtc64_12.dll', 'nvrtc64_9.dll'};
end
