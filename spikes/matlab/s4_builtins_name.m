function name = s4_builtins_name(major, minor)
%S4_BUILTINS_NAME File name of the NVRTC builtins library that goes with NVRTC major.minor.
name = sprintf('nvrtc-builtins64_%d%d.dll', major, minor);
end
