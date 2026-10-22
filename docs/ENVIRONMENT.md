# Development environment

This file records what the development machine has, as found on 2026-10-21 during Phase 0. The tables show the state after the Phase 0 installs later that day (see the last section). Each section names the commands that produced its values. Re-run them after a toolchain change and update this file.

## Host

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 255H; Windows reports 16 cores and 16 logical processors |
| Windows | Windows 11, `ver` prints `Microsoft Windows [Version 10.0.26300.9711]` |
| WSL distro | Fedora Linux 45 (WSL), `VARIANT="WSL"`, `RELEASE_TYPE=stable`. It read "45 (WSL Prerelease)" until the `fedora-release` 45-13 packages were installed at 15:12Z on 2026-10-21. Its WSL name is `FedoraLinux-44` (`wsl.exe -l -v`), so Windows sees the repo as `\\wsl.localhost\FedoraLinux-44\home\nub04\dev\phasecont` |
| WSL kernel | `6.18.54.1-microsoft-standard-WSL2` |
| WSL limits | `.wslconfig` sets `processors=12`, `memory=20GB`, `swap=8GB` and `networkingMode=consomme` |
| Free disk | 856 GB on the WSL root file system, 130 GB on `C:` |
| Smart App Control | on: `VerifiedAndReputablePolicyState` is 1 under `HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`. It blocked one test executable that `cargo test` built for `x86_64-pc-windows-gnu` (CodeIntegrity events 3033 and 3077). It has not blocked `pckern.dll` or `pckern.mexw64` |

WSL sees 12 processors (`nproc` prints 12), while Windows MATLAB sees all 16 (`maxNumCompThreads` is 16). A parallel timing taken in WSL is not directly comparable with one taken in MATLAB.

Commands: `grep -E '^(PRETTY_NAME|VERSION_ID|VARIANT|RELEASE_TYPE)=' /etc/os-release`, `rpm -q --qf '%{NAME} %{VERSION}-%{RELEASE} %{INSTALLTIME:date}\n' fedora-release-common`, `wsl.exe -l -v`, `uname -r`, `nproc`, `lscpu`, `df -h --output=target,avail ~ /mnt/c`, `cat /mnt/c/Users/<user>/.wslconfig`, `cmd.exe /c ver`, and, through `powershell.exe`, `Get-CimInstance Win32_Processor`, `Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy'` and `Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational'`.

## Rust in WSL

| Item | Value |
|---|---|
| rustc | `rustc 1.99.1 (58cf8f9c3 2026-10-13) (Fedora 1.99.1-2.fc45)`, host `x86_64-unknown-linux-gnu`, LLVM 23.1.2 |
| cargo | `cargo 1.99.1 (5f94df478 2026-08-27) (Fedora 1.99.1-2.fc45)` |
| Other tools | rustfmt 1.10.0, clippy 0.1.99, rust-analyzer 1.99.1 |
| Packages | `rust`, `cargo`, `rust-std-static`, `rust-std-static-x86_64-pc-windows-gnu`, `rust-src`, `rustfmt`, `clippy` and `rust-analyzer`, all at `1.99.1-2.fc45` |
| Targets | `x86_64-unknown-linux-gnu` (host) and `x86_64-pc-windows-gnu` |
| rustup | not installed |

The toolchain comes from Fedora packages, from the `updates-testing` repo, which is disabled by default on this machine. The install commands in the last section enable it for one call at a time.

Discovery found the set at `1.99.0-1.fc45` with only the host std. Fedora offered the Windows std (`rust-std-static-x86_64-pc-windows-gnu`) only at 1.98.1 in `fedora` and 1.99.1 in `updates-testing`, and a target std must come from the same build as `rustc`. The install therefore moved the whole set to 1.99.1. Spike results from WSL record the `rustc` of each run, and runs before 16:20Z used 1.99.0.

`updates-testing` also has 1.99.1 std packages for `i686-pc-windows-gnu`, `wasm32-unknown-unknown`, `wasm32-wasip1`, `x86_64-unknown-none`, `x86_64-unknown-uefi`, `aarch64-unknown-uefi` and `aarch64-unknown-none-softfloat`.

Commands: `rustc -vV`, `cargo -V`, `rustfmt -V`, `cargo clippy -V`, `rust-analyzer --version`, `rpm -q rust cargo rust-std-static rust-std-static-x86_64-pc-windows-gnu rust-src rustfmt clippy rust-analyzer`, `dnf list --installed 'rust-std-static*'`, and `dnf list --available --showduplicates --enablerepo=updates-testing 'rust-std-static*'`.

## C toolchains in WSL

| Tool | Version |
|---|---|
| gcc | `gcc (GCC) 16.2.1 20260819 (Red Hat 16.2.1-2)` |
| clang | `clang version 23.1.3 (Fedora 23.1.3-1.fc45)` |
| cmake | 4.3.0 |
| g++-15 | `g++-15 (GCC) 15.3.1 20260722 (Red Hat 15.3.1-1)`, the CUDA host compiler; `NVCC_CCBIN=g++-15` is set |
| x86_64-w64-mingw32-gcc | `x86_64-w64-mingw32-gcc (GCC) 16.2.1 20260819 (Fedora MinGW 16.2.1-1.fc45)`, from `mingw64-gcc` 16.2.1-1.fc45 |
| MinGW binutils | `mingw64-binutils` 2.46.0-2.fc45, which has `x86_64-w64-mingw32-dlltool` and `x86_64-w64-mingw32-objdump` |
| MinGW runtime | `mingw64-crt` and `mingw64-winpthreads-static` 14.0.0-3.fc45, `mingw64-headers` 14.0.0-2.fc45 |

The linker for `x86_64-pc-windows-gnu` is set in `spikes/.cargo/config.toml`. `x86_64-w64-mingw32-objdump -p pckern.dll | grep 'DLL Name'` lists the DLLs a build imports.

## Cargo tools

| Tool | Installed | Fedora package |
|---|---|---|
| cargo-deny | yes, 0.19.9 | `cargo-deny` 0.19.9-2.fc45 |
| cbindgen | no | `cbindgen` 0.29.4-2.fc45 |
| cargo-zigbuild | no | none |
| cargo-xwin | no | none |
| zig | no | `zig` 0.16.0-2.fc45 |

Commands: `command -v` for each tool, and `dnf list --available` for the package names.

## git

WSL has git 2.56.0 with `user.name` and `user.email` set globally and `init.defaultBranch` unset. The repo was created with `git init -b main`. Git for Windows is also installed, at `C:\Program Files\Git\cmd\git.exe`.

## Windows toolchains

| Item | Value |
|---|---|
| Rust | `cargo`, `rustc` and `rustup` are not on the Windows `PATH` |
| Visual Studio | Visual Studio Community 2026, version 18.11.0 (`installationVersion` 18.11.12306.390), at `C:\Program Files\Microsoft Visual Studio\18\Community` |
| MSVC toolsets | 14.51.36231 and 14.52.36815 |
| Windows SDK | 10.0.26100.0 |

Commands: `Get-Command cargo, rustc, rustup, git, nvcc` through `powershell.exe`, `vswhere.exe -all -products '*' -format text`, and directory listings of `VC\Tools\MSVC` and `C:\Program Files (x86)\Windows Kits\10\Lib`.

## MATLAB

| Item | Value |
|---|---|
| Release | `26.2.0.3386108 (R2026b)` |
| `matlabroot` | `C:\Program Files\MATLAB\R2026b` |
| `computer('arch')` | `win64` |
| MEX C compiler | Microsoft Visual C++ 2026 (C), version 18.0, at `C:\Program Files\Microsoft Visual Studio\18\Community\` |
| `maxNumCompThreads` | 16 |
| `gpuDevice` | NVIDIA GeForce RTX 5060 Laptop GPU, `ComputeCapability` 12.0, `DeviceSupported` 1, `DriverVersion` 13.4, `ToolkitVersion` 13.1 |
| Import libraries | `extern\lib\win64\microsoft` and `extern\lib\win64\mingw64` each hold `libmx.lib`, `libmex.lib`, `libmat.lib`, `libMatlabDataArray.lib` and `libMatlabEngine.lib`; `extern\lib\win64` also holds `libMatlabCppSharedLib.lib` |
| CUDA DLLs in `bin\win64` | `nvrtc64_130_0.dll`, `nvrtc-builtins64_131.dll`, `cudart64_13.dll` and `cublas64_13.dll` |

`scripts/matlab-win "mex -setup C"` printed `MEX configured to use 'Microsoft Visual C++ 2026 (C)' for C language compilation.` and exited 0 after about 16 s. `mex.getCompilerConfigurations('C','Installed')` lists only this compiler, so the MEX gateway builds with MSVC and no MinGW cross-compile of the gateway is needed.

A failed MATLAB command reaches WSL as a nonzero exit code. `scripts/matlab-win "error('phasecont:check','x')"` printed `ERROR: MATLAB error Exit Status: 0x00000001`, and the shell saw exit code 1. A MATLAB test script run this way can therefore report failure with `error` or `assert`.

Every run whose current folder is the repo prints `Warning: Unable to obtain a change notification handle for \\wsl.localhost\fedoralinux-44\home\nub04\dev\phasecont.` MATLAB cannot watch a UNC folder for changes, so a desktop session may not see files edited from WSL until `rehash` runs. Each `-batch` run starts fresh, so `scripts/matlab-win` is not affected.

`ver` lists 120 products in all, each at 26.2, among them Parallel Computing Toolbox, Symbolic Math Toolbox, Optimization Toolbox, MATLAB Coder, GPU Coder and MATLAB Test. Rule 8 of `CLAUDE.md` forbids depending on most of these. They are all installed here, so an accidental dependency will not fail on this machine; the "Base MATLAB check" test in section 12 of `docs/SPEC.md` catches one.

Commands: `version`, `matlabroot`, `computer('arch')`, `mex.getCompilerConfigurations('C','Selected')`, `maxNumCompThreads`, `gpuDevice` and `ver`, all through `scripts/matlab-win`, plus directory listings of `extern\lib\win64` and `bin\win64`.

## GPU and CUDA

| Item | WSL | Windows |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, compute capability 12.0, 8151 MiB | same |
| Driver | 617.42 | 617.42 |
| `nvidia-smi` header | `NVIDIA-SMI 615.78.05  KMD Version: 617.42  CUDA UMD Version: 13.4` | `NVIDIA-SMI 617.42  KMD Version: 617.42  CUDA UMD Version: 13.4` |
| Driver library | `/usr/lib/wsl/lib/libcuda.so`, `libcuda.so.1`, `libcuda.so.1.1` | installed with the driver |
| CUDA toolkit | `/usr/local/cuda`, nvcc release 13.4, V13.4.92 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4` |
| NVRTC | `libnvrtc.so.13.4.92` | `bin\x64\nvrtc64_130_0.dll` with `nvrtc-builtins64_134.dll` |

On Windows, `CUDA_PATH` is unset and `CUDA_PATH_V13_4` is `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`. The toolkit's NVRTC and the copy in MATLAB's `bin\win64` share the file name `nvrtc64_130_0.dll`. Their builtins files (`nvrtc-builtins64_134.dll` and `nvrtc-builtins64_131.dll`) suggest NVRTC 13.4 and 13.1. Spike S4 confirmed both with `nvrtcVersion`: a DLL inside MATLAB that loads the bare name `nvrtc64_130_0.dll` gets MATLAB's copy, NVRTC 13.1, and gets the toolkit's 13.4 only when it loads that file by absolute path first (`spikes/results/s4_matlab.txt`, `spikes/results/s4_matlab_preload.txt`).

`gpuDevice` reports `DeviceSupported` 1 for this compute capability 12.0 card. `gpuArray` still needs Parallel Computing Toolbox, which rule 8 of `CLAUDE.md` allows only on the optional `gpuArray` route.

Commands: `nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv` and plain `nvidia-smi`, in WSL and as `/mnt/c/Windows/System32/nvidia-smi.exe`; `/usr/local/cuda/bin/nvcc --version`; `cmd.exe /c 'echo %CUDA_PATH% & echo %CUDA_PATH_V13_4%'`; and directory listings of the toolkit and `/usr/lib/wsl/lib`.

## Staging folder

Windows MATLAB loads compiled binaries from the Windows file system, so Windows build outputs are staged under `/mnt/c/Users/<user>/phasecont-stage/` (`C:\Users\<user>\phasecont-stage\`). The spikes use its `spikes\` subfolder, which `spikes/stage.sh` creates.

`cmd.exe` cannot use a `\\wsl.localhost\...` path as its working directory. Run from the repo folder, `scripts/matlab-win "[s,o]=system('cd'); disp(o)"` prints `UNC paths are not supported.  Defaulting to Windows directory.` followed by `C:\Windows`. MATLAB code that runs commands with `system` therefore changes to a folder on `C:` first. `mex` is not affected: spike S1 built the gateway from the repo folder with relative source paths, and the build succeeded (`spikes/results/s1_gateway_nodll.txt`).

## Installed tools and install commands

Phase 0 needed Windows build route 1 (`x86_64-pc-windows-gnu` with MinGW) and `cargo deny check`. These commands were run on 2026-10-21, and `rpm` dates the packages between 16:20:30Z and 16:21:05Z:

```
sudo dnf install --enablerepo=updates-testing rust-std-static-x86_64-pc-windows-gnu mingw64-gcc mingw64-winpthreads-static
sudo dnf upgrade --enablerepo=updates-testing rust-src
sudo dnf install cargo-deny
```

The first command also installed other MinGW packages, among them `mingw64-binutils`, `mingw64-crt` and `mingw64-headers`, and upgraded `rust`, `cargo`, `rust-std-static`, `rustfmt`, `clippy` and `rust-analyzer` from 1.99.0 to 1.99.1. The second command keeps `rust-src`, which rust-analyzer reads, at the compiler's version.

After any later Rust update, `rustc -vV` and `rpm -q rust-std-static-x86_64-pc-windows-gnu` must report the same version, or Windows builds fail. No version lock is set (`dnf versionlock list` prints nothing), and none is planned. `rpm -q --requires rust-std-static-x86_64-pc-windows-gnu` lists `rust = 1.99.1-2.fc45`, so dnf cannot move `rust` to another version while the Windows std of the old version stays installed, and a lock would also hold back Rust security updates. Instead, the `cargo` build task compares the two versions before a Windows build (`docs/SPEC.md`, section 12). If a Fedora update ever moves `rust` before a matching Windows std is available, this command pins the set:

```
sudo dnf versionlock add rust cargo rust-std-static rust-std-static-x86_64-pc-windows-gnu
```

These tools are still missing. Later phases need them, and each one needs approval before it is installed:

- `sudo dnf install cbindgen` for the C header of `pc-ffi`.
- `sudo dnf install zig` and `cargo install --locked cargo-zigbuild` (Fedora has no package) for Linux release builds against glibc 2.28.
- `cargo install --locked cargo-xwin` (Fedora has no package), only if the `x86_64-pc-windows-msvc` route replaces route 1.
