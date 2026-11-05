//! The backends an ensemble run can use and the devices each one found.

/// Where an ensemble run executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    OpenCl,
    Metal,
}

impl Backend {
    pub const ALL: [Backend; 4] = [Backend::Cpu, Backend::Cuda, Backend::OpenCl, Backend::Metal];

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::OpenCl => "opencl",
            Backend::Metal => "metal",
        }
    }

    pub fn from_name(name: &str) -> Option<Backend> {
        Backend::ALL.into_iter().find(|b| b.name() == name)
    }
}

/// One device a backend can run on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub backend: Backend,
    /// Position among the devices of this backend.
    pub index: usize,
    pub name: String,
    pub driver: String,
    pub fp32: bool,
    pub fp64: bool,
    pub memory_bytes: Option<u64>,
}

/// Whether a backend can run in this process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendState {
    Available,
    /// The binary does not contain the backend.
    NotBuilt,
    /// The backend is built in but cannot start. The text says why, for example that the
    /// driver library is missing.
    Unavailable(String),
}

#[derive(Clone, Debug)]
pub struct DeviceList {
    pub devices: Vec<DeviceInfo>,
    /// One entry per backend, in the order of [`Backend::ALL`].
    pub backends: Vec<(Backend, BackendState)>,
}

/// The driver string of the CPU backend.
pub fn cpu_driver() -> String {
    format!("pc-ensemble {}", env!("CARGO_PKG_VERSION"))
}

fn cpu_device() -> DeviceInfo {
    DeviceInfo {
        backend: Backend::Cpu,
        index: 0,
        name: format!("CPU (rayon, {} threads)", rayon::current_num_threads()),
        driver: cpu_driver(),
        fp32: true,
        fp64: true,
        memory_bytes: None,
    }
}

/// Every device that can run an ensemble now.
///
/// The first call probes the GPU libraries, so it can take a moment on a machine with a
/// driver. A library that is missing shows up as [`BackendState::Unavailable`].
pub fn list_devices() -> DeviceList {
    let mut devices = vec![cpu_device()];
    devices.extend(crate::cuda::devices());
    devices.extend(crate::opencl::devices());
    devices.extend(crate::metal::devices());
    let backends = Backend::ALL
        .into_iter()
        .map(|b| {
            let state = match b {
                Backend::Cpu => BackendState::Available,
                Backend::Cuda => crate::cuda::backend_state(),
                Backend::OpenCl => crate::opencl::backend_state(),
                Backend::Metal => crate::metal::backend_state(),
            };
            (b, state)
        })
        .collect();
    DeviceList { devices, backends }
}

/// The device `index` of `backend`, or the reason there is none.
///
/// A backend that is not built into this binary, one that cannot start and an index past
/// the end of the list each get their own message, so a caller can print it as it is.
pub fn find_device(backend: Backend, index: usize) -> Result<DeviceInfo, String> {
    let list = list_devices();
    if let Some(d) = list
        .devices
        .iter()
        .find(|d| d.backend == backend && d.index == index)
    {
        return Ok(d.clone());
    }
    let name = backend.name();
    let state = list
        .backends
        .iter()
        .find(|(b, _)| *b == backend)
        .map(|(_, s)| s);
    Err(match state {
        Some(BackendState::NotBuilt) => {
            format!("no {name} device: the {name} backend is not built into this binary")
        }
        Some(BackendState::Unavailable(why)) => {
            format!("no {name} device: the {name} backend cannot start: {why}")
        }
        _ => {
            let found = list.devices.iter().filter(|d| d.backend == backend).count();
            format!("no {name} device {index}: the backend lists {found}")
        }
    })
}
