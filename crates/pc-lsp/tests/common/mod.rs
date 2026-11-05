//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A fresh folder under the system temp folder, removed on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> TempDir {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("pc-lsp-{label}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).expect("cannot create the temp folder");
        // The canonical form, so that tests compare the paths the loader sees.
        let path = std::fs::canonicalize(&path).expect("cannot resolve the temp folder");
        TempDir { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `text` to `rel` under the folder and returns the full path as
    /// text with `/` separators.
    pub fn write(&self, rel: &str, text: &str) -> String {
        let full = self.path.join(rel);
        if let Some(dir) = full.parent() {
            std::fs::create_dir_all(dir).expect("cannot create a folder");
        }
        std::fs::write(&full, text).expect("cannot write a file");
        full.display().to_string().replace('\\', "/")
    }

    /// The full path of `rel` as text, without creating anything.
    pub fn name(&self, rel: &str) -> String {
        self.path.join(rel).display().to_string().replace('\\', "/")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
