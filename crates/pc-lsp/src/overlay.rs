//! The loader behind `#include`: open buffers first, then disk, and only
//! inside the folders the user opened.
//!
//! A downloaded model can say `#include /etc/passwd`. The engine's own
//! `FsLoader` would open it, which is right for a command line tool and wrong
//! for an editor that opens files on its own. This loader reads an include
//! only when the file lies under a workspace folder or under the folder of the
//! document that holds the `#include`. A symbolic link that leads out is
//! refused too, because the check runs on the canonical path.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use pc_ode::{FileId, SourceLoader};

/// The largest included file the server reads from disk.
pub const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;

/// Open buffers and the workspace folders. Paths are in the text form that
/// [`crate::uri::path_from_uri`] gives, and compared after [`normalize`].
#[derive(Debug, Clone, Default)]
pub struct Overlay {
    buffers: HashMap<String, String>,
    roots: Vec<String>,
}

impl Overlay {
    pub fn new() -> Overlay {
        Overlay::default()
    }

    /// Replaces the workspace folders.
    pub fn set_roots(&mut self, roots: impl IntoIterator<Item = String>) {
        self.roots = roots.into_iter().map(|r| normalize(&r)).collect();
    }

    /// Sets the text of an open buffer.
    pub fn set_buffer(&mut self, path: &str, text: String) {
        self.buffers.insert(normalize(path), text);
    }

    pub fn remove_buffer(&mut self, path: &str) {
        self.buffers.remove(&normalize(path));
    }

    /// A loader for a document in `top_dir` (the document's own folder).
    pub fn loader<'a>(&'a self, top_dir: Option<&str>) -> OverlayLoader<'a> {
        OverlayLoader {
            overlay: self,
            top_dir: top_dir.map(normalize),
        }
    }
}

/// Loads includes for one analysis. See the module comment.
#[derive(Debug)]
pub struct OverlayLoader<'a> {
    overlay: &'a Overlay,
    top_dir: Option<String>,
}

impl OverlayLoader<'_> {
    fn allowed(&self, path: &str) -> bool {
        self.overlay
            .roots
            .iter()
            .chain(self.top_dir.iter())
            .any(|root| is_under(path, root))
    }

    /// True when the canonical form of `path` lies under a canonical allowed
    /// folder. A folder that cannot be resolved allows nothing.
    fn allowed_on_disk(&self, canonical: &Path) -> bool {
        self.overlay
            .roots
            .iter()
            .chain(self.top_dir.iter())
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .any(|root| canonical.starts_with(root))
    }
}

impl SourceLoader for OverlayLoader<'_> {
    fn load(
        &mut self,
        _from: FileId,
        from_name: &str,
        path: &str,
    ) -> Result<(String, String), String> {
        if !from_name.starts_with('/') && !from_name.starts_with("//") && !is_drive(from_name) {
            return Err("save the file before it includes another file".to_string());
        }
        let from_dir = from_name.rsplit_once('/').map_or("", |(dir, _)| dir);
        let path = path.replace('\\', "/");
        let joined = if path.starts_with('/') || is_drive(&path) {
            path
        } else {
            format!("{from_dir}/{path}")
        };
        let name = normalize(&joined);
        if !self.allowed(&name) {
            return Err(format!(
                "{name}: the file is outside the workspace folders and the folder of the document"
            ));
        }
        if let Some(text) = self.overlay.buffers.get(&name) {
            return Ok((name, text.clone()));
        }
        let canonical = std::fs::canonicalize(&name).map_err(|e| format!("{name}: {e}"))?;
        if !self.allowed_on_disk(&canonical) {
            return Err(format!(
                "{name}: the file is a link that leads outside the workspace folders"
            ));
        }
        let meta = std::fs::metadata(&canonical).map_err(|e| format!("{name}: {e}"))?;
        if meta.len() > MAX_FILE_BYTES {
            return Err(format!(
                "{name}: the file is larger than {} MiB",
                MAX_FILE_BYTES / (1024 * 1024)
            ));
        }
        let text = std::fs::read_to_string(&canonical).map_err(|e| format!("{name}: {e}"))?;
        Ok((name, text))
    }

    fn identity(&mut self, name: &str) -> String {
        match std::fs::canonicalize(name) {
            Ok(path) => path.display().to_string(),
            Err(_) => name.to_string(),
        }
    }
}

/// `path` with `.` removed, `dir/..` folded as text and separators turned
/// into `/`. The text is the key for buffers and the display name of files.
pub fn normalize(path: &str) -> String {
    let path = path.replace('\\', "/");
    let mut out = PathBuf::new();
    let mut depth = 0usize;
    for part in Path::new(&path).components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                if depth > 0 {
                    out.pop();
                    depth -= 1;
                }
            }
            Component::Normal(name) => {
                out.push(name);
                depth += 1;
            }
            other => out.push(other.as_os_str()),
        }
    }
    out.display().to_string().replace('\\', "/")
}

/// True when `path` is `root` or lies below it. Both are normalized.
fn is_under(path: &str, root: &str) -> bool {
    if root.is_empty() {
        return false;
    }
    let root = root.trim_end_matches('/');
    path == root
        || path
            .strip_prefix(root)
            .is_some_and(|rest| rest.starts_with('/'))
}

fn is_drive(path: &str) -> bool {
    let b = path.as_bytes();
    b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':'
}

#[cfg(test)]
mod tests;
