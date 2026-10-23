//! Pass that resolves `#include` lines.
//!
//! The parser keeps `#include name.ode` as a [`StmtKind::Include`] statement
//! and never reads files. [`resolve_includes`] replaces each one with the
//! statements of the named file, which it gets from a [`SourceLoader`]. The
//! loader decides where text comes from: [`FsLoader`] reads the file system
//! and [`MapLoader`] holds text in memory, for tests and for callers such as
//! MATLAB or the language server that already have the text.
//!
//! Each included file gets its own [`FileId`] in the [`SourceMap`], so its
//! diagnostics point into it. The rules are in `docs/language.md`,
//! section "Includes".

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use crate::ast::{Model, Stmt, StmtKind};
use crate::diag::{Code, Diagnostic, FixIt};
use crate::parser::parse;
use crate::span::{FileId, SourceMap, Span};

/// An `#include` file name in quotes.
pub const INCLUDE_QUOTED: Code = Code(59);
/// An `#include` file the loader cannot read.
pub const INCLUDE_UNREADABLE: Code = Code(60);
/// An `#include` that includes a file already being included.
pub const INCLUDE_CYCLE: Code = Code(61);
/// An `#include` nested deeper than [`MAX_INCLUDE_DEPTH`].
pub const INCLUDE_TOO_DEEP: Code = Code(62);
/// An `#include` inside an array block.
pub const INCLUDE_IN_BLOCK: Code = Code(63);

/// The deepest an included file may sit. The top file is at depth 0 and a
/// file it includes is at depth 1.
pub const MAX_INCLUDE_DEPTH: usize = 16;

/// Supplies the text of included files.
pub trait SourceLoader {
    /// Loads `path`, as written after `#include` in the file `from`.
    /// Returns the display name, which also identifies the file when
    /// looking for cycles, and the text. The error is a short reason shown
    /// to the user.
    fn load(&mut self, from: FileId, path: &str) -> Result<(String, String), String>;

    /// Called right after the text from the last successful [`load`] is
    /// stored in the source map as `file`. A loader that resolves paths
    /// relative to the including file records the id here. A load that is
    /// then rejected as a cycle gets no `added` call, so calls do not pair
    /// one to one with loads.
    ///
    /// [`load`]: SourceLoader::load
    fn added(&mut self, file: FileId) {
        let _ = file;
    }
}

/// A loader that holds file text in memory, keyed by the path as written.
/// The including file plays no part, so all files share one flat namespace.
#[derive(Debug, Clone, Default)]
pub struct MapLoader {
    files: HashMap<String, String>,
}

impl MapLoader {
    pub fn new() -> MapLoader {
        MapLoader::default()
    }

    /// Stores `text` under `path`, replacing any earlier text.
    pub fn insert(&mut self, path: impl Into<String>, text: impl Into<String>) {
        self.files.insert(path.into(), text.into());
    }
}

impl SourceLoader for MapLoader {
    fn load(&mut self, _from: FileId, path: &str) -> Result<(String, String), String> {
        match self.files.get(path) {
            Some(text) => Ok((path.to_string(), text.clone())),
            None => Err("there is no such file".to_string()),
        }
    }
}

/// A loader that reads files with `std::fs`.
///
/// A relative path is taken relative to the directory of the including
/// file, never the working directory; an absolute path is used as given.
/// Paths are cleaned up as text (`.` dropped, `dir/..` removed) without
/// asking the file system, so the display names stay as the user wrote
/// them. Only the top file, given to [`FsLoader::load_top`], may be
/// relative to the working directory.
#[derive(Debug, Clone, Default)]
pub struct FsLoader {
    /// The path of each file this loader read, by id.
    paths: HashMap<FileId, PathBuf>,
    /// The path of the last successful `load`, waiting for its id.
    pending: Option<PathBuf>,
}

impl FsLoader {
    pub fn new() -> FsLoader {
        FsLoader::default()
    }

    /// Reads the top file, stores it in `map` and returns its id. The
    /// display name is the cleaned-up `path`.
    pub fn load_top(
        &mut self,
        map: &mut SourceMap,
        path: impl AsRef<Path>,
    ) -> Result<FileId, String> {
        let path = normalize(path.as_ref());
        let text = read(&path)?;
        let id = map.add(path.display().to_string(), text);
        self.paths.insert(id, path);
        Ok(id)
    }
}

impl SourceLoader for FsLoader {
    fn load(&mut self, from: FileId, path: &str) -> Result<(String, String), String> {
        let Some(including) = self.paths.get(&from) else {
            return Err(format!(
                "file {} was not loaded by this loader, so its directory is unknown",
                from.0
            ));
        };
        let dir = including.parent().unwrap_or(Path::new(""));
        let full = normalize(&dir.join(path));
        let text = read(&full)?;
        let name = full.display().to_string();
        self.pending = Some(full);
        Ok((name, text))
    }

    fn added(&mut self, file: FileId) {
        if let Some(path) = self.pending.take() {
            self.paths.insert(file, path);
        }
    }
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| e.to_string())
}

/// Removes `.` components and folds `dir/..` pairs, as text. A `..` with
/// nothing left to fold stays in a relative path and is dropped after the
/// root of an absolute one.
fn normalize(path: &Path) -> PathBuf {
    let mut parts: Vec<Component> = Vec::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => match parts.last() {
                Some(Component::Normal(_)) => {
                    parts.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => parts.push(part),
            },
            _ => parts.push(part),
        }
    }
    parts.iter().collect()
}

/// Replaces every `#include` in `model` with the statements of the file it
/// names, recursively, and returns the result.
///
/// `model` is the template and is not changed; it keeps its `Include`
/// statements so the writer can write them back. `file` is the id of the
/// file `model` was parsed from. Included files are added to `map` and
/// parsed, and their parse diagnostics are part of the returned list, in
/// the order the files are read. A `done` line ends the included file only.
///
/// An include that cannot be resolved (unreadable file, cycle, depth over
/// [`MAX_INCLUDE_DEPTH`]) is reported at its line and dropped. An include
/// inside an array block is reported and left in place, unresolved.
pub fn resolve_includes(
    model: &Model,
    file: FileId,
    map: &mut SourceMap,
    loader: &mut dyn SourceLoader,
) -> (Model, Vec<Diagnostic>) {
    let name = map.name(file).to_string();
    let mut resolver = Resolver {
        map,
        loader,
        diags: Vec::new(),
        stack: vec![Open {
            file,
            name,
            via: None,
        }],
    };
    let mut statements = Vec::new();
    resolver.statements(&model.statements, &mut statements);
    (Model { statements }, resolver.diags)
}

/// A file being included.
struct Open {
    file: FileId,
    name: String,
    /// The `#include` line that opened it; `None` for the top file.
    via: Option<Span>,
}

struct Resolver<'a> {
    map: &'a mut SourceMap,
    loader: &'a mut dyn SourceLoader,
    diags: Vec<Diagnostic>,
    /// The chain of files from the top file to the one being read.
    stack: Vec<Open>,
}

impl Resolver<'_> {
    fn statements(&mut self, input: &[Stmt], out: &mut Vec<Stmt>) {
        for stmt in input {
            match &stmt.kind {
                StmtKind::Include { path, path_span } => self.include(stmt, path, *path_span, out),
                StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => {
                    for inner in body {
                        if matches!(inner.kind, StmtKind::Include { .. }) {
                            let message = "`#include` cannot be used inside an array block, \
                                           since the block would copy the file once per index; \
                                           move it outside the block"
                                .to_string();
                            self.report(INCLUDE_IN_BLOCK, inner.span, message);
                        }
                    }
                    out.push(stmt.clone());
                }
                _ => out.push(stmt.clone()),
            }
        }
    }

    fn include(&mut self, stmt: &Stmt, path: &str, path_span: Span, out: &mut Vec<Stmt>) {
        let path = self.unquote(path, path_span);
        if self.stack.len() > MAX_INCLUDE_DEPTH {
            let message = format!(
                "includes are nested more than {MAX_INCLUDE_DEPTH} deep: {}",
                self.chain(path)
            );
            self.report(INCLUDE_TOO_DEEP, stmt.span, message);
            return;
        }
        let from = self
            .stack
            .last()
            .expect("the top file is on the stack")
            .file;
        let (name, text) = match self.loader.load(from, path) {
            Ok(loaded) => loaded,
            Err(reason) => {
                let message = format!(
                    "cannot read the included file `{path}`: {reason} (include chain {})",
                    self.chain(path)
                );
                self.report(INCLUDE_UNREADABLE, stmt.span, message);
                return;
            }
        };
        if self.stack.iter().any(|open| open.name == name) {
            let message = format!(
                "`{name}` includes itself through the chain {}",
                self.chain(&name)
            );
            self.report(INCLUDE_CYCLE, stmt.span, message);
            return;
        }
        let id = self.map.add(name.clone(), text);
        self.loader.added(id);
        let (mut model, diags) = parse(self.map.text(id), id);
        self.diags.extend(diags);
        // `done` ends this file only (the parser stops there).
        if model
            .statements
            .last()
            .is_some_and(|s| s.kind == StmtKind::Done)
        {
            model.statements.pop();
        }
        self.stack.push(Open {
            file: id,
            name,
            via: Some(stmt.span),
        });
        self.statements(&model.statements, out);
        self.stack.pop();
    }

    /// `path` without matching quotes around it. Quotes are reported with a
    /// fix-it that removes them: XPPAUT would not load the file.
    fn unquote<'p>(&mut self, path: &'p str, path_span: Span) -> &'p str {
        let quoted = path.len() >= 2
            && ((path.starts_with('"') && path.ends_with('"'))
                || (path.starts_with('\'') && path.ends_with('\'')));
        if !quoted {
            return path;
        }
        let inner = &path[1..path.len() - 1];
        let message = format!(
            "XPPAUT does not load a file name in quotes; remove the quotes, as in \
             `#include {inner}`"
        );
        let mut d = Diagnostic::new(INCLUDE_QUOTED, path_span, message);
        d.fix_it = Some(FixIt {
            span: path_span,
            replacement: inner.to_string(),
        });
        self.diags.push(d);
        inner
    }

    /// The file names from the top file down, then `last`.
    fn chain(&self, last: &str) -> String {
        let mut names: Vec<&str> = self.stack.iter().map(|o| o.name.as_str()).collect();
        names.push(last);
        names.join(" -> ")
    }

    /// Reports at `span`, with a note at each `#include` line that led to
    /// the current file.
    fn report(&mut self, code: Code, span: Span, message: String) {
        let mut d = Diagnostic::new(code, span, message);
        d.notes = self
            .stack
            .iter()
            .filter_map(|o| o.via)
            .map(|via| (via, "included here".to_string()))
            .collect();
        self.diags.push(d);
    }
}

#[cfg(test)]
mod tests;
