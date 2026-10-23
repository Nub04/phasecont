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
use crate::parser::stmt::MALFORMED_STATEMENT;
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
/// More than [`MAX_INCLUDES`] includes in one model.
pub const INCLUDE_TOO_MANY: Code = Code(64);

/// The deepest an included file may sit. The top file is at depth 0 and a
/// file it includes is at depth 1.
pub const MAX_INCLUDE_DEPTH: usize = 16;

/// The most includes one [`resolve_includes`] call resolves, counting every
/// `#include` line each time it is reached.
pub const MAX_INCLUDES: usize = 4096;

/// Supplies the text of included files.
pub trait SourceLoader {
    /// Loads `path`, as written after `#include` in the file `from`, whose
    /// display name in the source map is `from_name`. Returns the display
    /// name of the loaded file and its text. The error is a short reason
    /// shown to the user.
    fn load(
        &mut self,
        from: FileId,
        from_name: &str,
        path: &str,
    ) -> Result<(String, String), String>;

    /// The key that says whether two display names are the same file. It
    /// is used to find cycles and to read and parse each file once per
    /// [`resolve_includes`] call. The default is the name itself.
    fn identity(&mut self, name: &str) -> String {
        name.to_string()
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
    fn load(
        &mut self,
        _from: FileId,
        _from_name: &str,
        path: &str,
    ) -> Result<(String, String), String> {
        match self.files.get(path) {
            Some(text) => Ok((path.to_string(), text.clone())),
            None => Err("there is no such file".to_string()),
        }
    }
}

/// A loader that reads files with `std::fs`. It holds no state.
///
/// A relative path is taken relative to the directory of the including
/// file's display name, never the working directory; an absolute path is
/// used as given. The joined path is opened as it is, so the system follows
/// symbolic links as it would for any program. The display name is the
/// joined path cleaned up as text (`.` dropped, `dir/..` removed), for
/// reading only. The identity of a file is its canonical path, where the
/// system can give one.
#[derive(Debug, Clone, Copy, Default)]
pub struct FsLoader;

impl FsLoader {
    pub fn new() -> FsLoader {
        FsLoader
    }

    /// Reads the top file, stores it in `map` under its cleaned-up path
    /// and returns its id. Only this path may be relative to the working
    /// directory.
    pub fn load_top(&self, map: &mut SourceMap, path: impl AsRef<Path>) -> Result<FileId, String> {
        let path = path.as_ref();
        let text = read(path)?;
        Ok(map.add(normalize(path).display().to_string(), text))
    }
}

impl SourceLoader for FsLoader {
    fn load(
        &mut self,
        _from: FileId,
        from_name: &str,
        path: &str,
    ) -> Result<(String, String), String> {
        let path = Path::new(path);
        if let Some(message) = drive_relative(path) {
            return Err(message);
        }
        let dir = Path::new(from_name).parent().unwrap_or(Path::new(""));
        let full = dir.join(path);
        let text = read(&full)?;
        Ok((normalize(&full).display().to_string(), text))
    }

    fn identity(&mut self, name: &str) -> String {
        match std::fs::canonicalize(name) {
            Ok(path) => path.display().to_string(),
            Err(_) => normalize(Path::new(name)).display().to_string(),
        }
    }
}

/// An error message when `path` names a drive but no root (`C:foo.ode` on
/// Windows), which the system reads relative to that drive's working
/// directory.
fn drive_relative(path: &Path) -> Option<String> {
    let mut parts = path.components();
    let Some(Component::Prefix(_)) = parts.next() else {
        return None;
    };
    if matches!(parts.next(), Some(Component::RootDir)) {
        return None;
    }
    Some(format!(
        "{}: the name gives a drive but no root folder, so it depends on the working \
         directory; write a full path or one relative to the including file",
        show(&path.display().to_string())
    ))
}

/// Reads `path`. The error names the path.
fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", show(&path.display().to_string())))
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

/// `text` with control characters escaped (a tab shows as `\t`), so a
/// message stays on one line. Backslashes and quotes are left alone.
fn show(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// Replaces every `#include` in `model` with the statements of the file it
/// names, recursively, and returns the result.
///
/// `model` is the template and is not changed; it keeps its `Include`
/// statements so the writer can write them back. `file` is the id of the
/// file `model` was parsed from. Each included file is read, added to `map`
/// and parsed once per call, however often it is included; its statements
/// are copied in at each include, and its parse diagnostics are part of the
/// returned list once, in the order the files are read. A `done` line ends
/// the included file only.
///
/// An include that cannot be resolved (unreadable file, cycle, depth over
/// [`MAX_INCLUDE_DEPTH`], more than [`MAX_INCLUDES`] in total) is reported
/// at its line and dropped. An include inside an array block is reported
/// and left in place, unresolved.
pub fn resolve_includes(
    model: &Model,
    file: FileId,
    map: &mut SourceMap,
    loader: &mut dyn SourceLoader,
) -> (Model, Vec<Diagnostic>) {
    let name = map.name(file).to_string();
    let key = loader.identity(&name);
    let mut resolver = Resolver {
        map,
        loader,
        diags: Vec::new(),
        stack: vec![Open {
            file,
            name,
            key,
            via: None,
        }],
        loads: HashMap::new(),
        parsed: HashMap::new(),
        count: 0,
        too_many: false,
    };
    let mut statements = Vec::new();
    resolver.statements(&model.statements, &mut statements);
    (Model { statements }, resolver.diags)
}

/// A file being included.
struct Open {
    file: FileId,
    name: String,
    key: String,
    /// The `#include` line that opened it; `None` for the top file.
    via: Option<Span>,
}

/// What loading one path from one file gave: the display name, identity
/// key and text, or the loader's error.
type Loaded = Result<(String, String, String), String>;

struct Resolver<'a> {
    map: &'a mut SourceMap,
    loader: &'a mut dyn SourceLoader,
    diags: Vec<Diagnostic>,
    /// The chain of files from the top file to the one being read.
    stack: Vec<Open>,
    /// Load results by including file and path as written, so each path is
    /// read once per including file.
    loads: HashMap<(FileId, String), Loaded>,
    /// Each parsed file by identity key, with its `done` removed.
    parsed: HashMap<String, (FileId, Model)>,
    /// Includes resolved so far.
    count: usize,
    /// Whether the include count limit was reported.
    too_many: bool,
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
                            self.report(INCLUDE_IN_BLOCK, inner.span, message, None);
                        }
                    }
                    out.push(stmt.clone());
                }
                _ => out.push(stmt.clone()),
            }
        }
    }

    fn include(&mut self, stmt: &Stmt, path: &str, path_span: Span, out: &mut Vec<Stmt>) {
        let Some(path) = self.unquote(path, path_span) else {
            return;
        };
        if self.count >= MAX_INCLUDES {
            if !self.too_many {
                self.too_many = true;
                let message = format!(
                    "more than {MAX_INCLUDES} includes in total; this `#include` and every \
                     later one are not resolved"
                );
                self.report(INCLUDE_TOO_MANY, stmt.span, message, None);
            }
            return;
        }
        if self.stack.len() > MAX_INCLUDE_DEPTH {
            let message = format!(
                "includes are nested more than {MAX_INCLUDE_DEPTH} deep: {}",
                self.chain(path)
            );
            self.report(INCLUDE_TOO_DEEP, stmt.span, message, None);
            return;
        }
        let (name, key, text) = match self.load(path) {
            Ok(loaded) => loaded,
            Err(reason) => {
                let message = format!(
                    "cannot read the included file `{}`: {} (include chain {})",
                    show(path),
                    show(&reason),
                    self.chain(path)
                );
                self.report(INCLUDE_UNREADABLE, stmt.span, message, None);
                return;
            }
        };
        if self.stack.iter().any(|open| open.key == key) {
            let message = format!(
                "`{}` includes itself through the chain {}",
                show(&name),
                self.chain(&name)
            );
            self.report(INCLUDE_CYCLE, stmt.span, message, None);
            return;
        }
        let (id, model) = match self.parsed.get(&key) {
            Some((id, model)) => (*id, model.clone()),
            None => {
                let id = self.map.add(name.clone(), text);
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
                self.parsed.insert(key.clone(), (id, model.clone()));
                (id, model)
            }
        };
        self.count += 1;
        self.stack.push(Open {
            file: id,
            name,
            key,
            via: Some(stmt.span),
        });
        self.statements(&model.statements, out);
        self.stack.pop();
    }

    /// Loads `path` from the file on top of the stack, once per including
    /// file.
    fn load(&mut self, path: &str) -> Loaded {
        let open = self.stack.last().expect("the top file is on the stack");
        let slot = (open.file, path.to_string());
        if let Some(loaded) = self.loads.get(&slot) {
            return loaded.clone();
        }
        let loaded = self
            .loader
            .load(open.file, &open.name, path)
            .map(|(name, text)| {
                let key = self.loader.identity(&name);
                (name, key, text)
            });
        self.loads.insert(slot, loaded.clone());
        loaded
    }

    /// `path` without matching quotes around it. Quotes are reported with a
    /// fix-it that removes them: XPPAUT would not load the file. Returns
    /// `None` after reporting a quoted empty name.
    fn unquote<'p>(&mut self, path: &'p str, path_span: Span) -> Option<&'p str> {
        let quoted = path.len() >= 2
            && ((path.starts_with('"') && path.ends_with('"'))
                || (path.starts_with('\'') && path.ends_with('\'')));
        if !quoted {
            return Some(path);
        }
        let inner = &path[1..path.len() - 1];
        if inner.is_empty() {
            let message = "this `#include` names no file; write a file name, as in \
                           `#include other.ode`"
                .to_string();
            self.report(MALFORMED_STATEMENT, path_span, message, None);
            return None;
        }
        let message = format!(
            "XPPAUT does not load a file name in quotes; remove the quotes, as in \
             `#include {}`",
            show(inner)
        );
        let fix = FixIt {
            span: path_span,
            replacement: inner.to_string(),
        };
        self.report(INCLUDE_QUOTED, path_span, message, Some(fix));
        Some(inner)
    }

    /// The file names from the top file down, then `last`, with control
    /// characters escaped.
    fn chain(&self, last: &str) -> String {
        let mut names: Vec<String> = self.stack.iter().map(|o| show(&o.name)).collect();
        names.push(show(last));
        names.join(" -> ")
    }

    /// Reports at `span`, with a note at each `#include` line that led to
    /// the current file.
    fn report(&mut self, code: Code, span: Span, message: String, fix_it: Option<FixIt>) {
        let mut d = Diagnostic::new(code, span, message);
        d.fix_it = fix_it;
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
