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
use std::rc::Rc;

use crate::ast::{Model, Stmt, StmtKind};
use crate::diag::{Code, Diagnostic, FixIt, Sink, with_summary};
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
/// Included files that add more than [`MAX_INCLUDED_STATEMENTS`] statements
/// or [`MAX_INCLUDED_BYTES`] bytes of statement text.
pub const INCLUDE_TOO_MUCH: Code = Code(72);

/// The deepest an included file may sit. The top file is at depth 0 and a
/// file it includes is at depth 1.
pub const MAX_INCLUDE_DEPTH: usize = 16;

/// The most includes one [`resolve_includes`] call resolves, counting every
/// `#include` line each time it is reached.
pub const MAX_INCLUDES: usize = 4096;

/// The most statements the included files of one [`resolve_includes`] call
/// add, counting each statement of an included file once per copy. A block
/// counts as 1 plus the statements of its body. The top file's own
/// statements are not counted.
pub const MAX_INCLUDED_STATEMENTS: usize = 100_000;

/// The most bytes of statement text the included files of one
/// [`resolve_includes`] call add, counted once per copy: from the start of
/// each statement to the end of its trailing comment. A statement count alone
/// does not bound the work, since one statement can be a megabyte long; the
/// syntax tree takes several hundred bytes of memory per byte of text.
pub const MAX_INCLUDED_BYTES: usize = 1024 * 1024;

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
/// joined path with only `.` components dropped; `..` is kept, since
/// folding it as text past a symbolic link would name a different file.
/// The display name therefore always names the opened file, and the file's
/// own includes are taken relative to it. The identity of a file is its
/// canonical path, where the system can give one.
///
/// A path that is not valid UTF-8 gets a lossy display name. The loader
/// remembers the real path of every such file, so the file's own includes
/// and its identity still use the real path.
#[derive(Debug, Clone, Default)]
pub struct FsLoader {
    real: HashMap<String, PathBuf>,
}

impl FsLoader {
    pub fn new() -> FsLoader {
        FsLoader::default()
    }

    /// Records the real path behind `shown` when the display name lost
    /// bytes, and returns `shown`.
    fn remember(&mut self, shown: PathBuf) -> String {
        let name = shown.display().to_string();
        if shown.to_str().is_none() {
            self.real.insert(name.clone(), shown);
        }
        name
    }

    /// The real path of the file shown as `name`.
    fn real_path(&self, name: &str) -> PathBuf {
        self.real
            .get(name)
            .cloned()
            .unwrap_or_else(|| PathBuf::from(name))
    }

    /// Reads the top file, stores it in `map` under its path with `.`
    /// components dropped and returns its id. Only this path may be relative to the working
    /// directory.
    pub fn load_top(
        &mut self,
        map: &mut SourceMap,
        path: impl AsRef<Path>,
    ) -> Result<FileId, String> {
        let path = path.as_ref();
        let text = read(path)?;
        let name = self.remember(display_name(path));
        Ok(map.add(name, text))
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
        let from = self.real_path(from_name);
        let dir = from.parent().unwrap_or(Path::new(""));
        let full = dir.join(path);
        let text = read(&full)?;
        Ok((self.remember(display_name(&full)), text))
    }

    // Known limitation (P11): when `canonicalize` fails, the fallback folds `..`
    // textually, which differs from the file system if a link is in the path.
    fn identity(&mut self, name: &str) -> String {
        let real = self.real_path(name);
        match std::fs::canonicalize(&real) {
            Ok(path) => path.display().to_string(),
            Err(_) => normalize(&real).display().to_string(),
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

/// `path` without its `.` components. `..` is kept.
fn display_name(path: &Path) -> PathBuf {
    path.components()
        .filter(|part| *part != Component::CurDir)
        .collect()
}

/// Removes `.` components and folds `dir/..` pairs, as text. Only the
/// identity fallback uses it, when the system cannot give a canonical path. A `..` with
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

/// What one copy of a file with `statements` adds, as (statements, bytes):
/// every statement except an `#include` (which is replaced by its file) plus
/// the body of each block, and the text those statements cover.
fn added_by(statements: &[Stmt]) -> (usize, usize) {
    let mut count = 0;
    let mut bytes = 0;
    for s in statements {
        if matches!(s.kind, StmtKind::Include { .. }) {
            continue;
        }
        count += match &s.kind {
            StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => 1 + body.len(),
            _ => 1,
        };
        let end = s
            .comment
            .as_ref()
            .map_or(s.span.end, |c| c.span.end.max(s.span.end));
        bytes += end.saturating_sub(s.span.start);
    }
    (count, bytes)
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
/// [`MAX_INCLUDE_DEPTH`]) is reported at its line and dropped. An include
/// inside an array block is reported and left in place, unresolved.
///
/// Every `#include` statement reached counts toward [`MAX_INCLUDES`],
/// resolved or not, including those in array blocks and every repeat of a
/// file included more than once. The first one past the limit is reported
/// once; it and every later one are dropped with no other diagnostic and no
/// loader call.
///
/// The statements the included files add, and their text, are counted too,
/// once per copy ([`MAX_INCLUDED_STATEMENTS`], [`MAX_INCLUDED_BYTES`]). An include whose file would take the total
/// past the limit is reported once ([`INCLUDE_TOO_MUCH`]); it and every
/// later include are dropped, with no other diagnostic and no loader call.
///
/// The list holds at most [`crate::diag::MAX_DIAGNOSTICS`] diagnostics; if there were
/// more, an error ends it that counts the rest.
pub fn resolve_includes(
    model: &Model,
    file: FileId,
    map: &mut SourceMap,
    loader: &mut dyn SourceLoader,
) -> (Model, Vec<Diagnostic>) {
    let (resolved, diags, dropped) = resolve_counted(model, file, map, loader);
    (resolved, with_summary(diags, dropped, file))
}

/// [`resolve_includes`] without the summary error; also returns the number
/// of diagnostics that were not stored.
pub(crate) fn resolve_counted(
    model: &Model,
    file: FileId,
    map: &mut SourceMap,
    loader: &mut dyn SourceLoader,
) -> (Model, Vec<Diagnostic>, usize) {
    let name = map.name(file).to_string();
    let key = loader.identity(&name);
    let mut resolver = Resolver {
        map,
        loader,
        diags: Sink::default(),
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
        emitted: (0, 0),
        too_much: false,
    };
    let mut statements = Vec::new();
    resolver.statements(&model.statements, &mut statements);
    let (diags, dropped) = resolver.diags.finish();
    (
        Model {
            statements,
            suppressed: Vec::new(),
        },
        diags,
        dropped,
    )
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

/// A parsed file: its id, its statements and what one copy adds (statements,
/// bytes).
type ParsedFile = (FileId, Rc<Model>, (usize, usize));

struct Resolver<'a> {
    map: &'a mut SourceMap,
    loader: &'a mut dyn SourceLoader,
    diags: Sink,
    /// The chain of files from the top file to the one being read.
    stack: Vec<Open>,
    /// Load results by the including site's display name and the path as
    /// written, so each path is read once per including name. The name, not
    /// the file id, is the key: one file id can stand for two names whose
    /// relative paths lead to different files.
    loads: HashMap<(String, String), Loaded>,
    /// Each parsed file by identity key, with its `done` removed and the
    /// number of statements one copy of it adds.
    parsed: HashMap<String, ParsedFile>,
    /// `#include` statements reached so far, resolved or not.
    count: usize,
    /// Whether the include count limit was reported.
    too_many: bool,
    /// Statements and bytes of text added by included files so far.
    emitted: (usize, usize),
    /// Whether the statement limit was reached.
    too_much: bool,
}

impl Resolver<'_> {
    fn statements(&mut self, input: &[Stmt], out: &mut Vec<Stmt>) {
        for stmt in input {
            match &stmt.kind {
                StmtKind::Include { path, path_span } => self.include(stmt, path, *path_span, out),
                StmtKind::Block { body, .. } | StmtKind::BadBlock { body } => {
                    for inner in body {
                        if matches!(inner.kind, StmtKind::Include { .. }) && self.counted(inner) {
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

    /// Counts the `#include` statement `stmt` and returns whether it is
    /// within [`MAX_INCLUDES`]. The first one past the limit is reported;
    /// it and every later one get no other diagnostic and no loader call.
    fn counted(&mut self, stmt: &Stmt) -> bool {
        if self.too_much {
            return false;
        }
        self.count += 1;
        if self.count <= MAX_INCLUDES {
            return true;
        }
        if !self.too_many {
            self.too_many = true;
            let message = format!(
                "more than {MAX_INCLUDES} includes in total; this `#include` and every \
                 later one are not resolved"
            );
            self.report(INCLUDE_TOO_MANY, stmt.span, message, None);
        }
        false
    }

    fn include(&mut self, stmt: &Stmt, path: &str, path_span: Span, out: &mut Vec<Stmt>) {
        if !self.counted(stmt) {
            return;
        }
        let Some(path) = self.unquote(path, path_span) else {
            return;
        };
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
        if !self.parsed.contains_key(&key) {
            let id = self.map.add(name.clone(), text);
            let (mut model, diags) = parse(self.map.text(id), id);
            for d in diags {
                self.diags.push(d);
            }
            // `done` ends this file only (the parser stops there).
            if model
                .statements
                .last()
                .is_some_and(|s| s.kind == StmtKind::Done)
            {
                model.statements.pop();
            }
            let weight = added_by(&model.statements);
            self.parsed
                .insert(key.clone(), (id, Rc::new(model), weight));
        }
        let (id, model, weight) = self.parsed[&key].clone();
        if weight.0 > MAX_INCLUDED_STATEMENTS - self.emitted.0
            || weight.1 > MAX_INCLUDED_BYTES - self.emitted.1
        {
            self.too_much = true;
            let message = format!(
                "the included files add more than {MAX_INCLUDED_STATEMENTS} statements or \
                 {MAX_INCLUDED_BYTES} bytes of text in total; this `#include` and every later one are not resolved"
            );
            self.report(INCLUDE_TOO_MUCH, stmt.span, message, None);
            return;
        }
        self.emitted.0 += weight.0;
        self.emitted.1 += weight.1;
        // The stack keeps this site's own display name, even when the parse
        // is shared with another name for the same file, so nested includes
        // resolve against this site's directory.
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
    /// display name (the cache key is the name of the including file and the
    /// path as written).
    fn load(&mut self, path: &str) -> Loaded {
        let open = self.stack.last().expect("the top file is on the stack");
        let slot = (open.name.clone(), path.to_string());
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
    /// `None` after reporting a quoted name that is empty or blank.
    fn unquote<'p>(&mut self, path: &'p str, path_span: Span) -> Option<&'p str> {
        let quoted = path.len() >= 2
            && ((path.starts_with('"') && path.ends_with('"'))
                || (path.starts_with('\'') && path.ends_with('\'')));
        if !quoted {
            return Some(path);
        }
        let inner = &path[1..path.len() - 1];
        if inner.trim().is_empty() {
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
