//! The server loop: documents, analyses and the LSP messages around them.
//!
//! The loop blocks on the first message, then takes every message that is
//! already queued and applies them in order. Text changes only mark a document
//! dirty. Each dirty document is analysed once, after the batch, so 100
//! queued edits cost one `pc_ode::load`. A request that arrives in the middle
//! of a batch flushes the dirty documents first, so it sees the current text.
//!
//! One analysis can report diagnostics for the document and for the files it
//! includes. The server keeps what each open document reported per target
//! file and publishes the merged list for a target, so two documents that
//! include the same file do not erase each other's diagnostics.

use std::collections::{HashMap, HashSet};

use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::{
    CodeActionKind, CodeActionOptions, CodeActionParams, CodeActionProviderCapability,
    CompletionOptions, CompletionParams, CompletionResponse, Diagnostic,
    DidChangeTextDocumentParams, DidChangeWorkspaceFoldersParams, DidCloseTextDocumentParams,
    DidOpenTextDocumentParams, DocumentSymbolParams, DocumentSymbolResponse, GotoDefinitionParams,
    GotoDefinitionResponse, HoverParams, HoverProviderCapability, OneOf, PublishDiagnosticsParams,
    ServerCapabilities, TextDocumentContentChangeEvent, TextDocumentSyncCapability,
    TextDocumentSyncKind, TextDocumentSyncOptions, Uri, WorkspaceFoldersServerCapabilities,
    WorkspaceServerCapabilities,
};
use serde_json::{Value, json};

use crate::analysis::{Analysis, analyze};
use crate::convert::LineIndex;
use crate::diagnostics::to_lsp;
use crate::features;
use crate::guard::guarded;
use crate::overlay::{Overlay, normalize};
use crate::uri::{path_from_uri, uri_from_path};

/// How a session ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    /// True when the client sent `shutdown` before it ended the session.
    pub shutdown: bool,
    /// How many times a document was analysed. Tests read this.
    pub analyses: usize,
}

/// Reads the parameters of a notification. A notification has no response,
/// so bad parameters are logged and the notification is skipped.
macro_rules! parse_params {
    ($method:expr, $ty:ty, $params:expr) => {
        match serde_json::from_value::<$ty>($params) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("pc-lsp: bad parameters for {}: {e}", $method);
                None
            }
        }
    };
}

type Sent = Result<(), String>;
type Failure = (ErrorCode, String);

/// The capabilities the server announces.
pub fn capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Options(
            TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(TextDocumentSyncKind::FULL),
                ..TextDocumentSyncOptions::default()
            },
        )),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec!["@".to_string()]),
            ..CompletionOptions::default()
        }),
        definition_provider: Some(OneOf::Left(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        code_action_provider: Some(CodeActionProviderCapability::Options(CodeActionOptions {
            code_action_kinds: Some(vec![CodeActionKind::QUICKFIX]),
            ..CodeActionOptions::default()
        })),
        workspace: Some(WorkspaceServerCapabilities {
            workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(OneOf::Left(true)),
            }),
            file_operations: None,
        }),
        ..ServerCapabilities::default()
    }
}

/// Runs a session on `conn` until `exit`, or until the client goes away.
/// An error means the connection broke while the server was sending.
pub fn run(conn: &Connection) -> Result<Outcome, String> {
    let mut state = State::default();
    if !handshake(conn, &mut state)? {
        return Ok(state.outcome());
    }
    loop {
        let Ok(first) = conn.receiver.recv() else {
            return Ok(state.outcome());
        };
        let mut batch = vec![first];
        while let Ok(next) = conn.receiver.try_recv() {
            batch.push(next);
        }
        for message in batch {
            match message {
                Message::Request(request) => state.request(conn, request)?,
                Message::Notification(note) => {
                    if note.method == "exit" {
                        return Ok(state.outcome());
                    }
                    state.notification(note);
                }
                // The server sends no requests, so it expects no responses.
                Message::Response(_) => {}
            }
        }
        state.refresh(conn)?;
    }
}

/// Waits for `initialize`. Returns false when the client leaves first.
fn handshake(conn: &Connection, state: &mut State) -> Result<bool, String> {
    loop {
        let Ok(message) = conn.receiver.recv() else {
            return Ok(false);
        };
        match message {
            Message::Request(request) if request.method == "initialize" => {
                state.initialize(&request.params);
                let result = json!({
                    "capabilities": capabilities(),
                    "serverInfo": { "name": "pc-lsp", "version": env!("CARGO_PKG_VERSION") },
                });
                return send(
                    conn,
                    Message::Response(Response::new_ok(request.id, result)),
                )
                .map(|()| true);
            }
            Message::Request(request) => send(
                conn,
                Message::Response(Response::new_err(
                    request.id,
                    ErrorCode::ServerNotInitialized as i32,
                    "the server is not initialized".to_string(),
                )),
            )?,
            Message::Notification(note) if note.method == "exit" => return Ok(false),
            _ => {}
        }
    }
}

fn send(conn: &Connection, message: Message) -> Sent {
    conn.sender
        .send(message)
        .map_err(|_| "the client connection is closed".to_string())
}

/// Answers `id` with the result of `f`. A panic in `f` becomes an internal
/// error response and the server keeps running.
pub(crate) fn respond(
    conn: &Connection,
    id: RequestId,
    f: impl FnOnce() -> Result<Value, Failure>,
) -> Sent {
    let response = match guarded(f) {
        Ok(Ok(value)) => Response::new_ok(id, value),
        Ok(Err((code, message))) => Response::new_err(id, code as i32, message),
        Err(panic) => {
            eprintln!("pc-lsp: a request handler panicked: {panic}");
            Response::new_err(
                id,
                ErrorCode::InternalError as i32,
                format!("internal error: {panic}"),
            )
        }
    };
    send(conn, Message::Response(response))
}

/// One open document.
struct Doc {
    version: i32,
    text: String,
    /// The display name used for the analysis: the normalized path, or the
    /// URI text for a document that is not a file.
    name: String,
    /// The path of a file document, as `Overlay` keys it.
    path: Option<String>,
    /// The folder of a file document.
    dir: Option<String>,
}

#[derive(Default)]
struct State {
    overlay: Overlay,
    roots: Vec<String>,
    docs: HashMap<Uri, Doc>,
    /// Display name to the URI the client used, for open documents.
    names: HashMap<String, Uri>,
    analyses: HashMap<Uri, Analysis>,
    /// The files each analysis read, by display name.
    reads: HashMap<Uri, HashSet<String>>,
    /// What each open document reported, per target file.
    reported: HashMap<Uri, HashMap<Uri, Vec<Diagnostic>>>,
    /// Documents to analyse, in the order they became dirty.
    dirty: Vec<Uri>,
    /// Files whose diagnostics must be published at the next flush.
    to_publish: Vec<Uri>,
    shutdown: bool,
    count: usize,
}

impl State {
    fn outcome(&self) -> Outcome {
        Outcome {
            shutdown: self.shutdown,
            analyses: self.count,
        }
    }

    fn initialize(&mut self, params: &Value) {
        let mut roots = Vec::new();
        if let Some(folders) = params.get("workspaceFolders").and_then(Value::as_array) {
            for folder in folders {
                roots.extend(
                    folder
                        .get("uri")
                        .and_then(Value::as_str)
                        .and_then(path_from_uri),
                );
            }
        }
        if roots.is_empty()
            && let Some(path) = params
                .get("rootUri")
                .and_then(Value::as_str)
                .and_then(path_from_uri)
        {
            roots.push(path);
        }
        self.set_roots(roots);
    }

    fn set_roots(&mut self, roots: Vec<String>) {
        self.overlay.set_roots(roots.iter().cloned());
        self.roots = roots;
    }

    fn mark_dirty(&mut self, uri: &Uri) {
        if !self.dirty.contains(uri) {
            self.dirty.push(uri.clone());
        }
    }

    /// Marks every open document that read `name`, except `skip`.
    fn mark_readers_dirty(&mut self, name: &str, skip: &Uri) {
        let readers: Vec<Uri> = self
            .reads
            .iter()
            .filter(|(uri, files)| *uri != skip && files.contains(name))
            .map(|(uri, _)| uri.clone())
            .collect();
        for uri in readers {
            self.mark_dirty(&uri);
        }
    }

    // Notifications.

    fn notification(&mut self, note: Notification) {
        let Notification { method, params } = note;
        match method.as_str() {
            "textDocument/didOpen" => {
                if let Some(p) = parse_params!(method, DidOpenTextDocumentParams, params) {
                    let item = p.text_document;
                    self.open(item.uri, item.version, item.text);
                }
            }
            "textDocument/didChange" => {
                if let Some(p) = parse_params!(method, DidChangeTextDocumentParams, params) {
                    self.change(
                        &p.text_document.uri,
                        p.text_document.version,
                        p.content_changes,
                    );
                }
            }
            "textDocument/didClose" => {
                if let Some(p) = parse_params!(method, DidCloseTextDocumentParams, params) {
                    self.close(&p.text_document.uri);
                }
            }
            "workspace/didChangeWorkspaceFolders" => {
                if let Some(p) = parse_params!(method, DidChangeWorkspaceFoldersParams, params) {
                    self.change_folders(p);
                }
            }
            // `initialized`, `$/cancelRequest`, `$/setTrace`, configuration
            // changes and anything else the server does not use.
            _ => {}
        }
    }

    fn open(&mut self, uri: Uri, version: i32, text: String) {
        let path = path_from_uri(uri.as_str());
        let name = path
            .as_deref()
            .map_or_else(|| uri.as_str().to_string(), normalize);
        let dir = path.as_ref().and_then(|_| {
            name.rsplit_once('/')
                .map(|(dir, _)| if dir.is_empty() { "/" } else { dir }.to_string())
        });
        if let Some(path) = &path {
            self.overlay.set_buffer(path, text.clone());
        }
        self.names.insert(name.clone(), uri.clone());
        self.docs.insert(
            uri.clone(),
            Doc {
                version,
                text,
                name: name.clone(),
                path,
                dir,
            },
        );
        self.mark_dirty(&uri);
        self.mark_readers_dirty(&name, &uri);
    }

    fn change(&mut self, uri: &Uri, version: i32, changes: Vec<TextDocumentContentChangeEvent>) {
        let Some(doc) = self.docs.get_mut(uri) else {
            return;
        };
        for change in changes {
            apply_change(&mut doc.text, change);
        }
        doc.version = version;
        let name = doc.name.clone();
        if let Some(path) = doc.path.clone() {
            self.overlay.set_buffer(&path, doc.text.clone());
        }
        self.mark_dirty(uri);
        self.mark_readers_dirty(&name, uri);
    }

    fn close(&mut self, uri: &Uri) {
        let Some(doc) = self.docs.remove(uri) else {
            return;
        };
        if let Some(path) = &doc.path {
            self.overlay.remove_buffer(path);
        }
        self.names.remove(&doc.name);
        self.analyses.remove(uri);
        self.reads.remove(uri);
        self.dirty.retain(|u| u != uri);
        let old = self.reported.remove(uri).unwrap_or_default();
        self.to_publish.extend(old.into_keys());
        self.to_publish.push(uri.clone());
        // Readers now see the file on disk, not the buffer.
        self.mark_readers_dirty(&doc.name, uri);
    }

    fn change_folders(&mut self, params: DidChangeWorkspaceFoldersParams) {
        let removed: Vec<String> = params
            .event
            .removed
            .iter()
            .filter_map(|f| path_from_uri(f.uri.as_str()))
            .collect();
        let mut roots: Vec<String> = self
            .roots
            .iter()
            .filter(|r| !removed.iter().any(|x| normalize(x) == **r))
            .cloned()
            .collect();
        roots.extend(
            params
                .event
                .added
                .iter()
                .filter_map(|f| path_from_uri(f.uri.as_str())),
        );
        self.set_roots(roots);
        // What an include may read has changed, so check everything again.
        let all: Vec<Uri> = self.docs.keys().cloned().collect();
        for uri in all {
            self.mark_dirty(&uri);
        }
    }

    // Analysis and publishing.

    /// Analyses the dirty documents and publishes what changed.
    fn refresh(&mut self, conn: &Connection) -> Sent {
        let dirty = std::mem::take(&mut self.dirty);
        for top in dirty {
            self.analyse(&top);
        }
        let mut seen = HashSet::new();
        let targets: Vec<Uri> = std::mem::take(&mut self.to_publish)
            .into_iter()
            .filter(|uri| seen.insert(uri.clone()))
            .collect();
        for uri in targets {
            self.publish(conn, &uri)?;
        }
        Ok(())
    }

    fn analyse(&mut self, top: &Uri) {
        let Some(doc) = self.docs.get(top) else {
            return;
        };
        let (name, text) = (doc.name.clone(), doc.text.clone());
        let result = guarded(|| {
            let mut loader = self.overlay.loader(doc.dir.as_deref());
            let analysis = analyze(&name, text, &mut loader);
            let names = &self.names;
            let mut uri_of = |n: &str| names.get(n).cloned().or_else(|| uri_from_path(n));
            let pairs = to_lsp(&analysis, &mut uri_of);
            (analysis, pairs)
        });
        self.count += 1;
        let (analysis, pairs) = match result {
            Ok(done) => done,
            Err(panic) => {
                eprintln!("pc-lsp: analysis of {name} panicked: {panic}");
                return;
            }
        };
        let mut by_target: HashMap<Uri, Vec<Diagnostic>> = HashMap::new();
        for (uri, diagnostic) in pairs {
            by_target.entry(uri).or_default().push(diagnostic);
        }
        self.to_publish.extend(by_target.keys().cloned());
        if let Some(old) = self.reported.insert(top.clone(), by_target) {
            self.to_publish.extend(old.into_keys());
        }
        self.to_publish.push(top.clone());
        self.reads
            .insert(top.clone(), analysis.file_names().into_iter().collect());
        self.analyses.insert(top.clone(), analysis);
    }

    /// Publishes the merged diagnostics of `target`.
    fn publish(&self, conn: &Connection, target: &Uri) -> Sent {
        // The target's own report comes first, then the others by URI.
        let mut tops: Vec<&Uri> = self.reported.keys().collect();
        tops.sort_by_key(|u| (*u != target, u.as_str().to_string()));
        let lists: Vec<&Vec<Diagnostic>> = tops
            .into_iter()
            .filter_map(|t| self.reported[t].get(target))
            .collect();
        let diagnostics = match lists.as_slice() {
            [] => Vec::new(),
            [one] => (*one).clone(),
            many => merge(many),
        };
        let version = self.docs.get(target).map(|d| d.version);
        let params = PublishDiagnosticsParams::new(target.clone(), diagnostics, version);
        send(
            conn,
            Message::Notification(Notification::new(
                "textDocument/publishDiagnostics".to_string(),
                params,
            )),
        )
    }

    // Requests.

    fn request(&mut self, conn: &Connection, request: Request) -> Sent {
        let Request { id, method, params } = request;
        if self.shutdown {
            return send(
                conn,
                Message::Response(Response::new_err(
                    id,
                    ErrorCode::InvalidRequest as i32,
                    "the server is shutting down".to_string(),
                )),
            );
        }
        if method == "shutdown" {
            self.shutdown = true;
            return send(conn, Message::Response(Response::new_ok(id, Value::Null)));
        }
        self.refresh(conn)?;
        respond(conn, id, || self.answer(&method, params))
    }

    fn analysis(&self, uri: &Uri) -> Option<&Analysis> {
        self.analyses.get(uri)
    }

    fn answer(&self, method: &str, params: Value) -> Result<Value, Failure> {
        macro_rules! params {
            ($ty:ty) => {
                serde_json::from_value::<$ty>(params)
                    .map_err(|e| (ErrorCode::InvalidParams, e.to_string()))?
            };
        }
        macro_rules! value {
            ($e:expr) => {
                serde_json::to_value($e).map_err(|e| (ErrorCode::InternalError, e.to_string()))
            };
        }
        match method {
            "textDocument/hover" => {
                let p = params!(HoverParams);
                let at = p.text_document_position_params;
                let hover = self
                    .analysis(&at.text_document.uri)
                    .and_then(|a| features::hover(a, a.top, at.position));
                value!(hover)
            }
            "textDocument/completion" => {
                let p = params!(CompletionParams);
                let at = p.text_document_position;
                let items = self
                    .analysis(&at.text_document.uri)
                    .map(|a| features::completion(a, a.top, at.position))
                    .unwrap_or_default();
                value!(CompletionResponse::Array(items))
            }
            "textDocument/definition" => {
                let p = params!(GotoDefinitionParams);
                let at = p.text_document_position_params;
                let names = &self.names;
                let mut uri_of = |n: &str| names.get(n).cloned().or_else(|| uri_from_path(n));
                let location = self
                    .analysis(&at.text_document.uri)
                    .and_then(|a| features::definition(a, a.top, at.position, &mut uri_of));
                value!(location.map(GotoDefinitionResponse::Scalar))
            }
            "textDocument/documentSymbol" => {
                let p = params!(DocumentSymbolParams);
                let symbols = self
                    .analysis(&p.text_document.uri)
                    .map(|a| features::document_symbols(a, a.top))
                    .unwrap_or_default();
                value!(DocumentSymbolResponse::Nested(symbols))
            }
            "textDocument/codeAction" => {
                let p = params!(CodeActionParams);
                value!(features::code_actions(
                    &p.text_document.uri,
                    &p.context.diagnostics
                ))
            }
            other => Err((
                ErrorCode::MethodNotFound,
                format!("the server does not handle {other}"),
            )),
        }
    }
}

/// Applies one change event to `text`. An event without a range replaces the
/// whole text. One with a range is applied as an edit, for clients that send
/// incremental changes although the server asked for full text.
fn apply_change(text: &mut String, change: TextDocumentContentChangeEvent) {
    let Some(range) = change.range else {
        *text = change.text;
        return;
    };
    let (start, end) = {
        let index = LineIndex::new(text);
        (index.offset(range.start), index.offset(range.end))
    };
    let end = end.max(start);
    text.replace_range(start..end, &change.text);
}

/// The diagnostics of several reports with the repeats left out. Two
/// documents that include one file both report its problems.
fn merge(lists: &[&Vec<Diagnostic>]) -> Vec<Diagnostic> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for diagnostic in lists.iter().flat_map(|l| l.iter()) {
        let r = diagnostic.range;
        let key = (
            r.start.line,
            r.start.character,
            r.end.line,
            r.end.character,
            diagnostic.code.clone().map(|c| format!("{c:?}")),
            diagnostic.message.clone(),
        );
        if seen.insert(key) {
            out.push(diagnostic.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests;
