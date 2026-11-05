//! The server loop over an in-memory connection.

mod common;

use std::thread::{self, JoinHandle};
use std::time::Duration;

use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use pc_lsp::server::{Outcome, run};
use pc_lsp::uri::uri_from_path;
use serde_json::{Value, json};

const WAIT: Duration = Duration::from_secs(10);

const GOOD: &str = "par a=1\nx'=-a*x\ninit x=1\ndone\n";
const BAD: &str = "par a=1.2.3\nx'=-a*x\ninit x=1\ndone\n";
const URI: &str = "file:///work/m.ode";

struct Client {
    conn: Connection,
    server: Option<JoinHandle<Result<Outcome, String>>>,
    next_id: i32,
    notes: Vec<Notification>,
}

impl Client {
    /// Starts a server on its own thread and completes the handshake.
    fn start(init_params: Value) -> Client {
        let (server_side, conn) = Connection::memory();
        let server = thread::spawn(move || run(&server_side));
        let mut c = Client {
            conn,
            server: Some(server),
            next_id: 0,
            notes: Vec::new(),
        };
        let r = c.request("initialize", init_params);
        assert!(r.response_result.is_ok(), "{r:?}");
        c.notify("initialized", json!({}));
        c
    }

    fn plain() -> Client {
        Client::start(json!({ "capabilities": {} }))
    }

    fn notify(&self, method: &str, params: Value) {
        let n = Notification::new(method.to_string(), params);
        self.conn.sender.send(Message::Notification(n)).unwrap();
    }

    fn send_request(&mut self, method: &str, params: Value) -> RequestId {
        self.next_id += 1;
        let id = RequestId::from(self.next_id);
        let r = Request::new(id.clone(), method.to_string(), params);
        self.conn.sender.send(Message::Request(r)).unwrap();
        id
    }

    fn next_message(&self) -> Message {
        self.conn
            .receiver
            .recv_timeout(WAIT)
            .expect("the server did not answer in time")
    }

    fn request(&mut self, method: &str, params: Value) -> Response {
        let id = self.send_request(method, params);
        loop {
            match self.next_message() {
                Message::Response(r) if r.id == id => return r,
                Message::Response(r) => panic!("response for another request: {r:?}"),
                Message::Notification(n) => self.notes.push(n),
                Message::Request(r) => panic!("the server sent a request: {r:?}"),
            }
        }
    }

    fn result(&mut self, method: &str, params: Value) -> Value {
        self.request(method, params)
            .response_result
            .unwrap_or_else(|e| panic!("{method} failed: {e:?}"))
    }

    /// The next `publishDiagnostics` for `uri`, from what is already
    /// buffered or from the connection.
    fn diagnostics(&mut self, uri: &str) -> Value {
        loop {
            if let Some(i) = self.notes.iter().position(|n| is_for(n, uri)) {
                return self.notes.remove(i).params;
            }
            match self.next_message() {
                Message::Notification(n) => self.notes.push(n),
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    fn open(&self, uri: &str, version: i32, text: &str) {
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {
                "uri": uri, "languageId": "ode", "version": version, "text": text
            }}),
        );
    }

    fn change(&self, uri: &str, version: i32, text: &str) {
        self.notify(
            "textDocument/didChange",
            json!({
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}]
            }),
        );
    }

    fn finish(mut self) -> Outcome {
        self.notify("exit", Value::Null);
        self.server.take().unwrap().join().unwrap().unwrap()
    }
}

fn is_for(n: &Notification, uri: &str) -> bool {
    n.method == "textDocument/publishDiagnostics" && n.params["uri"] == uri
}

fn at(uri: &str, line: u32, character: u32) -> Value {
    json!({"textDocument": {"uri": uri}, "position": {"line": line, "character": character}})
}

#[test]
fn initialize_announces_the_capabilities() {
    let (server_side, conn) = Connection::memory();
    let server = thread::spawn(move || run(&server_side));
    conn.sender
        .send(Message::Request(Request::new(
            RequestId::from(1),
            "initialize".to_string(),
            json!({"capabilities": {}}),
        )))
        .unwrap();
    let Message::Response(r) = conn.receiver.recv_timeout(WAIT).unwrap() else {
        panic!("not a response");
    };
    let result = r.response_result.unwrap();
    assert_eq!(result["serverInfo"]["name"], "pc-lsp");
    let caps = &result["capabilities"];
    assert_eq!(caps["textDocumentSync"]["change"], 1, "full sync");
    assert_eq!(caps["textDocumentSync"]["openClose"], true);
    for key in [
        "hoverProvider",
        "completionProvider",
        "definitionProvider",
        "documentSymbolProvider",
        "codeActionProvider",
    ] {
        assert!(!caps[key].is_null(), "no {key}");
    }
    conn.sender
        .send(Message::Notification(Notification::new(
            "exit".to_string(),
            Value::Null,
        )))
        .unwrap();
    let outcome = server.join().unwrap().unwrap();
    assert!(!outcome.shutdown);
}

#[test]
fn a_request_before_initialize_is_refused() {
    let (server_side, conn) = Connection::memory();
    let server = thread::spawn(move || run(&server_side));
    conn.sender
        .send(Message::Request(Request::new(
            RequestId::from(1),
            "textDocument/hover".to_string(),
            json!({}),
        )))
        .unwrap();
    let Message::Response(r) = conn.receiver.recv_timeout(WAIT).unwrap() else {
        panic!("not a response");
    };
    assert_eq!(
        r.response_result.unwrap_err().code,
        ErrorCode::ServerNotInitialized as i32
    );
    drop(conn);
    let outcome = server.join().unwrap().unwrap();
    assert_eq!(outcome.analyses, 0);
}

#[test]
fn an_error_is_published_and_a_fix_clears_it() {
    let mut c = Client::plain();
    c.open(URI, 1, BAD);
    let p = c.diagnostics(URI);
    assert_eq!(p["version"], 1);
    let list = p["diagnostics"].as_array().unwrap();
    assert!(!list.is_empty(), "{p}");
    assert_eq!(list[0]["source"], "phasecont");
    assert!(list[0]["code"].as_str().unwrap().starts_with("ODE"));
    assert_eq!(list[0]["range"]["start"]["line"], 0);

    c.change(URI, 2, GOOD);
    let p = c.diagnostics(URI);
    assert_eq!(p["version"], 2);
    assert_eq!(p["diagnostics"], json!([]));
    assert!(!c.finish().shutdown);
}

#[test]
fn closing_a_document_clears_its_diagnostics() {
    let mut c = Client::plain();
    c.open(URI, 1, BAD);
    assert!(
        !c.diagnostics(URI)["diagnostics"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    c.notify(
        "textDocument/didClose",
        json!({"textDocument": {"uri": URI}}),
    );
    // A request after the close flushes the queue.
    let _ = c.request(
        "textDocument/documentSymbol",
        json!({"textDocument": {"uri": URI}}),
    );
    let p = c.diagnostics(URI);
    assert_eq!(p["diagnostics"], json!([]));
    assert!(p["version"].is_null());
    c.finish();
}

#[test]
fn hover_completion_symbols_and_definition_answer_over_the_protocol() {
    let mut c = Client::plain();
    c.open(URI, 1, GOOD);
    // `a` in `x'=-a*x` is at line 1, column 4.
    let hover = c.result("textDocument/hover", at(URI, 1, 4));
    let text = hover["contents"]["value"].as_str().unwrap();
    assert!(text.contains("`a`") && text.contains("parameter"), "{text}");

    let items = c.result("textDocument/completion", at(URI, 1, 4));
    let labels: Vec<&str> = items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    assert!(
        labels.contains(&"sin") && labels.contains(&"a"),
        "{labels:?}"
    );

    let symbols = c.result(
        "textDocument/documentSymbol",
        json!({"textDocument": {"uri": URI}}),
    );
    let names: Vec<&str> = symbols
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"a") && names.contains(&"x"), "{names:?}");

    let def = c.result("textDocument/definition", at(URI, 1, 4));
    assert_eq!(def["uri"], URI);
    assert_eq!(def["range"]["start"], json!({"line": 0, "character": 4}));
    c.finish();
}

#[test]
fn a_fix_it_comes_back_as_a_quick_fix() {
    let tmp = common::TempDir::new("session-fix");
    tmp.write("part.ode", "par a=1\n");
    let text = "#include \"part.ode\"\nx'=-a*x\ndone\n";
    let main_path = tmp.write("main.ode", text);
    let main_uri = uri_from_path(&main_path).unwrap();
    let dir = tmp.path().display().to_string();
    let mut c = Client::start(folder_params(&dir));
    c.open(main_uri.as_str(), 1, text);
    let p = c.diagnostics(main_uri.as_str());
    let d = p["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| !d["data"].is_null())
        .unwrap_or_else(|| panic!("a quoted include carries a fix-it: {p}"))
        .clone();
    let actions = c.result(
        "textDocument/codeAction",
        json!({
            "textDocument": {"uri": main_uri.as_str()},
            "range": d["range"],
            "context": {"diagnostics": [d]}
        }),
    );
    let first = &actions.as_array().unwrap()[0];
    assert_eq!(first["kind"], "quickfix");
    let edits = &first["edit"]["changes"][main_uri.as_str()];
    assert_eq!(edits[0]["newText"], "part.ode");
    assert_eq!(edits[0]["range"], d["range"]);
    c.finish();
}

#[test]
fn queued_edits_cost_one_analysis_of_the_last_text() {
    let (server_side, conn) = Connection::memory();
    let send = |m: Message| conn.sender.send(m).unwrap();
    send(Message::Request(Request::new(
        RequestId::from(1),
        "initialize".to_string(),
        json!({"capabilities": {}}),
    )));
    let note = |method: &str, params: Value| {
        Message::Notification(Notification::new(method.to_string(), params))
    };
    send(note(
        "textDocument/didOpen",
        json!({"textDocument": {"uri": URI, "languageId": "ode", "version": 1, "text": GOOD}}),
    ));
    for v in 2..=101 {
        // Every text but the last is clean; the last one has the error.
        let text = if v == 101 { BAD } else { GOOD };
        send(note(
            "textDocument/didChange",
            json!({
                "textDocument": {"uri": URI, "version": v},
                "contentChanges": [{"text": text}]
            }),
        ));
    }
    // A request flushes the queue; `exit` alone would end the session
    // without analysing, since nobody is left to read the result.
    send(Message::Request(Request::new(
        RequestId::from(2),
        "textDocument/documentSymbol".to_string(),
        json!({"textDocument": {"uri": URI}}),
    )));
    send(note("exit", Value::Null));
    let outcome = run(&server_side).unwrap();
    assert_eq!(outcome.analyses, 1, "one load for 101 queued messages");
    let published: Vec<Value> = conn
        .receiver
        .try_iter()
        .filter_map(|m| match m {
            Message::Notification(n) if is_for(&n, URI) => Some(n.params),
            _ => None,
        })
        .collect();
    assert_eq!(published.len(), 1, "one publish");
    assert_eq!(published[0]["version"], 101);
    assert!(!published[0]["diagnostics"].as_array().unwrap().is_empty());
}

#[test]
fn shutdown_then_exit_reports_a_clean_end() {
    let mut c = Client::plain();
    assert_eq!(c.result("shutdown", Value::Null), Value::Null);
    let r = c.request("textDocument/hover", at(URI, 0, 0));
    assert_eq!(
        r.response_result.unwrap_err().code,
        ErrorCode::InvalidRequest as i32
    );
    assert!(c.finish().shutdown);
}

#[test]
fn losing_the_client_without_shutdown_is_not_clean() {
    let c = Client::plain();
    let Client { conn, server, .. } = c;
    drop(conn);
    let outcome = server.unwrap().join().unwrap().unwrap();
    assert!(!outcome.shutdown);
}

#[test]
fn an_unknown_method_and_bad_parameters_get_errors() {
    let mut c = Client::plain();
    let r = c.request("textDocument/rename", json!({}));
    assert_eq!(
        r.response_result.unwrap_err().code,
        ErrorCode::MethodNotFound as i32
    );
    let r = c.request("textDocument/hover", json!({"textDocument": 5}));
    assert_eq!(
        r.response_result.unwrap_err().code,
        ErrorCode::InvalidParams as i32
    );
    // Both left the session usable.
    c.open(URI, 1, GOOD);
    assert!(c.result("textDocument/hover", at(URI, 1, 4)).is_object());
    c.finish();
}

#[test]
fn requests_on_unopened_documents_answer_empty() {
    let mut c = Client::plain();
    assert_eq!(c.result("textDocument/hover", at(URI, 0, 0)), Value::Null);
    assert_eq!(
        c.result("textDocument/definition", at(URI, 0, 0)),
        Value::Null
    );
    assert_eq!(
        c.result(
            "textDocument/documentSymbol",
            json!({"textDocument": {"uri": URI}})
        ),
        json!([])
    );
    c.finish();
}

#[test]
fn a_bad_notification_is_skipped() {
    let mut c = Client::plain();
    c.notify("textDocument/didOpen", json!({"textDocument": 1}));
    c.notify("$/somethingNew", json!({}));
    c.open(URI, 1, GOOD);
    assert!(c.result("textDocument/hover", at(URI, 1, 4)).is_object());
    c.finish();
}

fn folder_params(dir: &str) -> Value {
    json!({
        "capabilities": {},
        "workspaceFolders": [{"uri": uri_from_path(dir).unwrap().as_str(), "name": "w"}]
    })
}

#[test]
fn an_include_reports_to_its_own_file_and_follows_the_open_buffer() {
    let tmp = common::TempDir::new("session-include");
    let main_path = tmp.write("main.ode", "#include part.ode\nx'=-x\ndone\n");
    let part_path = tmp.write("part.ode", "par a=1.2.3\n");
    let main_uri = uri_from_path(&main_path).unwrap();
    let part_uri = uri_from_path(&part_path).unwrap();
    let dir = tmp.path().display().to_string();
    let mut c = Client::start(folder_params(&dir));

    c.open(main_uri.as_str(), 1, "#include part.ode\nx'=-x\ndone\n");
    let p = c.diagnostics(part_uri.as_str());
    assert!(
        !p["diagnostics"].as_array().unwrap().is_empty(),
        "the include has an error on disk: {p}"
    );
    assert!(p["version"].is_null(), "the include is not open");

    // Opening the included file with fixed text replaces what is on disk.
    c.open(part_uri.as_str(), 1, "par a=1\n");
    let p = c.diagnostics(part_uri.as_str());
    assert_eq!(p["version"], 1);
    assert_eq!(p["diagnostics"], json!([]), "{p}");
    c.finish();
}

#[test]
fn an_include_outside_the_folders_is_not_read() {
    let tmp = common::TempDir::new("session-outside");
    let outside = common::TempDir::new("session-secret");
    let secret = outside.write("secret.ode", "par leaked=1\n");
    let main_text = format!("#include {secret}\nx'=-x\ndone\n");
    let main_path = tmp.write("main.ode", &main_text);
    let main_uri = uri_from_path(&main_path).unwrap();
    let dir = tmp.path().display().to_string();
    let mut c = Client::start(folder_params(&dir));
    c.open(main_uri.as_str(), 1, &main_text);
    let p = c.diagnostics(main_uri.as_str());
    let messages: Vec<&str> = p["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["message"].as_str().unwrap())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("outside")),
        "the refusal is reported: {messages:?}"
    );
    // The file never entered the analysis, so `leaked` has no definition.
    let hover = c.result("textDocument/hover", at(main_uri.as_str(), 1, 0));
    assert!(hover.is_null() || !hover.to_string().contains("leaked"));
    c.finish();
}
