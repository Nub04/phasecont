//! The `pc-lsp` binary over real pipes.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

const WAIT: Duration = Duration::from_secs(20);
const URI: &str = "file:///work/m.ode";
const GOOD: &str = "par a=1\nx'=-a*x\ninit x=1\ndone\n";
const BAD: &str = "par a=1.2.3\nx'=-a*x\ninit x=1\ndone\n";

struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    messages: Receiver<Value>,
}

impl Session {
    fn start() -> Session {
        let mut child = Command::new(env!("CARGO_BIN_EXE_pc-lsp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("cannot start pc-lsp");
        let stdin = child.stdin.take();
        let mut out = BufReader::new(child.stdout.take().unwrap());
        let (tx, messages) = mpsc::channel();
        thread::spawn(move || {
            while let Some(message) = read_one(&mut out) {
                if tx.send(message).is_err() {
                    break;
                }
            }
        });
        Session {
            child,
            stdin,
            messages,
        }
    }

    fn send_raw(&mut self, bytes: &[u8]) {
        let stdin = self.stdin.as_mut().expect("stdin is closed");
        stdin.write_all(bytes).unwrap();
        stdin.flush().unwrap();
    }

    fn send(&mut self, value: Value) {
        let body = value.to_string();
        let framed = format!("Content-Length: {}\r\n\r\n{body}", body.len());
        self.send_raw(framed.as_bytes());
    }

    fn request(&mut self, id: i32, method: &str, params: Value) {
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
    }

    fn notify(&mut self, method: &str, params: Value) {
        self.send(json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    /// The next message for which `want` is true; others are dropped.
    fn wait_for(&self, what: &str, want: impl Fn(&Value) -> bool) -> Value {
        loop {
            let message = self
                .messages
                .recv_timeout(WAIT)
                .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
            if want(&message) {
                return message;
            }
        }
    }

    fn response(&self, id: i32) -> Value {
        self.wait_for(&format!("the response to {id}"), |m| m["id"] == id)
    }

    fn diagnostics(&self) -> Value {
        self.wait_for("publishDiagnostics", |m| {
            m["method"] == "textDocument/publishDiagnostics"
        })
    }

    fn handshake(&mut self) {
        self.request(1, "initialize", json!({"capabilities": {}}));
        let r = self.response(1);
        assert_eq!(r["result"]["serverInfo"]["name"], "pc-lsp", "{r}");
        self.notify("initialized", json!({}));
    }

    fn open(&mut self, version: i32, text: &str) {
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {
                "uri": URI, "languageId": "ode", "version": version, "text": text
            }}),
        );
    }

    /// Ends the session and returns the exit code, killing the child if it
    /// does not stop in time.
    fn finish(mut self, shutdown: bool) -> Option<i32> {
        if shutdown {
            self.request(99, "shutdown", Value::Null);
            assert_eq!(self.response(99)["result"], Value::Null);
        }
        self.notify("exit", Value::Null);
        self.stdin = None;
        let (tx, rx) = mpsc::channel();
        let mut child = self.child;
        thread::spawn(move || {
            let _ = tx.send(child.wait());
        });
        match rx.recv_timeout(WAIT) {
            Ok(status) => status.unwrap().code(),
            Err(_) => panic!("pc-lsp did not exit"),
        }
    }
}

fn read_one(out: &mut impl BufRead) -> Option<Value> {
    let mut length = None;
    loop {
        let mut line = String::new();
        if out.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(n) = line.strip_prefix("Content-Length:") {
            length = n.trim().parse::<usize>().ok();
        }
    }
    let mut body = vec![0u8; length?];
    out.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

#[test]
fn an_error_is_published_and_a_fix_clears_it() {
    let mut s = Session::start();
    s.handshake();
    s.open(1, BAD);
    let p = s.diagnostics();
    assert_eq!(p["params"]["uri"], URI);
    assert_eq!(p["params"]["version"], 1);
    assert!(!p["params"]["diagnostics"].as_array().unwrap().is_empty());

    s.notify(
        "textDocument/didChange",
        json!({
            "textDocument": {"uri": URI, "version": 2},
            "contentChanges": [{"text": GOOD}]
        }),
    );
    let p = s.diagnostics();
    assert_eq!(p["params"]["version"], 2);
    assert_eq!(p["params"]["diagnostics"], json!([]));
    assert_eq!(s.finish(true), Some(0));
}

#[test]
fn a_malformed_message_does_not_end_the_session() {
    let mut s = Session::start();
    s.handshake();
    // A well-framed body that is not JSON, then one that is JSON but no
    // message.
    s.send_raw(b"Content-Length: 8\r\n\r\nnot json");
    s.send_raw(b"Content-Length: 2\r\n\r\n{}");
    s.open(1, BAD);
    let p = s.diagnostics();
    assert!(!p["params"]["diagnostics"].as_array().unwrap().is_empty());
    assert_eq!(s.finish(true), Some(0));
}

#[test]
fn hover_works_over_the_pipes() {
    let mut s = Session::start();
    s.handshake();
    s.open(1, GOOD);
    s.request(
        2,
        "textDocument/hover",
        json!({"textDocument": {"uri": URI}, "position": {"line": 1, "character": 4}}),
    );
    let r = s.response(2);
    let text = r["result"]["contents"]["value"].as_str().unwrap();
    assert!(text.contains("parameter"), "{r}");
    assert_eq!(s.finish(true), Some(0));
}

#[test]
fn exit_without_shutdown_is_a_failure_code() {
    let mut s = Session::start();
    s.handshake();
    assert_eq!(s.finish(false), Some(1));
}

#[test]
fn closing_stdin_ends_the_server() {
    let mut s = Session::start();
    s.handshake();
    s.stdin = None;
    let mut child = s.child;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(child.wait());
    });
    let status = rx.recv_timeout(WAIT).expect("pc-lsp did not exit").unwrap();
    assert_eq!(status.code(), Some(1), "no shutdown was received");
}

#[test]
fn version_flag_prints_and_exits() {
    let out = Command::new(env!("CARGO_BIN_EXE_pc-lsp"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.starts_with("pc-lsp "), "{text}");
}
