use lsp_server::Connection;
use lsp_types::{Position, Range};

use super::*;

fn response_of(client: &Connection) -> Response {
    match client.receiver.recv().unwrap() {
        Message::Response(r) => r,
        other => panic!("expected a response, got {other:?}"),
    }
}

#[test]
fn a_result_is_sent_as_the_response() {
    let (server, client) = Connection::memory();
    respond(&server, RequestId::from(7), || Ok(json!({"ok": true}))).unwrap();
    let r = response_of(&client);
    assert_eq!(r.id, RequestId::from(7));
    assert_eq!(r.response_result.unwrap(), json!({"ok": true}));
}

#[test]
fn a_failure_is_sent_with_its_code() {
    let (server, client) = Connection::memory();
    respond(&server, RequestId::from(1), || {
        Err((ErrorCode::InvalidParams, "bad".to_string()))
    })
    .unwrap();
    let e = response_of(&client).response_result.unwrap_err();
    assert_eq!(e.code, ErrorCode::InvalidParams as i32);
    assert_eq!(e.message, "bad");
}

#[test]
fn a_panic_in_a_handler_becomes_an_internal_error() {
    let (server, client) = Connection::memory();
    respond(&server, RequestId::from(2), || panic!("handler broke")).unwrap();
    let e = response_of(&client).response_result.unwrap_err();
    assert_eq!(e.code, ErrorCode::InternalError as i32);
    assert!(e.message.contains("handler broke"), "{}", e.message);
    // The same connection still carries the next answer.
    respond(&server, RequestId::from(3), || Ok(Value::Null)).unwrap();
    assert!(response_of(&client).response_result.is_ok());
}

#[test]
fn sending_to_a_closed_connection_is_an_error() {
    let (server, client) = Connection::memory();
    drop(client);
    assert!(respond(&server, RequestId::from(4), || Ok(Value::Null)).is_err());
}

fn full(text: &str) -> TextDocumentContentChangeEvent {
    TextDocumentContentChangeEvent {
        range: None,
        range_length: None,
        text: text.to_string(),
    }
}

fn edit(sl: u32, sc: u32, el: u32, ec: u32, text: &str) -> TextDocumentContentChangeEvent {
    TextDocumentContentChangeEvent {
        range: Some(Range::new(Position::new(sl, sc), Position::new(el, ec))),
        range_length: None,
        text: text.to_string(),
    }
}

#[test]
fn a_full_change_replaces_the_text() {
    let mut t = "old".to_string();
    apply_change(&mut t, full("new\ntext"));
    assert_eq!(t, "new\ntext");
}

#[test]
fn a_ranged_change_edits_the_text() {
    let mut t = "x'=-x\ny'=-y\n".to_string();
    apply_change(&mut t, edit(1, 3, 1, 5, "2*y"));
    assert_eq!(t, "x'=-x\ny'=2*y\n");
    apply_change(&mut t, edit(0, 0, 0, 0, "# hi\n"));
    assert_eq!(t, "# hi\nx'=-x\ny'=2*y\n");
}

#[test]
fn a_ranged_change_counts_utf16_columns_and_clamps() {
    let mut t = "\u{1f600}ab".to_string();
    // The emoji is two UTF-16 units, so column 2 is just after it.
    apply_change(&mut t, edit(0, 2, 0, 3, "X"));
    assert_eq!(t, "\u{1f600}Xb");
    apply_change(&mut t, edit(5, 0, 9, 9, "!"));
    assert_eq!(t, "\u{1f600}Xb!");
    // A reversed range inserts at its start.
    apply_change(&mut t, edit(0, 3, 0, 2, "?"));
    assert!(t.contains('?'));
}

#[test]
fn merge_drops_repeats_and_keeps_order() {
    let d = |line: u32, message: &str| Diagnostic {
        range: Range::new(Position::new(line, 0), Position::new(line, 1)),
        message: message.to_string(),
        ..Diagnostic::default()
    };
    let a = vec![d(0, "a"), d(1, "b")];
    let b = vec![d(1, "b"), d(2, "c")];
    let merged = merge(&[&a, &b]);
    let messages: Vec<&str> = merged.iter().map(|x| x.message.as_str()).collect();
    assert_eq!(messages, ["a", "b", "c"]);
}
