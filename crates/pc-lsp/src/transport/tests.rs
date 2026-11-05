use lsp_server::Notification;

use super::*;

fn frame(body: &str) -> Vec<u8> {
    format!("Content-Length: {}\r\n\r\n{body}", body.len()).into_bytes()
}

fn notification(method: &str) -> String {
    format!(r#"{{"jsonrpc":"2.0","method":"{method}"}}"#)
}

fn read_all(bytes: Vec<u8>) -> Vec<Message> {
    let mut input = io::Cursor::new(bytes);
    let mut got = Vec::new();
    read_messages(&mut input, |m| {
        got.push(m);
        true
    });
    got
}

fn methods(messages: &[Message]) -> Vec<&str> {
    messages
        .iter()
        .map(|m| match m {
            Message::Notification(n) => n.method.as_str(),
            _ => "other",
        })
        .collect()
}

#[test]
fn good_messages_arrive_in_order() {
    let mut bytes = frame(&notification("a"));
    bytes.extend(frame(&notification("b")));
    assert_eq!(methods(&read_all(bytes)), ["a", "b"]);
}

#[test]
fn a_payload_that_is_not_json_is_skipped() {
    let mut bytes = frame(&notification("a"));
    bytes.extend(frame("{not json"));
    bytes.extend(frame(&notification("b")));
    assert_eq!(methods(&read_all(bytes)), ["a", "b"]);
}

#[test]
fn json_that_is_not_a_message_is_skipped() {
    let mut bytes = frame(&notification("a"));
    bytes.extend(frame(r#"{"jsonrpc":"2.0"}"#));
    bytes.extend(frame("[1,2,3]"));
    bytes.extend(frame(&notification("b")));
    assert_eq!(methods(&read_all(bytes)), ["a", "b"]);
}

#[test]
fn a_truncated_message_ends_the_input() {
    let mut bytes = frame(&notification("a"));
    bytes.extend(b"Content-Length: 100\r\n\r\n{\"jsonrpc\"");
    assert_eq!(methods(&read_all(bytes)), ["a"]);
}

#[test]
fn delivery_stops_when_the_receiver_is_gone() {
    let mut bytes = frame(&notification("a"));
    bytes.extend(frame(&notification("b")));
    let mut input = io::Cursor::new(bytes);
    let mut count = 0;
    read_messages(&mut input, |_| {
        count += 1;
        false
    });
    assert_eq!(count, 1);
}

#[test]
fn written_messages_read_back() {
    let sent = vec![
        Message::Notification(Notification::new(
            "x".to_string(),
            serde_json::json!({"n": 1}),
        )),
        Message::Notification(Notification::new("y".to_string(), serde_json::json!(null))),
    ];
    let mut out = Vec::new();
    write_messages(sent.into_iter(), &mut out);
    assert_eq!(methods(&read_all(out)), ["x", "y"]);
}

#[test]
fn long_log_text_is_clipped_on_a_character_boundary() {
    let text = "\u{e9}".repeat(LOG_LIMIT);
    let clipped = clip(&text);
    assert!(clipped.ends_with("..."));
    assert!(clipped.len() < text.len());
    assert_eq!(clip("short"), "short");
}
