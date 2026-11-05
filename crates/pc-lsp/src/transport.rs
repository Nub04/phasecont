//! Stdin and stdout for the server.
//!
//! The server talks to an in-memory connection. One thread reads messages
//! from stdin into it and one thread writes what the server sends to stdout,
//! so the server never blocks on a pipe. Nothing else in the process may
//! print to stdout, because the client reads it as the protocol.

use std::io::{self, BufRead, Write};
use std::thread;

use lsp_server::{Connection, Message};

use crate::server;

/// Longest piece of a bad message that goes into the log.
const LOG_LIMIT: usize = 200;

/// Reads messages from `input` and gives each to `deliver`, until the input
/// ends, `deliver` returns false or a read fails for a reason other than bad
/// data. A message that is not valid is logged and skipped, so one bad
/// message does not end the session.
pub fn read_messages(input: &mut impl BufRead, mut deliver: impl FnMut(Message) -> bool) {
    loop {
        match Message::read(input) {
            Ok(Some(message)) => {
                if !deliver(message) {
                    return;
                }
            }
            Ok(None) => return,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                eprintln!("pc-lsp: skipped a message: {}", clip(&e.to_string()));
            }
            Err(e) => {
                eprintln!("pc-lsp: stopped reading: {e}");
                return;
            }
        }
    }
}

/// Writes each message to `out` until the messages end or a write fails.
pub fn write_messages(messages: impl Iterator<Item = Message>, out: &mut impl Write) {
    for message in messages {
        if let Err(e) = message.write(out) {
            eprintln!("pc-lsp: stopped writing: {e}");
            return;
        }
    }
}

fn clip(text: &str) -> String {
    let mut end = text.len().min(LOG_LIMIT);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    if end < text.len() {
        format!("{}...", &text[..end])
    } else {
        text.to_string()
    }
}

/// Runs the server on stdin and stdout and returns the exit code: 0 after
/// `shutdown` and `exit`, 1 otherwise.
pub fn serve_stdio() -> i32 {
    let (server_side, io_side) = Connection::memory();
    let to_server = io_side.sender;
    let from_server = io_side.receiver;
    let reader = thread::Builder::new()
        .name("pc-lsp-reader".to_string())
        .spawn(move || {
            let stdin = io::stdin();
            read_messages(&mut stdin.lock(), |m| to_server.send(m).is_ok());
        });
    let writer = thread::Builder::new()
        .name("pc-lsp-writer".to_string())
        .spawn(move || {
            let stdout = io::stdout();
            write_messages(from_server.iter(), &mut stdout.lock());
        });
    let (Ok(_reader), Ok(writer)) = (reader, writer) else {
        eprintln!("pc-lsp: could not start the reader and writer threads");
        return 1;
    };
    let result = server::run(&server_side);
    // Dropping the server's end closes the writer's input. The writer then
    // sends what is queued and returns. The reader may still wait on stdin;
    // the process exit ends it.
    drop(server_side);
    let _ = writer.join();
    match result {
        Ok(outcome) if outcome.shutdown => 0,
        Ok(_) => 1,
        Err(e) => {
            eprintln!("pc-lsp: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests;
