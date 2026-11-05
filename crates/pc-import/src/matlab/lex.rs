//! Tokens of the MATLAB subset the importer reads.
//!
//! Element-wise operators (`.*`, `./`, `.^`) lex as their plain forms, because every value in
//! the subset is a scalar. A string, a complex literal or a character outside ASCII is an error.

use crate::error::{ImportError, Kind, Result, line_col};

#[derive(Debug, Clone, PartialEq)]
pub enum T {
    Num(f64),
    Id(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    LBrack,
    RBrack,
    Comma,
    Semi,
    Newline,
    Assign,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    And,
    Or,
    Not,
    Quote,
    At,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Tok {
    pub t: T,
    /// Byte offset in the source.
    pub at: usize,
    /// True when a blank or tab sits right before the token.
    pub ws: bool,
}

fn err(src: &str, at: usize, kind: Kind, msg: impl Into<String>) -> ImportError {
    let (l, c) = line_col(src, at);
    ImportError::at(kind, msg, l, c)
}

pub fn lex(src: &str) -> Result<Vec<Tok>> {
    let b = src.as_bytes();
    let mut out: Vec<Tok> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let at = i;
        let next = b.get(i + 1).copied();
        match c {
            b' ' | b'\t' | b'\r' => i += 1,
            b'\n' => {
                out.push(Tok {
                    t: T::Newline,
                    at,
                    ws: false,
                });
                i += 1;
            }
            b'%' => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'.' if b[i..].starts_with(b"...") => {
                // Continuation: the rest of the line and its newline are skipped.
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                i += 1;
            }
            b'0'..=b'9' | b'.' if c != b'.' || next.is_some_and(|n| n.is_ascii_digit()) => {
                i = number(src, i, &mut out)?;
            }
            b'.' => match next {
                Some(b'*') => {
                    out.push(Tok {
                        t: T::Star,
                        at,
                        ws: false,
                    });
                    i += 2;
                }
                Some(b'/') => {
                    out.push(Tok {
                        t: T::Slash,
                        at,
                        ws: false,
                    });
                    i += 2;
                }
                Some(b'^') => {
                    out.push(Tok {
                        t: T::Caret,
                        at,
                        ws: false,
                    });
                    i += 2;
                }
                Some(b'\'') => {
                    out.push(Tok {
                        t: T::Quote,
                        at,
                        ws: false,
                    });
                    i += 2;
                }
                _ => return Err(err(src, at, Kind::Syntax, "unexpected `.`")),
            },
            b'a'..=b'z' | b'A'..=b'Z' => {
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(Tok {
                    t: T::Id(src[at..i].to_string()),
                    at,
                    ws: false,
                });
            }
            b'\'' => {
                // Transpose after a value, a string otherwise.
                let after_value = matches!(
                    out.last().map(|t| &t.t),
                    Some(T::Num(_) | T::Id(_) | T::RParen | T::RBrack | T::Quote)
                );
                if !after_value {
                    return Err(err(
                        src,
                        at,
                        Kind::Unsupported,
                        "a character string is not part of the translated subset",
                    ));
                }
                out.push(Tok {
                    t: T::Quote,
                    at,
                    ws: false,
                });
                i += 1;
            }
            b'"' => {
                return Err(err(
                    src,
                    at,
                    Kind::Unsupported,
                    "a string is not part of the translated subset",
                ));
            }
            _ => {
                let (t, len) = match (c, next) {
                    (b'+', _) => (T::Plus, 1),
                    (b'-', _) => (T::Minus, 1),
                    (b'*', _) => (T::Star, 1),
                    (b'/', _) => (T::Slash, 1),
                    (b'^', _) => (T::Caret, 1),
                    (b'(', _) => (T::LParen, 1),
                    (b')', _) => (T::RParen, 1),
                    (b'[', _) => (T::LBrack, 1),
                    (b']', _) => (T::RBrack, 1),
                    (b',', _) => (T::Comma, 1),
                    (b';', _) => (T::Semi, 1),
                    (b'@', _) => (T::At, 1),
                    (b':', _) => {
                        return Err(err(
                            src,
                            at,
                            Kind::Unsupported,
                            "a range or colon index is not part of the translated subset",
                        ));
                    }
                    (b'=', Some(b'=')) => (T::EqEq, 2),
                    (b'=', _) => (T::Assign, 1),
                    (b'~' | b'!', Some(b'=')) => (T::Ne, 2),
                    (b'~' | b'!', _) => (T::Not, 1),
                    (b'<', Some(b'=')) => (T::Le, 2),
                    (b'<', _) => (T::Lt, 1),
                    (b'>', Some(b'=')) => (T::Ge, 2),
                    (b'>', _) => (T::Gt, 1),
                    (b'&', Some(b'&')) => (T::AndAnd, 2),
                    (b'&', _) => (T::And, 1),
                    (b'|', Some(b'|')) => (T::OrOr, 2),
                    (b'|', _) => (T::Or, 1),
                    _ => {
                        let ch = src[at..].chars().next().unwrap_or('?');
                        return Err(err(
                            src,
                            at,
                            Kind::Syntax,
                            format!("unexpected character `{ch}`"),
                        ));
                    }
                };
                out.push(Tok { t, at, ws: false });
                i += len;
            }
        }
    }
    out.push(Tok {
        t: T::Eof,
        at: src.len(),
        ws: false,
    });
    for t in &mut out {
        t.ws = t.at > 0 && matches!(b[t.at - 1], b' ' | b'\t');
    }
    Ok(out)
}

/// Reads a number that starts at `start` and returns the offset after it.
fn number(src: &str, start: usize, out: &mut Vec<Tok>) -> Result<usize> {
    let b = src.as_bytes();
    let mut i = start;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        // `1.*x` is `1 .* x`, not `1. * x`.
        let op = matches!(b.get(i + 1), Some(b'*' | b'/' | b'^' | b'\''));
        let dots = b[i..].starts_with(b"...");
        if !op && !dots {
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E' || b[i] == b'd' || b[i] == b'D') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        if j < b.len() && b[j].is_ascii_digit() {
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    if i < b.len()
        && (b[i] == b'i' || b[i] == b'j')
        && !b.get(i + 1).is_some_and(u8::is_ascii_alphanumeric)
    {
        return Err(err(
            src,
            start,
            Kind::Unsupported,
            "a complex number is not part of the translated subset",
        ));
    }
    let text = src[start..i].replace(['d', 'D'], "e");
    let v: f64 = text.parse().map_err(|_| {
        err(
            src,
            start,
            Kind::Syntax,
            format!("`{}` is not a number", &src[start..i]),
        )
    })?;
    out.push(Tok {
        t: T::Num(v),
        at: start,
        ws: false,
    });
    Ok(i)
}
