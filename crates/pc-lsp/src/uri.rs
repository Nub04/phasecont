//! `file:` URIs against path text.
//!
//! `lsp_types::Uri` only parses and prints a URI. The server needs the path
//! behind a `file:` URI, and a URI for a path it found by itself (an included
//! file). Paths are handled as text with `/` separators, which every
//! platform's file functions accept. A Windows drive letter is written in
//! upper case, because an editor sends `file:///c%3A/...` and the same file
//! must give the same key either way.

use std::str::FromStr;

use lsp_types::Uri;

/// The path behind a `file:` URI, or `None` for another scheme, a bad
/// escape, or bytes that are not UTF-8.
///
/// `file:///C:/a%20b.ode` gives `C:/a b.ode`. A host other than `localhost`
/// gives a network path, `//host/share/a.ode`. A query or fragment is dropped.
pub fn path_from_uri(uri: &str) -> Option<String> {
    let (scheme, rest) = uri.split_once(':')?;
    if !scheme.eq_ignore_ascii_case("file") {
        return None;
    }
    let rest = rest.strip_prefix("//")?;
    let rest = rest.split(['?', '#']).next().unwrap_or("");
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let path = decode(path)?;
    if path.is_empty() {
        return None;
    }
    let host = decode(host)?;
    if !host.is_empty() && !host.eq_ignore_ascii_case("localhost") {
        return Some(format!("//{host}{path}"));
    }
    Some(upper_drive(drop_slash_before_drive(path)))
}

/// A `file:` URI for `path`. `None` when the path is empty or relative.
pub fn uri_from_path(path: &str) -> Option<Uri> {
    let path = path.replace('\\', "/");
    let (host, rest) = match path.strip_prefix("//") {
        Some(unc) => {
            let i = unc.find('/')?;
            (&unc[..i], &unc[i..])
        }
        None => ("", path.as_str()),
    };
    let body = if host.is_empty() && is_drive_path(rest) {
        format!("/{}", upper_drive(rest.to_string()))
    } else if rest.starts_with('/') {
        rest.to_string()
    } else {
        return None;
    };
    let mut text = String::from("file://");
    text.push_str(&encode(host, false));
    text.push_str(&encode(&body, true));
    Uri::from_str(&text).ok()
}

/// True for `C:` or `C:/...`.
fn is_drive_path(path: &str) -> bool {
    let b = path.as_bytes();
    b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b.len() == 2 || b[2] == b'/')
}

/// `/C:/x` to `C:/x`; any other path is unchanged.
fn drop_slash_before_drive(path: String) -> String {
    match path.strip_prefix('/') {
        Some(rest) if is_drive_path(rest) => rest.to_string(),
        _ => path,
    }
}

fn upper_drive(mut path: String) -> String {
    if is_drive_path(&path) {
        path[..1].make_ascii_uppercase();
    }
    path
}

/// Percent-decodes `text`. `None` for a bad escape or a result that is not
/// UTF-8.
fn decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Percent-encodes `text`, keeping the unreserved characters. A `/` stays
/// when `keep_slash` is set, and so does the `:` of a drive letter.
fn encode(text: &str, keep_slash: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, b) in text.bytes().enumerate() {
        let drive_colon = b == b':' && i == 2 && is_drive_path(&text[1..]);
        let plain = b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'.' | b'_' | b'~')
            || (keep_slash && b == b'/')
            || drive_colon;
        if plain {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests;
