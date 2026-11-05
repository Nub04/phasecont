use proptest::prelude::*;

use super::*;

#[test]
fn posix_path() {
    assert_eq!(
        path_from_uri("file:///home/me/a%20b.ode").as_deref(),
        Some("/home/me/a b.ode")
    );
}

#[test]
fn windows_drive_in_both_editor_forms() {
    assert_eq!(
        path_from_uri("file:///c%3A/Users/me/m.ode").as_deref(),
        Some("C:/Users/me/m.ode")
    );
    assert_eq!(
        path_from_uri("file:///C:/Users/me/m.ode").as_deref(),
        Some("C:/Users/me/m.ode")
    );
}

#[test]
fn network_path_and_localhost() {
    assert_eq!(
        path_from_uri("file://server/share/m.ode").as_deref(),
        Some("//server/share/m.ode")
    );
    assert_eq!(
        path_from_uri("file://localhost/tmp/m.ode").as_deref(),
        Some("/tmp/m.ode")
    );
}

#[test]
fn other_schemes_and_bad_input_give_none() {
    assert_eq!(path_from_uri("untitled:Untitled-1"), None);
    assert_eq!(path_from_uri("https://example.com/a.ode"), None);
    assert_eq!(path_from_uri("file:///a%zz"), None);
    assert_eq!(path_from_uri("file:///a%ff"), None);
    assert_eq!(path_from_uri("file:///a%4"), None);
    assert_eq!(path_from_uri("file://"), None);
    assert_eq!(path_from_uri(""), None);
}

#[test]
fn query_and_fragment_are_dropped() {
    assert_eq!(
        path_from_uri("file:///tmp/m.ode?x=1#top").as_deref(),
        Some("/tmp/m.ode")
    );
}

#[test]
fn uri_from_path_encodes() {
    let uri = uri_from_path("/home/me/a b#1.ode").unwrap();
    assert_eq!(uri.as_str(), "file:///home/me/a%20b%231.ode");
    let uri = uri_from_path("C:\\Users\\me\\m.ode").unwrap();
    assert_eq!(uri.as_str(), "file:///C:/Users/me/m.ode");
    let uri = uri_from_path("//server/share/m.ode").unwrap();
    assert_eq!(uri.as_str(), "file://server/share/m.ode");
}

#[test]
fn relative_and_empty_paths_give_none() {
    assert!(uri_from_path("").is_none());
    assert!(uri_from_path("a/b.ode").is_none());
}

proptest! {
    #[test]
    fn a_path_round_trips(
        parts in proptest::collection::vec("[a-zA-Z0-9 _.#%&+\u{e9}\u{4e2d}-]{1,8}", 1..5),
        drive in proptest::option::of(proptest::char::range('a', 'z')),
    ) {
        let body = parts.join("/");
        let path = match drive {
            Some(d) => format!("{}:/{body}", d.to_ascii_uppercase()),
            None => format!("/{body}"),
        };
        let uri = uri_from_path(&path).unwrap();
        prop_assert_eq!(path_from_uri(uri.as_str()), Some(path));
    }

    #[test]
    fn path_from_uri_never_panics(text in ".{0,40}") {
        let _ = path_from_uri(&text);
        let _ = path_from_uri(&format!("file://{text}"));
    }
}
