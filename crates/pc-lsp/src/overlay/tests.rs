use super::*;

#[test]
fn normalize_folds_dots_and_separators() {
    assert_eq!(normalize("/a/./b/../c.ode"), "/a/c.ode");
    assert_eq!(normalize("/a/b/../../c.ode"), "/c.ode");
    assert_eq!(normalize("/a/../../c.ode"), "/c.ode");
    assert_eq!(normalize("/a\\b/c.ode"), "/a/b/c.ode");
}

#[test]
fn is_under_matches_whole_components() {
    assert!(is_under("/w/a.ode", "/w"));
    assert!(is_under("/w", "/w"));
    assert!(is_under("/w/x/y.ode", "/w/"));
    assert!(!is_under("/w2/a.ode", "/w"));
    assert!(!is_under("/other/a.ode", "/w"));
    assert!(!is_under("/w/a.ode", ""));
}

#[test]
fn a_dotdot_cannot_climb_out_of_a_root_as_text() {
    let mut overlay = Overlay::new();
    overlay.set_roots(["/w/proj".to_string()]);
    let loader = overlay.loader(None);
    assert!(loader.allowed(&normalize("/w/proj/a/../b.ode")));
    assert!(!loader.allowed(&normalize("/w/proj/../secret.ode")));
}

#[test]
fn buffers_are_found_under_any_spelling_of_the_path() {
    let mut overlay = Overlay::new();
    overlay.set_buffer("/w/./a.ode", "par a=1\n".to_string());
    let mut loader = overlay.loader(Some("/w"));
    let (name, text) = loader
        .load(FileId(0), "/w/main.ode", "sub/../a.ode")
        .unwrap();
    assert_eq!(name, "/w/a.ode");
    assert_eq!(text, "par a=1\n");
}
