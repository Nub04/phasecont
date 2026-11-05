//! The include loader: open buffers win over disk, and nothing outside the
//! allowed folders is read.

mod common;

use common::TempDir;
use pc_lsp::analysis::analyze;
use pc_lsp::overlay::Overlay;

/// The codes of an analysis, as `ODEnnnn` strings.
fn codes(a: &pc_lsp::analysis::Analysis) -> Vec<String> {
    a.loaded
        .diagnostics
        .iter()
        .map(|d| d.code.to_string())
        .collect()
}

fn dir_of(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

#[test]
fn an_include_reads_the_file_on_disk() {
    let tmp = TempDir::new("disk");
    let main = tmp.write("main.ode", "#include sub.ode\nx'=a-x\n");
    tmp.write("sub.ode", "par a=1\n");
    let overlay = Overlay::new();
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, "#include sub.ode\nx'=a-x\n".to_string(), &mut loader);
    assert_eq!(codes(&a), Vec::<String>::new());
    assert!(a.loaded.info.symbol("a").is_some());
}

#[test]
fn an_unsaved_buffer_wins_over_the_file_on_disk() {
    let tmp = TempDir::new("buffer");
    let main = tmp.write("main.ode", "#include sub.ode\nx'=a-x\n");
    // The saved sub.ode declares `a`; the buffer in the editor declares `b`.
    let sub = tmp.write("sub.ode", "par a=1\n");
    let mut overlay = Overlay::new();
    overlay.set_buffer(&sub, "par b=2\n".to_string());
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, "#include sub.ode\nx'=a-x\n".to_string(), &mut loader);
    assert!(
        a.loaded.info.symbol("b").is_some(),
        "the buffer was not used"
    );
    assert!(
        a.loaded.info.symbol("a").is_none(),
        "the file on disk was read"
    );
}

#[test]
fn a_buffer_of_a_file_that_is_not_on_disk_loads() {
    let tmp = TempDir::new("nofile");
    let main = tmp.name("main.ode");
    let sub = tmp.name("new.ode");
    let mut overlay = Overlay::new();
    overlay.set_buffer(&sub, "par a=1\n".to_string());
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, "#include new.ode\nx'=a-x\n".to_string(), &mut loader);
    assert_eq!(codes(&a), Vec::<String>::new());
}

#[test]
fn an_absolute_include_outside_the_allowed_folders_is_refused() {
    let tmp = TempDir::new("outside");
    let other = TempDir::new("secret");
    let secret = other.write("secret.ode", "par leaked=1\n");
    let main = tmp.name("main.ode");
    let text = format!("#include {secret}\nx'=-x\n");
    let overlay = Overlay::new();
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, text, &mut loader);
    assert!(
        codes(&a).contains(&"ODE0060".to_string()),
        "{:?}",
        codes(&a)
    );
    assert!(a.loaded.info.symbol("leaked").is_none());
    let message = &a.loaded.diagnostics[0].message;
    assert!(message.contains("outside"), "{message}");
}

#[test]
fn a_parent_folder_include_is_refused_unless_a_workspace_folder_covers_it() {
    let tmp = TempDir::new("parent");
    tmp.write("shared.ode", "par s=1\n");
    let main = tmp.name("proj/main.ode");
    let text = "#include ../shared.ode\nx'=s-x\n".to_string();

    let overlay = Overlay::new();
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, text.clone(), &mut loader);
    assert!(codes(&a).contains(&"ODE0060".to_string()));

    let mut overlay = Overlay::new();
    overlay.set_roots([tmp.name("")]);
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, text, &mut loader);
    assert_eq!(codes(&a), Vec::<String>::new());
}

#[cfg(unix)]
#[test]
fn a_link_that_leads_outside_is_refused() {
    let tmp = TempDir::new("link");
    let other = TempDir::new("target");
    let secret = other.write("secret.ode", "par leaked=1\n");
    let main = tmp.write("main.ode", "");
    std::os::unix::fs::symlink(&secret, tmp.path().join("link.ode")).unwrap();
    let overlay = Overlay::new();
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, "#include link.ode\nx'=-x\n".to_string(), &mut loader);
    assert!(codes(&a).contains(&"ODE0060".to_string()));
    assert!(a.loaded.info.symbol("leaked").is_none());
    assert!(a.loaded.diagnostics[0].message.contains("link"));
}

#[test]
fn a_file_over_the_size_limit_is_refused() {
    let tmp = TempDir::new("big");
    let main = tmp.name("main.ode");
    let big = "# filler\n".repeat((pc_lsp::overlay::MAX_FILE_BYTES as usize) / 9 + 10);
    tmp.write("big.ode", &big);
    let overlay = Overlay::new();
    let mut loader = overlay.loader(Some(dir_of(&main)));
    let a = analyze(&main, "#include big.ode\nx'=-x\n".to_string(), &mut loader);
    assert!(codes(&a).contains(&"ODE0060".to_string()));
    assert!(a.loaded.diagnostics[0].message.contains("larger"));
}

#[test]
fn an_unsaved_document_cannot_include() {
    let overlay = Overlay::new();
    let mut loader = overlay.loader(None);
    let a = analyze(
        "untitled:Untitled-1",
        "#include sub.ode\nx'=-x\n".to_string(),
        &mut loader,
    );
    assert!(codes(&a).contains(&"ODE0060".to_string()));
    assert!(a.loaded.diagnostics[0].message.contains("save the file"));
}

#[test]
fn an_include_cycle_ends_with_a_diagnostic() {
    let tmp = TempDir::new("cycle");
    let a_path = tmp.write("a.ode", "#include b.ode\n");
    tmp.write("b.ode", "#include a.ode\n");
    let overlay = Overlay::new();
    let mut loader = overlay.loader(Some(dir_of(&a_path)));
    let a = analyze(&a_path, "#include b.ode\n".to_string(), &mut loader);
    assert!(
        codes(&a).contains(&"ODE0061".to_string()),
        "{:?}",
        codes(&a)
    );
}
