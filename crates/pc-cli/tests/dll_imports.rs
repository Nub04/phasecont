//! Runs `pc-cli dll-imports` on the PE fixture in `tests/fixtures/pe`.

mod common;

use std::path::{Path, PathBuf};
use std::process::Output;

use common::{code, stderr, stdout};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run(args: &[&str]) -> Output {
    common::run(&fixtures(), args)
}

#[test]
fn fixture_lists_its_imports_and_fails_on_user32() {
    let o = run(&["dll-imports", "pe/fixture.dll"]);
    assert_eq!(stdout(&o), "KERNEL32.dll\nmsvcrt.dll\nUSER32.dll\n");
    assert_eq!(code(&o), 1);
    let err = stderr(&o);
    assert!(
        err.trim_end()
            .ends_with("imports DLLs outside the allow-list: USER32.dll"),
        "{err}"
    );
    assert!(err.starts_with("pc-cli: "), "{err}");
    assert_eq!(err.lines().count(), 1, "{err}");
}

#[test]
fn a_missing_file_is_a_read_error() {
    let o = run(&["dll-imports", "pe/missing.dll"]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("pc-cli: cannot read pe/missing.dll"));
    assert!(stdout(&o).is_empty());
}

#[test]
fn a_text_file_is_a_malformed_pe_file() {
    let o = run(&["dll-imports", "plain.ode"]);
    assert_eq!(code(&o), 1);
    let err = stderr(&o);
    assert!(err.starts_with("pc-cli: plain.ode: not a PE file"), "{err}");
    assert!(stdout(&o).is_empty());
}

/// A file over the size limit is refused before it is read. A sparse file keeps the test cheap.
#[test]
fn a_file_over_the_size_limit_is_refused_unread() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dll-imports-big");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let big = std::fs::File::create(dir.join("big.dll")).expect("create file");
    big.set_len(256 * 1024 * 1024 + 1).expect("set length");
    drop(big);
    let o = common::run(&dir, &["dll-imports", "big.dll"]);
    assert_eq!(code(&o), 1);
    let err = stderr(&o);
    assert!(err.starts_with("pc-cli: big.dll: "), "{err}");
    assert!(err.contains("larger than 256 MiB"), "{err}");
    assert!(!err.contains("not a PE file"), "{err}");
    assert!(stdout(&o).is_empty());
}

#[test]
fn zero_or_two_files_are_usage_errors() {
    for args in [
        &["dll-imports"][..],
        &["dll-imports", "pe/fixture.dll", "plain.ode"],
    ] {
        let o = run(args);
        assert_eq!(code(&o), 2, "{args:?}");
        assert!(stderr(&o).contains("usage: pc-cli"), "{args:?}");
    }
}

#[test]
fn help_prints_the_tool_help() {
    let o = run(&["dll-imports", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(
        stdout(&o).starts_with("usage: pc-cli dll-imports"),
        "{}",
        stdout(&o)
    );
}
