//! Runs `pc-cli licenses` on fixture workspaces written under the target
//! directory. The fixtures use path dependencies only, so no network is needed.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::{code, stderr, stdout};

/// A fresh empty directory for one test.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("licenses-{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("create dir");
    fs::write(path, text).expect("write file");
}

/// Writes a library crate `name` next to the workspace.
fn lib_crate(root: &Path, name: &str, license: Option<&str>, extra: &str) {
    write(
        &root.join(name).join("Cargo.toml"),
        &format!(
            "[package]\nname = \"{name}\"\nversion = \"1.2.3\"\nedition = \"2021\"\n\
             license = \"MIT\"\n{extra}"
        ),
    );
    write(&root.join(name).join("src/lib.rs"), "");
    if let Some(text) = license {
        write(&root.join(name).join("LICENSE-MIT"), text);
    }
}

/// Writes the workspace `root/ws` with one member `app` and returns its path.
/// `app` depends on `withlic`, builds with `buildlic` and tests with `devonly`
/// (which has no license file); `extra_dep` adds one more normal dependency.
fn fixture(root: &Path, extra_dep: Option<&str>) -> PathBuf {
    lib_crate(root, "withlic", Some("with license\r\ntext\n"), "");
    lib_crate(root, "buildlic", Some("build license\n"), "");
    lib_crate(root, "devonly", None, "");
    lib_crate(root, "nolic", None, "");
    let ws = root.join("ws");
    write(
        &ws.join("Cargo.toml"),
        "[workspace]\nmembers = [\"app\"]\nresolver = \"2\"\n",
    );
    let extra = extra_dep
        .map(|n| format!("{n} = {{ path = \"../../{n}\" }}\n"))
        .unwrap_or_default();
    write(
        &ws.join("app/Cargo.toml"),
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\
             [dependencies]\nwithlic = {{ path = \"../../withlic\" }}\n{extra}\
             [build-dependencies]\nbuildlic = {{ path = \"../../buildlic\" }}\n\
             [dev-dependencies]\ndevonly = {{ path = \"../../devonly\" }}\n"
        ),
    );
    write(&ws.join("app/src/lib.rs"), "");
    write(&ws.join("app/build.rs"), "fn main() {}\n");
    let lock = Command::new(env!("CARGO"))
        .args(["generate-lockfile", "--offline"])
        .current_dir(&ws)
        .output()
        .expect("run cargo");
    assert!(lock.status.success(), "{}", stderr(&lock));
    ws
}

const TARGET: &str = "x86_64-pc-windows-gnu";

#[test]
fn writes_sorted_sections_and_skips_dev_only_and_workspace_crates() {
    let root = scratch("ok");
    let ws = fixture(&root, None);
    let out = root.join("out.txt");
    let o = common::run(
        &ws,
        &[
            "licenses",
            "--target",
            TARGET,
            "--package",
            "app",
            "--out",
            out.to_str().expect("utf-8 path"),
        ],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stdout(&o).is_empty());
    let text = fs::read_to_string(&out).expect("output file");
    let want = "== buildlic 1.2.3 (MIT) ==\n--- LICENSE-MIT ---\nbuild license\n\n\
                == withlic 1.2.3 (MIT) ==\n--- LICENSE-MIT ---\nwith license\ntext\n\n";
    assert_eq!(text, want);
}

#[test]
fn a_crate_without_a_license_file_fails_and_is_named() {
    let root = scratch("missing");
    let ws = fixture(&root, Some("nolic"));
    let out = root.join("out.txt");
    let o = common::run(
        &ws,
        &[
            "licenses",
            "--target",
            TARGET,
            "--package",
            "app",
            "--out",
            out.to_str().expect("utf-8 path"),
        ],
    );
    assert_eq!(code(&o), 1);
    let err = stderr(&o);
    assert!(err.starts_with("pc-cli: "), "{err}");
    assert!(err.contains("nolic 1.2.3"), "{err}");
    assert!(!out.exists(), "no output file on failure");
}

#[test]
fn unknown_package_is_an_error() {
    let root = scratch("nopkg");
    let ws = fixture(&root, None);
    let out = root.join("out.txt");
    let o = common::run(
        &ws,
        &[
            "licenses",
            "--target",
            TARGET,
            "--package",
            "nothere",
            "--out",
            out.to_str().expect("utf-8 path"),
        ],
    );
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("nothere"), "{}", stderr(&o));
}

#[test]
fn missing_target_or_out_is_a_usage_error() {
    let root = scratch("usage");
    for args in [
        &["licenses", "--out", "x.txt"][..],
        &["licenses", "--target", TARGET],
        &["licenses", "--target", TARGET, "--out", "x", "extra"],
        &["licenses", "--target", TARGET, "--out", "x", "--frob"],
    ] {
        let o = common::run(&root, args);
        assert_eq!(code(&o), 2, "{args:?}");
        assert!(stderr(&o).contains("usage: pc-cli"), "{args:?}");
    }
}

/// Writes `root/ov/<name>/` with a license file and a SOURCE file.
fn override_for(root: &Path, name: &str, text: &str) {
    let dir = root.join("ov").join(name);
    write(&dir.join("LICENSE"), text);
    write(&dir.join("SOURCE"), "https://example.invalid/repo tag v9\n");
}

fn run_with_overrides(root: &Path, ws: &Path) -> std::process::Output {
    let out = root.join("out.txt");
    let ov = root.join("ov");
    common::run(
        ws,
        &[
            "licenses",
            "--target",
            TARGET,
            "--package",
            "app",
            "--out",
            out.to_str().expect("utf-8 path"),
            "--overrides",
            ov.to_str().expect("utf-8 path"),
        ],
    )
}

#[test]
fn an_override_stands_in_for_a_missing_license_file() {
    let root = scratch("ov-used");
    let ws = fixture(&root, Some("nolic"));
    override_for(&root, "nolic", "override text\n");
    let o = run_with_overrides(&root, &ws);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = fs::read_to_string(root.join("out.txt")).expect("output file");
    assert!(text.contains("== nolic 1.2.3 (MIT) =="), "{text}");
    assert!(text.contains("supplied by PhaseCont"), "{text}");
    assert!(
        text.contains("https://example.invalid/repo tag v9"),
        "{text}"
    );
    assert!(text.contains("--- LICENSE ---\noverride text\n"), "{text}");
}

#[test]
fn own_license_files_win_over_an_override() {
    let root = scratch("ov-own");
    let ws = fixture(&root, None);
    override_for(&root, "withlic", "override text\n");
    let o = run_with_overrides(&root, &ws);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = fs::read_to_string(root.join("out.txt")).expect("output file");
    assert!(text.contains("with license\ntext\n"), "{text}");
    assert!(!text.contains("override text"), "{text}");
    assert!(!text.contains("supplied by PhaseCont"), "{text}");
}

#[test]
fn an_override_folder_without_the_crate_still_fails() {
    let root = scratch("ov-other");
    let ws = fixture(&root, Some("nolic"));
    override_for(&root, "someother", "other text\n");
    let o = run_with_overrides(&root, &ws);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("nolic 1.2.3"), "{}", stderr(&o));
    assert!(!root.join("out.txt").exists());
}
