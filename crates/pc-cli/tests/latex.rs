//! `pc-cli latex` on the golden documents of `pc-latex`.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../pc-latex/tests/golden")
}

fn golden(name: &str) -> String {
    fs::read_to_string(golden_dir().join(name)).expect("read golden file")
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("latex-{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn path(p: &Path) -> &str {
    p.to_str().expect("UTF-8 path")
}

#[test]
fn inline_figure_goes_to_stdout() {
    let dir = scratch("stdout");
    let input = golden_dir().join("fig_branch.json");
    let o = run(&dir, &["latex", "pgfplots", path(&input)]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(stdout(&o), golden("fig_branch.tex"));
}

#[test]
fn table_to_a_file() {
    let dir = scratch("table");
    let input = golden_dir().join("table_special.json");
    let out = dir.join("table.tex");
    let o = run(&dir, &["latex", "table", "--out", path(&out), path(&input)]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(stdout(&o), "");
    assert_eq!(
        fs::read_to_string(&out).unwrap(),
        golden("table_special.tex")
    );
}

#[test]
fn standalone_flag_wraps_the_fragment() {
    let dir = scratch("standalone");
    let input = golden_dir().join("table_special.json");
    let o = run(&dir, &["latex", "table", "--standalone", path(&input)]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let text = stdout(&o);
    assert!(
        text.starts_with("\\documentclass[border=5pt]{standalone}\n"),
        "{text}"
    );
    assert!(text.contains(&golden("table_special.tex")));
    assert!(text.ends_with("\\end{document}\n"));
}

#[test]
fn external_data_is_written_beside_the_output() {
    let dir = scratch("external");
    let input = golden_dir().join("fig_external.json");
    let out = dir.join("fig.tex");
    let o = run(&dir, &["latex", "pgfplots", "-o", path(&out), path(&input)]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(
        fs::read_to_string(&out).unwrap(),
        golden("fig_external.tex")
    );
    for n in ["fold-curve-s1.dat", "fold-curve-s2.dat"] {
        assert_eq!(
            fs::read_to_string(dir.join(n)).unwrap(),
            golden(&format!("fig_external.{n}")),
            "{n}"
        );
    }
}

#[test]
fn external_data_without_out_is_an_error() {
    let dir = scratch("external-stdout");
    let input = golden_dir().join("fig_external.json");
    let o = run(&dir, &["latex", "pgfplots", path(&input)]);
    assert_eq!(code(&o), 1);
    assert_eq!(stdout(&o), "");
    assert!(stderr(&o).contains("--out"), "{}", stderr(&o));
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 0, "nothing is written");
}

#[test]
fn a_bad_document_names_the_kind_and_writes_nothing() {
    let dir = scratch("bad");
    let input = dir.join("nan.json");
    fs::write(
        &input,
        r#"{"xlabel":"a","ylabel":"x","digits":4,
            "series":[{"class":"stable","x":[0,1],"y":[0,null]}]}"#,
    )
    .unwrap();
    let out = dir.join("never.tex");
    let o = run(
        &dir,
        &["latex", "pgfplots", "--out", path(&out), path(&input)],
    );
    assert_eq!(code(&o), 1);
    let err = stderr(&o);
    assert!(
        err.contains("nonFinite") && err.contains("series 1"),
        "{err}"
    );
    assert!(!out.exists());
}

#[test]
fn missing_digits_is_an_error_not_a_default() {
    let dir = scratch("digits");
    let input = dir.join("t.json");
    fs::write(
        &input,
        r#"{"columns":["a"],"rows":[{"label":"p","type":"t","values":[1]}]}"#,
    )
    .unwrap();
    let o = run(&dir, &["latex", "table", path(&input)]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("missingDigits"), "{}", stderr(&o));
}

#[test]
fn unreadable_and_malformed_input() {
    let dir = scratch("io");
    let o = run(&dir, &["latex", "table", "no-such-file.json"]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("cannot read"), "{}", stderr(&o));
    let bad = dir.join("bad.json");
    fs::write(&bad, "{not json").unwrap();
    let o = run(&dir, &["latex", "table", path(&bad)]);
    assert_eq!(code(&o), 1);
    assert!(stderr(&o).contains("badInput"), "{}", stderr(&o));
}

#[test]
fn usage_errors_exit_2() {
    let dir = scratch("usage");
    for args in [
        &["latex"][..],
        &["latex", "figure", "x.json"],
        &["latex", "table"],
    ] {
        let o = run(&dir, args);
        assert_eq!(code(&o), 2, "{args:?}");
        assert!(stderr(&o).contains("usage:"), "{args:?}");
    }
    let o = run(&dir, &["latex", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).contains("usage: pc-cli latex"));
}
