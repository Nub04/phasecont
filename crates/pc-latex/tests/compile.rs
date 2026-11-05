//! Compiles the golden outputs with the local `latexmk`. Prints `SKIPPED: no latexmk`
//! when it is missing. `PC_REQUIRE_LATEX=1` turns that skip into a failure.

use pc_latex::{Kind, Rendered, render, render_value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

fn golden(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn have_latexmk() -> bool {
    Command::new("latexmk")
        .arg("-v")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// True when the test should run. A missing `latexmk` skips, or fails under
/// `PC_REQUIRE_LATEX=1`.
fn available() -> bool {
    if have_latexmk() {
        return true;
    }
    assert!(
        std::env::var_os("PC_REQUIRE_LATEX").is_none_or(|v| v != "1"),
        "PC_REQUIRE_LATEX=1 but latexmk is not installed"
    );
    eprintln!("SKIPPED: no latexmk");
    false
}

/// A fresh directory under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "pc-latex-{}-{}-{tag}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Compiles `main.tex` in `dir` and returns the PDF path. Panics with the log on failure.
fn compile(dir: &Path) -> PathBuf {
    let out = Command::new("latexmk")
        .args([
            "-pdf",
            "-norc",
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-directory=out",
            "main.tex",
        ])
        .current_dir(dir)
        .output()
        .expect("run latexmk");
    let pdf = dir.join("out").join("main.pdf");
    if !out.status.success() || !pdf.is_file() {
        let log = fs::read_to_string(dir.join("out").join("main.log")).unwrap_or_default();
        panic!(
            "latexmk failed ({}):\n{}\n{}\n--- main.log ---\n{log}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    pdf
}

fn write_data(dir: &Path, r: &Rendered) {
    for (name, content) in &r.data_files {
        fs::write(dir.join(name), content).unwrap();
    }
}

const ARTICLE_HEAD: &str = "\\documentclass{article}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\usepackage{booktabs}\n\\begin{document}\n";

/// Each golden as a fragment read with `\input`, inside an ordinary article.
#[test]
fn fragments_compile_inside_an_article() {
    if !available() {
        return;
    }
    for (kind, name) in [
        (Kind::Pgfplots, "fig_branch"),
        (Kind::Pgfplots, "fig_external"),
        (Kind::Table, "table_special"),
    ] {
        let r = render(kind, &golden(&format!("{name}.json"))).unwrap();
        let dir = Scratch::new(name);
        fs::write(dir.path().join("fragment.tex"), &r.tex).unwrap();
        write_data(dir.path(), &r);
        fs::write(
            dir.path().join("main.tex"),
            format!("{ARTICLE_HEAD}\\input{{fragment.tex}}\n\\end{{document}}\n"),
        )
        .unwrap();
        let pdf = compile(dir.path());
        assert!(fs::metadata(pdf).unwrap().len() > 1000, "{name}: tiny pdf");
    }
}

/// Each golden rendered as a `standalone` document.
#[test]
fn standalone_documents_compile() {
    if !available() {
        return;
    }
    for (kind, name) in [
        (Kind::Pgfplots, "fig_branch"),
        (Kind::Pgfplots, "fig_external"),
        (Kind::Table, "table_special"),
    ] {
        let mut v: serde_json::Value =
            serde_json::from_str(&golden(&format!("{name}.json"))).unwrap();
        v["standalone"] = true.into();
        let r = render_value(kind, &v).unwrap();
        let dir = Scratch::new(&format!("{name}-standalone"));
        fs::write(dir.path().join("main.tex"), &r.tex).unwrap();
        write_data(dir.path(), &r);
        compile(dir.path());
    }
}

/// Labels full of TeX special characters compile after `escape`.
#[test]
fn escaped_text_compiles() {
    if !available() {
        return;
    }
    let json = r##"{"digits":4,"standalone":true,
        "xlabel":"a_1 & b% #c {d} ~e ^f \\g $h [i] <j> |k",
        "ylabel":"x_2",
        "title":"50% of $5 & #1 {ok}",
        "series":[{"name":"s_1 & t","class":"stable","x":[0,1],"y":[0,1]}],
        "marks":[{"label":"BP#3 {x} _y","x":0.5,"y":0.5}]}"##;
    let r = render(Kind::Pgfplots, json).unwrap();
    let dir = Scratch::new("escaped");
    fs::write(dir.path().join("main.tex"), &r.tex).unwrap();
    compile(dir.path());
}
