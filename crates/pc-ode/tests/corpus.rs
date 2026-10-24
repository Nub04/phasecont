//! Tests over the model files in `tests/corpus/`: each loads, round-trips
//! through the writer and matches its golden file.
//!
//! The golden file of `foo.ode` is `foo.golden` in the same folder. Run
//! `PC_BLESS=1 cargo test -p pc-ode --test corpus` to rewrite the golden
//! files after a change that is meant to alter them, and read the diff.

mod common;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use common::{assert_expanded_round_trip, assert_round_trip, without_array_marks};
use pc_ode::{
    Diagnostic, FsLoader, Loaded, ModelKind, Severity, SourceMap, StateKind, SymbolKind, load,
    write, write_expanded,
};

// ---------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

/// The `*.ode` files directly in the corpus folder, sorted by name.
fn corpus_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(corpus_dir())
        .expect("cannot read the corpus folder")
        .map(|e| e.expect("cannot read a corpus entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "ode"))
        .collect();
    files.sort();
    files
}

/// The file name without its folder, for messages and tables.
fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .expect("corpus file names are UTF-8")
        .to_string()
}

fn load_path(path: &Path) -> (SourceMap, Loaded) {
    let mut map = SourceMap::new();
    let top = FsLoader::new()
        .load_top(&mut map, path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let loaded = load(top, &mut map, &mut FsLoader::new());
    (map, loaded)
}

fn codes_of(diags: &[Diagnostic]) -> Vec<(u16, &'static str)> {
    let mut v: Vec<(u16, &'static str)> = diags
        .iter()
        .map(|d| {
            let sev = match d.severity {
                Severity::Error => "error",
                Severity::Warning => "warning",
                Severity::Note => "note",
            };
            (d.code.0, sev)
        })
        .collect();
    v.sort_unstable();
    v
}

// ---------------------------------------------------------------------
// The corpus list
// ---------------------------------------------------------------------

/// Models the corpus must hold (brief of Task 10).
const REQUIRED: &[&str] = &[
    "fitzhugh_nagumo",
    "morris_lecar",
    "hodgkin_huxley",
    "lorenz",
    "rossler",
    "brusselator",
    "fold_normal_form",
    "hopf_normal_form",
    "cusp_normal_form",
    "bogdanov_takens",
    "henon_map",
    "logistic_map",
    "inline_table",
    "sets_include",
    "stochastic",
    "options_heavy",
    "unsupported",
    "abbreviations",
    "chain_range",
    "chain_block",
    "cable",
];

/// The warnings a corpus file may give, as codes. Every other file gives
/// none. Each entry is there because the construct needs it.
const EXPECTED_WARNINGS: &[(&str, &[u16])] = &[
    // `2delta*Y` is implicit multiplication.
    ("abbreviations.ode", &[13]),
    // One warning per unsupported line: ODE0054 `delay`, 0055 `shift`,
    // 0056 `del_shft`, 0028 Volterra (two lines), 0029 `bndry`, 0030 `0=`
    // and `solv`, 0031 `special`, 0032 `export`, 0033 `only`, 0034
    // `options`, 0035 `hom_bcs`, 0037 `sum(..)of(..)`, 0036 file table,
    // 0038 active comment.
    (
        "unsupported.ode",
        &[
            28, 28, 29, 30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 54, 55, 56,
        ],
    ),
];

#[test]
fn corpus_holds_every_required_model() {
    let names: Vec<String> = corpus_files().iter().map(|p| file_name(p)).collect();
    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|r| !names.contains(&format!("{r}.ode")))
        .collect();
    assert!(missing.is_empty(), "missing corpus models: {missing:?}");
}

#[test]
fn every_corpus_file_has_a_source_comment_and_ends_with_done() {
    for path in corpus_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        let first = text.lines().next().unwrap_or("");
        assert!(
            first.starts_with("# ") && first.len() > 20,
            "{}: the first line must be a comment naming the model and its source",
            file_name(&path)
        );
        assert!(
            text.trim_end().to_lowercase().ends_with("done"),
            "{}: the last line should be `done`",
            file_name(&path)
        );
    }
}

#[test]
fn corpus_and_golden_files_use_lf_line_endings() {
    let mut paths = corpus_files();
    paths.extend(
        std::fs::read_dir(corpus_dir().join("include"))
            .unwrap()
            .map(|e| e.unwrap().path()),
    );
    paths.extend(
        std::fs::read_dir(corpus_dir())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "golden")),
    );
    for path in paths.iter().filter(|p| p.is_file()) {
        let text = std::fs::read_to_string(path).unwrap();
        assert!(!text.contains('\r'), "{} has a CR", path.display());
        assert!(
            text.is_empty() || text.ends_with('\n'),
            "{} has no final newline",
            path.display()
        );
    }
}

// ---------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------

#[test]
fn corpus_files_load_with_only_the_expected_warnings() {
    for path in corpus_files() {
        let name = file_name(&path);
        let (_, loaded) = load_path(&path);
        let errors: Vec<&Diagnostic> = loaded
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "{name}: errors {errors:#?}");
        let mut want: Vec<u16> = EXPECTED_WARNINGS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, w)| w.to_vec())
            .unwrap_or_default();
        want.sort_unstable();
        let mut got: Vec<u16> = loaded.diagnostics.iter().map(|d| d.code.0).collect();
        got.sort_unstable();
        assert_eq!(got, want, "{name}: warning codes");
        assert!(
            loaded
                .diagnostics
                .iter()
                .all(|d| d.severity == Severity::Warning),
            "{name}: expected warnings only"
        );
    }
}

#[test]
fn model_kinds_match_the_equations() {
    let want = [
        ("henon_map.ode", ModelKind::Map),
        ("logistic_map.ode", ModelKind::Map),
        ("lorenz.ode", ModelKind::Ode),
        ("stochastic.ode", ModelKind::Ode),
    ];
    for (name, kind) in want {
        let (_, loaded) = load_path(&corpus_dir().join(name));
        assert_eq!(loaded.info.kind, kind, "{name}");
    }
}

#[test]
fn simulation_only_and_smooth_flags_follow_the_constructs() {
    let flags = |name: &str| {
        let (_, loaded) = load_path(&corpus_dir().join(name));
        (loaded.info.smooth, loaded.info.simulation_only)
    };
    // A table makes a model non-smooth.
    assert_eq!(flags("inline_table.ode"), (false, false));
    // `global`, `markov` and `wiener` make a model simulation only; the
    // comparison inside `global` is not part of any formula that counts.
    assert!(flags("stochastic.ode").1);
    assert_eq!(flags("lorenz.ode"), (true, false));
    assert_eq!(flags("hodgkin_huxley.ode"), (true, false));
}

// ---------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------

#[test]
fn corpus_files_round_trip_through_the_writer() {
    for path in corpus_files() {
        let name = file_name(&path);
        let (_, loaded) = load_path(&path);
        // The checks of Task 9: `parse(write(parse(s))) == parse(s)` for
        // the template, and the same for the flat text of the expanded
        // model with its array marks cleared.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_round_trip(&loaded.template);
            assert_expanded_round_trip(&loaded.expanded);
        }));
        assert!(result.is_ok(), "{name}: round trip failed (see above)");
    }
}

/// The codes and severities of a load of `text` under the name of `path`,
/// so that its includes resolve as they do for the original.
fn codes_of_text(path: &Path, text: &str) -> Vec<(u16, &'static str)> {
    let mut map = SourceMap::new();
    // The display name of the top file is the path with `.` components
    // dropped, which is what `load_top` stores.
    let top = map.add(path.display().to_string(), text);
    let loaded = load(top, &mut map, &mut FsLoader::new());
    codes_of(&loaded.diagnostics)
}

#[test]
fn written_corpus_text_loads_with_the_same_diagnostics() {
    for path in corpus_files() {
        let name = file_name(&path);
        let (_, loaded) = load_path(&path);
        let want = codes_of(&loaded.diagnostics);
        let template_text = write(&loaded.template);
        assert_eq!(
            codes_of_text(&path, &template_text),
            want,
            "{name}: template text"
        );
        // The flat text has no includes and no arrays, so it can differ
        // only in the diagnostics that arrays collapse. The corpus has
        // none of those, so the codes must agree.
        let flat_text = write_expanded(&without_array_marks(&loaded.expanded));
        assert_eq!(
            codes_of_text(&path, &flat_text),
            want,
            "{name}: expanded text"
        );
    }
}

// ---------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------

/// The flat text of a loaded file as sorted lines without comments.
fn sorted_flat_lines(name: &str) -> Vec<String> {
    let (_, loaded) = load_path(&corpus_dir().join(name));
    let flat = without_array_marks(&loaded.expanded).strip_spans();
    let mut lines: Vec<String> = write_expanded(&flat)
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    lines.sort();
    lines
}

#[test]
fn chain_range_and_block_declare_the_same_model() {
    let range = sorted_flat_lines("chain_range.ode");
    let block = sorted_flat_lines("chain_block.ode");
    assert_eq!(range, block);
    // 20 cells: v and w for each, plus 40 initial values, one parameter
    // line, the options line and `done`.
    let odes = range.iter().filter(|l| l.contains("'=")).count();
    assert_eq!(odes, 40);
    let (_, loaded) = load_path(&corpus_dir().join("chain_range.ode"));
    let states = loaded
        .info
        .symbols
        .iter()
        .filter(|s| matches!(s.kind, SymbolKind::State(StateKind::Ode)))
        .count();
    assert_eq!(states, 40);
}

#[test]
fn cable_writes_its_ends_out_and_uses_neighbor_indices() {
    let text = std::fs::read_to_string(corpus_dir().join("cable.ode")).unwrap();
    assert!(text.contains("v[j-1]") && text.contains("v[j+1]"));
    assert!(text.contains("\nv1'=") && text.contains("\nv20'="));
    let (_, loaded) = load_path(&corpus_dir().join("cable.ode"));
    assert!(loaded.info.symbol("v20").is_some());
    assert!(loaded.info.symbol("v10").is_some());
}

// ---------------------------------------------------------------------
// Golden files
// ---------------------------------------------------------------------

/// `path` relative to the corpus folder, with `/` separators.
fn relative(path: &str, root: &Path) -> String {
    let rel = Path::new(path)
        .strip_prefix(root)
        .unwrap_or_else(|_| panic!("{path} is outside the corpus folder"));
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn kind_label(kind: SymbolKind) -> String {
    match kind {
        SymbolKind::State(StateKind::Ode) => "state-ode".to_string(),
        SymbolKind::State(StateKind::Map) => "state-map".to_string(),
        SymbolKind::State(StateKind::Markov) => "state-markov".to_string(),
        SymbolKind::Parameter => "parameter".to_string(),
        SymbolKind::Number => "number".to_string(),
        SymbolKind::Derived => "derived".to_string(),
        SymbolKind::Fixed => "fixed".to_string(),
        SymbolKind::Aux => "aux".to_string(),
        SymbolKind::Function { params } => format!("function({params})"),
        SymbolKind::Table => "table".to_string(),
        SymbolKind::Wiener => "wiener".to_string(),
    }
}

/// The golden text of one loaded corpus file.
fn render(map: &SourceMap, loaded: &Loaded, root: &Path) -> String {
    let at = |span: pc_ode::Span| {
        let (line, col) = map.line_col(span);
        format!("{}:{line}:{col}", relative(map.name(span.file), root))
    };
    let mut out = String::new();
    out.push_str("diagnostics\n");
    for d in &loaded.diagnostics {
        let sev = match d.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        };
        writeln!(out, "{}: {sev}[{}]: {}", at(d.span), d.code, d.message).unwrap();
        for (span, text) in &d.notes {
            writeln!(out, "  note: {}: {text}", at(*span)).unwrap();
        }
        if let Some(fix) = &d.fix_it {
            let old = &map.text(fix.span.file)[fix.span.start..fix.span.end];
            writeln!(
                out,
                "  fix-it: {}: replace {old:?} with {:?}",
                at(fix.span),
                fix.replacement
            )
            .unwrap();
        }
    }
    let info = &loaded.info;
    let count = |pred: fn(SymbolKind) -> bool| info.symbols.iter().filter(|s| pred(s.kind)).count();
    out.push_str("summary\n");
    writeln!(out, "kind: {:?}", info.kind).unwrap();
    writeln!(out, "smooth: {}", info.smooth).unwrap();
    writeln!(out, "simulation_only: {}", info.simulation_only).unwrap();
    writeln!(
        out,
        "states: {}",
        count(|k| matches!(k, SymbolKind::State(_)))
    )
    .unwrap();
    writeln!(
        out,
        "parameters: {}",
        count(|k| matches!(k, SymbolKind::Parameter))
    )
    .unwrap();
    writeln!(
        out,
        "numbers: {}",
        count(|k| matches!(k, SymbolKind::Number))
    )
    .unwrap();
    writeln!(
        out,
        "derived: {}",
        count(|k| matches!(k, SymbolKind::Derived))
    )
    .unwrap();
    writeln!(out, "fixed: {}", count(|k| matches!(k, SymbolKind::Fixed))).unwrap();
    writeln!(
        out,
        "functions: {}",
        count(|k| matches!(k, SymbolKind::Function { .. }))
    )
    .unwrap();
    writeln!(out, "aux: {}", count(|k| matches!(k, SymbolKind::Aux))).unwrap();
    writeln!(out, "tables: {}", count(|k| matches!(k, SymbolKind::Table))).unwrap();
    writeln!(
        out,
        "wiener: {}",
        count(|k| matches!(k, SymbolKind::Wiener))
    )
    .unwrap();
    out.push_str("symbols\n");
    for s in &info.symbols {
        writeln!(out, "{} {}", kind_label(s.kind), s.name.text).unwrap();
    }
    out
}

/// The first line where two texts differ, for the failure message.
fn first_difference(want: &str, got: &str) -> String {
    let mut w = want.lines();
    let mut g = got.lines();
    for n in 1.. {
        match (w.next(), g.next()) {
            (None, None) => return "no differing line (trailing text differs)".to_string(),
            (a, b) if a != b => {
                return format!(
                    "line {n}:\n  golden: {}\n  actual: {}",
                    a.unwrap_or("<end of file>"),
                    b.unwrap_or("<end of file>")
                );
            }
            _ => {}
        }
    }
    unreachable!()
}

#[test]
fn corpus_files_match_their_golden_files() {
    let bless = std::env::var_os("PC_BLESS").is_some_and(|v| v == "1");
    let root = corpus_dir();
    let mut failures = Vec::new();
    for path in corpus_files() {
        let (map, loaded) = load_path(&path);
        let actual = render(&map, &loaded, &root);
        assert!(
            !actual.contains(env!("CARGO_MANIFEST_DIR")),
            "{}: the golden text names an absolute path",
            file_name(&path)
        );
        let golden = path.with_extension("golden");
        if bless {
            std::fs::write(&golden, &actual).unwrap();
            continue;
        }
        match std::fs::read_to_string(&golden) {
            Err(_) => failures.push(format!(
                "{}: golden file is missing (run with PC_BLESS=1 to create it)",
                golden.display()
            )),
            Ok(want) if want != actual => failures.push(format!(
                "{}: golden file differs at {}\nrun with PC_BLESS=1 to rewrite it, then read the diff",
                golden.display(),
                first_difference(&want, &actual)
            )),
            Ok(_) => {}
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}

#[test]
fn every_golden_file_has_a_corpus_file() {
    for entry in std::fs::read_dir(corpus_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "golden") {
            assert!(
                path.with_extension("ode").is_file(),
                "{} has no .ode file",
                path.display()
            );
        }
    }
}

#[test]
fn golden_text_has_the_documented_shape() {
    let (map, loaded) = load_path(&corpus_dir().join("abbreviations.ode"));
    let text = render(&map, &loaded, &corpus_dir());
    assert!(text.starts_with("diagnostics\nabbreviations.ode:"));
    assert!(text.contains(": warning[ODE0013]: "));
    assert!(text.contains("\nsummary\nkind: Ode\nsmooth: true\nsimulation_only: false\n"));
    assert!(text.contains("\nsymbols\nparameter Om\n"));
    assert!(text.ends_with('\n') && !text.contains('\r'));
}
