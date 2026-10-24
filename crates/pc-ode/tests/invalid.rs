//! One invalid file per diagnostic code.
//!
//! Each `tests/invalid/odeNNNN.ode` (or `odeNNNN_b.ode` for a second file)
//! starts with a comment `# ODEnnnn ...` that names the code the file must
//! produce. The test loads every file through `FsLoader` and lists, in one
//! failure, every code in `CODES` that has no file and every file whose code
//! did not appear. A new file needs no change here.
//!
//! A code whose input cannot be a file that starts with a comment goes in
//! `BUILT_IN` with the reason.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use pc_ode::{CODES, FsLoader, SourceMap, load};

fn invalid_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/invalid")
}

/// Codes tested from text built in this file, each with the reason that a
/// file cannot do it.
const BUILT_IN: &[(u16, &str, &str)] = &[(
    18,
    // The old-style check looks at the first character of the file, and
    // every file in `tests/invalid/` must start with its `# ODEnnnn` comment.
    "0.5 old format\nx'=1\n",
    "a file is old style only when its first line starts with a digit, so it cannot start with the naming comment",
)];

/// The `odeNNNN.ode` and `odeNNNN_b.ode` files in the folder, sorted.
fn invalid_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(invalid_dir())
        .expect("cannot read the invalid folder")
        .map(|e| e.expect("cannot read an entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "ode"))
        .collect();
    files.sort();
    files
}

/// The code named by the first line `# ODEnnnn ...`, or `None`.
fn named_code(first_line: &str) -> Option<u16> {
    let rest = first_line.strip_prefix("# ODE")?;
    let digits = rest.get(..4)?;
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The code in a file name such as `ode0042.ode` or `ode0042_b.ode`.
fn code_in_file_name(name: &str) -> Option<u16> {
    let rest = name.strip_prefix("ode")?;
    let digits = rest.get(..4)?;
    let tail = &rest[4..];
    let tail_ok =
        tail == ".ode" || (tail.starts_with('_') && tail.ends_with(".ode") && tail.len() > 5);
    if tail_ok && digits.bytes().all(|b| b.is_ascii_digit()) {
        digits.parse().ok()
    } else {
        None
    }
}

fn load_file(path: &Path) -> pc_ode::Loaded {
    let mut map = SourceMap::new();
    let top = FsLoader::new()
        .load_top(&mut map, path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    load(top, &mut map, &mut FsLoader::new())
}

/// Files whose named code is not the first diagnostic of its severity, each
/// with the reason.
const NOT_FIRST: &[(&str, &str)] = &[(
    "ode0057.ode",
    "the special line that makes the name possibly declared gives its own ODE0031 warning",
)];

#[test]
fn every_code_has_an_invalid_file_that_produces_it() {
    let mut problems = Vec::new();
    let mut covered: BTreeSet<u16> = BTreeSet::new();

    for path in invalid_files() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).unwrap();
        let first = text.lines().next().unwrap_or("");
        let Some(named) = named_code(first) else {
            problems.push(format!(
                "{name}: the first line must be a comment `# ODEnnnn ...`, found {first:?}"
            ));
            continue;
        };
        if code_in_file_name(&name) != Some(named) {
            problems.push(format!(
                "{name}: the file name does not match ODE{named:04}"
            ));
        }
        if !CODES.iter().any(|(c, _, _)| c.0 == named) {
            problems.push(format!("{name}: ODE{named:04} is not in CODES"));
            continue;
        }
        let loaded = load_file(&path);
        let Some(first) = loaded.diagnostics.iter().find(|d| d.code.0 == named) else {
            problems.push(format!("{name}: did not produce ODE{named:04}"));
            continue;
        };
        covered.insert(named);
        let lead = loaded
            .diagnostics
            .iter()
            .find(|d| d.severity == first.severity)
            .unwrap();
        let exempt = NOT_FIRST.iter().any(|(f, _)| *f == name);
        if lead.code.0 != named && !exempt {
            problems.push(format!(
                "{name}: ODE{:04} comes before ODE{named:04} among the {:?} diagnostics",
                lead.code.0, first.severity
            ));
        }
    }

    for (code, text, _reason) in BUILT_IN {
        let mut map = SourceMap::new();
        let top = map.add("built-in.ode", *text);
        let loaded = load(top, &mut map, &mut FsLoader::new());
        if loaded.diagnostics.iter().any(|d| d.code.0 == *code) {
            covered.insert(*code);
        } else {
            problems.push(format!("built-in input did not produce ODE{code:04}"));
        }
    }

    for (code, _, _) in CODES {
        if !covered.contains(&code.0) {
            problems.push(format!("{code} has no invalid file that produces it"));
        }
    }
    assert!(problems.is_empty(), "\n{}", problems.join("\n"));
}

#[test]
fn built_in_table_names_only_codes_without_a_file() {
    for (code, _, reason) in BUILT_IN {
        assert!(!reason.is_empty());
        let has_file = invalid_files()
            .iter()
            .any(|p| code_in_file_name(&p.file_name().unwrap().to_string_lossy()) == Some(*code));
        assert!(
            !has_file,
            "ODE{code:04} has a file, so drop its BUILT_IN row"
        );
    }
}

#[test]
fn invalid_files_are_lf_and_helpers_are_not_listed_as_cases() {
    for path in invalid_files() {
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains('\r'), "{} has a CR", path.display());
    }
    // Helper files live in `include/` and are never loaded as a case.
    assert!(
        invalid_files()
            .iter()
            .all(|p| p.parent() == Some(invalid_dir().as_path()))
    );
}

#[test]
fn helper_files_under_include_are_lf() {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&invalid_dir().join("include"), &mut files);
    assert!(!files.is_empty());
    for p in files {
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(!text.contains('\r'), "{} has a CR", p.display());
    }
}

#[test]
fn not_first_table_names_existing_files() {
    for (f, reason) in NOT_FIRST {
        assert!(!reason.is_empty());
        assert!(invalid_dir().join(f).is_file(), "{f} does not exist");
    }
}
