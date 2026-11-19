//! The `ir`, `check-derivs` and `emit-c` commands.
//!
//! All three load the model the way `parse` does, print the load diagnostics
//! and the lowering errors to stderr, and stop with exit code 1 when the model
//! has errors or the IR refuses it. Their results go to stdout (or to the file
//! given with `-o`).

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use pc_ir::check::{EntryReport, Options, check_model, slot_label};
use pc_ir::eval::live_set;
use pc_ir::{Entry, EntryKind, Fallback, IrModel, Slot, lower};
use pc_jit::check::{Row, jit_rows};
use pc_ode::{Loaded, Severity};

use crate::{load_file, out, text};

/// A loaded and lowered model.
pub(crate) struct Prepared {
    pub(crate) loaded: Loaded,
    pub(crate) ir: IrModel,
}

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

/// Loads `file`, prints its diagnostics and lowers it. `Err` carries the exit
/// code when the model has errors, cannot be read or is refused by the IR.
pub(crate) fn prepare(file: &Path) -> Result<Prepared, ExitCode> {
    let (map, loaded) = load_file(file)?;
    to_stderr(&text::diagnostics(&map, &loaded));
    if loaded
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error)
    {
        return Err(ExitCode::from(1));
    }
    match lower(&loaded.expanded, &loaded.info) {
        Ok(ir) => Ok(Prepared { loaded, ir }),
        Err(errors) => {
            to_stderr(&text::ir_errors(&map, &file.to_string_lossy(), &errors));
            Err(ExitCode::from(1))
        }
    }
}

fn dir(slots: &[Slot]) -> (EntryKind, String) {
    (EntryKind::Dir(slots.to_vec()), slot_label(slots))
}

/// `ir`: the shape of the model and the live DAG nodes of each entry.
pub fn ir(file: &Path) -> ExitCode {
    let Prepared { mut ir, .. } = match prepare(file) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let mut s = String::new();
    let _ = writeln!(s, "n: {}", ir.n());
    let _ = writeln!(s, "n_p: {}", ir.n_p());
    let _ = writeln!(s, "n_aux: {}", ir.aux.len());
    let _ = writeln!(s, "autonomous: {}", ir.autonomous);
    let _ = writeln!(s, "smooth: {}", ir.smooth);
    let _ = writeln!(s, "entries (live DAG nodes, outputs):");
    let mut kinds = vec![
        (EntryKind::F, "F".to_string()),
        (EntryKind::Aux, "Aux".to_string()),
        (EntryKind::JacU, "JacU".to_string()),
        (EntryKind::JacP, "JacP".to_string()),
    ];
    for k in 2..=5 {
        kinds.push(dir(&vec![Slot::U; k]));
    }
    for (kind, label) in &kinds {
        match ir.entry(kind) {
            Ok(e) => {
                let nodes = live_set(&ir.dag, &e.outputs).iter().filter(|b| **b).count();
                let _ = writeln!(
                    s,
                    "  {label:<14} {nodes:>8} nodes {:>6} outputs",
                    e.outputs.len()
                );
            }
            Err(e) => {
                let _ = writeln!(s, "  {label:<14} not built: {e}");
            }
        }
    }
    let _ = writeln!(s, "dag nodes after building the entries: {}", ir.dag.len());
    out(&s, 0)
}

/// `emit-c`: C code for F, JacU and JacP.
pub fn emit_c(file: &Path, target: Option<&Path>) -> ExitCode {
    let Prepared { mut ir, .. } = match prepare(file) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let mut entries: Vec<(String, Entry)> = Vec::new();
    for (name, kind) in [
        ("F", EntryKind::F),
        ("JacU", EntryKind::JacU),
        ("JacP", EntryKind::JacP),
    ] {
        match ir.entry(&kind) {
            Ok(e) => entries.push((name.to_string(), e)),
            Err(e) => {
                to_stderr(&format!("pc-cli: cannot build entry {name}: {e}\n"));
                return ExitCode::from(1);
            }
        }
    }
    let named: Vec<(String, &Entry)> = entries.iter().map(|(n, e)| (n.clone(), e)).collect();
    let code = match pc_codegen::emit_c(&ir, &named) {
        Ok(code) => code,
        Err(e) => {
            to_stderr(&format!("pc-cli: cannot emit C: {e}\n"));
            return ExitCode::from(1);
        }
    };
    match target {
        None => out(&code, 0),
        Some(path) => match std::fs::write(path, &code) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                to_stderr(&format!("pc-cli: cannot write {}: {e}\n", path.display()));
                ExitCode::from(1)
            }
        },
    }
}

/// One line for a finite-difference row of a model that is not smooth.
fn info_row(e: &EntryReport) -> String {
    let mut s = format!(
        "{:<22} info (non-smooth, not counted)  worst {:.3e}  s {:.3e}  ratio {:.3e}  {} points, {} values",
        e.label, e.worst_err, e.scale, e.ratio, e.points, e.compared
    );
    if let Some(n) = &e.note {
        let _ = write!(s, "  ({n})");
    }
    s
}

/// The printed rows and the final line of `check-derivs`, and how many counted
/// checks failed. The finite-difference rows of `check_model` (labels ending
/// in `(FD)`) do not count when the model is not smooth, since a kink between
/// the two sides of a difference makes them unreliable; every other row, and
/// every JIT row, counts. A `SKIP` row (an entry over the node limit) is printed
/// and counted apart: it adds to neither the checks nor the failures.
fn tally(entries: &[EntryReport], jit: &[Row], smooth: bool) -> (String, usize) {
    let mut s = String::new();
    let (mut counted, mut bad, mut skipped) = (0_usize, 0_usize, 0_usize);
    let mut too_large = 0_usize;
    for e in entries {
        if e.skip {
            too_large += 1;
            let _ = writeln!(s, "{e}");
            continue;
        }
        if !smooth && e.label.ends_with("(FD)") {
            skipped += 1;
            let _ = writeln!(s, "{}", info_row(e));
            continue;
        }
        counted += 1;
        bad += usize::from(!e.pass);
        let _ = writeln!(s, "{e}");
    }
    for r in jit {
        if r.skip {
            too_large += 1;
            let _ = writeln!(s, "{}", r.text);
            continue;
        }
        counted += 1;
        bad += usize::from(!r.pass);
        let _ = writeln!(s, "{}", r.text);
    }
    if bad == 0 {
        let _ = write!(s, "all {counted} checks pass");
    } else {
        let _ = write!(s, "{bad} of {counted} checks FAIL");
    }
    if too_large > 0 {
        let _ = write!(s, "; {too_large} skipped (entry too large)");
    }
    if skipped > 0 {
        let _ = write!(
            s,
            "; {skipped} finite-difference lines not counted (the model is not smooth)"
        );
    }
    s.push('\n');
    (s, bad)
}

/// `check-derivs`: `check_model` plus the JIT against the DAG and the fallback.
pub fn check_derivs(file: &Path, points: usize, seed: u64) -> ExitCode {
    let Prepared { loaded, mut ir } = match prepare(file) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let fb = Fallback::new(&loaded.expanded, &loaded.info, &ir);
    let opt = Options::new(points, seed);
    let report = check_model(&mut ir, &fb, points, seed);
    let smooth = ir.smooth;
    let jit = jit_rows(ir, &fb, points, seed, opt.half_width);
    let (body, bad) = tally(&report.entries, &jit, smooth);
    let head = format!("{} points per entry, seed {}\n", report.points, report.seed);
    out(&(head + &body), u8::from(bad > 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pc_jit::check::RowKind;

    fn entry(label: &str, pass: bool, note: Option<&str>) -> EntryReport {
        EntryReport {
            label: label.to_string(),
            pass,
            worst_err: 0.0,
            scale: 1.0,
            ratio: if pass { 0.0 } else { 2.0 },
            x: vec![0.5],
            p: vec![1.0],
            points: 3,
            compared: 6,
            note: note.map(str::to_string),
            skip: false,
            skipped_points: 0,
        }
    }

    fn jit(label: &str, pass: bool, detail: &str) -> Row {
        pc_jit::check::row(RowKind::JitVsDag, label, pass, detail)
    }

    const TOO_BIG: &str = "entry not built: too big (TooLarge)";

    fn skip(label: &str) -> Row {
        pc_jit::check::skip_row(RowKind::JitVsDag, label, TOO_BIG)
    }

    fn skipped_entry(label: &str) -> EntryReport {
        let mut e = entry(label, true, Some(TOO_BIG));
        e.skip = true;
        e
    }

    #[test]
    fn skipped_rows_are_printed_but_neither_counted_nor_failed() {
        let entries = [entry("F", true, None), skipped_entry("JacU")];
        let rows = [
            jit("F (JIT vs DAG, 0 ulp)", true, "3 points"),
            skip("JacU (JIT vs DAG, 0 ulp)"),
        ];
        for smooth in [true, false] {
            let (text, bad) = tally(&entries, &rows, smooth);
            assert_eq!(bad, 0, "smooth {smooth}: {text}");
            assert!(text.contains("SKIP"), "{text}");
            assert!(!text.contains("FAIL"), "{text}");
            assert!(
                text.ends_with("all 2 checks pass; 2 skipped (entry too large)\n"),
                "{text}"
            );
        }
    }

    #[test]
    fn a_real_failure_next_to_skipped_rows_still_fails() {
        let entries = [entry("F", false, None), skipped_entry("JacU")];
        let rows = [skip("JacU (JIT vs DAG, 0 ulp)")];
        let (text, bad) = tally(&entries, &rows, true);
        assert_eq!(bad, 1, "{text}");
        assert!(text.contains("1 of 1 checks FAIL"), "{text}");
        assert!(text.contains("2 skipped (entry too large)"), "{text}");
    }

    #[test]
    fn a_row_that_compared_nothing_fails_and_a_non_smooth_info_row_says_so() {
        let mut empty = entry(
            "JacP",
            false,
            Some("nothing was compared: no point was drawn"),
        );
        empty.points = 0;
        empty.compared = 0;
        let (text, bad) = tally(&[empty.clone()], &[], true);
        assert_eq!(bad, 1, "{text}");
        assert!(text.contains("FAIL"), "{text}");
        assert!(text.contains("0 points, 0 values"), "{text}");
        assert!(text.contains("nothing was compared"), "{text}");
        assert!(text.ends_with("1 of 1 checks FAIL\n"), "{text}");

        // A finite-difference row of a non-smooth model is printed as info with its counts
        // and its note, and is not counted.
        let mut fd = empty;
        fd.label = "A (FD)".to_string();
        let (text, bad) = tally(&[fd], &[], false);
        assert_eq!(bad, 0, "{text}");
        assert!(text.contains("info (non-smooth, not counted)"), "{text}");
        assert!(text.contains("0 points, 0 values"), "{text}");
        assert!(text.contains("nothing was compared"), "{text}");

        let ok = entry("F", true, None);
        let (text, _) = tally(&[ok], &[], true);
        assert!(text.contains("3 points, 6 values"), "{text}");
    }

    #[test]
    fn all_rows_passing_gives_no_failures() {
        let entries = [entry("F", true, None), entry("A (FD)", true, None)];
        let rows = [jit("F (JIT vs DAG, 0 ulp)", true, "3 points")];
        for smooth in [true, false] {
            let (text, bad) = tally(&entries, &rows, smooth);
            assert_eq!(bad, 0);
            assert!(text.contains("pass"), "{text}");
        }
        let (text, _) = tally(&entries, &rows, true);
        assert!(text.ends_with("all 3 checks pass\n"), "{text}");
        let (text, _) = tally(&entries, &rows, false);
        assert!(
            text.ends_with(
                "all 2 checks pass; 1 finite-difference lines not counted (the model is not smooth)\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn a_failing_exact_row_counts_on_smooth_and_non_smooth_models() {
        for smooth in [true, false] {
            let (text, bad) = tally(&[entry("JacU", false, None)], &[], smooth);
            assert_eq!(bad, 1, "smooth {smooth}");
            assert!(text.ends_with("1 of 1 checks FAIL\n"), "{text}");
        }
    }

    #[test]
    fn a_failing_fd_row_counts_only_on_a_smooth_model() {
        for label in ["A (FD)", "f_alpha (FD)", "B (FD)", "C (FD)"] {
            let entries = [entry("F", true, None), entry(label, false, None)];
            let (text, bad) = tally(&entries, &[], true);
            assert_eq!(bad, 1, "{label}");
            assert!(text.contains("FAIL"), "{text}");
            let (text, bad) = tally(&entries, &[], false);
            assert_eq!(bad, 0, "{label}");
            assert!(text.contains("info (non-smooth, not counted)"), "{text}");
            assert!(!text.contains("FAIL"), "{text}");
            assert!(text.starts_with("F "), "{text}");
        }
    }

    #[test]
    fn the_plain_fd_row_for_an_unbuilt_entry_counts_on_a_non_smooth_model() {
        let r = entry("FD", false, Some("entry not built: too large"));
        let (_, bad) = tally(&[r], &[], false);
        assert_eq!(bad, 1);
    }

    #[test]
    fn unbuilt_entries_and_failed_jit_calls_count() {
        let unbuilt = entry("Dir[U,U]", false, Some("entry not built: too large"));
        let (text, bad) = tally(&[unbuilt], &[], true);
        assert_eq!(bad, 1);
        assert!(text.contains("entry not built"), "{text}");
        let rows = [
            jit("F (JIT vs DAG, 0 ulp)", false, "entry not built: x"),
            jit("F (JIT vs Fallback)", false, "JIT call failed: bad length"),
            jit("JacU (JIT vs DAG, 0 ulp)", true, "3 points"),
        ];
        for smooth in [true, false] {
            let (text, bad) = tally(&[], &rows, smooth);
            assert_eq!(bad, 2, "smooth {smooth}");
            assert!(text.ends_with("2 of 3 checks FAIL\n"), "{text}");
            assert!(text.contains("JIT call failed"), "{text}");
        }
    }
}
