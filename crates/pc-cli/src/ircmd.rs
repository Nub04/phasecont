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

use pc_ir::check::{
    EntryReport, Options, Rng, check_model, close, draw_point, fmt_vec, scale, slot_label,
};
use pc_ir::eval::{eval, live_set};
use pc_ir::{Entry, EntryKind, Fallback, IrModel, Slot, lower};
use pc_jit::CompiledModel;
use pc_ode::{Loaded, Severity};

use crate::{load_file, out, text};

/// A loaded and lowered model.
struct Prepared {
    loaded: Loaded,
    ir: IrModel,
}

fn to_stderr(text: &str) {
    let _ = std::io::stderr().write_all(text.as_bytes());
}

/// Loads `file`, prints its diagnostics and lowers it. `Err` carries the exit
/// code when the model has errors, cannot be read or is refused by the IR.
fn prepare(file: &Path) -> Result<Prepared, ExitCode> {
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
    let code = pc_codegen::emit_c(&ir, &named);
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

/// The entries the JIT is checked on: F, Aux (if the model has any), JacU, JacP
/// (if it has parameters), `Dir` of orders 1 to 5 with U slots, and for a
/// model with parameters a few mixes with P and Full slots.
fn jit_kinds(ir: &IrModel) -> Vec<(EntryKind, String)> {
    let mut kinds = vec![(EntryKind::F, "F".to_string())];
    if !ir.aux.is_empty() {
        kinds.push((EntryKind::Aux, "Aux".to_string()));
    }
    kinds.push((EntryKind::JacU, "JacU".to_string()));
    if ir.n_p() > 0 {
        kinds.push((EntryKind::JacP, "JacP".to_string()));
    }
    for k in 1..=5 {
        kinds.push(dir(&vec![Slot::U; k]));
    }
    if ir.n_p() > 0 {
        for mix in [
            vec![Slot::P],
            vec![Slot::Full],
            vec![Slot::U, Slot::P],
            vec![Slot::Full, Slot::Full],
            vec![Slot::U, Slot::U, Slot::P],
        ] {
            kinds.push(dir(&mix));
        }
    }
    kinds
}

/// One printed check line.
struct Row {
    pass: bool,
    text: String,
}

fn row(label: &str, pass: bool, detail: &str) -> Row {
    let verdict = if pass { "pass" } else { "FAIL" };
    Row {
        pass,
        text: format!("{label:<36} {verdict}  {detail}"),
    }
}

/// Same bits, except that a NaN matches any NaN: IEEE leaves the sign of a NaN
/// unspecified.
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The mismatches of one comparison: how many values differ and where the
/// first one was.
#[derive(Default)]
struct Mismatch {
    bad: usize,
    first: String,
}

impl Mismatch {
    /// Records the output `i` of the JIT, `jit`, against `other` (named
    /// `name`) at the point `x`, `p`.
    fn note(&mut self, i: usize, (x, p): (&[f64], &[f64]), jit: f64, other: f64, name: &str) {
        self.bad += 1;
        if self.first.is_empty() {
            self.first = format!(
                "first at output {i}, x = {}, p = {}: jit {jit:e}, {name} {other:e}",
                fmt_vec(x),
                fmt_vec(p)
            );
        }
    }

    fn row(&self, label: &str, points: usize, values: usize) -> Row {
        let detail = if self.bad == 0 {
            format!("{points} points, {values} values")
        } else {
            format!("{} of {values} values differ; {}", self.bad, self.first)
        };
        row(label, self.bad == 0, &detail)
    }
}

/// Compares the JIT with the DAG evaluator (bit for bit) and with the f64
/// fallback (the SPEC section 3 tolerance), two rows per entry. Points come
/// from the box of `check_model` with half-width `hw`.
fn jit_rows(mut ir: IrModel, fb: &Fallback, points: usize, seed: u64, hw: f64) -> Vec<Row> {
    let built: Vec<(EntryKind, String, Result<Entry, String>)> = jit_kinds(&ir)
        .into_iter()
        .map(|(k, l)| {
            let e = ir.entry(&k).map_err(|e| e.to_string());
            (k, l, e)
        })
        .collect();
    let mut cm = CompiledModel::new(ir);
    let mut rng = Rng(seed);
    let mut rows = Vec::new();
    for (kind, label, entry) in built {
        let l_dag = format!("{label} (JIT vs DAG, 0 ulp)");
        let l_fb = format!("{label} (JIT vs Fallback)");
        let entry = match entry {
            Ok(e) => e,
            Err(msg) => {
                let d = format!("entry not built: {msg}");
                rows.push(row(&l_dag, false, &d));
                rows.push(row(&l_fb, false, &d));
                continue;
            }
        };
        let (n, n_p) = (cm.ir().n(), cm.ir().n_p());
        let v_len = match &kind {
            EntryKind::Dir(slots) => slots.iter().map(|s| s.len(n, n_p)).sum(),
            _ => 0,
        };
        let (mut vs_dag, mut vs_fb) = (Mismatch::default(), Mismatch::default());
        let mut values = 0_usize;
        let mut call_error = None;
        for _ in 0..points {
            let (x, p) = draw_point(cm.ir(), &mut rng, hw);
            let v: Vec<f64> = (0..v_len).map(|_| rng.range(-1.0, 1.0)).collect();
            let mut want = vec![f64::NAN; entry.outputs.len()];
            eval(&cm.ir().dag, &entry.outputs, &x, &p, &v, &mut want);
            let mut got = vec![f64::NAN; entry.outputs.len()];
            if let Err(e) = cm.call(&kind, &x, &p, &v, &mut got) {
                call_error = Some(e.to_string());
                break;
            }
            let mut fbv = vec![f64::NAN; entry.outputs.len()];
            fb.eval(&kind, &x, &p, &v, &mut fbv);
            let s = scale(&fbv);
            for (i, ((&g, &w), &f)) in got.iter().zip(&want).zip(&fbv).enumerate() {
                values += 1;
                if !same_bits(g, w) {
                    vs_dag.note(i, (&x, &p), g, w, "dag");
                }
                if !close(g, f, s) {
                    vs_fb.note(i, (&x, &p), g, f, "fallback");
                }
            }
        }
        if let Some(msg) = call_error {
            let d = format!("JIT call failed: {msg}");
            rows.push(row(&l_dag, false, &d));
            rows.push(row(&l_fb, false, &d));
            continue;
        }
        rows.push(vs_dag.row(&l_dag, points, values));
        rows.push(vs_fb.row(&l_fb, points, values));
    }
    rows
}

/// One line for a finite-difference row of a model that is not smooth.
fn info_row(e: &EntryReport) -> String {
    format!(
        "{:<22} info (non-smooth, not counted)  worst {:.3e}  s {:.3e}  ratio {:.3e}",
        e.label, e.worst_err, e.scale, e.ratio
    )
}

/// The printed rows and the final line of `check-derivs`, and how many counted
/// checks failed. The finite-difference rows of `check_model` (labels ending
/// in `(FD)`) do not count when the model is not smooth, since a kink between
/// the two sides of a difference makes them unreliable; every other row, and
/// every JIT row, counts.
fn tally(entries: &[EntryReport], jit: &[Row], smooth: bool) -> (String, usize) {
    let mut s = String::new();
    let (mut counted, mut bad, mut skipped) = (0_usize, 0_usize, 0_usize);
    for e in entries {
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
        counted += 1;
        bad += usize::from(!r.pass);
        let _ = writeln!(s, "{}", r.text);
    }
    if bad == 0 {
        let _ = write!(s, "all {counted} checks pass");
    } else {
        let _ = write!(s, "{bad} of {counted} checks FAIL");
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

    #[test]
    fn same_bits_is_exact_except_for_nan() {
        assert!(same_bits(1.5, 1.5));
        assert!(!same_bits(1.0, 1.0 + f64::EPSILON));
        // Zeros of opposite sign differ in bits.
        assert!(!same_bits(0.0, -0.0));
        // Any NaN matches any NaN, whatever its sign or payload.
        assert!(same_bits(f64::NAN, -f64::NAN));
        assert!(same_bits(f64::NAN, f64::from_bits(0x7ff8_0000_0000_0001)));
        assert!(!same_bits(f64::NAN, 1.0));
        assert!(same_bits(f64::INFINITY, f64::INFINITY));
        assert!(!same_bits(f64::INFINITY, f64::NEG_INFINITY));
    }

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
            note: note.map(str::to_string),
        }
    }

    fn jit(label: &str, pass: bool, detail: &str) -> Row {
        row(label, pass, detail)
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

    #[test]
    fn a_mismatch_keeps_the_first_point_and_counts_all() {
        let mut m = Mismatch::default();
        m.note(1, (&[0.5], &[1.0, 0.25]), 2.0, 3.0, "dag");
        m.note(0, (&[9.0], &[9.0]), 5.0, 6.0, "dag");
        assert_eq!(m.bad, 2);
        assert!(
            m.first.starts_with("first at output 1, x = [5.000000e-1]"),
            "{}",
            m.first
        );
        assert!(m.first.contains("jit 2e0, dag 3e0"), "{}", m.first);
        let r = m.row("F (JIT vs DAG, 0 ulp)", 4, 8);
        assert!(!r.pass);
        assert!(
            r.text.contains("2 of 8 values differ; first at output 1"),
            "{}",
            r.text
        );
        let ok = Mismatch::default().row("F (JIT vs DAG, 0 ulp)", 4, 8);
        assert!(ok.pass);
        assert!(ok.text.ends_with("pass  4 points, 8 values"), "{}", ok.text);
    }
}
