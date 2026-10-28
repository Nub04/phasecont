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

use pc_ir::check::{EntryReport, Rng, check_model, close, scale};
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

fn slot_label(slots: &[Slot]) -> String {
    let names: Vec<&str> = slots
        .iter()
        .map(|s| match s {
            Slot::U => "U",
            Slot::P => "P",
            Slot::Full => "Full",
        })
        .collect();
    format!("Dir[{}]", names.join(","))
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
                eprintln!("pc-cli: cannot build entry {name}: {e}");
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
                eprintln!("pc-cli: cannot write {}: {e}", path.display());
                ExitCode::from(1)
            }
        },
    }
}

fn fmt_vec(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.6e}")).collect();
    format!("[{}]", parts.join(", "))
}

/// A random point in the box around the model's initial state and parameters,
/// with `t` last in `p`. It follows the points of `pc_ir::check`.
fn draw_point(ir: &IrModel, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
    let x = (0..ir.n())
        .map(|j| ir.init.get(j).copied().unwrap_or(0.0) + 0.5 * rng.range(-1.0, 1.0))
        .collect();
    let mut p: Vec<f64> = (0..ir.n_p())
        .map(|j| {
            let b = ir.par_values.get(j).copied().unwrap_or(0.0);
            if b == 0.0 {
                0.2 * rng.range(-1.0, 1.0)
            } else {
                b * rng.range(0.9, 1.1)
            }
        })
        .collect();
    p.push(rng.unit());
    (x, p)
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
        text: format!("{label:<30} {verdict}  {detail}"),
    }
}

/// Same bits, except that a NaN matches any NaN: IEEE leaves the sign of a NaN
/// unspecified.
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Compares the JIT with the DAG evaluator (bit for bit) and with the f64
/// fallback (the SPEC section 3 tolerance), two rows per entry.
fn jit_rows(mut ir: IrModel, fb: &Fallback, points: usize, seed: u64) -> Vec<Row> {
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
        let (mut bit_bad, mut fb_bad, mut values) = (0_usize, 0_usize, 0_usize);
        let mut first_bit = String::new();
        let mut first_fb = String::new();
        let mut call_error = None;
        for _ in 0..points {
            let (x, p) = draw_point(cm.ir(), &mut rng);
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
                    bit_bad += 1;
                    if first_bit.is_empty() {
                        first_bit = format!(
                            "first at output {i}, x = {}, p = {}: jit {g:e}, dag {w:e}",
                            fmt_vec(&x),
                            fmt_vec(&p)
                        );
                    }
                }
                if !close(g, f, s) {
                    fb_bad += 1;
                    if first_fb.is_empty() {
                        first_fb = format!(
                            "first at output {i}, x = {}, p = {}: jit {g:e}, fallback {f:e}",
                            fmt_vec(&x),
                            fmt_vec(&p)
                        );
                    }
                }
            }
        }
        if let Some(msg) = call_error {
            let d = format!("JIT call failed: {msg}");
            rows.push(row(&l_dag, false, &d));
            rows.push(row(&l_fb, false, &d));
            continue;
        }
        for (l, bad, first) in [(&l_dag, bit_bad, first_bit), (&l_fb, fb_bad, first_fb)] {
            let detail = if bad == 0 {
                format!("{points} points, {values} values")
            } else {
                format!("{bad} of {values} values differ; {first}")
            };
            rows.push(row(l, bad == 0, &detail));
        }
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

/// `check-derivs`: `check_model` plus the JIT against the DAG and the fallback.
pub fn check_derivs(file: &Path, points: usize, seed: u64) -> ExitCode {
    let Prepared { loaded, mut ir } = match prepare(file) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let fb = Fallback::new(&loaded.expanded, &loaded.info, &ir);
    let report = check_model(&mut ir, &fb, points, seed);
    let smooth = ir.smooth;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{} points per entry, seed {}",
        report.points, report.seed
    );
    let (mut counted, mut bad, mut skipped) = (0_usize, 0_usize, 0_usize);
    for e in &report.entries {
        if !smooth && e.label.ends_with("(FD)") {
            skipped += 1;
            let _ = writeln!(s, "{}", info_row(e));
            continue;
        }
        counted += 1;
        bad += usize::from(!e.pass);
        let _ = writeln!(s, "{e}");
    }
    for r in jit_rows(ir, &fb, points, seed) {
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
    out(&s, u8::from(bad > 0))
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

    #[test]
    fn slot_labels() {
        assert_eq!(slot_label(&[Slot::U, Slot::P, Slot::Full]), "Dir[U,P,Full]");
    }
}
