//! The JIT checks of SPEC section 4: the JIT against the DAG evaluator bit for bit, and
//! against the f64 fallback within the section 3 tolerance, with the running error bound of
//! the DAG evaluation of the same entry.
//!
//! `pc-cli check-derivs` and the corpus test share this code.

use crate::CompiledModel;
use pc_ir::check::{Rng, close_err, draw_point, fmt_vec, scale, slot_label};
use pc_ir::eval::eval_err;
use pc_ir::{Entry, EntryKind, Fallback, IrModel, Slot};

fn dir(slots: &[Slot]) -> (EntryKind, String) {
    (EntryKind::Dir(slots.to_vec()), slot_label(slots))
}

/// The entries the JIT is checked on: F, Aux (if the model has any), JacU, JacP
/// (if it has parameters), `Dir` of orders 1 to 5 with U slots, and for a
/// model with parameters a few mixes with P and Full slots.
pub fn jit_kinds(ir: &IrModel) -> Vec<(EntryKind, String)> {
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
pub struct Row {
    pub pass: bool,
    pub text: String,
}

pub fn row(label: &str, pass: bool, detail: &str) -> Row {
    let verdict = if pass { "pass" } else { "FAIL" };
    Row {
        pass,
        text: format!("{label:<36} {verdict}  {detail}"),
    }
}

/// Same bits, except that a NaN matches any NaN: IEEE leaves the sign of a NaN
/// unspecified.
pub fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The mismatches of one comparison: how many values differ and where the
/// first one was.
#[derive(Default)]
pub struct Mismatch {
    pub bad: usize,
    pub first: String,
}

impl Mismatch {
    /// Records the output `i` of the JIT, `jit`, against `other` (named
    /// `name`) at the point `x`, `p`.
    pub fn note(&mut self, i: usize, (x, p): (&[f64], &[f64]), jit: f64, other: f64, name: &str) {
        self.bad += 1;
        if self.first.is_empty() {
            self.first = format!(
                "first at output {i}, x = {}, p = {}: jit {jit:e}, {name} {other:e}",
                fmt_vec(x),
                fmt_vec(p)
            );
        }
    }

    pub fn row(&self, label: &str, points: usize, values: usize) -> Row {
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
pub fn jit_rows(mut ir: IrModel, fb: &Fallback, points: usize, seed: u64, hw: f64) -> Vec<Row> {
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
            let mut err = vec![f64::NAN; entry.outputs.len()];
            eval_err(
                &cm.ir().dag,
                &entry.outputs,
                &x,
                &p,
                &v,
                &mut want,
                &mut err,
            );
            let mut got = vec![f64::NAN; entry.outputs.len()];
            if let Err(e) = cm.call(&kind, &x, &p, &v, &mut got) {
                call_error = Some(e.to_string());
                break;
            }
            let mut fbv = vec![f64::NAN; entry.outputs.len()];
            fb.eval(&kind, &x, &p, &v, &mut fbv);
            let s = scale(&fbv);
            for (i, (((&g, &w), &f), &e)) in got.iter().zip(&want).zip(&fbv).zip(&err).enumerate() {
                values += 1;
                if !same_bits(g, w) {
                    vs_dag.note(i, (&x, &p), g, w, "dag");
                }
                if !close_err(g, f, s, e) {
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
