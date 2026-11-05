//! A booktabs `tabular` of special points: label, type, then one column per coordinate.

use crate::error::{ErrorKind, LatexError};
use crate::escape::escape;
use crate::input::{
    Object, as_object, number_array, reject_unknown, take_array, take_digits, take_string,
};
use crate::num::format_tex_math;
use crate::{Rendered, standalone_document, take_standalone};

pub(crate) fn render(mut obj: Object) -> Result<Rendered, LatexError> {
    let standalone = take_standalone(&mut obj)?;
    let digits = take_digits(&mut obj)?;
    let columns = take_array(&mut obj, "columns")?
        .iter()
        .map(|c| {
            c.as_str().map(str::to_string).ok_or_else(|| {
                LatexError::new(ErrorKind::BadInput, "`columns` must be an array of strings")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rows = take_array(&mut obj, "rows")?;
    reject_unknown(&obj, "the table")?;

    if rows.is_empty() {
        return Err(LatexError::new(
            ErrorKind::EmptyTable,
            "the table has no rows",
        ));
    }

    let mut body = String::new();
    body.push_str("% PhaseCont table. Needs \\usepackage{booktabs}.\n");
    body.push_str(&format!(
        "\\begin{{tabular}}{{ll{}}}\n\\toprule\n",
        "r".repeat(columns.len())
    ));
    let mut header = vec!["Label".to_string(), "Type".to_string()];
    header.extend(columns.iter().map(|c| escape(c)));
    body.push_str(&format!("{} \\\\\n\\midrule\n", header.join(" & ")));

    for (i, row) in rows.iter().enumerate() {
        let n = i + 1;
        let mut r = as_object(row, &format!("row {n}"))?.clone();
        let label = take_string(&mut r, "label")
            .map_err(|e| LatexError::new(e.kind(), format!("row {n}: {e}")))?;
        let kind = take_string(&mut r, "type")
            .map_err(|e| LatexError::new(e.kind(), format!("row {n}: {e}")))?;
        let values = r.remove("values").ok_or_else(|| {
            LatexError::new(
                ErrorKind::BadInput,
                format!("row {n}: `values` is required"),
            )
        })?;
        reject_unknown(&r, &format!("row {n}"))?;
        let values = number_array(&values, &format!("row {n}, values"))?;
        if values.len() != columns.len() {
            return Err(LatexError::new(
                ErrorKind::LengthMismatch,
                format!(
                    "row {n}: {} values for {} columns",
                    values.len(),
                    columns.len()
                ),
            ));
        }
        let mut cells = vec![escape(&label), escape(&kind)];
        cells.extend(values.iter().map(|v| format_tex_math(*v, digits)));
        body.push_str(&format!("{} \\\\\n", cells.join(" & ")));
    }
    body.push_str("\\bottomrule\n\\end{tabular}\n");

    let tex = if standalone {
        standalone_document("\\usepackage{booktabs}\n", &body)
    } else {
        body
    };
    Ok(Rendered {
        tex,
        data_files: Vec::new(),
    })
}
