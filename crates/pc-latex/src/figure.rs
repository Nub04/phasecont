//! A pgfplots `axis` with one line per series and one mark per special point.

use crate::error::{ErrorKind, LatexError};
use crate::escape::escape;
use crate::input::{
    Object, as_object, number, number_array, reject_unknown, take_array, take_digits,
    take_opt_string, take_string,
};
use crate::num::format_significant;
use crate::{PGFPLOTS_COMPAT, Rendered, standalone_document, take_standalone};
use serde_json::Value;

struct Series {
    name: Option<String>,
    unstable: bool,
    color: Option<(u8, u8, u8)>,
    x: Vec<f64>,
    y: Vec<f64>,
}

struct Mark {
    label: String,
    x: f64,
    y: f64,
}

pub(crate) fn render(mut obj: Object) -> Result<Rendered, LatexError> {
    let standalone = take_standalone(&mut obj)?;
    let digits = take_digits(&mut obj)?;
    let xlabel = take_string(&mut obj, "xlabel")?;
    let ylabel = take_string(&mut obj, "ylabel")?;
    let title = take_opt_string(&mut obj, "title")?;
    let external = take_data_mode(&mut obj)?;
    let basename = take_opt_string(&mut obj, "basename")?;
    let series = take_series(take_array(&mut obj, "series")?)?;
    let marks = match obj.remove("marks") {
        None => Vec::new(),
        Some(Value::Array(a)) => take_marks(&a)?,
        Some(_) => {
            return Err(LatexError::new(
                ErrorKind::BadInput,
                "`marks` must be an array",
            ));
        }
    };
    reject_unknown(&obj, "the figure")?;

    if series.is_empty() {
        return Err(LatexError::new(
            ErrorKind::EmptySeries,
            "the figure has no series",
        ));
    }
    let basename = match (external, basename) {
        (true, Some(b)) => {
            check_basename(&b)?;
            Some(b)
        }
        (true, None) => {
            return Err(LatexError::new(
                ErrorKind::BadName,
                "`basename` is required when `data` is \"external\"",
            ));
        }
        (false, Some(_)) => {
            return Err(LatexError::new(
                ErrorKind::BadInput,
                "`basename` is only used when `data` is \"external\"",
            ));
        }
        (false, None) => None,
    };

    let fmt = |x: f64| format_significant(x, digits);
    let data_name = |i: usize| format!("{}-s{}.dat", basename.as_deref().unwrap_or(""), i + 1);

    let mut body = String::new();
    body.push_str(&format!(
        "% PhaseCont figure. Needs \\usepackage{{pgfplots}} and \\pgfplotsset{{compat={PGFPLOTS_COMPAT}}}.\n"
    ));
    let mut data_files = Vec::new();
    if external {
        let names: Vec<String> = (0..series.len()).map(data_name).collect();
        body.push_str(&format!(
            "% Data files: {} (keep them next to the document that reads this file).\n",
            names.join(", ")
        ));
    }
    body.push_str("\\begin{tikzpicture}\n\\begin{axis}[\n");
    body.push_str(&format!("  xlabel={{{}}},\n", escape(&xlabel)));
    body.push_str(&format!("  ylabel={{{}}},\n", escape(&ylabel)));
    if let Some(t) = &title {
        body.push_str(&format!("  title={{{}}},\n", escape(t)));
    }
    body.push_str("]\n");

    for (i, s) in series.iter().enumerate() {
        let mut opts = vec!["mark=none".to_string()];
        if s.unstable {
            opts.push("dashed".to_string());
        }
        if let Some((r, g, b)) = s.color {
            opts.push(format!("color={{rgb,255:red,{r};green,{g};blue,{b}}}"));
        }
        if s.name.is_none() {
            opts.push("forget plot".to_string());
        }
        body.push_str(&format!("\\addplot+[{}]", opts.join(", ")));
        if external {
            let file = data_name(i);
            body.push_str(&format!(" table[x=x, y=y] {{{file}}};\n"));
            let mut content = String::from("x y\n");
            for (x, y) in s.x.iter().zip(&s.y) {
                content.push_str(&format!("{} {}\n", fmt(*x), fmt(*y)));
            }
            data_files.push((file, content));
        } else {
            body.push_str(" coordinates {\n");
            for (x, y) in s.x.iter().zip(&s.y) {
                body.push_str(&format!("  ({},{})\n", fmt(*x), fmt(*y)));
            }
            body.push_str("};\n");
        }
        if let Some(name) = &s.name {
            body.push_str(&format!("\\addlegendentry{{{}}}\n", escape(name)));
        }
    }
    for m in &marks {
        let (x, y) = (fmt(m.x), fmt(m.y));
        body.push_str(&format!(
            "\\addplot[only marks, mark=*, forget plot] coordinates {{({x},{y})}};\n"
        ));
        body.push_str(&format!(
            "\\node[above right] at (axis cs:{x},{y}) {{{}}};\n",
            escape(&m.label)
        ));
    }
    body.push_str("\\end{axis}\n\\end{tikzpicture}\n");

    let tex = if standalone {
        standalone_document(
            &format!("\\usepackage{{pgfplots}}\n\\pgfplotsset{{compat={PGFPLOTS_COMPAT}}}\n"),
            &body,
        )
    } else {
        body
    };
    Ok(Rendered { tex, data_files })
}

/// `data` is `"inline"` (the default) or `"external"`. Returns true for external.
fn take_data_mode(obj: &mut Object) -> Result<bool, LatexError> {
    match obj.remove("data") {
        None => Ok(false),
        Some(Value::String(s)) if s == "inline" => Ok(false),
        Some(Value::String(s)) if s == "external" => Ok(true),
        Some(v) => Err(LatexError::new(
            ErrorKind::BadInput,
            format!("`data` must be \"inline\" or \"external\", got {v}"),
        )),
    }
}

fn check_basename(b: &str) -> Result<(), LatexError> {
    let ok = !b.is_empty()
        && b.len() <= 100
        && b.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if ok {
        Ok(())
    } else {
        Err(LatexError::new(
            ErrorKind::BadName,
            format!("`basename` must have 1 to 100 characters from A-Z a-z 0-9 _ -, got {b:?}"),
        ))
    }
}

fn take_series(items: Vec<Value>) -> Result<Vec<Series>, LatexError> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let n = i + 1;
        let mut s = as_object(item, &format!("series {n}"))?.clone();
        let name = take_opt_string(&mut s, "name")?;
        let class = take_string(&mut s, "class")
            .map_err(|e| LatexError::new(e.kind(), format!("series {n}: {e}")))?;
        let unstable = match class.as_str() {
            "stable" => false,
            "unstable" => true,
            other => {
                return Err(LatexError::new(
                    ErrorKind::BadInput,
                    format!(
                        "series {n}: `class` must be \"stable\" or \"unstable\", got {other:?}"
                    ),
                ));
            }
        };
        let color = match take_opt_string(&mut s, "color")? {
            None => None,
            Some(c) => Some(parse_color(&c).ok_or_else(|| {
                LatexError::new(
                    ErrorKind::BadInput,
                    format!("series {n}: `color` must look like #RRGGBB, got {c:?}"),
                )
            })?),
        };
        let x = s.remove("x").ok_or_else(|| {
            LatexError::new(ErrorKind::BadInput, format!("series {n}: `x` is required"))
        })?;
        let y = s.remove("y").ok_or_else(|| {
            LatexError::new(ErrorKind::BadInput, format!("series {n}: `y` is required"))
        })?;
        reject_unknown(&s, &format!("series {n}"))?;
        let x = number_array(&x, &format!("series {n}, x"))?;
        let y = number_array(&y, &format!("series {n}, y"))?;
        if x.len() != y.len() {
            return Err(LatexError::new(
                ErrorKind::LengthMismatch,
                format!("series {n}: x has {} points and y has {}", x.len(), y.len()),
            ));
        }
        if x.is_empty() {
            return Err(LatexError::new(
                ErrorKind::EmptySeries,
                format!("series {n} has no points"),
            ));
        }
        out.push(Series {
            name,
            unstable,
            color,
            x,
            y,
        });
    }
    Ok(out)
}

fn take_marks(items: &[Value]) -> Result<Vec<Mark>, LatexError> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let n = i + 1;
        let mut m = as_object(item, &format!("mark {n}"))?.clone();
        let label = take_string(&mut m, "label")
            .map_err(|e| LatexError::new(e.kind(), format!("mark {n}: {e}")))?;
        // `type` belongs to the table rows, which share this object. A figure ignores it.
        take_opt_string(&mut m, "type")?;
        let x = m.remove("x").ok_or_else(|| {
            LatexError::new(ErrorKind::BadInput, format!("mark {n}: `x` is required"))
        })?;
        let y = m.remove("y").ok_or_else(|| {
            LatexError::new(ErrorKind::BadInput, format!("mark {n}: `y` is required"))
        })?;
        reject_unknown(&m, &format!("mark {n}"))?;
        out.push(Mark {
            label,
            x: number(&x, &format!("mark {n}, x"))?,
            y: number(&y, &format!("mark {n}, y"))?,
        });
    }
    Ok(out)
}

fn parse_color(s: &str) -> Option<(u8, u8, u8)> {
    let hex = s.strip_prefix('#')?;
    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    Some((byte(0)?, byte(2)?, byte(4)?))
}
