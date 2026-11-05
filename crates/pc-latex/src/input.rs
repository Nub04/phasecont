//! Reading the JSON documents. Every reader removes the keys it takes, so that
//! [`reject_unknown`] can report a key nobody read, such as a misspelled option.

use crate::error::{ErrorKind, LatexError};
use serde_json::{Map, Value};

pub(crate) type Object = Map<String, Value>;

fn bad(message: impl Into<String>) -> LatexError {
    LatexError::new(ErrorKind::BadInput, message)
}

/// `digits` is required and must be an integer from 1 to 17.
pub(crate) fn take_digits(obj: &mut Object) -> Result<usize, LatexError> {
    let Some(v) = obj.remove("digits") else {
        return Err(LatexError::new(
            ErrorKind::MissingDigits,
            "`digits` is required: the number of significant digits to print",
        ));
    };
    match v.as_u64() {
        Some(d @ 1..=17) => Ok(d as usize),
        _ => Err(LatexError::new(
            ErrorKind::BadDigits,
            format!("`digits` must be an integer from 1 to 17, got {v}"),
        )),
    }
}

pub(crate) fn take_string(obj: &mut Object, key: &str) -> Result<String, LatexError> {
    take_opt_string(obj, key)?.ok_or_else(|| bad(format!("`{key}` is required")))
}

pub(crate) fn take_opt_string(obj: &mut Object, key: &str) -> Result<Option<String>, LatexError> {
    match obj.remove(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(bad(format!("`{key}` must be a string"))),
    }
}

pub(crate) fn take_array(obj: &mut Object, key: &str) -> Result<Vec<Value>, LatexError> {
    match obj.remove(key) {
        Some(Value::Array(a)) => Ok(a),
        Some(_) => Err(bad(format!("`{key}` must be an array"))),
        None => Err(bad(format!("`{key}` is required"))),
    }
}

/// One finite number. `null` is what MATLAB's `jsonencode` writes for NaN and Inf, so it
/// gets its own message.
pub(crate) fn number(v: &Value, what: &str) -> Result<f64, LatexError> {
    match v {
        Value::Null => Err(LatexError::new(
            ErrorKind::NonFinite,
            format!("{what} is null, NaN or infinite"),
        )),
        Value::Number(n) => match n.as_f64() {
            Some(x) if x.is_finite() => Ok(x),
            _ => Err(LatexError::new(
                ErrorKind::NonFinite,
                format!("{what} is not a finite number"),
            )),
        },
        _ => Err(bad(format!("{what} must be a number"))),
    }
}

/// An array of finite numbers. `what` names the array, and the message gives the
/// 1-based index of the first bad entry as `what(i)`.
pub(crate) fn number_array(v: &Value, what: &str) -> Result<Vec<f64>, LatexError> {
    let Value::Array(a) = v else {
        return Err(bad(format!("{what} must be an array of numbers")));
    };
    a.iter()
        .enumerate()
        .map(|(i, e)| number(e, &format!("{what}({})", i + 1)))
        .collect()
}

pub(crate) fn reject_unknown(obj: &Object, context: &str) -> Result<(), LatexError> {
    match obj.keys().next() {
        None => Ok(()),
        Some(k) => Err(bad(format!("unknown key `{k}` in {context}"))),
    }
}

pub(crate) fn as_object<'a>(v: &'a Value, what: &str) -> Result<&'a Object, LatexError> {
    v.as_object()
        .ok_or_else(|| bad(format!("{what} must be an object")))
}
