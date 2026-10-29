//! Checks that `include/pckern.h` and the Rust `PcApiV1` table agree.

use core::mem::{offset_of, size_of};

use pckern::error::*;
use pckern::{PC_ABI_VERSION, PcApiV1, pc_api_v1};

const HEADER: &str = include_str!("../include/pckern.h");

/// The integer value of `#define NAME VALUE`, ignoring other lines and non-integer values.
fn define(name: &str) -> Option<u64> {
    HEADER.lines().find_map(|line| {
        let mut words = line.split_whitespace();
        if words.next()? != "#define" || words.next()? != name {
            return None;
        }
        words.next()?.parse().ok()
    })
}

fn need(name: &str) -> u64 {
    define(name).unwrap_or_else(|| panic!("pckern.h has no integer #define {name}"))
}

macro_rules! fields {
    ($($f:ident),* $(,)?) => {
        const FIELDS: &[&str] = &[$(stringify!($f)),*];

        #[cfg(target_pointer_width = "64")]
        fn rust_offsets() -> Vec<(&'static str, usize)> {
            vec![$((stringify!($f), offset_of!(PcApiV1, $f))),*]
        }
    };
}

fields!(
    abi_version,
    struct_size,
    model_from_ode,
    model_compile,
    model_rhs,
    run_start,
    run_poll,
    run_copy,
    run_cancel,
    free_handle,
    free_string,
    last_error,
    model_info,
    model_equilibrium,
    debug_panic,
    version,
);

#[test]
fn abi_version_matches() {
    assert_eq!(need("PC_ABI_VERSION"), PC_ABI_VERSION as u64);
    assert_eq!(
        need("PC_ABI_VERSION"),
        unsafe { (*pc_api_v1()).abi_version } as u64
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn size_matches() {
    assert_eq!(need("PC_API_V1_SIZE"), size_of::<PcApiV1>() as u64);
    assert_eq!(
        need("PC_API_V1_SIZE"),
        unsafe { (*pc_api_v1()).struct_size } as u64
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn offsets_match() {
    for (name, off) in rust_offsets() {
        let key = format!("PC_API_V1_OFFSET_{}", name.to_uppercase());
        assert_eq!(need(&key), off as u64, "{key}");
    }
}

#[test]
fn header_has_one_offset_per_field() {
    let n = HEADER
        .lines()
        .filter(|l| l.trim_start().starts_with("#define PC_API_V1_OFFSET_"))
        .count();
    assert_eq!(n, FIELDS.len());
}

#[test]
fn member_names_follow_rust_order() {
    let start = HEADER
        .find("typedef struct pc_api_v1_t")
        .expect("typedef struct pc_api_v1_t");
    let body = &HEADER[start..];
    let body = &body[body.find('{').expect("open brace") + 1..];
    let body = &body[..body.find("} pc_api_v1_t;").expect("closing line")];
    // Drop comments, then split into members at ';'.
    let mut code = String::new();
    let mut rest = body;
    while let Some(i) = rest.find("/*") {
        code.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("unterminated comment");
        rest = &rest[i + j + 2..];
    }
    code.push_str(rest);
    let names: Vec<String> = code
        .split(';')
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .map(|m| {
            let tail = match m.find("(*") {
                Some(i) => &m[i + 2..],
                None => m.rsplit(char::is_whitespace).next().unwrap(),
            };
            tail.chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect()
        })
        .collect();
    assert_eq!(names, FIELDS);
}

#[test]
fn codes_match() {
    let codes: [(&str, i32); 9] = [
        ("PC_OK", PC_OK),
        ("PC_ERR_PANIC", PC_ERR_PANIC),
        ("PC_ERR_BAD_ARG", PC_ERR_BAD_ARG),
        ("PC_ERR_BAD_HANDLE", PC_ERR_BAD_HANDLE),
        ("PC_ERR_INTERNAL", PC_ERR_INTERNAL),
        ("PC_ERR_MODEL", PC_ERR_MODEL),
        ("PC_ERR_BAD_PROBLEM", PC_ERR_BAD_PROBLEM),
        ("PC_ERR_NON_AUTONOMOUS", PC_ERR_NON_AUTONOMOUS),
        ("PC_ERR_NUMERIC", PC_ERR_NUMERIC),
    ];
    for (name, value) in codes {
        assert_eq!(need(name), value as u64, "{name}");
    }
    assert_eq!(need("PC_COPY_POINTS"), 0);
    assert_eq!(need("PC_COPY_SPECIALS"), 1);
}
