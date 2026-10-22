//! S4 checks through the function table that the MEX gateway gets from
//! `pc_api_v1()`. Bad arguments fail before any CUDA call, so those checks
//! run on any machine. The GPU run skips when the CUDA driver or NVRTC
//! library is missing, and otherwise checks the report against contract
//! section 5.

use std::ffi::{CStr, CString, c_char};

use pckern::ffi::{PC_ERR_BAD_ARG, PC_OK};
use pckern::{PcApiV1, pc_api_v1};

fn api() -> &'static PcApiV1 {
    unsafe { &*pc_api_v1() }
}

fn last_error() -> String {
    let p = (api().last_error)();
    assert!(!p.is_null());
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Calls `run_json("s4", args)` and returns the code and the report, which
/// it frees with `free_string`.
fn run_s4(args: &str) -> (i32, Option<String>) {
    let args = CString::new(args).unwrap();
    let mut out: *mut c_char = std::ptr::dangling_mut();
    let rc = unsafe { (api().run_json)(c"s4".as_ptr(), args.as_ptr(), &mut out) };
    if out.is_null() {
        return (rc, None);
    }
    let text = unsafe { CStr::from_ptr(out) }.to_str().unwrap().to_owned();
    unsafe { (api().free_string)(out) };
    (rc, Some(text))
}

#[test]
fn bad_arguments_fail_before_any_cuda_call() {
    // Each token appears only in the message for that argument: the range
    // checks in s4_cuda.rs, then the ffi::Args parse and key errors.
    let cases = [
        ("n=0", "s4: n=0 is out of range"),
        ("n=134217729", "s4: n=134217729 is out of range"),
        ("reps=0", "s4: reps=0 is out of range"),
        ("reps=10001", "s4: reps=10001 is out of range"),
        ("n=abc", "n=abc is not a non-negative integer"),
        ("size=10", r#"unknown key "size""#),
    ];
    for (args, token) in cases {
        let (rc, out) = run_s4(args);
        let msg = last_error();
        assert_eq!(rc, PC_ERR_BAD_ARG, "{args}: {msg}");
        assert!(out.is_none(), "{args}");
        assert!(msg.contains(token), "{args}: {msg}");
    }
}

const TOP_KEYS: [&str; 16] = [
    "spike",
    "target",
    "profile",
    "driver_version",
    "nvrtc_major",
    "nvrtc_minor",
    "nvrtc_path",
    "device",
    "cc_major",
    "cc_minor",
    "n",
    "a",
    "reps",
    "chosen",
    "paths",
    "pass",
];

const PATH_KEYS: [&str; 10] = [
    "name",
    "arch",
    "compile_ms",
    "load_ms",
    "error",
    "max_ulp_fma",
    "max_ulp_unfused",
    "gbps",
    "timings",
    "pass",
];

const TIMING_KEYS: [&str; 4] = ["name", "median_ns", "min_ns", "max_ns"];

#[test]
fn small_gpu_run_follows_the_contract() {
    let driver = unsafe { cudarc::driver::sys::is_culib_present() };
    let nvrtc = unsafe { cudarc::nvrtc::sys::is_culib_present() };
    if !driver || !nvrtc {
        println!("skipped: CUDA driver library found: {driver}, NVRTC library found: {nvrtc}");
        return;
    }
    let (rc, out) = run_s4("n=4097 reps=2");
    assert_eq!(rc, PC_OK, "{}", last_error());
    let text = out.expect("a report");
    let r = Json::parse(&text).unwrap_or_else(|e| panic!("{e}: {text}"));

    assert_eq!(r.keys(), TOP_KEYS, "{text}");
    assert_eq!(r.get("spike").str(), "s4");
    assert!(!r.get("target").str().is_empty());
    assert!(!r.get("profile").str().is_empty());
    let dv = r.get("driver_version").int();
    let (nj, nn) = (r.get("nvrtc_major").int(), r.get("nvrtc_minor").int());
    let (maj, min) = (r.get("cc_major").int(), r.get("cc_minor").int());
    assert!(dv > 0 && nj > 0 && maj > 0, "{text}");
    if cfg!(target_os = "linux") {
        assert!(r.get("nvrtc_path").str().contains("libnvrtc.so"), "{text}");
    }
    assert!(!r.get("device").str().is_empty());
    assert_eq!(r.get("n").int(), 4097);
    assert_eq!(r.get("a").num(), f64::from(1.7f32));
    assert_eq!(r.get("reps").int(), 2);

    // The selection rule from the contract, applied to the reported versions.
    let expected = if dv >= nj * 1000 + nn * 10 {
        "ptx"
    } else if dv >= 13000 {
        "cubin"
    } else {
        ""
    };
    let chosen = r.get("chosen").str();
    assert_eq!(chosen, expected, "{text}");
    assert!(chosen == "ptx" || chosen == "cubin", "{text}");

    let paths = r.get("paths").arr();
    assert_eq!(paths.len(), 2, "{text}");
    let archs = [format!("compute_{maj}{min}"), format!("sm_{maj}{min}")];
    for ((p, name), arch) in paths.iter().zip(["ptx", "cubin"]).zip(&archs) {
        assert_eq!(p.keys(), PATH_KEYS, "{text}");
        assert_eq!(p.get("name").str(), name);
        assert_eq!(p.get("arch").str(), arch);
        for k in [
            "compile_ms",
            "load_ms",
            "max_ulp_fma",
            "max_ulp_unfused",
            "gbps",
        ] {
            assert!(p.get(k).is_num_or_null(), "{name}.{k}: {text}");
        }
        let error = p.get("error").str();
        let timings = p.get("timings").arr();
        let fma = p.get("max_ulp_fma").num_or_nan();
        assert_eq!(
            p.get("pass").bool(),
            error.is_empty() && fma <= 1.0,
            "{text}"
        );
        if error.is_empty() {
            assert_eq!(timings.len(), 1, "{text}");
            let t = &timings[0];
            assert_eq!(t.keys(), TIMING_KEYS, "{text}");
            assert_eq!(t.get("name").str(), "kernel");
            let (lo, mid, hi) = (
                t.get("min_ns").num(),
                t.get("median_ns").num(),
                t.get("max_ns").num(),
            );
            assert!(0.0 < lo && lo <= mid && mid <= hi, "{text}");
            assert!(p.get("compile_ms").num() > 0.0, "{text}");
            assert!(p.get("load_ms").num() > 0.0, "{text}");
            assert!(p.get("gbps").num() > 0.0, "{text}");
        } else {
            assert!(timings.is_empty(), "{text}");
            assert!(p.get("gbps").is_null(), "{text}");
            println!("{name} path failed: {error}");
        }
    }
    let chosen_path = paths.iter().find(|p| p.get("name").str() == chosen);
    assert!(chosen_path.unwrap().get("pass").bool(), "{text}");
    assert!(r.get("pass").bool(), "{text}");
    println!("{text}");
}

/// The JSON values that `ffi::Json` writes. Numbers written by `Json::int`
/// have no exponent and parse as `Int`; `Json::num` always writes one.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser {
            s: text.as_bytes(),
            i: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("trailing text at byte {}", p.i));
        }
        Ok(v)
    }

    fn keys(&self) -> Vec<&str> {
        match self {
            Json::Obj(kv) => kv.iter().map(|(k, _)| k.as_str()).collect(),
            other => panic!("not an object: {other:?}"),
        }
    }

    fn get(&self, key: &str) -> &Json {
        match self {
            Json::Obj(kv) => kv
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("no key {key}")),
            other => panic!("not an object: {other:?}"),
        }
    }

    fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            other => panic!("not a string: {other:?}"),
        }
    }

    fn int(&self) -> i64 {
        match self {
            Json::Int(v) => *v,
            other => panic!("not an int: {other:?}"),
        }
    }

    fn num(&self) -> f64 {
        match self {
            Json::Num(v) => *v,
            other => panic!("not a num: {other:?}"),
        }
    }

    fn num_or_nan(&self) -> f64 {
        match self {
            Json::Null => f64::NAN,
            _ => self.num(),
        }
    }

    fn is_num_or_null(&self) -> bool {
        matches!(self, Json::Num(_) | Json::Null)
    }

    fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    fn bool(&self) -> bool {
        match self {
            Json::Bool(b) => *b,
            other => panic!("not a bool: {other:?}"),
        }
    }

    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            other => panic!("not an array: {other:?}"),
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(|c| c.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", c as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    let k = self.string()?;
                    self.eat(b':')?;
                    kv.push((k, self.value()?));
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        self.eat(b'}')?;
                        return Ok(Json::Obj(kv));
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut v = Vec::new();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    v.push(self.value()?);
                    if self.peek() == Some(b',') {
                        self.i += 1;
                    } else {
                        self.eat(b']')?;
                        return Ok(Json::Arr(v));
                    }
                }
            }
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'n') => self.word("null", Json::Null),
            Some(_) => self.number(),
            None => Err("unexpected end of text".to_owned()),
        }
    }

    fn word(&mut self, w: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(w.as_bytes()) {
            self.i += w.len();
            Ok(v)
        } else {
            Err(format!("bad literal at byte {}", self.i))
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || b"+-.eE".contains(c))
        {
            self.i += 1;
        }
        let t = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        if t.contains(['.', 'e', 'E']) {
            t.parse().map(Json::Num).map_err(|e| format!("{t:?}: {e}"))
        } else {
            t.parse().map(Json::Int).map_err(|e| format!("{t:?}: {e}"))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hex = self.s.get(self.i..self.i + 4).ok_or("short \\u escape")?;
                            self.i += 4;
                            let code = u32::from_str_radix(std::str::from_utf8(hex).unwrap(), 16)
                                .map_err(|e| e.to_string())?;
                            char::from_u32(code).ok_or("bad \\u escape")?
                        }
                        _ => return Err(format!("bad escape at byte {}", self.i)),
                    };
                    out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes());
                }
                _ => out.push(c),
            }
        }
    }
}

#[test]
fn the_test_parser_reads_what_ffi_json_writes() {
    let mut inner = pckern::ffi::Json::default();
    inner.str("name", "kernel").num("median_ns", 8.4e5);
    let mut j = pckern::ffi::Json::default();
    j.str("s", "a\"b\\c\n\u{1}")
        .int("i", -13040)
        .num("x", 1.7f32 as f64)
        .num("nan", f64::NAN)
        .boolean("b", false)
        .objs("rows", vec![inner])
        .objs("none", vec![]);
    let v = Json::parse(&j.finish()).unwrap();
    assert_eq!(v.keys(), ["s", "i", "x", "nan", "b", "rows", "none"]);
    assert_eq!(v.get("s").str(), "a\"b\\c\n\u{1}");
    assert_eq!(v.get("i").int(), -13040);
    assert_eq!(v.get("x").num(), f64::from(1.7f32));
    assert!(v.get("nan").is_null());
    assert!(!v.get("b").bool());
    assert_eq!(v.get("rows").arr()[0].get("median_ns").num(), 8.4e5);
    assert!(v.get("none").arr().is_empty());
}
