//! Runs `pc-cli dde-roots` on x'(t) = -(pi/2) x(t - 1), whose roots with real part zero
//! are exactly +-i pi/2, and on bad problem files.

mod common;

use std::f64::consts::FRAC_PI_2;
use std::path::{Path, PathBuf};

use common::{code, run, stderr, stdout};
use serde_json::Value;

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dde_roots");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Writes `text` to a file called `name` in the scratch directory and returns its path.
fn write(name: &str, text: &str) -> String {
    let path = scratch().join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn neutral(n: usize) -> String {
    format!(
        r#"{{"dim":1,"a0":[0.0],"delays":[{{"tau":1.0,"a":[{:?}]}}],"n":{n},"re_min":-3.0}}"#,
        -FRAC_PI_2
    )
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

#[test]
fn the_neutral_pair_is_found() {
    let file = write("neutral.json", &neutral(20));
    let o = run(&scratch(), &["dde-roots", &file]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["n"], 20);
    assert!(v["unrefined"].as_array().unwrap().is_empty(), "{v}");
    let roots = v["roots"].as_array().unwrap();
    assert!(!roots.is_empty());
    for r in roots {
        assert!(num(&r["residual"]) <= 1e-12, "{r}");
    }
    let top = num(&roots[0]["re"]);
    assert!(top.abs() <= 1e-12, "largest real part {top}");
    for sign in [1.0, -1.0] {
        let hit = roots.iter().any(|r| {
            num(&r["re"]).abs() <= 1e-12 && (num(&r["im"]) - sign * FRAC_PI_2).abs() <= 1e-12
        });
        assert!(hit, "no root at {}i in {v}", sign * FRAC_PI_2);
    }
    for pair in roots.windows(2) {
        assert!(
            num(&pair[0]["re"]) >= num(&pair[1]["re"]),
            "not sorted: {v}"
        );
    }
}

/// Runs a problem text, expecting exit 2, and returns stderr.
fn bad(name: &str, text: &str) -> String {
    let file = write(name, text);
    let o = run(&scratch(), &["dde-roots", &file]);
    assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
    assert!(stdout(&o).is_empty(), "{name}: stdout {}", stdout(&o));
    stderr(&o)
}

#[test]
fn bad_problem_files_exit_two_and_name_the_key() {
    let good = neutral(20);
    let no_n = good.replace(r#","n":20"#, "");
    assert!(bad("no_n.json", &no_n).contains("missing key \"n\""));
    let no_r = good.replace(r#","re_min":-3.0"#, "");
    assert!(bad("no_r.json", &no_r).contains("missing key \"re_min\""));
    let extra = good.replace(r#""dim":1,"#, r#""dim":1,"extra":0,"#);
    assert!(bad("extra.json", &extra).contains("unknown key \"extra\""));
    assert!(bad("json.json", "{\"dim\":").contains("invalid JSON"));
    // The engine refuses these as BadProblem: tau <= 0 and d (n + 1) over the limit.
    let zero_tau = good.replace(r#""tau":1.0"#, r#""tau":0.0"#);
    assert!(bad("tau.json", &zero_tau).contains("delay 0 is 0, it must be positive"));
    let big = good.replace(r#""n":20"#, r#""n":4000"#);
    assert!(bad("big.json", &big).contains("dim * (n + 1) must be at most 4000"));
    let short = good.replace("\"a0\":[0.0]", "\"a0\":[0.0,0.0]");
    assert!(bad("short.json", &short).contains("a0"));
}

#[test]
fn an_unreadable_file_exits_two() {
    let missing = scratch().join("no_such_file.json");
    let o = run(&scratch(), &["dde-roots", &missing.to_string_lossy()]);
    assert_eq!(code(&o), 2);
    assert!(
        stderr(&o).contains("cannot read problem file"),
        "{}",
        stderr(&o)
    );
}

#[test]
fn usage_errors_and_help() {
    let o = run(&scratch(), &["dde-roots"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("dde-roots needs a problem file"));
    let o = run(&scratch(), &["dde-roots", "--help"]);
    assert_eq!(code(&o), 0);
    assert!(stdout(&o).starts_with("usage: pc-cli dde-roots"));
}
