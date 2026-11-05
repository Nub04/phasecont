//! Golden outputs. The `.tex` files in `golden/` were written by hand from the format in
//! the SPEC, so they check the renderer and are not a copy of its output.

use pc_latex::{Kind, render, render_value};
use std::fs;
use std::path::PathBuf;

fn golden(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn figure_with_stable_and_unstable_series_and_marks() {
    let out = render(Kind::Pgfplots, &golden("fig_branch.json")).unwrap();
    assert_eq!(out.tex, golden("fig_branch.tex"));
    assert!(out.data_files.is_empty());
}

#[test]
fn figure_with_external_data() {
    let out = render(Kind::Pgfplots, &golden("fig_external.json")).unwrap();
    assert_eq!(out.tex, golden("fig_external.tex"));
    let names: Vec<&str> = out.data_files.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["fold-curve-s1.dat", "fold-curve-s2.dat"]);
    for (name, content) in &out.data_files {
        assert_eq!(content, &golden(&format!("fig_external.{name}")), "{name}");
    }
}

#[test]
fn table_of_special_points() {
    let out = render(Kind::Table, &golden("table_special.json")).unwrap();
    assert_eq!(out.tex, golden("table_special.tex"));
    assert!(out.data_files.is_empty());
}

/// `standalone: true` wraps the same fragment in a document.
#[test]
fn standalone_wraps_the_fragment() {
    for (kind, name, packages) in [
        (
            Kind::Pgfplots,
            "fig_branch",
            "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n",
        ),
        (Kind::Table, "table_special", "\\usepackage{booktabs}\n"),
    ] {
        let mut v: serde_json::Value =
            serde_json::from_str(&golden(&format!("{name}.json"))).unwrap();
        v["standalone"] = serde_json::Value::Bool(true);
        let out = render_value(kind, &v).unwrap();
        let expected = format!(
            "\\documentclass[border=5pt]{{standalone}}\n{packages}\\begin{{document}}\n{}\\end{{document}}\n",
            golden(&format!("{name}.tex"))
        );
        assert_eq!(out.tex, expected, "{name}");
    }
}

#[test]
fn json_form_for_the_c_abi() {
    let out = render(Kind::Pgfplots, &golden("fig_external.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&out.to_json()).unwrap();
    assert_eq!(v["tex"], golden("fig_external.tex"));
    let files = v["data_files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0]["name"], "fold-curve-s1.dat");
    assert_eq!(
        files[0]["content"],
        golden("fig_external.fold-curve-s1.dat")
    );
}

#[test]
fn kind_names_round_trip() {
    for k in [Kind::Pgfplots, Kind::Table] {
        assert_eq!(Kind::from_name(k.name()), Some(k));
    }
    assert_eq!(Kind::from_name("figure"), None);
}

#[test]
fn inline_is_the_default_and_can_be_named() {
    let mut v: serde_json::Value = serde_json::from_str(&golden("fig_branch.json")).unwrap();
    let plain = render_value(Kind::Pgfplots, &v).unwrap();
    v["data"] = "inline".into();
    assert_eq!(render_value(Kind::Pgfplots, &v).unwrap(), plain);
}
