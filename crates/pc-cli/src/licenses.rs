//! `pc-cli licenses`: collect the license texts of the crates a package links.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use serde_json::Value;

/// Prefixes of the file names that count as license files, lower case.
const PREFIXES: &[&str] = &["license", "licence", "copying", "notice"];

/// One crate of the closure.
struct Krate {
    name: String,
    version: String,
    license: String,
    dir: PathBuf,
}

/// Writes the license texts of the closure of `package` for `target` to `out`.
pub fn licenses(target: &str, package: &str, out: &Path, overrides: Option<&Path>) -> ExitCode {
    match run(target, package, out, overrides) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("pc-cli: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(target: &str, package: &str, out: &Path, overrides: Option<&Path>) -> Result<(), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .args(["metadata", "--format-version", "1", "--filter-platform"])
        .arg(target)
        .arg("--locked")
        .output()
        .map_err(|e| format!("cannot run cargo: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim_end()
        ));
    }
    let meta: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("cannot read cargo metadata: {e}"))?;
    let krates = closure(&meta, package)?;
    let mut text = String::new();
    let mut missing: Vec<String> = Vec::new();
    for k in &krates {
        let mut files =
            license_files(&k.dir).map_err(|e| format!("cannot read {}: {e}", k.dir.display()))?;
        let mut note = String::new();
        let fallback = overrides.map(|o| o.join(&k.name)).filter(|d| d.is_dir());
        if let (true, Some(dir)) = (files.is_empty(), fallback) {
            files =
                license_files(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
            let source = std::fs::read_to_string(dir.join("SOURCE")).unwrap_or_default();
            note = format!(
                "(license files supplied by PhaseCont, taken from: {})\n",
                source.trim().replace("\r\n", "\n").replace('\n', "; ")
            );
        }
        if files.is_empty() {
            missing.push(format!("{} {}", k.name, k.version));
            continue;
        }
        text.push_str(&format!("== {} {} ({}) ==\n", k.name, k.version, k.license));
        text.push_str(&note);
        for (name, body) in files {
            text.push_str(&format!("--- {name} ---\n{body}"));
            if !body.ends_with('\n') {
                text.push('\n');
            }
            text.push('\n');
        }
    }
    if !missing.is_empty() {
        return Err(format!("no license file in: {}", missing.join(", ")));
    }
    std::fs::write(out, text).map_err(|e| format!("cannot write {}: {e}", out.display()))
}

fn str_of<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// The non-member crates in the normal and build dependency closure of
/// `package`, sorted by name, then version.
fn closure(meta: &Value, package: &str) -> Result<Vec<Krate>, String> {
    let bad = || "unexpected cargo metadata layout".to_string();
    let packages = meta
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(bad)?;
    let nodes = meta
        .get("resolve")
        .and_then(|r| r.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(bad)?;
    let members: BTreeSet<&str> = meta
        .get("workspace_members")
        .and_then(Value::as_array)
        .ok_or_else(bad)?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let by_id: BTreeMap<&str, &Value> = packages
        .iter()
        .filter_map(|p| str_of(p, "id").map(|id| (id, p)))
        .collect();
    let edges: BTreeMap<&str, &Value> = nodes
        .iter()
        .filter_map(|n| str_of(n, "id").map(|id| (id, n)))
        .collect();
    let root = packages
        .iter()
        .find(|p| {
            str_of(p, "name") == Some(package)
                && str_of(p, "id").is_some_and(|id| members.contains(id))
        })
        .and_then(|p| str_of(p, "id"))
        .ok_or_else(|| format!("package {package} is not in the workspace"))?;

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let deps = edges
            .get(id)
            .and_then(|n| n.get("deps"))
            .and_then(Value::as_array)
            .ok_or_else(bad)?;
        for dep in deps {
            let Some(pkg) = str_of(dep, "pkg") else {
                continue;
            };
            // A dependency is kept when at least one edge kind is not `dev`.
            let kinds = dep.get("dep_kinds").and_then(Value::as_array);
            let linked = kinds.is_none_or(|ks| ks.iter().any(|k| str_of(k, "kind") != Some("dev")));
            if linked {
                stack.push(pkg);
            }
        }
    }

    let mut krates = Vec::new();
    for id in seen {
        if members.contains(id) {
            continue;
        }
        let p = by_id.get(id).ok_or_else(bad)?;
        let manifest = str_of(p, "manifest_path").ok_or_else(bad)?;
        krates.push(Krate {
            name: str_of(p, "name").ok_or_else(bad)?.to_string(),
            version: str_of(p, "version").ok_or_else(bad)?.to_string(),
            license: str_of(p, "license")
                .unwrap_or("(none declared)")
                .to_string(),
            dir: Path::new(manifest).parent().ok_or_else(bad)?.to_path_buf(),
        });
    }
    krates.sort_by(|a, b| (&a.name, &a.version).cmp(&(&b.name, &b.version)));
    Ok(krates)
}

/// Name and text of each license file in `dir`, sorted by name. Line endings
/// become LF and bytes that are not UTF-8 are replaced.
fn license_files(dir: &Path) -> std::io::Result<Vec<(String, String)>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let lower = name.to_ascii_lowercase();
        if entry.path().is_file() && PREFIXES.iter().any(|p| lower.starts_with(p)) {
            let bytes = std::fs::read(entry.path())?;
            let body = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
            files.push((name, body));
        }
    }
    files.sort();
    Ok(files)
}
