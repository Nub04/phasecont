//! S2: the JIT, the Rust references and the C reference agree bit for bit.

#[test]
fn jit_matches_rust_and_c_bit_for_bit() {
    let json = pckern::s2_jit::run("calls=10000 reps=1 samples=10000").unwrap();
    for key in ["jit_rhs", "jit_jac", "jit_shim_exp", "c_rhs", "c_jac"] {
        assert!(json.contains(&format!("\"{key}\":0e0")), "{key}: {json}");
    }
    assert!(json.contains("\"rhs_checksums_match\":true"), "{json}");
    assert!(json.contains("\"jac_checksums_match\":true"), "{json}");
}

#[test]
fn rejects_zero_calls_and_unknown_keys() {
    assert!(pckern::s2_jit::run("calls=0").is_err());
    assert!(pckern::s2_jit::run("call=10").is_err());
    assert!(pckern::s2_jit::run("calls=1000 reps=1 samples=0").is_err());
}

/// Finds `"name":"<name>"` for each name in turn, each after the one before.
fn assert_names_in_order(text: &str, names: &[&str]) {
    let mut from = 0;
    for name in names {
        let key = format!("\"name\":\"{name}\"");
        let at = text[from..]
            .find(&key)
            .unwrap_or_else(|| panic!("{key} is missing or out of order in {text}"));
        from += at + key.len();
    }
    assert_eq!(text.matches("\"name\":").count(), names.len(), "{text}");
}

#[test]
fn ratios_follow_the_timings_in_a_fixed_order() {
    let json = pckern::s2_jit::run("calls=1000 reps=3 samples=100").unwrap();
    let has_c = !json.contains("\"c_compiler\":\"none\"");
    let (head, ratios) = json.split_once("\"ratios\":[").expect("no ratios key");
    let (_, timings) = head.split_once("\"timings\":[").expect("no timings key");

    let mut names = vec!["jit rhs", "rust rhs via pointer", "rust rhs inlined"];
    if has_c {
        names.push("c rhs");
    }
    names.extend(["jit jac", "rust jac via pointer", "rust jac inlined"]);
    if has_c {
        names.push("c jac");
    }
    assert_names_in_order(timings, &names);

    let mut ratio_names = vec!["rhs jit/pointer"];
    if has_c {
        ratio_names.push("rhs jit/c");
    }
    ratio_names.push("jac jit/pointer");
    if has_c {
        ratio_names.push("jac jit/c");
    }
    assert_names_in_order(ratios, &ratio_names);
    let per_rep: Vec<&str> = ratios
        .split("\"per_rep\":[")
        .skip(1)
        .map(|s| s.split_once(']').expect("unclosed per_rep").0)
        .collect();
    assert_eq!(per_rep.len(), ratio_names.len(), "{ratios}");
    for values in per_rep {
        assert_eq!(values.split(',').count(), 3, "{values}");
    }
}
