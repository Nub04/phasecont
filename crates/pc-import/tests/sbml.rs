//! SBML and MathML: files become `.ode` text that the real parser reads. The models are our own.

mod common;

use common::load;
use pc_import::{Kind, from_sbml, sbml};
use proptest::prelude::*;

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-13 * (1.0 + b.abs()), "{a} vs {b}");
}

const MML: &str = "http://www.w3.org/1998/Math/MathML";

fn math(inner: &str) -> String {
    format!("<math xmlns=\"{MML}\">{inner}</math>")
}

fn doc(level: u32, body: &str) -> String {
    let ns = if level == 3 {
        "http://www.sbml.org/sbml/level3/version1/core"
    } else {
        "http://www.sbml.org/sbml/level2/version4"
    };
    let version = if level == 3 { 1 } else { 4 };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <sbml xmlns=\"{ns}\" level=\"{level}\" version=\"{version}\">\
         <model id=\"m\" name=\"test model\">{body}</model></sbml>"
    )
}

fn import(level: u32, body: &str) -> pc_import::Imported {
    let text = doc(level, body);
    from_sbml(text.as_bytes()).unwrap_or_else(|e| panic!("import failed: {e:?}\n{text}"))
}

fn fail(level: u32, body: &str) -> pc_import::ImportError {
    let text = doc(level, body);
    match from_sbml(text.as_bytes()) {
        Ok(i) => panic!("expected an error, got:\n{}", i.ode),
        Err(e) => e,
    }
}

const COMP: &str =
    "<listOfCompartments><compartment id=\"C\" size=\"2\" constant=\"true\"/></listOfCompartments>";

/// `S -> ` with rate `k*S` in a compartment of size 2.
fn decay(species_attrs: &str) -> String {
    format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" {species_attrs}/></listOfSpecies>\
         <listOfParameters><parameter id=\"k\" value=\"0.5\" constant=\"true\"/></listOfParameters>\
         <listOfReactions><reaction id=\"R\" reversible=\"false\">\
         <listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
         <kineticLaw>{}</kineticLaw></reaction></listOfReactions>",
        math("<apply><times/><ci>k</ci><ci>S</ci></apply>")
    )
}

#[test]
fn concentration_decay_divides_by_the_compartment() {
    let imp = import(3, &decay("initialConcentration=\"4\""));
    let mut m = load(&imp.ode);
    assert_eq!((m.n(), m.n_p()), (1, 2));
    // par order: C, k
    near(m.f(&[3.0], &[2.0, 0.5], 0.0)[0], -0.5 * 3.0 / 2.0);
}

#[test]
fn amount_species_skips_the_division_and_converts_the_start_value() {
    let imp = import(
        3,
        &decay("initialAmount=\"4\" hasOnlySubstanceUnits=\"true\""),
    );
    let mut m = load(&imp.ode);
    near(m.f(&[3.0], &[2.0, 0.5], 0.0)[0], -0.5 * 3.0);
    assert!(imp.ode.contains("S=4"), "{}", imp.ode);

    // An amount given for a concentration species is divided by the compartment size (4 / 2).
    let imp = import(3, &decay("initialAmount=\"4\""));
    assert!(imp.ode.contains("S=2"), "{}", imp.ode);
    // A concentration given for an amount species is multiplied by it (4 * 2).
    let imp = import(
        3,
        &decay("initialConcentration=\"4\" hasOnlySubstanceUnits=\"true\""),
    );
    assert!(imp.ode.contains("S=8"), "{}", imp.ode);
}

#[test]
fn level_two_reactions_default_to_stoichiometry_one() {
    let body = decay("initialConcentration=\"1\"").replace(" stoichiometry=\"1\"", "");
    let imp = import(2, &body);
    let mut m = load(&imp.ode);
    near(m.f(&[3.0], &[2.0, 0.5], 0.0)[0], -0.75);
    // Level 3 has no default.
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::MissingValue, "{e:?}");
}

#[test]
fn michaelis_menten_pair_matches_the_rate_law() {
    let body = format!(
        "<listOfCompartments><compartment id=\"V\" size=\"1\"/></listOfCompartments>\
         <listOfSpecies>\
         <species id=\"S\" compartment=\"V\" initialConcentration=\"5\"/>\
         <species id=\"P\" compartment=\"V\" initialConcentration=\"0\"/></listOfSpecies>\
         <listOfParameters><parameter id=\"vmax\" value=\"2\"/><parameter id=\"km\" value=\"0.7\"/></listOfParameters>\
         <listOfReactions><reaction id=\"conv\">\
         <listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
         <listOfProducts><speciesReference species=\"P\" stoichiometry=\"1\"/></listOfProducts>\
         <kineticLaw>{}</kineticLaw></reaction></listOfReactions>",
        math(
            "<apply><divide/><apply><times/><ci>vmax</ci><ci>S</ci></apply>\
              <apply><plus/><ci>km</ci><ci>S</ci></apply></apply>"
        )
    );
    let imp = import(3, &body);
    let mut m = load(&imp.ode);
    assert_eq!(m.n(), 2);
    // par order: V, vmax, km; states in species order S, P.
    let (s, rate) = (1.3, 2.0 * 1.3 / (0.7 + 1.3));
    let f = m.f(&[s, 0.4], &[1.0, 2.0, 0.7], 0.0);
    near(f[0], -rate);
    near(f[1], rate);
}

#[test]
fn stoichiometry_scales_the_term() {
    let body = format!(
        "<listOfCompartments><compartment id=\"V\" size=\"1\"/></listOfCompartments>\
         <listOfSpecies>\
         <species id=\"A\" compartment=\"V\" initialConcentration=\"1\"/>\
         <species id=\"B\" compartment=\"V\" initialConcentration=\"0\"/></listOfSpecies>\
         <listOfParameters><parameter id=\"k\" value=\"3\"/></listOfParameters>\
         <listOfReactions><reaction id=\"dim\">\
         <listOfReactants><speciesReference species=\"A\" stoichiometry=\"2\"/></listOfReactants>\
         <listOfProducts><speciesReference species=\"B\" stoichiometry=\"1\"/></listOfProducts>\
         <kineticLaw>{}</kineticLaw></reaction></listOfReactions>",
        math("<apply><times/><ci>k</ci><ci>A</ci><ci>A</ci></apply>")
    );
    let imp = import(3, &body);
    let mut m = load(&imp.ode);
    let f = m.f(&[2.0, 0.0], &[1.0, 3.0], 0.0);
    near(f[0], -2.0 * 12.0);
    near(f[1], 12.0);
}

#[test]
fn local_parameters_get_unique_names() {
    let body = format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>\
         <listOfParameters><parameter id=\"R_k\" value=\"99\"/></listOfParameters>\
         <listOfReactions><reaction id=\"R\">\
         <listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
         <kineticLaw>{}<listOfLocalParameters>\
         <localParameter id=\"k\" value=\"0.25\"/><localParameter id=\"unused\" value=\"7\"/>\
         </listOfLocalParameters></kineticLaw></reaction></listOfReactions>",
        math("<apply><times/><ci>k</ci><ci>S</ci></apply>")
    );
    let imp = import(3, &body);
    // The global `R_k` is not read by anything, so it has no effect, and the local gets its own
    // name. The local that nothing reads is left out.
    assert!(imp.ode.contains("R_k_=0.25"), "{}", imp.ode);
    assert!(!imp.ode.contains("unused"), "{}", imp.ode);
    let mut m = load(&imp.ode);
    // par order: C, R_k, R_k_
    near(m.f(&[3.0], &[2.0, 99.0, 0.25], 0.0)[0], -0.25 * 3.0 / 2.0);
}

#[test]
fn level_two_local_parameters_work() {
    let body = format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>\
         <listOfReactions><reaction id=\"R\">\
         <listOfReactants><speciesReference species=\"S\"/></listOfReactants>\
         <kineticLaw>{}<listOfParameters><parameter id=\"k\" value=\"0.25\"/></listOfParameters>\
         </kineticLaw></reaction></listOfReactions>",
        math("<apply><times/><ci>k</ci><ci>S</ci></apply>")
    );
    let imp = import(2, &body);
    let mut m = load(&imp.ode);
    near(m.f(&[3.0], &[2.0, 0.25], 0.0)[0], -0.25 * 3.0 / 2.0);
}

#[test]
fn assignment_rules_are_ordered_by_what_they_read() {
    // `a` reads `b`, `b` reads `base`, and the file lists `a` first.
    let body = format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>\
         <listOfParameters>\
         <parameter id=\"a\" constant=\"false\"/><parameter id=\"b\" constant=\"false\"/>\
         <parameter id=\"base\" value=\"3\"/></listOfParameters>\
         <listOfRules>\
         <assignmentRule variable=\"a\">{}</assignmentRule>\
         <assignmentRule variable=\"b\">{}</assignmentRule></listOfRules>\
         <listOfReactions><reaction id=\"R\">\
         <listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
         <kineticLaw>{}</kineticLaw></reaction></listOfReactions>",
        math("<apply><times/><cn>2</cn><ci>b</ci></apply>"),
        math("<apply><plus/><ci>base</ci><cn>1</cn></apply>"),
        math("<apply><times/><ci>a</ci><ci>S</ci></apply>")
    );
    let imp = import(3, &body);
    let mut m = load(&imp.ode);
    // a = 2 * (c + 1) = 8; R = a * S; S' = -R / C.
    near(m.f(&[1.5], &[2.0, 3.0], 0.0)[0], -8.0 * 1.5 / 2.0);
}

#[test]
fn rate_rule_parameter_becomes_a_state() {
    let body = format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>\
         <listOfParameters><parameter id=\"g\" value=\"0.5\" constant=\"false\"/></listOfParameters>\
         <listOfRules><rateRule variable=\"g\">{}</rateRule></listOfRules>\
         <listOfReactions><reaction id=\"R\">\
         <listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
         <kineticLaw>{}</kineticLaw></reaction></listOfReactions>",
        math("<apply><minus/><ci>S</ci></apply>"),
        math("<apply><times/><ci>g</ci><ci>S</ci></apply>")
    );
    let imp = import(3, &body);
    let mut m = load(&imp.ode);
    assert_eq!(m.n(), 2);
    // States: S then g.
    let f = m.f(&[1.5, 0.5], &[2.0], 0.0);
    near(f[0], -0.5 * 1.5 / 2.0);
    near(f[1], -1.5);
}

#[test]
fn rate_rule_species_needs_an_initial_value() {
    let body = format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" boundaryCondition=\"true\"/></listOfSpecies>\
         <listOfRules><rateRule variable=\"S\">{}</rateRule></listOfRules>",
        math("<cn>1</cn>")
    );
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::MissingValue, "{e:?}");
}

#[test]
fn time_symbol_reads_t() {
    let body = format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\" boundaryCondition=\"true\"/></listOfSpecies>\
         <listOfRules><rateRule variable=\"S\">{}</rateRule></listOfRules>",
        math(
            "<apply><times/><cn>2</cn><csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\">time</csymbol></apply>"
        )
    );
    let imp = import(3, &body);
    let mut m = load(&imp.ode);
    near(m.f(&[1.0], &[2.0], 1.5)[0], 3.0);
}

fn piecewise_model(with_otherwise: bool) -> String {
    let otherwise = if with_otherwise {
        "<otherwise><cn>-1</cn></otherwise>"
    } else {
        ""
    };
    format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\" boundaryCondition=\"true\"/></listOfSpecies>\
         <listOfRules><rateRule variable=\"S\">{}</rateRule></listOfRules>",
        math(&format!(
            "<piecewise><piece><cn>5</cn><apply><gt/><ci>S</ci><cn>0</cn></apply></piece>{otherwise}</piecewise>"
        ))
    )
}

#[test]
fn piecewise_with_otherwise_selects_by_condition() {
    let imp = import(3, &piecewise_model(true));
    let mut m = load(&imp.ode);
    near(m.f(&[1.0], &[2.0], 0.0)[0], 5.0);
    near(m.f(&[-1.0], &[2.0], 0.0)[0], -1.0);
}

#[test]
fn piecewise_without_otherwise_is_refused() {
    let e = fail(3, &piecewise_model(false));
    assert_eq!(e.kind, Kind::Unsupported, "{e:?}");
    assert!(e.message.contains("otherwise"), "{}", e.message);
}

fn rate_rule_with(expr: &str) -> String {
    format!(
        "{COMP}\
         <listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\" boundaryCondition=\"true\"/></listOfSpecies>\
         <listOfRules><rateRule variable=\"S\">{}</rateRule></listOfRules>",
        math(expr)
    )
}

#[test]
fn mathml_functions_match_their_values() {
    let cases: &[(&str, f64)] = &[
        ("<apply><root/><cn>9</cn></apply>", 3.0),
        (
            "<apply><root/><degree><cn>3</cn></degree><cn>27</cn></apply>",
            3.0,
        ),
        ("<apply><log/><cn>100</cn></apply>", 2.0),
        (
            "<apply><log/><logbase><cn>2</cn></logbase><cn>8</cn></apply>",
            3.0,
        ),
        ("<apply><power/><cn>2</cn><cn>10</cn></apply>", 1024.0),
        ("<apply><floor/><cn>2.7</cn></apply>", 2.0),
        ("<apply><ceiling/><cn>2.1</cn></apply>", 3.0),
        ("<apply><sec/><cn>0</cn></apply>", 1.0),
        ("<apply><neq/><cn>1</cn><cn>2</cn></apply>", 1.0),
        ("<apply><and/><cn>1</cn><cn>1</cn><cn>0</cn></apply>", 0.0),
        ("<apply><minus/><cn>7</cn><cn>2</cn></apply>", 5.0),
        ("<cn type=\"e-notation\">1.5<sep/>2</cn>", 150.0),
        ("<cn type=\"rational\">1<sep/>4</cn>", 0.25),
        ("<apply><plus/></apply>", 0.0),
        ("<apply><times/></apply>", 1.0),
        ("<pi/>", std::f64::consts::PI),
    ];
    for (expr, want) in cases {
        let imp = import(3, &rate_rule_with(expr));
        let mut m = load(&imp.ode);
        let got = m.f(&[1.0], &[2.0], 0.0)[0];
        assert!(
            (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "{expr}: {got} vs {want}\n{}",
            imp.ode
        );
    }
}

#[test]
fn ceiling_warns_that_xppaut_rejects_it() {
    let imp = import(3, &rate_rule_with("<apply><ceiling/><cn>2.1</cn></apply>"));
    assert_eq!(
        imp.warnings.iter().filter(|w| w.contains("ceil")).count(),
        1,
        "{:?}",
        imp.warnings
    );
}

#[test]
fn unsupported_operators_are_named() {
    for op in ["factorial", "xor", "quotient", "rem", "arcsinh", "arcsec"] {
        let expr = format!("<apply><{op}/><cn>3</cn><cn>2</cn></apply>");
        let e = fail(3, &rate_rule_with(&expr));
        assert_eq!(e.kind, Kind::Unsupported, "{op}: {e:?}");
        assert!(e.message.contains(op), "{op}: {}", e.message);
    }
}

#[test]
fn chained_comparisons_are_refused() {
    let e = fail(
        3,
        &rate_rule_with("<apply><lt/><cn>1</cn><cn>2</cn><cn>3</cn></apply>"),
    );
    assert_eq!(e.kind, Kind::Unsupported, "{e:?}");
}

#[test]
fn delay_and_rate_of_symbols_are_refused() {
    let e = fail(
        3,
        &rate_rule_with(
            "<apply><csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/delay\">delay</csymbol><ci>S</ci><cn>1</cn></apply>",
        ),
    );
    assert_eq!(e.kind, Kind::Unsupported, "{e:?}");
}

#[test]
fn events_are_refused_by_name_with_a_position() {
    let body = format!(
        "{}<listOfEvents><event id=\"e1\"/></listOfEvents>",
        decay("initialConcentration=\"1\"")
    );
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Unsupported, "{e:?}");
    assert!(e.message.contains("event"), "{}", e.message);
    assert!(e.line.is_some(), "{e:?}");
}

#[test]
fn other_unsupported_features_are_refused_by_name() {
    let base = decay("initialConcentration=\"1\"");
    let cases: &[(&str, &str)] = &[
        (
            "<listOfFunctionDefinitions><functionDefinition id=\"f\"/></listOfFunctionDefinitions>",
            "functionDefinition",
        ),
        (
            "<listOfInitialAssignments><initialAssignment symbol=\"k\"/></listOfInitialAssignments>",
            "initialAssignment",
        ),
        (
            "<listOfRules><algebraicRule/></listOfRules>",
            "algebraicRule",
        ),
    ];
    for (extra, name) in cases {
        let e = fail(3, &format!("{base}{extra}"));
        assert_eq!(e.kind, Kind::Unsupported, "{name}: {e:?}");
        assert!(e.message.contains(name), "{name}: {}", e.message);
    }
}

#[test]
fn compartment_rules_are_refused() {
    let body = format!(
        "{}<listOfRules><rateRule variable=\"C\">{}</rateRule></listOfRules>",
        decay("initialConcentration=\"1\""),
        math("<cn>1</cn>")
    );
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Unsupported, "{e:?}");
}

#[test]
fn level_one_and_unknown_levels_are_refused() {
    for level in ["1", "4"] {
        let text =
            format!("<sbml xmlns=\"x\" level=\"{level}\" version=\"1\"><model id=\"m\"/></sbml>");
        let e = from_sbml(text.as_bytes()).unwrap_err();
        assert_eq!(e.kind, Kind::Unsupported, "level {level}: {e:?}");
    }
    let e = from_sbml(b"<sbml version=\"1\"><model id=\"m\"/></sbml>").unwrap_err();
    assert_eq!(e.kind, Kind::Syntax, "{e:?}");
}

#[test]
fn required_packages_are_refused() {
    let text = "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" \
                xmlns:comp=\"http://www.sbml.org/sbml/level3/version1/comp/version1\" \
                level=\"3\" version=\"1\" comp:required=\"true\"><model id=\"m\"/></sbml>";
    let e = from_sbml(text.as_bytes()).unwrap_err();
    assert_eq!(e.kind, Kind::Unsupported, "{e:?}");
}

#[test]
fn missing_values_raise_named_errors() {
    // A species in a reaction with no start value.
    let e = fail(3, &decay(""));
    assert_eq!(e.kind, Kind::MissingValue, "{e:?}");
    assert!(e.message.contains('S'), "{}", e.message);

    // A parameter the kinetic law reads, with no value.
    let body = decay("initialConcentration=\"1\"").replace(" value=\"0.5\"", "");
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::MissingValue, "{e:?}");
    assert!(e.message.contains('k'), "{}", e.message);

    // A concentration given for an amount species, in a compartment with no size.
    let body = decay("initialConcentration=\"1\" hasOnlySubstanceUnits=\"true\"")
        .replace(" size=\"2\"", "");
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::MissingValue, "{e:?}");

    // A reaction without a kinetic law.
    let body = decay("initialConcentration=\"1\"");
    let start = body.find("<kineticLaw>").unwrap();
    let end = body.find("</kineticLaw>").unwrap() + "</kineticLaw>".len();
    let e = fail(3, &format!("{}{}", &body[..start], &body[end..]));
    assert_eq!(e.kind, Kind::MissingValue, "{e:?}");
}

#[test]
fn an_unused_parameter_without_a_value_is_dropped_with_a_warning() {
    let body = format!(
        "{}<listOfParameters><parameter id=\"spare\"/></listOfParameters>",
        decay("initialConcentration=\"1\"")
    );
    let imp = import(3, &body);
    assert!(!imp.ode.contains("spare"), "{}", imp.ode);
    assert!(
        imp.warnings.iter().any(|w| w.contains("spare")),
        "{:?}",
        imp.warnings
    );
}

#[test]
fn duplicate_ids_are_a_name_collision() {
    let body =
        decay("initialConcentration=\"1\"").replace("<parameter id=\"k\"", "<parameter id=\"S\"");
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::NameCollision, "{e:?}");
    assert!(e.line.is_some(), "{e:?}");
}

#[test]
fn names_that_differ_only_in_case_collide() {
    let body =
        decay("initialConcentration=\"1\"").replace("<parameter id=\"k\"", "<parameter id=\"s\"");
    let e = fail(3, &body);
    assert!(matches!(e.kind, Kind::NameCollision), "{e:?}");
}

#[test]
fn undeclared_species_and_compartments_are_named() {
    let body = decay("initialConcentration=\"1\"").replace("species=\"S\"", "species=\"Ghost\"");
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Undeclared, "{e:?}");
    assert!(e.message.contains("Ghost"), "{}", e.message);

    let body =
        decay("initialConcentration=\"1\"").replace("compartment=\"C\"", "compartment=\"Nowhere\"");
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Undeclared, "{e:?}");

    // A kinetic law that reads a name nothing defines.
    let body = decay("initialConcentration=\"1\"").replace("<ci>k</ci>", "<ci>nope</ci>");
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Undeclared, "{e:?}");
    assert!(e.message.contains("nope"), "{}", e.message);
}

#[test]
fn a_model_without_dynamics_is_refused() {
    let body = format!(
        "{COMP}<listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>"
    );
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::NoDynamics, "{e:?}");
}

#[test]
fn conflicting_species_attributes_are_refused() {
    let e = fail(3, &decay("initialConcentration=\"1\" initialAmount=\"2\""));
    assert_eq!(e.kind, Kind::Mismatch, "{e:?}");

    // Constant, not a boundary species, and in a reaction.
    let e = fail(3, &decay("initialConcentration=\"1\" constant=\"true\""));
    assert_eq!(e.kind, Kind::Mismatch, "{e:?}");
}

#[test]
fn boundary_species_stay_constant_inside_reactions() {
    // The only species is a boundary species, so no state changes and there is no dynamics.
    let e = fail(
        3,
        &decay("initialConcentration=\"1\" boundaryCondition=\"true\""),
    );
    assert_eq!(e.kind, Kind::NoDynamics, "{e:?}");

    // Next to a species that does change, the boundary species is a constant and its rate is
    // left out of its own equation.
    let body = decay("initialConcentration=\"1\" boundaryCondition=\"true\"").replace(
        "</listOfSpecies>",
        "<species id=\"T\" compartment=\"C\" initialConcentration=\"0\"/></listOfSpecies>",
    );
    let body = body.replace(
        "</listOfReactants>",
        "</listOfReactants><listOfProducts><speciesReference species=\"T\" stoichiometry=\"1\"/></listOfProducts>",
    );
    let imp = import(3, &body);
    let mut m = load(&imp.ode);
    assert_eq!(m.n(), 1);
    // par order: C, S, k; the one state is T.
    near(m.f(&[0.0], &[2.0, 3.0, 0.5], 0.0)[0], 0.5 * 3.0 / 2.0);
}

#[test]
fn rule_cycles_are_refused() {
    let body = format!(
        "{}<listOfParameters><parameter id=\"p\" constant=\"false\"/><parameter id=\"q\" constant=\"false\"/></listOfParameters>\
         <listOfRules><assignmentRule variable=\"p\">{}</assignmentRule>\
         <assignmentRule variable=\"q\">{}</assignmentRule></listOfRules>",
        decay("initialConcentration=\"1\""),
        math("<apply><plus/><ci>q</ci><cn>1</cn></apply>"),
        math("<apply><plus/><ci>p</ci><cn>1</cn></apply>")
    );
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Mismatch, "{e:?}");
    assert!(
        e.message.contains('p') && e.message.contains('q'),
        "{}",
        e.message
    );
}

#[test]
fn two_rules_on_one_variable_are_refused() {
    let body = format!(
        "{}<listOfParameters><parameter id=\"p\" constant=\"false\" value=\"1\"/></listOfParameters>\
         <listOfRules><rateRule variable=\"p\">{}</rateRule>\
         <assignmentRule variable=\"p\">{}</assignmentRule></listOfRules>",
        decay("initialConcentration=\"1\""),
        math("<cn>1</cn>"),
        math("<cn>2</cn>")
    );
    let e = fail(3, &body);
    assert_eq!(e.kind, Kind::Mismatch, "{e:?}");
}

#[test]
fn titles_survive_as_a_comment() {
    let imp = import(3, &decay("initialConcentration=\"1\""));
    assert!(imp.ode.starts_with("# test model\n"), "{}", imp.ode);
}

#[test]
fn error_ids_follow_the_import_namespace() {
    let e = fail(3, &decay(""));
    assert_eq!(e.error_id(), "phasecont:import:missingValue");
}

// Hostile input.

#[test]
fn a_dtd_is_refused() {
    let text = "<?xml version=\"1.0\"?>\
        <!DOCTYPE lolz [<!ENTITY lol \"lol\"><!ENTITY lol2 \"&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;\">]>\
        <sbml level=\"3\" version=\"1\"><model id=\"&lol2;\"/></sbml>";
    let e = from_sbml(text.as_bytes()).unwrap_err();
    assert_eq!(e.kind, Kind::Xml, "{e:?}");
    assert!(e.message.contains("DOCTYPE"), "{}", e.message);
}

#[test]
fn bad_encodings_and_syntax_are_xml_errors() {
    let e = from_sbml(&[0xff, 0xfe, b'<', b's']).unwrap_err();
    assert_eq!(e.kind, Kind::Xml, "{e:?}");
    let e = from_sbml(b"<sbml level=\"3\"").unwrap_err();
    assert_eq!(e.kind, Kind::Xml, "{e:?}");
    assert!(e.line.is_some());
    let e = from_sbml(b"").unwrap_err();
    assert_eq!(e.kind, Kind::Xml, "{e:?}");
}

#[test]
fn a_byte_order_mark_is_accepted() {
    let mut bytes = vec![0xef, 0xbb, 0xbf];
    bytes.extend_from_slice(doc(3, &decay("initialConcentration=\"1\"")).as_bytes());
    from_sbml(&bytes).unwrap();
}

/// MATLAB decodes a file with a non-UTF-8 `encoding="..."` declaration and hands over UTF-8, so
/// the reader must not trust the declaration.
#[test]
fn a_declared_latin1_encoding_is_ignored_for_utf8_text() {
    let text = doc(3, &decay("initialConcentration=\"1\"")).replace("UTF-8", "ISO-8859-1");
    assert!(text.contains("ISO-8859-1"));
    from_sbml(text.as_bytes()).unwrap();
}

#[test]
fn oversize_files_are_refused() {
    let big = vec![b' '; sbml::MAX_XML + 1];
    let e = from_sbml(&big).unwrap_err();
    assert_eq!(e.kind, Kind::TooLarge, "{e:?}");
}

#[test]
fn deep_mathml_nesting_is_an_error_not_an_overflow() {
    let depth = 5000;
    let mut s = String::new();
    for _ in 0..depth {
        s.push_str("<apply><plus/><cn>1</cn>");
    }
    s.push_str("<ci>S</ci>");
    for _ in 0..depth {
        s.push_str("</apply>");
    }
    let e = fail(3, &rate_rule_with(&s));
    assert!(
        matches!(e.kind, Kind::TooLarge | Kind::Syntax | Kind::Xml),
        "{e:?}"
    );
}

#[test]
fn deep_xml_nesting_is_an_error_not_an_overflow() {
    let depth = 100_000;
    let text = format!("{}{}", "<a>".repeat(depth), "</a>".repeat(depth));
    let e = from_sbml(text.as_bytes()).unwrap_err();
    // roxmltree stops at its own depth cap or the root is not `sbml`; either way it is an error.
    assert!(
        matches!(e.kind, Kind::Xml | Kind::Syntax | Kind::TooLarge),
        "{e:?}"
    );
}

#[test]
fn a_huge_sum_builds_a_balanced_tree() {
    let n = 20_000;
    let mut s = String::from("<apply><plus/>");
    for i in 0..n {
        s.push_str(&format!("<cn>{}</cn>", i % 7));
    }
    s.push_str("</apply>");
    let imp = import(3, &rate_rule_with(&s));
    let mut m = load(&imp.ode);
    let want: f64 = (0..n).map(|i| (i % 7) as f64).sum();
    near(m.f(&[1.0], &[2.0], 0.0)[0], want);
}

#[test]
fn a_long_chain_of_rules_takes_linear_time() {
    let n = 20_000;
    let mut params = String::from("<listOfParameters><parameter id=\"p0\" value=\"1\"/>");
    let mut rules = String::from("<listOfRules>");
    for i in 1..=n {
        params.push_str(&format!("<parameter id=\"p{i}\" constant=\"false\"/>"));
        rules.push_str(&format!(
            "<assignmentRule variable=\"p{i}\">{}</assignmentRule>",
            math(&format!(
                "<apply><plus/><ci>p{}</ci><cn>1</cn></apply>",
                i - 1
            ))
        ));
    }
    params.push_str("</listOfParameters>");
    rules.push_str("</listOfRules>");
    let body = format!(
        "{COMP}<listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>\
         {params}{rules}\
         <listOfReactions><reaction id=\"R\"><listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
         <kineticLaw>{}</kineticLaw></reaction></listOfReactions>",
        math(&format!("<ci>p{n}</ci>"))
    );
    let start = std::time::Instant::now();
    let imp = import(3, &body);
    assert!(
        start.elapsed() < std::time::Duration::from_secs(10),
        "took {:?}",
        start.elapsed()
    );
    assert!(imp.ode.len() < pc_import::emit::MAX_OUTPUT);
}

fn valid_doc() -> Vec<u8> {
    doc(3, &decay("initialConcentration=\"1\"")).into_bytes()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = from_sbml(&bytes);
    }

    #[test]
    fn mutated_documents_never_panic_and_accepted_ones_load(
        edits in proptest::collection::vec((0usize..4000, any::<u8>()), 0..6),
        cut in proptest::option::of(0usize..4000),
    ) {
        let mut bytes = valid_doc();
        for (at, b) in edits {
            let i = at % bytes.len();
            bytes[i] = b;
        }
        if let Some(c) = cut {
            bytes.truncate(c % (bytes.len() + 1));
        }
        if let Ok(imp) = from_sbml(&bytes) {
            // Whatever the importer accepts, the real parser must accept too.
            let _ = load(&imp.ode);
        }
    }

    #[test]
    fn mutated_text_tokens_never_panic(
        swaps in proptest::collection::vec((0usize..12, 0usize..12), 1..4),
    ) {
        const TOKENS: [&str; 12] = [
            "<ci>k</ci>", "<cn>3</cn>", "<apply>", "</apply>", "<times/>", "<piecewise>",
            "<otherwise>", "<csymbol>time</csymbol>", "<ci>S</ci>", "<minus/>", "<power/>", "<cn>1e999</cn>",
        ];
        let mut text = String::new();
        for (a, b) in swaps {
            text.push_str(TOKENS[a]);
            text.push_str(TOKENS[b]);
        }
        let body = format!(
            "{COMP}<listOfSpecies><species id=\"S\" compartment=\"C\" initialConcentration=\"1\"/></listOfSpecies>\
             <listOfParameters><parameter id=\"k\" value=\"1\"/></listOfParameters>\
             <listOfReactions><reaction id=\"R\"><listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\"/></listOfReactants>\
             <kineticLaw><math xmlns=\"{MML}\">{text}</math></kineticLaw></reaction></listOfReactions>"
        );
        if let Ok(imp) = from_sbml(doc(3, &body).as_bytes()) {
            let _ = load(&imp.ode);
        }
    }
}
