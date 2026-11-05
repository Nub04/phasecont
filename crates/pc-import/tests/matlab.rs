//! The MATLAB subset: handles and function files become `.ode` text that the real parser reads.

mod common;

use common::load;
use pc_import::{FunctionInput, Kind, from_function};

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-13 * (1.0 + b.abs()), "{a} vs {b}");
}

#[test]
fn oscillator_handle_matches_the_closed_form() {
    let params = [("w".to_string(), 2.0), ("c".to_string(), 0.1)];
    let imp = from_function(&FunctionInput {
        text: "@(t,x,p) [x(2); -p(1)^2*x(1) - p(2)*x(2)]",
        params: &params,
        state_names: Some(&s(&["u", "v"])),
        init: &[1.0, 0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    assert_eq!((m.n(), m.n_p()), (2, 2));
    let f = m.f(&[0.3, -0.7], &[2.0, 0.1], 0.0);
    near(f[0], -0.7);
    near(f[1], -4.0 * 0.3 + 0.1 * 0.7);
}

#[test]
fn captured_constants_become_numbers() {
    let caps = [("a".to_string(), vec![1.5]), ("b".to_string(), vec![0.5])];
    let imp = from_function(&FunctionInput {
        text: "@(t,x) [a*x(1) - b*x(1)*x(2); b*x(1)*x(2) - x(2)]",
        captures: &caps,
        state_names: Some(&s(&["prey", "pred"])),
        init: &[1.0, 1.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    let f = m.f(&[2.0, 3.0], &[], 0.0);
    near(f[0], 1.5 * 2.0 - 0.5 * 6.0);
    near(f[1], 0.5 * 6.0 - 3.0);
}

#[test]
fn lorenz_handle_matches_the_hand_written_right_side() {
    let params = [
        ("sigma".to_string(), 10.0),
        ("rho".to_string(), 28.0),
        ("beta".to_string(), 8.0 / 3.0),
    ];
    let imp = from_function(&FunctionInput {
        text: "@(t,x,p) [p(1)*(x(2)-x(1)); x(1)*(p(2)-x(3)) - x(2); x(1)*x(2) - p(3)*x(3)]",
        params: &params,
        state_names: Some(&s(&["x", "y", "z"])),
        init: &[1.0, 1.0, 1.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    let (x, y, z) = (1.2, -0.4, 22.0);
    let f = m.f(&[x, y, z], &[10.0, 28.0, 8.0 / 3.0], 0.0);
    near(f[0], 10.0 * (y - x));
    near(f[1], x * (28.0 - z) - y);
    near(f[2], x * y - 8.0 / 3.0 * z);
}

#[test]
fn time_reaches_the_right_side() {
    let imp = from_function(&FunctionInput {
        text: "@(t,x) -x(1) + sin(t)",
        state_names: Some(&s(&["x"])),
        init: &[0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    near(m.f(&[0.5], &[], 1.3)[0], -0.5 + 1.3f64.sin());
}

#[test]
fn atan2_keeps_the_c_argument_order() {
    let imp = from_function(&FunctionInput {
        text: "@(t,x) atan2(1, 0) + 0*x(1)",
        state_names: Some(&s(&["x"])),
        init: &[0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    near(m.f(&[0.0], &[], 0.0)[0], std::f64::consts::FRAC_PI_2);
}

#[test]
fn logical_and_gives_one() {
    let imp = from_function(&FunctionInput {
        text: "@(t,x) (3 & 2) + 0*x(1)",
        state_names: Some(&s(&["x"])),
        init: &[0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    near(m.f(&[0.0], &[], 0.0)[0], 1.0);
}

#[test]
fn negative_numbers_and_powers_parse() {
    let imp = from_function(&FunctionInput {
        text: "@(t,x) [-3*x(1)^2 + 1e-6 - (-x(1))]",
        state_names: Some(&s(&["x"])),
        init: &[-0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    near(m.f(&[0.5], &[], 0.0)[0], -3.0 * 0.25 + 1e-6 + 0.5);
}

#[test]
fn function_file_with_locals_and_indexed_assignment() {
    let text = "function dydt = f(t, y, p)\n\
                % damped pendulum\n\
                th = y(1);\n\
                om = y(2);\n\
                dydt = zeros(2, 1);\n\
                dydt(1) = om;\n\
                dydt(2) = -sin(th) - p(1)*om;\n\
                end\n";
    let params = [("damp".to_string(), 0.2)];
    let imp = from_function(&FunctionInput {
        text,
        params: &params,
        state_names: Some(&s(&["th", "om"])),
        init: &[0.1, 0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    let f = m.f(&[0.4, 0.9], &[0.2], 0.0);
    near(f[0], 0.9);
    near(f[1], -(0.4f64).sin() - 0.2 * 0.9);
}

#[test]
fn function_file_with_a_vector_result() {
    let text = "function dydt = f(t, y)\n\
                dydt = [y(2); -y(1)];\n\
                end\n";
    let imp = from_function(&FunctionInput {
        text,
        state_names: Some(&s(&["a", "b"])),
        init: &[1.0, 0.0],
        ..Default::default()
    })
    .unwrap();
    let mut m = load(&imp.ode);
    let f = m.f(&[0.25, 0.5], &[], 0.0);
    near(f[0], 0.5);
    near(f[1], -0.25);
}

#[test]
fn missing_initial_value_is_an_error() {
    let err = from_function(&FunctionInput {
        text: "@(t,x) -x(1)",
        state_names: Some(&s(&["x"])),
        init: &[],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::MissingValue, "{err}");
}

#[test]
fn undeclared_name_is_an_error() {
    let err = from_function(&FunctionInput {
        text: "@(t,x) -k*x(1)",
        state_names: Some(&s(&["x"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::Undeclared, "{err}");
    assert!(err.message.contains('k'), "{err}");
}

#[test]
fn state_name_count_must_match() {
    let err = from_function(&FunctionInput {
        text: "@(t,x) [x(2); -x(1)]",
        state_names: Some(&s(&["only_one"])),
        init: &[1.0, 0.0],
        ..Default::default()
    })
    .unwrap_err();
    assert!(
        matches!(err.kind, Kind::Mismatch | Kind::Unsupported),
        "{err}"
    );
}

#[test]
fn control_flow_is_refused_with_a_position() {
    let text = "function dydt = f(t, y)\n\
                dydt = zeros(1, 1);\n\
                for i = 1:3\n\
                dydt(1) = dydt(1) - y(1);\n\
                end\n\
                end\n";
    let err = from_function(&FunctionInput {
        text,
        state_names: Some(&s(&["a"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::Unsupported, "{err}");
    assert!(err.line.is_some(), "{err}");
}

#[test]
fn if_is_refused() {
    let text = "function dydt = f(t, y)\n\
                if y(1) > 0\n\
                dydt = -y(1);\n\
                else\n\
                dydt = y(1);\n\
                end\n\
                end\n";
    let err = from_function(&FunctionInput {
        text,
        state_names: Some(&s(&["a"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::Unsupported, "{err}");
}

#[test]
fn mod_is_refused_by_name() {
    let err = from_function(&FunctionInput {
        text: "@(t,x) mod(x(1), 2)",
        state_names: Some(&s(&["x"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::Unsupported, "{err}");
    assert!(err.message.contains("mod"), "{err}");
}

#[test]
fn unknown_function_is_refused_by_name() {
    let err = from_function(&FunctionInput {
        text: "@(t,x) myhelper(x(1))",
        state_names: Some(&s(&["x"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::Unsupported, "{err}");
    assert!(err.message.contains("myhelper"), "{err}");
}

#[test]
fn blank_separated_vector_elements_are_refused() {
    let err = from_function(&FunctionInput {
        text: "@(t,x) [x(2) -x(1)]",
        state_names: Some(&s(&["a", "b"])),
        init: &[1.0, 0.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::Unsupported, "{err}");
}

#[test]
fn deep_nesting_is_an_error_not_a_crash() {
    let deep = format!("@(t,x) {}x(1){}", "(".repeat(5000), ")".repeat(5000));
    let err = from_function(&FunctionInput {
        text: &deep,
        state_names: Some(&s(&["x"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert!(
        matches!(err.kind, Kind::TooLarge | Kind::Unsupported | Kind::Syntax),
        "{err}"
    );

    let long: String = std::iter::once("x(1)".to_string())
        .chain((0..20_000).map(|_| "+x(1)".to_string()))
        .collect();
    let r = from_function(&FunctionInput {
        text: &format!("@(t,x) {long}"),
        state_names: Some(&s(&["x"])),
        init: &[1.0],
        ..Default::default()
    });
    // Either a clean refusal or a model the parser accepts; never a panic or abort.
    if let Ok(imp) = r {
        let _ = load(&imp.ode);
    }
}

#[test]
fn reserved_state_names_are_renamed_with_a_warning() {
    let imp = from_function(&FunctionInput {
        text: "@(t,x) -x(1)",
        state_names: Some(&s(&["sin"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap();
    assert!(
        imp.warnings.iter().any(|w| w.contains("sin")),
        "{:?}",
        imp.warnings
    );
    load(&imp.ode);
}

#[test]
fn colliding_names_are_an_error() {
    let params = [("K".to_string(), 1.0)];
    let err = from_function(&FunctionInput {
        text: "@(t,x,p) -p(1)*x(1)",
        params: &params,
        state_names: Some(&s(&["k"])),
        init: &[1.0],
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(err.kind, Kind::NameCollision, "{err}");
}

mod no_panic {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn arbitrary_text_never_panics(text in "\\PC{0,200}") {
            let _ = from_function(&FunctionInput {
                text: &text,
                state_names: Some(&["x".to_string()]),
                init: &[1.0],
                ..Default::default()
            });
        }

        #[test]
        fn handle_shaped_text_never_panics(body in "[-a-z0-9+*/^()\\[\\];,. 't&|~<>=]{0,120}") {
            let text = format!("@(t,x,p) {body}");
            let _ = from_function(&FunctionInput {
                text: &text,
                params: &[("a".to_string(), 1.0)],
                state_names: Some(&["x".to_string()]),
                init: &[1.0],
                ..Default::default()
            });
        }
    }
}
