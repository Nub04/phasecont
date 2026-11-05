use super::*;

#[test]
fn a_value_passes_through() {
    assert_eq!(guarded(|| 7), Ok(7));
}

#[test]
fn a_str_panic_becomes_its_message() {
    let r: Result<(), String> = guarded(|| panic!("boom"));
    assert_eq!(r, Err("boom".to_string()));
}

#[test]
fn a_formatted_panic_becomes_its_message() {
    let n = 3;
    let r: Result<(), String> = guarded(|| panic!("bad index {n}"));
    assert_eq!(r, Err("bad index 3".to_string()));
}

#[test]
fn a_payload_that_is_not_text_still_gives_an_error() {
    let r: Result<(), String> = guarded(|| std::panic::panic_any(42_u8));
    assert!(r.unwrap_err().contains("not text"));
}

#[test]
fn the_caller_can_keep_going_after_a_panic() {
    let mut count = 0;
    for i in 0..3 {
        let _ = guarded(|| {
            if i == 1 {
                panic!("middle");
            }
        });
        count += 1;
    }
    assert_eq!(count, 3);
}
