use proptest::prelude::*;

use super::*;

/// Text built from pieces that stress the conversion: ASCII, a two-byte
/// character, a three-byte character, an astral character (two UTF-16 units),
/// and every line terminator.
fn text() -> impl Strategy<Value = String> {
    let piece = prop_oneof![
        Just("a"),
        Just(" "),
        Just("x'=1"),
        Just("\u{e9}"),
        Just("\u{20ac}"),
        Just("\u{1f600}"),
        Just("\n"),
        Just("\r\n"),
        Just("\r"),
    ];
    proptest::collection::vec(piece, 0..40).prop_map(|p| p.concat())
}

/// True when `offset` is between the `\r` and the `\n` of a CRLF.
fn inside_crlf(text: &str, offset: usize) -> bool {
    offset > 0
        && text.as_bytes()[offset - 1] == b'\r'
        && text.as_bytes().get(offset) == Some(&b'\n')
}

proptest! {
    #[test]
    fn offset_of_position_round_trips(t in text()) {
        let index = LineIndex::new(&t);
        for o in 0..=t.len() {
            if !t.is_char_boundary(o) || inside_crlf(&t, o) {
                continue;
            }
            prop_assert_eq!(index.offset(index.position(o)), o, "text {:?}", t);
        }
    }

    #[test]
    fn position_never_panics_and_stays_in_the_text(t in text(), o in any::<usize>()) {
        let index = LineIndex::new(&t);
        let p = index.position(o);
        prop_assert!((p.line as usize) < index.line_count());
    }

    #[test]
    fn offset_never_panics_and_lands_on_a_character(
        t in text(), line in any::<u32>(), col in any::<u32>()
    ) {
        let index = LineIndex::new(&t);
        let o = index.offset(Position::new(line, col));
        prop_assert!(o <= t.len());
        prop_assert!(t.is_char_boundary(o));
    }

    #[test]
    fn positions_do_not_decrease_with_the_offset(t in text()) {
        let index = LineIndex::new(&t);
        let mut last = Position::new(0, 0);
        for o in 0..=t.len() {
            if !t.is_char_boundary(o) {
                continue;
            }
            let p = index.position(o);
            prop_assert!((p.line, p.character) >= (last.line, last.character), "text {:?}", t);
            last = p;
        }
    }

    #[test]
    fn lines_agree_with_the_source_map(t in text()) {
        let mut map = pc_ode::SourceMap::new();
        let file = map.add("t.ode", t.clone());
        let index = LineIndex::new(&t);
        for o in 0..=t.len() {
            if !t.is_char_boundary(o) {
                continue;
            }
            let (line, _) = map.line_col(pc_ode::Span::new(file, o, o));
            prop_assert_eq!(index.position(o).line as usize + 1, line, "text {:?} offset {}", t, o);
        }
    }
}

#[test]
fn astral_characters_count_two_units() {
    let t = "a\u{1f600}b";
    let index = LineIndex::new(t);
    assert_eq!(index.position(0), Position::new(0, 0));
    assert_eq!(index.position(1), Position::new(0, 1));
    assert_eq!(index.position(5), Position::new(0, 3));
    assert_eq!(index.position(6), Position::new(0, 4));
}

#[test]
fn a_column_inside_a_surrogate_pair_maps_to_the_character_start() {
    let t = "a\u{1f600}b";
    let index = LineIndex::new(t);
    assert_eq!(index.offset(Position::new(0, 2)), 1);
    assert_eq!(index.offset(Position::new(0, 3)), 5);
}

#[test]
fn every_terminator_starts_a_new_line() {
    let t = "a\nb\r\nc\rd";
    let index = LineIndex::new(t);
    assert_eq!(index.line_count(), 4);
    assert_eq!(index.position(2), Position::new(1, 0));
    assert_eq!(index.position(5), Position::new(2, 0));
    assert_eq!(index.position(7), Position::new(3, 0));
}

#[test]
fn a_position_inside_a_crlf_maps_to_the_line_end() {
    let t = "ab\r\ncd";
    let index = LineIndex::new(t);
    assert_eq!(index.position(3), Position::new(0, 2));
    assert_eq!(index.position(4), Position::new(1, 0));
}

#[test]
fn clamps_past_the_end() {
    let t = "ab\ncd";
    let index = LineIndex::new(t);
    assert_eq!(index.position(100), Position::new(1, 2));
    assert_eq!(index.offset(Position::new(9, 0)), t.len());
    assert_eq!(index.offset(Position::new(0, 50)), 2);
}

#[test]
fn an_offset_inside_a_character_moves_back() {
    let t = "\u{20ac}x";
    let index = LineIndex::new(t);
    assert_eq!(index.position(2), Position::new(0, 0));
}

#[test]
fn a_reversed_range_is_empty_at_its_start() {
    let index = LineIndex::new("abc");
    assert_eq!(
        index.range(2, 1),
        Range::new(Position::new(0, 2), Position::new(0, 2))
    );
}

#[test]
fn an_empty_text_has_one_line() {
    let index = LineIndex::new("");
    assert_eq!(index.line_count(), 1);
    assert_eq!(index.position(0), Position::new(0, 0));
    assert_eq!(index.offset(Position::new(0, 3)), 0);
}
