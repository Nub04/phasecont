/// Escapes `s` for LaTeX text mode.
///
/// The ten characters `# $ % & _ { } ~ ^ \` become their LaTeX forms. The brackets and
/// the three characters `< > |` are escaped as well, because the T1 and OT1 encodings
/// print `<`, `>` and `|` as other glyphs and a `[` at the start of a table cell or
/// after `\\` is read as an optional argument. Control characters, including line
/// breaks, become a space. Other characters pass through unchanged.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '#' | '$' | '%' | '&' | '_' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            '[' => out.push_str("{[}"),
            ']' => out.push_str("{]}"),
            '<' => out.push_str("\\textless{}"),
            '>' => out.push_str("\\textgreater{}"),
            '|' => out.push_str("\\textbar{}"),
            c if c.is_control() => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Every token `escape` can emit for one input character.
    const TOKENS: [(&str, char); 15] = [
        ("\\textbackslash{}", '\\'),
        ("\\textasciitilde{}", '~'),
        ("\\textasciicircum{}", '^'),
        ("\\textless{}", '<'),
        ("\\textgreater{}", '>'),
        ("\\textbar{}", '|'),
        ("\\#", '#'),
        ("\\$", '$'),
        ("\\%", '%'),
        ("\\&", '&'),
        ("\\_", '_'),
        ("\\{", '{'),
        ("\\}", '}'),
        ("{[}", '['),
        ("{]}", ']'),
    ];

    /// Test-only inverse: reads tokens off the front of `s`.
    fn unescape(s: &str) -> String {
        let mut out = String::new();
        let mut rest = s;
        'outer: while !rest.is_empty() {
            for (tok, c) in TOKENS {
                if let Some(r) = rest.strip_prefix(tok) {
                    out.push(c);
                    rest = r;
                    continue 'outer;
                }
            }
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
        out
    }

    #[test]
    fn the_ten_classic_characters() {
        assert_eq!(escape("#"), "\\#");
        assert_eq!(escape("$"), "\\$");
        assert_eq!(escape("%"), "\\%");
        assert_eq!(escape("&"), "\\&");
        assert_eq!(escape("_"), "\\_");
        assert_eq!(escape("{}"), "\\{\\}");
        assert_eq!(escape("~"), "\\textasciitilde{}");
        assert_eq!(escape("^"), "\\textasciicircum{}");
        assert_eq!(escape("\\"), "\\textbackslash{}");
    }

    #[test]
    fn the_extensions() {
        assert_eq!(escape("[a]"), "{[}a{]}");
        assert_eq!(
            escape("a<b>c|d"),
            "a\\textless{}b\\textgreater{}c\\textbar{}d"
        );
    }

    #[test]
    fn control_characters_become_spaces() {
        assert_eq!(escape("a\nb\r\tc\u{0}"), "a b  c ");
    }

    #[test]
    fn plain_text_is_unchanged() {
        assert_eq!(escape("I ext 1.5 (HB) λ"), "I ext 1.5 (HB) λ");
    }

    proptest! {
        #[test]
        fn no_raw_special_character_outside_a_token(s in "\\PC{0,40}") {
            let mut rest = escape(&s);
            // Strip every known token, then nothing special may be left.
            for (tok, _) in TOKENS {
                rest = rest.replace(tok, "");
            }
            for c in ['\\', '#', '$', '%', '&', '_', '{', '}', '~', '^', '[', ']', '<', '>', '|'] {
                prop_assert!(!rest.contains(c), "{c:?} left in {rest:?} from {s:?}");
            }
        }

        #[test]
        fn round_trip_without_control_characters(
            s in "[ -~\u{a0}-\u{2ff}\u{4e00}-\u{4e20}\u{1f600}-\u{1f610}]{0,40}"
        ) {
            prop_assert_eq!(unescape(&escape(&s)), s);
        }
    }
}
