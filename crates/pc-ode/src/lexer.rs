//! Tokenizer for `.ode` text.
//!
//! The lexer reads one whole file and returns its tokens, with newlines as
//! tokens of their own so that the statement parser can work line by line.
//! It follows the rules in `docs/language.md` for lines, names, numbers and
//! blanks. It never panics: bad input becomes a diagnostic plus an
//! [`TokenKind::Unknown`] token, and lexing goes on.
//!
//! Only ASCII letters, digits and `_` can appear in names and numbers, so
//! every other character outside the punctuation list is a stray character.

use crate::diag::{Code, Diagnostic, Severity};
use crate::span::{FileId, Span};

/// A number that is not well formed, such as `1e2.5`, `0x10` or `1_000`.
pub const MALFORMED_NUMBER: Code = Code(1);
/// A number literal that is too large for an `f64`, such as `1e999`.
pub const NUMBER_TOO_LARGE: Code = Code(2);
/// A character that cannot start any token, or a misplaced backslash.
pub const STRAY_CHARACTER: Code = Code(3);

/// What a token is. Text is not stored, except for names: the spelling comes
/// from the token's span.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// A number literal with its value. A literal too large for `f64` still
    /// gives a token, with value infinity, next to its diagnostic.
    Num(f64),
    /// A name. The string is the lower-case lookup key.
    Name(String),
    /// `'`
    Quote,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    /// `=`
    Eq,
    /// `%`
    Percent,
    /// `..`
    DotDot,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    /// `**`
    StarStar,
    Lt,
    Gt,
    /// `<=`
    Le,
    /// `>=`
    Ge,
    /// `==`
    EqEq,
    /// `&`
    Amp,
    /// `|`
    Pipe,
    /// `!`. The text `!=` gives `Bang` then `Eq`; the parser reports it.
    Bang,
    /// `@`, which starts option lines.
    At,
    /// `#` and the rest of the line, without the line terminator.
    Comment,
    /// One `\n` or `\r\n`.
    Newline,
    /// A stray character or a malformed number, reported by a diagnostic.
    Unknown,
    /// End of the text. Always the last token, with an empty span.
    Eof,
}

/// One token with its place in the text.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// True when blanks (spaces, tabs, a lone `\r` or a backslash-newline
    /// continuation) come directly before this token on the same logical
    /// line. The parser uses it to apply the blank-joining rule.
    pub preceded_by_blank: bool,
}

/// Splits `text` into tokens. The last token is always [`TokenKind::Eof`].
///
/// Two-character operators (`**`, `<=`, `>=`, `==`, `..`) are formed only
/// when no blank sits between the characters.
pub fn lex(file: FileId, text: &str) -> (Vec<Token>, Vec<Diagnostic>) {
    let mut lexer = Lexer {
        file,
        text,
        bytes: text.as_bytes(),
        pos: 0,
        blank: false,
        tokens: Vec::new(),
        diags: Vec::new(),
    };
    lexer.run();
    (lexer.tokens, lexer.diags)
}

struct Lexer<'a> {
    file: FileId,
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    /// Blanks have been seen since the last token.
    blank: bool,
    tokens: Vec<Token>,
    diags: Vec<Diagnostic>,
}

impl Lexer<'_> {
    fn run(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b' ' | b'\t' => self.skip_blank(1),
                b'\r' if self.peek(1) == Some(b'\n') => self.single(TokenKind::Newline, 2),
                b'\r' => self.skip_blank(1),
                b'\n' => self.single(TokenKind::Newline, 1),
                b'\\' => self.backslash(),
                b'#' => self.comment(),
                b'0'..=b'9' => self.number(),
                b'.' => match self.peek(1) {
                    Some(b'.') => self.single(TokenKind::DotDot, 2),
                    Some(b'0'..=b'9') => self.number(),
                    _ => self.stray(),
                },
                b'a'..=b'z' | b'A'..=b'Z' => self.name(),
                b'\'' => self.single(TokenKind::Quote, 1),
                b'(' => self.single(TokenKind::LParen, 1),
                b')' => self.single(TokenKind::RParen, 1),
                b'[' => self.single(TokenKind::LBracket, 1),
                b']' => self.single(TokenKind::RBracket, 1),
                b'{' => self.single(TokenKind::LBrace, 1),
                b'}' => self.single(TokenKind::RBrace, 1),
                b',' => self.single(TokenKind::Comma, 1),
                b'%' => self.single(TokenKind::Percent, 1),
                b'+' => self.single(TokenKind::Plus, 1),
                b'-' => self.single(TokenKind::Minus, 1),
                b'/' => self.single(TokenKind::Slash, 1),
                b'^' => self.single(TokenKind::Caret, 1),
                b'&' => self.single(TokenKind::Amp, 1),
                b'|' => self.single(TokenKind::Pipe, 1),
                b'!' => self.single(TokenKind::Bang, 1),
                b'@' => self.single(TokenKind::At, 1),
                b'*' => self.with_equals_or_double(b'*', TokenKind::Star, TokenKind::StarStar),
                b'=' => self.with_equals_or_double(b'=', TokenKind::Eq, TokenKind::EqEq),
                b'<' => self.with_equals_or_double(b'=', TokenKind::Lt, TokenKind::Le),
                b'>' => self.with_equals_or_double(b'=', TokenKind::Gt, TokenKind::Ge),
                _ => self.stray(),
            }
        }
        let end = self.text.len();
        self.push(TokenKind::Eof, end, end);
    }

    fn peek(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    fn span(&self, start: usize, end: usize) -> Span {
        Span::new(self.file, start, end)
    }

    /// Adds a token and clears the blank flag.
    fn push(&mut self, kind: TokenKind, start: usize, end: usize) {
        let span = self.span(start, end);
        self.tokens.push(Token {
            kind,
            span,
            preceded_by_blank: self.blank,
        });
        self.blank = false;
    }

    fn error(&mut self, code: Code, start: usize, end: usize, message: String) {
        let span = self.span(start, end);
        self.diags.push(Diagnostic {
            code,
            severity: Severity::Error,
            span,
            message,
            fix_it: None,
            notes: Vec::new(),
        });
    }

    fn skip_blank(&mut self, len: usize) {
        self.pos += len;
        self.blank = true;
    }

    /// A token of `len` bytes at the current position.
    fn single(&mut self, kind: TokenKind, len: usize) {
        let start = self.pos;
        self.pos += len;
        self.push(kind, start, self.pos);
    }

    /// `first` alone gives `single`; `first` followed directly by `second`
    /// gives `double`.
    fn with_equals_or_double(&mut self, second: u8, single: TokenKind, double: TokenKind) {
        if self.peek(1) == Some(second) {
            self.single(double, 2);
        } else {
            self.single(single, 1);
        }
    }

    /// A backslash joins lines when a line terminator follows it directly.
    /// Anywhere else it is a stray character.
    fn backslash(&mut self) {
        match (self.peek(1), self.peek(2)) {
            (Some(b'\n'), _) => self.skip_blank(2),
            (Some(b'\r'), Some(b'\n')) => self.skip_blank(3),
            _ => self.stray(),
        }
    }

    /// One unknown character, reported and kept as an `Unknown` token.
    fn stray(&mut self) {
        let start = self.pos;
        let ch = self.text[start..].chars().next().unwrap_or('\0');
        let end = start + ch.len_utf8().max(1);
        let message = if ch == '\\' {
            "a backslash must be the last character on a line to continue it".to_string()
        } else {
            format!("unexpected character {ch:?}")
        };
        self.error(STRAY_CHARACTER, start, end, message);
        self.pos = end;
        self.push(TokenKind::Unknown, start, end);
    }

    /// `#` to the end of the line, not including the line terminator.
    fn comment(&mut self) {
        let start = self.pos;
        let rest = &self.bytes[start..];
        let mut end = match rest.iter().position(|&b| b == b'\n') {
            Some(i) => start + i,
            None => self.bytes.len(),
        };
        if end > start && self.bytes.get(end) == Some(&b'\n') && self.bytes[end - 1] == b'\r' {
            end -= 1;
        }
        self.pos = end;
        self.push(TokenKind::Comment, start, end);
    }

    fn is_digit_at(&self, index: usize) -> bool {
        self.bytes.get(index).is_some_and(u8::is_ascii_digit)
    }

    /// A dot that is not the start of `..`.
    fn is_single_dot_at(&self, index: usize) -> bool {
        self.bytes.get(index) == Some(&b'.') && self.bytes.get(index + 1) != Some(&b'.')
    }

    fn number(&mut self) {
        let start = self.pos;
        let mut i = start;
        while self.is_digit_at(i) {
            i += 1;
        }
        if self.is_single_dot_at(i) {
            i += 1;
            while self.is_digit_at(i) {
                i += 1;
            }
        }
        // An exponent needs a digit, or a sign and a digit, after the `e`.
        if matches!(self.bytes.get(i), Some(b'e' | b'E')) {
            let mut j = i + 1;
            if matches!(self.bytes.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if self.is_digit_at(j) {
                while self.is_digit_at(j) {
                    j += 1;
                }
                i = j;
            }
        }

        let hex = &self.text[start..i] == "0"
            && matches!(self.bytes.get(i), Some(b'x' | b'X'))
            && self.bytes.get(i + 1).is_some_and(u8::is_ascii_hexdigit);
        // A dot here is a second dot, or a dot after an exponent.
        let malformed = hex || self.bytes.get(i) == Some(&b'_') || self.is_single_dot_at(i);
        if malformed {
            self.malformed_number(start, i);
            return;
        }

        let end = i;
        match self.text[start..end].parse::<f64>() {
            Ok(value) => {
                if value.is_infinite() {
                    let message = format!(
                        "the number {} is too large for a double precision value",
                        &self.text[start..end]
                    );
                    self.error(NUMBER_TOO_LARGE, start, end, message);
                }
                self.pos = end;
                self.push(TokenKind::Num(value), start, end);
            }
            // Cannot happen for the grammar above, but a malformed run is a
            // safe answer if it ever does.
            Err(_) => self.malformed_number(start, end),
        }
    }

    /// Reports one diagnostic for the whole run of number-like characters
    /// that starts at `start` and keeps it as one `Unknown` token.
    fn malformed_number(&mut self, start: usize, scanned: usize) {
        let mut i = scanned;
        while let Some(&b) = self.bytes.get(i) {
            if b.is_ascii_alphanumeric() || b == b'_' || self.is_single_dot_at(i) {
                i += 1;
            } else {
                break;
            }
        }
        let message = format!("malformed number {:?}", &self.text[start..i]);
        self.error(MALFORMED_NUMBER, start, i, message);
        self.pos = i;
        self.push(TokenKind::Unknown, start, i);
    }

    /// A letter, then letters, digits, `_` and `.`. A `.` that begins `..`
    /// ends the name.
    fn name(&mut self) {
        let start = self.pos;
        let mut i = start + 1;
        while let Some(&b) = self.bytes.get(i) {
            if b.is_ascii_alphanumeric() || b == b'_' || self.is_single_dot_at(i) {
                i += 1;
            } else {
                break;
            }
        }
        let key = self.text[start..i].to_ascii_lowercase();
        self.pos = i;
        self.push(TokenKind::Name(key), start, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diag::CODES;

    const F: FileId = FileId(0);

    fn n(v: f64) -> TokenKind {
        TokenKind::Num(v)
    }

    fn name(s: &str) -> TokenKind {
        TokenKind::Name(s.to_string())
    }

    /// Token kinds without the final `Eof`. Panics if there are diagnostics.
    fn kinds(text: &str) -> Vec<TokenKind> {
        let (tokens, diags) = lex(F, text);
        assert!(
            diags.is_empty(),
            "unexpected diagnostics for {text:?}: {diags:?}"
        );
        kinds_of(&tokens)
    }

    fn kinds_of(tokens: &[Token]) -> Vec<TokenKind> {
        let (last, rest) = tokens.split_last().unwrap();
        assert_eq!(last.kind, TokenKind::Eof);
        rest.iter().map(|t| t.kind.clone()).collect()
    }

    fn slice<'a>(text: &'a str, t: &Token) -> &'a str {
        &text[t.span.start..t.span.end]
    }

    fn blanks(text: &str) -> Vec<bool> {
        let (tokens, _) = lex(F, text);
        tokens.iter().map(|t| t.preceded_by_blank).collect()
    }

    #[test]
    fn array_range_in_a_subscript() {
        use TokenKind::*;
        assert_eq!(
            kinds("x[1..10]"),
            vec![name("x"), LBracket, n(1.0), DotDot, n(10.0), RBracket]
        );
    }

    #[test]
    fn percent_range() {
        use TokenKind::*;
        let expected = vec![Percent, LBracket, n(1.0), DotDot, n(3.0), RBracket];
        assert_eq!(kinds("%[1..3]"), expected);
        assert_eq!(kinds("%[1 .. 3]"), expected);
        // Only the blanks around the range mark tokens.
        assert_eq!(
            blanks("%[1 .. 3]"),
            vec![false, false, false, true, true, false, false]
        );
    }

    #[test]
    fn number_forms() {
        assert_eq!(kinds("5"), vec![n(5.0)]);
        assert_eq!(kinds(".5"), vec![n(0.5)]);
        assert_eq!(kinds("5."), vec![n(5.0)]);
        assert_eq!(kinds("1.e2"), vec![n(100.0)]);
        assert_eq!(kinds("1e+2"), vec![n(100.0)]);
        assert_eq!(kinds("1E2"), vec![n(100.0)]);
        assert_eq!(kinds("2.5e-3"), vec![n(0.0025)]);
        assert_eq!(kinds("007"), vec![n(7.0)]);
        assert_eq!(kinds("1e-400"), vec![n(0.0)]);
    }

    #[test]
    fn number_text_comes_from_the_span() {
        let text = "1.e2 .5";
        let (tokens, _) = lex(F, text);
        assert_eq!(slice(text, &tokens[0]), "1.e2");
        assert_eq!(slice(text, &tokens[1]), ".5");
    }

    #[test]
    fn dot_dot_ends_a_number() {
        use TokenKind::*;
        assert_eq!(kinds("1..10"), vec![n(1.0), DotDot, n(10.0)]);
        assert_eq!(kinds("1.5..10"), vec![n(1.5), DotDot, n(10.0)]);
        assert_eq!(kinds("1e2..3"), vec![n(100.0), DotDot, n(3.0)]);
    }

    #[test]
    fn e_is_an_exponent_only_before_a_digit() {
        use TokenKind::*;
        assert_eq!(
            kinds("2exp(x)"),
            vec![n(2.0), name("exp"), LParen, name("x"), RParen]
        );
        assert_eq!(kinds("2e-3x"), vec![n(0.002), name("x")]);
        assert_eq!(kinds("1e"), vec![n(1.0), name("e")]);
        assert_eq!(kinds("1e+"), vec![n(1.0), name("e"), Plus]);
        assert_eq!(kinds("1E-x"), vec![n(1.0), name("e"), Minus, name("x")]);
        assert_eq!(kinds("2e3e4"), vec![n(2000.0), name("e4")]);
    }

    #[test]
    fn names_keep_dots_and_fold_case() {
        let text = "2P.E.";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        assert_eq!(tokens[0].kind, n(2.0));
        assert_eq!(tokens[1].kind, name("p.e."));
        assert_eq!(slice(text, &tokens[1]), "P.E.");
        assert!(!tokens[1].preceded_by_blank);

        assert_eq!(kinds("Foo_Bar9"), vec![name("foo_bar9")]);
        assert_eq!(kinds("a.b"), vec![name("a.b")]);
    }

    #[test]
    fn dot_dot_ends_a_name() {
        use TokenKind::*;
        assert_eq!(kinds("a..b"), vec![name("a"), DotDot, name("b")]);
        assert_eq!(kinds("a.b..c"), vec![name("a.b"), DotDot, name("c")]);
    }

    #[test]
    fn names_start_with_a_letter_only() {
        use TokenKind::*;
        let (tokens, diags) = lex(F, "_x");
        assert_eq!(diags.len(), 1);
        assert_eq!(tokens[0].kind, Unknown);
        assert_eq!(tokens[1].kind, name("x"));
    }

    #[test]
    fn punctuation_and_operators() {
        use TokenKind::*;
        assert_eq!(
            kinds("' ( ) [ ] { } , = % + - * / ^ < > & | ! @"),
            vec![
                Quote, LParen, RParen, LBracket, RBracket, LBrace, RBrace, Comma, Eq, Percent,
                Plus, Minus, Star, Slash, Caret, Lt, Gt, Amp, Pipe, Bang, At
            ]
        );
        assert_eq!(
            kinds("** <= >= == .."),
            vec![StarStar, Le, Ge, EqEq, DotDot]
        );
    }

    #[test]
    fn two_character_operators_need_no_blank_inside() {
        use TokenKind::*;
        assert_eq!(kinds("* *"), vec![Star, Star]);
        assert_eq!(blanks("* *"), vec![false, true, false]);
        assert_eq!(kinds("< ="), vec![Lt, Eq]);
        assert_eq!(kinds("> ="), vec![Gt, Eq]);
        assert_eq!(kinds("= ="), vec![Eq, Eq]);
        let (tokens, diags) = lex(F, ". .");
        assert_eq!(diags.len(), 2);
        assert_eq!(kinds_of(&tokens), vec![Unknown, Unknown]);
    }

    #[test]
    fn bang_equals_is_two_tokens() {
        use TokenKind::*;
        assert_eq!(kinds("a!=b"), vec![name("a"), Bang, Eq, name("b")]);
        assert_eq!(kinds("!d=1"), vec![Bang, name("d"), Eq, n(1.0)]);
    }

    #[test]
    fn ode_and_option_lines() {
        use TokenKind::*;
        assert_eq!(
            kinds("x'=-x*y"),
            vec![name("x"), Quote, Eq, Minus, name("x"), Star, name("y")]
        );
        assert_eq!(
            kinds("@ total=10,dt=.05"),
            vec![
                At,
                name("total"),
                Eq,
                n(10.0),
                Comma,
                name("dt"),
                Eq,
                n(0.05)
            ]
        );
    }

    #[test]
    fn newlines_have_one_token_each() {
        use TokenKind::*;
        let text = "a\nb\r\n\nc";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        assert_eq!(
            kinds_of(&tokens),
            vec![name("a"), Newline, name("b"), Newline, Newline, name("c")]
        );
        assert_eq!(slice(text, &tokens[1]), "\n");
        assert_eq!(slice(text, &tokens[3]), "\r\n");
        assert_eq!(slice(text, &tokens[4]), "\n");
    }

    #[test]
    fn blank_flag_resets_after_a_newline() {
        // a, NL, b, NL, c, EOF
        assert_eq!(
            blanks("a  \n b\nc"),
            vec![false, true, true, false, false, false]
        );
        assert_eq!(blanks("a\tb"), vec![false, true, false]);
    }

    #[test]
    fn continuation_is_a_blank() {
        use TokenKind::*;
        let text = "a\\\nb";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        assert_eq!(kinds_of(&tokens), vec![name("a"), name("b")]);
        assert!(!tokens[0].preceded_by_blank);
        assert!(tokens[1].preceded_by_blank);

        let (tokens, diags) = lex(F, "a\\\r\nb");
        assert!(diags.is_empty());
        assert_eq!(kinds_of(&tokens), vec![name("a"), name("b")]);
        assert!(tokens[1].preceded_by_blank);

        // A continuation then a real newline.
        assert_eq!(
            kinds("a \\\n  + b\nc"),
            vec![name("a"), Plus, name("b"), Newline, name("c")]
        );
        // Two continuations in a row, and one at the very start of the line.
        assert_eq!(kinds("\\\n\\\nx"), vec![name("x")]);
        assert_eq!(blanks("\\\n\\\nx"), vec![true, false]);
    }

    #[test]
    fn continuation_inside_a_number_splits_it() {
        assert_eq!(kinds("1\\\n2"), vec![n(1.0), n(2.0)]);
    }

    #[test]
    fn lone_cr_is_a_blank() {
        assert_eq!(kinds("a\rb"), vec![name("a"), name("b")]);
        assert_eq!(blanks("a\rb"), vec![false, true, false]);
    }

    #[test]
    fn comment_runs_to_the_end_of_the_line() {
        use TokenKind::*;
        let text = "x=1 # hi there\r\n#include foo.ode\n# last";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        assert_eq!(
            kinds_of(&tokens),
            vec![
                name("x"),
                Eq,
                n(1.0),
                Comment,
                Newline,
                Comment,
                Newline,
                Comment
            ]
        );
        assert_eq!(slice(text, &tokens[3]), "# hi there");
        assert!(tokens[3].preceded_by_blank);
        assert_eq!(slice(text, &tokens[5]), "#include foo.ode");
        assert_eq!(slice(text, &tokens[7]), "# last");
    }

    #[test]
    fn comment_ignores_a_trailing_backslash_and_lone_cr() {
        use TokenKind::*;
        let text = "# a \\\nb";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        assert_eq!(kinds_of(&tokens), vec![Comment, Newline, name("b")]);
        assert_eq!(slice(text, &tokens[0]), "# a \\");
        let text = "# a\rb\nc";
        let (tokens, _) = lex(F, text);
        assert_eq!(slice(text, &tokens[0]), "# a\rb");
    }

    #[test]
    fn eof_token_is_empty_and_last() {
        for text in ["", "a", "a\n", "  ", "a \\\n"] {
            let (tokens, _) = lex(F, text);
            let eofs = tokens.iter().filter(|t| t.kind == TokenKind::Eof).count();
            assert_eq!(eofs, 1, "{text:?}");
            let last = tokens.last().unwrap();
            assert_eq!(last.kind, TokenKind::Eof);
            assert_eq!(last.span, Span::new(F, text.len(), text.len()));
        }
        assert!(blanks("a ")[1]);
        assert!(!blanks("")[0]);
    }

    fn one_error(text: &str) -> (Vec<Token>, Diagnostic) {
        let (tokens, mut diags) = lex(F, text);
        assert_eq!(diags.len(), 1, "{text:?}: {diags:?}");
        (tokens, diags.remove(0))
    }

    #[test]
    fn malformed_numbers_give_one_diagnostic_each() {
        for (text, run) in [
            ("1e2.5", "1e2.5"),
            ("0x10", "0x10"),
            ("0X1f", "0X1f"),
            ("1_000", "1_000"),
            ("1.2.3", "1.2.3"),
            (".5.5", ".5.5"),
            ("1e2.", "1e2."),
        ] {
            let wrapped = format!("a+{text}+b");
            let (tokens, d) = one_error(&wrapped);
            assert_eq!(d.code, MALFORMED_NUMBER, "{text}");
            assert_eq!(d.severity, Severity::Error);
            assert_eq!(&wrapped[d.span.start..d.span.end], run, "{text}");
            // The run is a single Unknown token between `+` and `+`.
            assert_eq!(
                kinds_of(&tokens),
                vec![
                    name("a"),
                    TokenKind::Plus,
                    TokenKind::Unknown,
                    TokenKind::Plus,
                    name("b")
                ],
                "{text}"
            );
            assert_eq!(tokens[2].span, d.span);
        }
    }

    #[test]
    fn near_misses_are_not_malformed() {
        use TokenKind::*;
        assert_eq!(kinds("0xy"), vec![n(0.0), name("xy")]);
        assert_eq!(kinds("10x"), vec![n(10.0), name("x")]);
        assert_eq!(kinds("5.x"), vec![n(5.0), name("x")]);
        assert_eq!(kinds("1.5.."), vec![n(1.5), DotDot]);
    }

    #[test]
    fn number_overflow_is_an_error_but_keeps_the_token() {
        let (tokens, d) = one_error("1e999");
        assert_eq!(d.code, NUMBER_TOO_LARGE);
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.span, Span::new(F, 0, 5));
        assert_eq!(tokens[0].kind, n(f64::INFINITY));
        assert_eq!(tokens[0].span, d.span);
        // The largest finite value is fine.
        assert_eq!(kinds("1.7976931348623157e308"), vec![n(f64::MAX)]);
    }

    #[test]
    fn stray_characters() {
        for (text, bad) in [
            ("$", "$"),
            ("a ? b", "?"),
            ("\\", "\\"),
            ("a\\ \nb", "\\"),
            (".", "."),
            ("é", "é"),
            ("x = Ω", "Ω"),
            (":", ":"),
            (";", ";"),
            ("\"", "\""),
            ("~", "~"),
        ] {
            let (tokens, d) = one_error(text);
            assert_eq!(d.code, STRAY_CHARACTER, "{text:?}");
            assert_eq!(d.severity, Severity::Error);
            assert_eq!(&text[d.span.start..d.span.end], bad, "{text:?}");
            let unknown: Vec<_> = tokens
                .iter()
                .filter(|t| t.kind == TokenKind::Unknown)
                .collect();
            assert_eq!(unknown.len(), 1, "{text:?}");
            assert_eq!(unknown[0].span, d.span);
        }
    }

    #[test]
    fn non_ascii_letters_are_not_name_characters() {
        use TokenKind::*;
        let (tokens, diags) = lex(F, "aé1");
        assert_eq!(diags.len(), 1);
        assert_eq!(kinds_of(&tokens), vec![name("a"), Unknown, n(1.0)]);
    }

    #[test]
    fn lone_dot_versus_dot_dot() {
        use TokenKind::*;
        let (tokens, diags) = lex(F, "...");
        assert_eq!(diags.len(), 1);
        assert_eq!(kinds_of(&tokens), vec![DotDot, Unknown]);
        assert_eq!(kinds(".."), vec![DotDot]);
        assert_eq!(kinds("....")[..], [DotDot, DotDot]);
    }

    #[test]
    fn codes_are_registered_in_order() {
        assert_eq!(MALFORMED_NUMBER, Code(1));
        assert_eq!(NUMBER_TOO_LARGE, Code(2));
        assert_eq!(STRAY_CHARACTER, Code(3));
        for code in [MALFORMED_NUMBER, NUMBER_TOO_LARGE, STRAY_CHARACTER] {
            let entry = CODES.iter().find(|(c, _, _)| *c == code).unwrap();
            assert_eq!(entry.1, Severity::Error);
        }
    }

    #[test]
    fn nasty_inputs_keep_spans_inside_the_text() {
        let inputs = [
            "",
            "\\",
            "\\\\",
            "\\\r",
            "\r",
            "\r\r\n",
            "é",
            "Ω",
            "aΩb",
            "😀",
            ".",
            "1e",
            "1e+",
            "1e-",
            "..",
            "...",
            "0x",
            "0x1_",
            "1_",
            "1.2.3.4",
            "#",
            "# é",
            "<",
            "!=",
            "x[1..",
            "\u{0}",
            "\u{feff}x",
            "1e999999999999999999999",
            ".e5",
            "a\\",
        ];
        for text in inputs {
            let (tokens, diags) = lex(F, text);
            assert!(!tokens.is_empty(), "{text:?}");
            assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof, "{text:?}");
            let spans = tokens
                .iter()
                .map(|t| t.span)
                .chain(diags.iter().map(|d| d.span));
            for s in spans {
                assert_eq!(s.file, F);
                assert!(s.start <= s.end, "{text:?}: {s:?}");
                assert!(s.end <= text.len(), "{text:?}: {s:?}");
                assert!(text.is_char_boundary(s.start), "{text:?}: {s:?}");
                assert!(text.is_char_boundary(s.end), "{text:?}: {s:?}");
            }
            // Tokens are in order and never overlap.
            for pair in tokens.windows(2) {
                assert!(pair[0].span.end <= pair[1].span.start, "{text:?}");
            }
        }
    }

    #[test]
    fn lexer_survives_every_short_ascii_string() {
        // All strings of length <= 3 over a small alphabet that hits each rule.
        let alphabet = [
            'a', 'e', '1', '.', '_', 'x', '+', '\\', '\n', '\r', '#', ' ', 'é',
        ];
        let mut texts = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..3 {
            let mut next = Vec::new();
            for s in &frontier {
                for c in alphabet {
                    let mut t = s.clone();
                    t.push(c);
                    next.push(t);
                }
            }
            texts.extend(next.iter().cloned());
            frontier = next;
        }
        for text in texts {
            let (tokens, diags) = lex(F, &text);
            assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof);
            for s in tokens
                .iter()
                .map(|t| t.span)
                .chain(diags.iter().map(|d| d.span))
            {
                assert!(s.start <= s.end && s.end <= text.len(), "{text:?}");
                assert!(text.is_char_boundary(s.start) && text.is_char_boundary(s.end));
            }
        }
    }
}
