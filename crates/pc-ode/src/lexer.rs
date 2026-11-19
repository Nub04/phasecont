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

use crate::diag::{Code, Diagnostic, FixIt, QuickFix, Severity};
use crate::parser::expr::clip;
use crate::span::{FileId, Span};

/// A number that is not well formed, such as `1e2.5`, `0x10` or `1_000`.
pub const MALFORMED_NUMBER: Code = Code(1);
/// A number literal that is too large for an `f64`, such as `1e999`.
pub const NUMBER_TOO_LARGE: Code = Code(2);
/// A character that cannot start any token.
pub const STRAY_CHARACTER: Code = Code(3);
/// A backslash followed by text on the same line. XPPAUT drops the text and
/// joins the next line; so does PhaseCont, with this warning.
pub const BACKSLASH_TEXT: Code = Code(75);
/// A backslash inside a `#` comment. XPPAUT drops the rest of the comment
/// line and joins the next line to it, so that line is part of the comment;
/// so does PhaseCont, with this warning.
pub const BACKSLASH_COMMENT: Code = Code(76);

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
    /// `;`, which separates the assignments in a `global` block.
    Semicolon,
    /// `@`, which starts option lines.
    At,
    /// `#` and the rest of the line, without the line terminator.
    Comment,
    /// One `\n`, `\r\n` or a `\r` that no `\n` follows.
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
    /// True when blanks (spaces, tabs or a backslash-newline
    /// continuation) come directly before this token on the same logical
    /// line. The parser uses it to apply the blank-joining rule.
    pub preceded_by_blank: bool,
}

/// Splits the text of a main file into tokens. The last token is always
/// [`TokenKind::Eof`].
///
/// Two-character operators (`**`, `<=`, `>=`, `==`, `..`) are formed only
/// when no blank sits between the characters. A comment on the first line
/// keeps a backslash as text (see [`lex_file`]).
pub fn lex(file: FileId, text: &str) -> (Vec<Token>, Vec<Diagnostic>) {
    lex_file(file, text, true)
}

/// [`lex`] for the text of a main file (`main` true) or an included file
/// (`main` false). XPPAUT 8.2.2 does not apply the backslash rule to the
/// first line of the main file, so a `#` comment there keeps a backslash
/// and does not join the next line; an included file has no such line
/// (probes `backslash-comment-line1` and `backslash-comment-inc`).
pub(crate) fn lex_file(file: FileId, text: &str, main: bool) -> (Vec<Token>, Vec<Diagnostic>) {
    let mut lexer = Lexer {
        file,
        text,
        bytes: text.as_bytes(),
        pos: 0,
        blank: false,
        first_line_end: 0,
        tokens: Vec::new(),
        diags: Vec::new(),
    };
    if main {
        lexer.first_line_end = lexer.line_end(0);
    }
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
    /// The end of the first line when it is the first line of a main file,
    /// else 0. A comment that starts before it keeps its backslashes.
    first_line_end: usize,
    tokens: Vec<Token>,
    diags: Vec<Diagnostic>,
}

impl Lexer<'_> {
    fn run(&mut self) {
        // A byte order mark at the very start of the file is ignored. Anywhere
        // else U+FEFF is a stray character.
        if self.text.starts_with('\u{feff}') {
            self.pos = '\u{feff}'.len_utf8();
        }
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b' ' | b'\t' => self.skip_blank(1),
                b'\r' if self.peek(1) == Some(b'\n') => self.single(TokenKind::Newline, 2),
                // A CR that no LF follows ends the line too.
                b'\r' => self.single(TokenKind::Newline, 1),
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
                b';' => self.single(TokenKind::Semicolon, 1),
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
            fixes: Vec::new(),
            notes: Vec::new(),
        });
    }

    /// A warning with quick fixes, the first of them preferred.
    fn warn(
        &mut self,
        code: Code,
        start: usize,
        end: usize,
        message: String,
        fixes: Vec<QuickFix>,
    ) {
        let span = self.span(start, end);
        self.diags.push(Diagnostic {
            code,
            severity: Severity::Warning,
            span,
            message,
            fix_it: fixes.first().map(|f| f.edit.clone()),
            fixes,
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

    /// The end of the line that holds `from`: the offset of the first `\n` or
    /// `\r` at or after it, or the end of the text.
    fn line_end(&self, from: usize) -> usize {
        match self.bytes[from.min(self.bytes.len())..]
            .iter()
            .position(|&b| b == b'\n' || b == b'\r')
        {
            Some(i) => from + i,
            None => self.bytes.len(),
        }
    }

    /// The length of the line terminator at `at`: 2 for `\r\n`, 1 for `\n` or
    /// a `\r` that no `\n` follows, 0 at the end of the text.
    fn terminator_len(&self, at: usize) -> usize {
        match (self.bytes.get(at), self.bytes.get(at + 1)) {
            (Some(b'\r'), Some(b'\n')) => 2,
            (Some(b'\r' | b'\n'), _) => 1,
            _ => 0,
        }
    }

    /// True when a line that starts at `start` holds text that matters to
    /// the model: something other than blanks or a `#` comment.
    fn line_has_content(&self, start: usize, end: usize) -> bool {
        let line = self.text[start..end].trim_matches([' ', '\t']);
        !line.is_empty() && !line.starts_with('#')
    }

    /// A backslash joins lines when only spaces and tabs sit between it and
    /// the line terminator or the end of the text (XPPAUT 8.2.2 measured:
    /// probes backslash-blanks and backslash-eof). With other text after it
    /// on the line, XPPAUT drops that text and the terminator and joins the
    /// next line (probes backslash-text, backslash-text-crlf,
    /// backslash-eof-text); PhaseCont does the same and gives
    /// [`BACKSLASH_TEXT`] so that the drop is not silent.
    fn backslash(&mut self) {
        let start = self.pos;
        let eol = self.line_end(start + 1);
        let term = self.terminator_len(eol);
        if self.bytes[start + 1..eol]
            .iter()
            .all(|b| matches!(b, b' ' | b'\t'))
        {
            self.skip_blank(eol + term - start);
            return;
        }
        self.drop_text(start, eol, term);
    }

    /// The backslash at `start` with text after it, up to `eol`, which a
    /// terminator of `term` bytes follows. Drops the text and the
    /// terminator, warns, and goes on with the next line.
    fn drop_text(&mut self, start: usize, eol: usize, term: usize) {
        let after = &self.text[start + 1..eol];
        let dropped = after.trim_matches([' ', '\t']);
        let what = if dropped.starts_with('#') {
            format!("the comment `{}`", clip(dropped))
        } else {
            format!("the text `{}`", clip(dropped))
        };
        let joins = term > 0 && eol + term < self.bytes.len();
        let message = if joins {
            format!(
                "XPPAUT drops {what} after this backslash and joins the next line, \
                 so the model does not contain it"
            )
        } else {
            format!("XPPAUT drops {what} after this backslash, so the model does not contain it")
        };
        let delete = QuickFix {
            title: "Delete the text after the backslash".to_string(),
            edit: FixIt {
                span: self.span(start + 1, eol),
                replacement: String::new(),
            },
        };
        // A backslash moved behind a `#` would sit in a comment and swallow
        // the next line (BACKSLASH_COMMENT). Moved behind text that holds a
        // backslash, it would leave that backslash in the middle of the line
        // and still drop what follows it. Text with either character only
        // gets the delete.
        let fixes = if dropped.contains(['#', '\\']) {
            vec![delete]
        } else {
            let mut moved = after.trim_end_matches([' ', '\t']).to_string();
            moved.push_str(" \\");
            let mv = QuickFix {
                title: "Move the backslash to the end of the line".to_string(),
                edit: FixIt {
                    span: self.span(start, eol),
                    replacement: moved,
                },
            };
            vec![mv, delete]
        };
        self.warn(BACKSLASH_TEXT, start, eol, message, fixes);
        self.pos = eol + term;
        self.blank = true;
    }

    /// One unknown character, reported and kept as an `Unknown` token.
    fn stray(&mut self) {
        let start = self.pos;
        let ch = self.text[start..].chars().next().unwrap_or('\0');
        let end = start + ch.len_utf8().max(1);
        self.error(
            STRAY_CHARACTER,
            start,
            end,
            format!("unexpected character {ch:?}"),
        );
        self.pos = end;
        self.push(TokenKind::Unknown, start, end);
    }

    /// `#` to the end of the line, not including the line terminator.
    ///
    /// A backslash anywhere in a comment ends the comment text there. As in
    /// XPPAUT 8.2.2 (probes backslash-comment, backslash-comment-mid,
    /// backslash-comment-chain), the rest of the line, its terminator and
    /// the whole next line are then part of the comment and give no tokens,
    /// and a backslash in that next line does the same to the line after it.
    /// The first line of a main file is not treated so.
    fn comment(&mut self) {
        let start = self.pos;
        let eol = self.line_end(start);
        let backslash = if start < self.first_line_end {
            None
        } else {
            self.bytes[start..eol].iter().position(|&b| b == b'\\')
        };
        let Some(i) = backslash else {
            self.pos = eol;
            self.push(TokenKind::Comment, start, eol);
            return;
        };
        let first = start + i;
        self.push(TokenKind::Comment, start, first);
        self.pos = first;
        self.join_comment_lines(first, eol);
    }

    /// The comment line that has its backslash at `bs` and ends at `eol`
    /// swallows the next line, and the one after it while those have a
    /// backslash too. Leaves `pos` at the terminator of the last line
    /// swallowed, or at the end of the text.
    fn join_comment_lines(&mut self, mut bs: usize, mut eol: usize) {
        loop {
            let term = self.terminator_len(eol);
            let next = eol + term;
            if term == 0 || next >= self.bytes.len() {
                // Nothing follows, so nothing is swallowed.
                self.pos = eol;
                return;
            }
            let next_end = self.line_end(next);
            if self.line_has_content(next, next_end) {
                self.warn_swallowed(bs, eol, next, next_end);
            }
            eol = next_end;
            match self.bytes[next..next_end].iter().position(|&b| b == b'\\') {
                Some(i) => bs = next + i,
                None => {
                    self.pos = eol;
                    return;
                }
            }
        }
    }

    /// The lines a comment line that ends at `eol` swallows, as (start, end)
    /// offsets in order: the next line, and the one after it while a
    /// swallowed line holds a backslash. Stops where the text ends.
    fn swallowed_lines(&self, mut eol: usize) -> Vec<(usize, usize)> {
        let mut lines = Vec::new();
        loop {
            let next = eol + self.terminator_len(eol);
            if next == eol || next >= self.bytes.len() {
                return lines;
            }
            let end = self.line_end(next);
            lines.push((next, end));
            if !self.bytes[next..end].contains(&b'\\') {
                return lines;
            }
            eol = end;
        }
    }

    /// [`BACKSLASH_COMMENT`] for the backslash at `bs` in a comment line that
    /// ends at `eol`, with the line `next..next_end` swallowed.
    fn warn_swallowed(&mut self, bs: usize, eol: usize, next: usize, next_end: usize) {
        let line = self.text[next..next_end].trim_matches([' ', '\t']);
        let message = format!(
            "XPPAUT reads the next line, `{}`, as part of this comment, \
             so the model does not contain it",
            clip(line)
        );
        // Every backslash of the line goes: one left behind would swallow
        // the next line just the same.
        let last = bs + self.text[bs..eol].rfind('\\').unwrap_or(0);
        let several = last > bs;
        let at_end = !several && self.text[bs + 1..eol].trim_matches([' ', '\t']).is_empty();
        let remove = QuickFix {
            title: if several {
                "Remove the backslashes from the comment"
            } else if at_end {
                "Remove the backslash at the end of the comment"
            } else {
                "Remove the backslash from the comment"
            }
            .to_string(),
            edit: FixIt {
                span: self.span(bs, last + 1),
                replacement: self.text[bs..=last].replace('\\', ""),
            },
        };
        let (span_end, replacement, count) = self.comment_out_swallowed(bs, eol);
        let comment_out = QuickFix {
            title: if count == 1 {
                "Comment out the next line".to_string()
            } else {
                format!("Comment out the next {count} lines")
            },
            edit: FixIt {
                span: self.span(bs, span_end),
                replacement,
            },
        };
        self.warn(
            BACKSLASH_COMMENT,
            bs,
            next_end,
            message,
            vec![remove, comment_out],
        );
    }

    /// The edit that keeps what XPPAUT reads and says it in the text: it
    /// replaces the text from the backslash at `bs` in the comment line that
    /// ends at `eol` through the last swallowed line. Gives the end of the
    /// replaced span, the replacement and the number of lines swallowed.
    ///
    /// The comment line and each swallowed line lose their backslashes. A
    /// swallowed line gets a `# ` unless it is blank or a comment already.
    /// The last swallowed line has no backslash unless the text ends in it,
    /// so it stays out of the span and only its `# ` is inserted.
    fn comment_out_swallowed(&self, bs: usize, eol: usize) -> (usize, String, usize) {
        let lines = self.swallowed_lines(eol);
        let mut replacement = self.text[bs + 1..eol].replace('\\', "");
        let mut prev = eol;
        let mut span_end = eol;
        for (i, &(start, end)) in lines.iter().enumerate() {
            // The terminator that ends the previous line, as written.
            replacement.push_str(&self.text[prev..start]);
            let line = &self.text[start..end];
            let trimmed = line.trim_matches([' ', '\t']);
            if !(trimmed.is_empty() || trimmed.starts_with('#')) {
                replacement.push_str("# ");
            }
            if i + 1 == lines.len() && !line.contains('\\') {
                span_end = start;
            } else {
                replacement.push_str(&line.replace('\\', ""));
                span_end = end;
            }
            prev = end;
        }
        (span_end, replacement, lines.len())
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

        // Only zeros before the `x` (`0x10`, `00x10`), with no dot or exponent.
        let hex = self.bytes[start..i].iter().all(|&b| b == b'0')
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
            let exponent_sign = matches!(b, b'+' | b'-')
                && matches!(self.bytes[i - 1], b'e' | b'E')
                && self.is_digit_at(i + 1);
            if b.is_ascii_alphanumeric() || b == b'_' || self.is_single_dot_at(i) || exponent_sign {
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
    fn semicolon_separates_global_assignments() {
        use TokenKind::*;
        let (tokens, diags) = lex(F, "{x=0;y=1}");
        assert!(diags.is_empty(), "{diags:#?}");
        assert_eq!(
            kinds_of(&tokens),
            vec![
                LBrace,
                name("x"),
                Eq,
                n(0.0),
                Semicolon,
                name("y"),
                Eq,
                n(1.0),
                RBrace
            ]
        );
        assert_eq!(tokens[4].span, Span::new(F, 4, 5));
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
    fn lone_cr_is_a_line_end() {
        assert_eq!(
            kinds("a\rb"),
            vec![name("a"), TokenKind::Newline, name("b")]
        );
        assert_eq!(blanks("a\rb"), vec![false; 4]);
        // The token is the one byte of the CR; a CRLF stays one token.
        let text = "a\rb\r\nc\r\r\nd";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        let spans: Vec<(usize, usize)> =
            tokens.iter().map(|t| (t.span.start, t.span.end)).collect();
        assert_eq!(
            spans,
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 5),
                (5, 6),
                (6, 7),
                (7, 9),
                (9, 10),
                (10, 10)
            ]
        );
        // A backslash before a lone CR continues the line.
        assert_eq!(kinds("1\\\r2"), vec![n(1.0), n(2.0)]);
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
    fn backslash_then_blanks_then_a_line_end_continues_the_line() {
        // XPPAUT 8.2.2 joins the lines when only spaces or tabs sit between
        // the backslash and the line end (probes backslash-blanks,
        // backslash-tab, backslash-mixed-blanks). The CRLF and lone CR line
        // ends in this list have no probe of their own.
        for text in [
            "1+\\ \n2",
            "1+\\  \n2",
            "1+\\\t\n2",
            "1+\\ \t \n2",
            "1+\\ \r\n2",
            "1+\\ \r2",
            "1+\\\n2",
        ] {
            let (tokens, diags) = lex(F, text);
            assert!(diags.is_empty(), "{text:?}: {diags:#?}");
            assert_eq!(
                kinds_of(&tokens),
                vec![n(1.0), TokenKind::Plus, n(2.0)],
                "{text:?}"
            );
            // The joined line leaves a blank before the 2, like a plain
            // backslash-newline.
            assert!(tokens[2].preceded_by_blank, "{text:?}");
        }
        // At the end of the text the backslash joins nothing and is not an
        // error (probe backslash-eof).
        for text in ["1+\\", "1+\\ ", "1+\\\t "] {
            let (tokens, diags) = lex(F, text);
            assert!(diags.is_empty(), "{text:?}: {diags:#?}");
            assert_eq!(kinds_of(&tokens), vec![n(1.0), TokenKind::Plus], "{text:?}");
        }
        // Blanks then a second line end do not continue it twice.
        let (tokens, _) = lex(F, "1\\ \n\n2");
        assert_eq!(kinds_of(&tokens), vec![n(1.0), TokenKind::Newline, n(2.0)]);
    }

    #[test]
    fn comment_stops_at_a_lone_cr() {
        use TokenKind::*;
        let text = "# a\rb\nc";
        let (tokens, _) = lex(F, text);
        assert_eq!(slice(text, &tokens[0]), "# a");
        assert_eq!(
            kinds_of(&tokens),
            vec![Comment, Newline, name("b"), Newline, name("c")]
        );
    }

    /// Applies one fix to `text`.
    fn apply(text: &str, fix: &QuickFix) -> String {
        let s = &fix.edit.span;
        format!(
            "{}{}{}",
            &text[..s.start],
            fix.edit.replacement,
            &text[s.end..]
        )
    }

    /// The titles of the fixes of a diagnostic, preferred first.
    fn titles(d: &Diagnostic) -> Vec<&str> {
        d.fixes.iter().map(|f| f.title.as_str()).collect()
    }

    #[test]
    fn text_after_a_backslash_is_dropped_with_a_warning() {
        // XPPAUT 8.2.2 drops the text after a backslash and the line end and
        // joins the next line (probes backslash-text, backslash-text-crlf).
        for (text, want) in [
            ("1+\\ x\n2", vec![n(1.0), TokenKind::Plus, n(2.0)]),
            ("1+\\5\n2", vec![n(1.0), TokenKind::Plus, n(2.0)]),
            ("1+\\ x\r\n2", vec![n(1.0), TokenKind::Plus, n(2.0)]),
            ("1+\\ x\r2", vec![n(1.0), TokenKind::Plus, n(2.0)]),
            ("1+\\ # c\n2", vec![n(1.0), TokenKind::Plus, n(2.0)]),
            // Text and a second backslash: the first one decides.
            ("1+\\ \\\n2", vec![n(1.0), TokenKind::Plus, n(2.0)]),
            // The next line has its own dropped text: two warnings.
            (
                "1+\\ a\n2\\ b\n3",
                vec![n(1.0), TokenKind::Plus, n(2.0), n(3.0)],
            ),
            // The end of the text after the dropped text.
            ("1+\\ x", vec![n(1.0), TokenKind::Plus]),
            ("1+\\ x\n", vec![n(1.0), TokenKind::Plus]),
        ] {
            let (tokens, diags) = lex(F, text);
            assert_eq!(kinds_of(&tokens), want, "{text:?}");
            assert!(!diags.is_empty(), "{text:?}");
            assert!(
                diags
                    .iter()
                    .all(|d| d.code == BACKSLASH_TEXT && d.severity == Severity::Warning),
                "{text:?}: {diags:#?}"
            );
            // The text and the line end are gone, the next token follows a blank.
            if let Some(next) = tokens
                .get(2)
                .filter(|t| matches!(t.kind, TokenKind::Num(_)))
            {
                assert!(next.preceded_by_blank, "{text:?}");
            }
        }
        // Warnings: one per backslash.
        assert_eq!(lex(F, "1+\\ a\n2\\ b\n3").1.len(), 2);
    }

    #[test]
    fn the_text_warning_quotes_the_text_and_says_what_is_lost() {
        let text = "x'=1+\\ 2*y\nz";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 1);
        let d = &diags[0];
        assert_eq!(&text[d.span.start..d.span.end], "\\ 2*y");
        assert!(d.message.contains("`2*y`"), "{}", d.message);
        assert!(d.message.contains("next line"), "{}", d.message);
        assert!(d.message.contains("does not contain"), "{}", d.message);
        // A long text is cut.
        let long = format!("1+\\ {}\n2", "abcdefghij".repeat(20));
        let (_, diags) = lex(F, &long);
        assert!(diags[0].message.len() < 200, "{}", diags[0].message);
        assert!(diags[0].message.contains("..."), "{}", diags[0].message);
        // A dropped comment is named as a comment.
        let (_, diags) = lex(F, "1+\\ # note\n2");
        assert!(
            diags[0].message.contains("comment `# note`"),
            "{}",
            diags[0].message
        );
        // At the end of the text no line is joined, and the message does not say so.
        let (_, diags) = lex(F, "1+\\ x");
        assert!(
            !diags[0].message.contains("next line"),
            "{}",
            diags[0].message
        );
    }

    #[test]
    fn the_text_warning_has_two_fixes() {
        let text = "x'=a+\\ b\ny=2\n";
        let (_, diags) = lex(F, text);
        let d = &diags[0];
        assert_eq!(
            titles(d),
            vec![
                "Move the backslash to the end of the line",
                "Delete the text after the backslash"
            ]
        );
        // The preferred fix is the one in `fix_it`.
        assert_eq!(d.fix_it.as_ref(), Some(&d.fixes[0].edit));
        // Moving the backslash keeps the text and joins the lines, as the
        // author most likely meant, and leaves no warning.
        let moved = apply(text, &d.fixes[0]);
        assert_eq!(moved, "x'=a+ b \\\ny=2\n");
        let (tokens, diags) = lex(F, &moved);
        assert!(diags.is_empty(), "{diags:#?}");
        assert_eq!(
            kinds_of(&tokens),
            vec![
                name("x"),
                TokenKind::Quote,
                TokenKind::Eq,
                name("a"),
                TokenKind::Plus,
                name("b"),
                name("y"),
                TokenKind::Eq,
                n(2.0),
                TokenKind::Newline
            ]
        );
        // Deleting the text keeps what XPPAUT reads, now without the warning.
        let deleted = apply(text, &d.fixes[1]);
        assert_eq!(deleted, "x'=a+\\\ny=2\n");
        let (tokens, diags) = lex(F, &deleted);
        assert!(diags.is_empty(), "{diags:#?}");
        let (want, _) = lex(F, text);
        assert_eq!(kinds_of(&tokens), kinds_of(&want));
    }

    #[test]
    fn a_dropped_comment_only_has_the_delete_fix() {
        // A backslash moved behind a `#` would sit in a comment.
        for text in ["1+\\ # note\n2", "1+\\ x # note\n2"] {
            let (_, diags) = lex(F, text);
            assert_eq!(
                titles(&diags[0]),
                vec!["Delete the text after the backslash"],
                "{text:?}"
            );
            let fixed = apply(text, &diags[0].fixes[0]);
            let (_, again) = lex(F, &fixed);
            assert!(again.is_empty(), "{fixed:?}: {again:#?}");
        }
    }

    #[test]
    fn dropped_text_with_a_backslash_only_has_the_delete_fix() {
        // The Move fix would put the backslash after text that already holds
        // one, so the fixed line would still carry a backslash in the middle
        // of its text (` b \ \`, `C:\dir \`). Only the delete is offered.
        for text in ["a=1+\\ b \\\nc", "a=1+\\ C:\\dir\nc"] {
            let (_, diags) = lex(F, text);
            assert_eq!(diags.len(), 1, "{text:?}: {diags:#?}");
            assert_eq!(diags[0].code, BACKSLASH_TEXT, "{text:?}");
            assert_eq!(
                titles(&diags[0]),
                vec!["Delete the text after the backslash"],
                "{text:?}"
            );
            let fixed = apply(text, &diags[0].fixes[0]);
            let (_, again) = lex(F, &fixed);
            assert!(again.is_empty(), "{fixed:?}: {again:#?}");
        }
    }

    #[test]
    fn the_move_fix_keeps_the_crlf_and_the_blanks() {
        for text in ["a+\\  b  \r\nc\r\n", "a+\\b\rc\r"] {
            let (_, diags) = lex(F, text);
            let moved = apply(text, &diags[0].fixes[0]);
            let (_, again) = lex(F, &moved);
            assert!(again.is_empty(), "{moved:?}: {again:#?}");
            assert!(moved.contains("b \\"), "{moved:?}");
        }
    }

    #[test]
    fn a_backslash_in_a_comment_joins_the_next_line_to_it() {
        use TokenKind::*;
        // Line 2, so that the exemption of the first line does not apply.
        let text = "x\n# a \\\nb=1\nc";
        let (tokens, diags) = lex(F, text);
        assert_eq!(
            kinds_of(&tokens),
            vec![name("x"), Newline, Comment, Newline, name("c")]
        );
        let comment = tokens.iter().find(|t| t.kind == Comment).unwrap();
        // The comment text stops before the backslash.
        assert_eq!(slice(text, comment), "# a ");
        assert_eq!(diags.len(), 1, "{diags:#?}");
        let d = &diags[0];
        assert_eq!(d.code, BACKSLASH_COMMENT);
        assert_eq!(d.severity, Severity::Warning);
        // The warning starts at the backslash and covers the swallowed line.
        assert_eq!(&text[d.span.start..d.span.end], "\\\nb=1");
        assert!(d.message.contains("`b=1`"), "{}", d.message);
        assert!(d.message.contains("does not contain"), "{}", d.message);
    }

    #[test]
    fn a_backslash_in_the_middle_of_a_comment_drops_the_rest_of_it() {
        let text = "x\n# a \\ b c\nd=1\ne";
        let (tokens, diags) = lex(F, text);
        let comment = tokens
            .iter()
            .find(|t| t.kind == TokenKind::Comment)
            .unwrap();
        assert_eq!(slice(text, comment), "# a ");
        assert_eq!(diags.len(), 1);
        assert_eq!(
            kinds_of(&tokens),
            vec![
                name("x"),
                TokenKind::Newline,
                TokenKind::Comment,
                TokenKind::Newline,
                name("e")
            ]
        );
        assert_eq!(
            titles(&diags[0]),
            vec![
                "Remove the backslash from the comment",
                "Comment out the next line"
            ]
        );
    }

    #[test]
    fn the_comment_warning_has_two_fixes() {
        let text = "x\n# a \\\nb=1\nc\n";
        let (_, diags) = lex(F, text);
        let d = &diags[0];
        assert_eq!(
            titles(d),
            vec![
                "Remove the backslash at the end of the comment",
                "Comment out the next line"
            ]
        );
        assert_eq!(d.fix_it.as_ref(), Some(&d.fixes[0].edit));
        // Removing the backslash keeps the next line as a statement.
        let removed = apply(text, &d.fixes[0]);
        assert_eq!(removed, "x\n# a \nb=1\nc\n");
        let (tokens, again) = lex(F, &removed);
        assert!(again.is_empty(), "{again:#?}");
        assert!(tokens.iter().any(|t| t.kind == name("b")));
        // Commenting the line out says in the text what XPPAUT reads: no
        // warning, and the same tokens as before the fix.
        let commented = apply(text, &d.fixes[1]);
        assert_eq!(commented, "x\n# a \n# b=1\nc\n");
        let (tokens, again) = lex(F, &commented);
        assert!(again.is_empty(), "{again:#?}");
        let (before, _) = lex(F, text);
        let names = |ts: &[Token]| -> Vec<TokenKind> {
            ts.iter()
                .map(|t| t.kind.clone())
                .filter(|k| !matches!(k, TokenKind::Comment | TokenKind::Newline))
                .collect()
        };
        assert_eq!(names(&tokens), names(&before));
        // With text after the backslash, that text stays in the comment.
        let text = "x\n# a \\ b\nc=1\n";
        let (_, diags) = lex(F, text);
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a  b\n# c=1\n");
    }

    #[test]
    fn comment_backslashes_chain_and_skip_blank_and_comment_lines() {
        use TokenKind::*;
        // The swallowed line has a backslash of its own, so it takes the
        // line after it too.
        let text = "x\n# a \\\nb \\\nc\nd";
        let (tokens, diags) = lex(F, text);
        assert_eq!(
            kinds_of(&tokens),
            vec![name("x"), Newline, Comment, Newline, name("d")]
        );
        // Both swallowed lines have content, so there are two warnings.
        assert_eq!(diags.len(), 2, "{diags:#?}");
        assert!(diags.iter().all(|d| d.code == BACKSLASH_COMMENT));
        // A blank or comment-only swallowed line loses nothing: no warning.
        for text in [
            "x\n# a \\\n\nd",
            "x\n# a \\\n# more\nd",
            "x\n# a \\\n   \nd",
        ] {
            let (tokens, diags) = lex(F, text);
            assert!(diags.is_empty(), "{text:?}: {diags:#?}");
            assert_eq!(
                kinds_of(&tokens),
                vec![name("x"), Newline, Comment, Newline, name("d")],
                "{text:?}"
            );
        }
        // At the end of the text nothing is swallowed.
        for text in ["x\n# a \\", "x\n# a \\\n", "x\n# a \\ b"] {
            let (tokens, diags) = lex(F, text);
            assert!(diags.is_empty(), "{text:?}: {diags:#?}");
            assert_eq!(kinds_of(&tokens)[..3], [name("x"), Newline, Comment]);
        }
    }

    #[test]
    fn comment_backslash_with_crlf_and_lone_cr() {
        use TokenKind::*;
        for text in ["x\r\n# a \\\r\nb=1\r\nc", "x\r# a \\\rb=1\rc"] {
            let (tokens, diags) = lex(F, text);
            assert_eq!(
                kinds_of(&tokens),
                vec![name("x"), Newline, Comment, Newline, name("c")],
                "{text:?}"
            );
            assert_eq!(diags.len(), 1, "{text:?}");
            let fixed = apply(text, &diags[0].fixes[1]);
            let (_, again) = lex(F, &fixed);
            assert!(again.is_empty(), "{fixed:?}: {again:#?}");
        }
    }

    /// What the parser gets from `text`: its tokens apart from comments, line
    /// ends and the final `Eof`.
    fn meaning(text: &str) -> Vec<TokenKind> {
        let (tokens, _) = lex(F, text);
        tokens
            .iter()
            .map(|t| t.kind.clone())
            .filter(|k| !matches!(k, TokenKind::Comment | TokenKind::Newline | TokenKind::Eof))
            .collect()
    }

    fn warnings(text: &str) -> usize {
        lex(F, text).1.len()
    }

    #[test]
    fn one_comment_out_covers_every_line_the_comment_swallows() {
        // `b` and `c` are both part of the comment for XPPAUT: `b` follows the
        // backslash, and `b` has a backslash of its own.
        let text = "x\n# a \\\nb \\\nc\nd\n";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 2, "{diags:#?}");
        assert_eq!(
            titles(&diags[0]),
            vec![
                "Remove the backslash at the end of the comment",
                "Comment out the next 2 lines"
            ]
        );
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a \n# b \n# c\nd\n");
        assert_eq!(warnings(&commented), 0, "{commented:?}");
        assert_eq!(meaning(&commented), meaning(text));
        assert_eq!(meaning(text), vec![name("x"), name("d")]);
        // The second warning is about `c` alone.
        assert_eq!(
            titles(&diags[1]),
            vec![
                "Remove the backslash at the end of the comment",
                "Comment out the next line"
            ]
        );
        let commented = apply(text, &diags[1].fixes[1]);
        assert_eq!(commented, "x\n# a \\\nb \n# c\nd\n");
    }

    #[test]
    fn each_fix_of_a_chain_ends_warning_free_or_keeps_the_meaning() {
        // (warnings left, same meaning as before the fix) for every fix of
        // every warning of the chain `# a \`, `b \`, `c`.
        let text = "x\n# a \\\nb \\\nc\nd\n";
        let (_, diags) = lex(F, text);
        let before = (warnings(text), meaning(text));
        assert_eq!(before.0, 2);
        let mut got = Vec::new();
        for d in &diags {
            for fix in &d.fixes {
                let fixed = apply(text, fix);
                got.push((warnings(&fixed), meaning(&fixed) == before.1));
            }
        }
        assert_eq!(
            got,
            vec![
                // Remove the backslash of the comment: `b \` and `c` join into
                // one live statement. No warning, and the meaning changes
                // because the model now has `b c`.
                (0, false),
                // Comment out `b` and `c`: no warning, same meaning.
                (0, true),
                // Remove the backslash of `b`, which is itself swallowed: the
                // first warning stays and `c` becomes live. Fewer warnings and
                // a different meaning, which fits neither rule (owner question
                // in the fix report).
                (1, false),
                // Comment out `c`: one warning fewer, same meaning.
                (1, true),
            ]
        );
    }

    #[test]
    fn the_comment_out_of_a_chain_drops_the_backslashes_after_the_first() {
        // Backslashes in the middle of the comment lines go too, so that a
        // commented line does not swallow the next one again.
        let text = "x\n# a \\ b\nc \\ d\ne\nf\n";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 2, "{diags:#?}");
        assert_eq!(titles(&diags[0])[1], "Comment out the next 2 lines");
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a  b\n# c  d\n# e\nf\n");
        assert_eq!(warnings(&commented), 0, "{commented:?}");
        assert_eq!(meaning(&commented), meaning(text));
        assert_eq!(meaning(text), vec![name("x"), name("f")]);
    }

    #[test]
    fn a_second_backslash_in_the_comment_line_goes_with_the_first() {
        let text = "x\n# a \\ b \\\nc\nd\n";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 1, "{diags:#?}");
        assert_eq!(
            titles(&diags[0]),
            vec![
                "Remove the backslashes from the comment",
                "Comment out the next line"
            ]
        );
        let removed = apply(text, &diags[0].fixes[0]);
        assert_eq!(removed, "x\n# a  b \nc\nd\n");
        assert_eq!(warnings(&removed), 0, "{removed:?}");
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a  b \n# c\nd\n");
        assert_eq!(warnings(&commented), 0, "{commented:?}");
        assert_eq!(meaning(&commented), meaning(text));
    }

    #[test]
    fn the_chain_comment_out_keeps_each_line_end() {
        for (text, fixed) in [
            (
                "x\r\n# a \\\r\nb \\\r\nc\r\nd\r\n",
                "x\r\n# a \r\n# b \r\n# c\r\nd\r\n",
            ),
            ("x\r# a \\\rb \\\rc\rd\r", "x\r# a \r# b \r# c\rd\r"),
        ] {
            let (_, diags) = lex(F, text);
            assert_eq!(diags.len(), 2, "{text:?}: {diags:#?}");
            let commented = apply(text, &diags[0].fixes[1]);
            assert_eq!(commented, fixed);
            assert_eq!(warnings(&commented), 0, "{commented:?}");
            assert_eq!(meaning(&commented), meaning(text), "{text:?}");
        }
    }

    #[test]
    fn the_chain_comment_out_leaves_blank_and_comment_lines_alone() {
        // `b \` is swallowed and takes the blank line after it. The blank line
        // gets no `#`, and `c` stays live.
        let text = "x\n# a \\\nb \\\n\nc\n";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 1, "{diags:#?}");
        assert_eq!(titles(&diags[0])[1], "Comment out the next 2 lines");
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a \n# b \n\nc\n");
        assert_eq!(warnings(&commented), 0, "{commented:?}");
        assert_eq!(meaning(&commented), meaning(text));
        assert_eq!(meaning(text), vec![name("x"), name("c")]);
        // A comment-only line in the chain keeps its `#`, loses its backslash
        // and passes the chain on to `c`.
        let text = "x\n# a \\\n# b \\\nc\nd\n";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 1, "{diags:#?}");
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a \\\n# b \n# c\nd\n");
        assert_eq!(warnings(&commented), 0, "{commented:?}");
        assert_eq!(meaning(&commented), meaning(text));
    }

    #[test]
    fn a_chain_that_runs_into_the_end_of_the_text_is_commented_out_to_its_end() {
        // The last line has a backslash of its own and nothing after it, so
        // it is part of the edit and loses the backslash.
        let text = "x\n# a \\\nb \\";
        let (_, diags) = lex(F, text);
        assert_eq!(diags.len(), 1, "{diags:#?}");
        assert_eq!(titles(&diags[0])[1], "Comment out the next line");
        let commented = apply(text, &diags[0].fixes[1]);
        assert_eq!(commented, "x\n# a \n# b ");
        assert_eq!(warnings(&commented), 0, "{commented:?}");
        assert_eq!(meaning(&commented), meaning(text));
    }

    #[test]
    fn the_first_line_of_a_main_file_keeps_a_backslash_in_a_comment() {
        use TokenKind::*;
        // XPPAUT 8.2.2 does not join after a comment on the first line of the
        // main file (probe backslash-comment-line1), but does in an included
        // file (probe backslash-comment-inc).
        let text = "# a \\\nb=1\n";
        let (tokens, diags) = lex_file(F, text, true);
        assert!(diags.is_empty(), "{diags:#?}");
        assert_eq!(slice(text, &tokens[0]), "# a \\");
        assert_eq!(
            kinds_of(&tokens),
            vec![Comment, Newline, name("b"), Eq, n(1.0), Newline]
        );
        let (tokens, diags) = lex_file(F, text, false);
        assert_eq!(diags.len(), 1, "{diags:#?}");
        assert_eq!(slice(text, &tokens[0]), "# a ");
        assert_eq!(kinds_of(&tokens), vec![Comment, Newline]);
        // A trailing comment on the first line is exempt as well.
        let (_, diags) = lex_file(F, "a=1 # c \\\nb=2\n", true);
        assert!(diags.is_empty(), "{diags:#?}");
        // The second line is not.
        let (_, diags) = lex_file(F, "\n# c \\\nb=2\n", true);
        assert_eq!(diags.len(), 1, "{diags:#?}");
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
            ("00x10", "00x10"),
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
    fn malformed_numbers_swallow_a_signed_exponent() {
        for text in [
            "1.2.3e-4", "1_000e+3", "1e2.5e+3", "1_0e+5", "1.2.3E-4", "0x1e+3",
        ] {
            let (tokens, d) = one_error(text);
            assert_eq!(d.code, MALFORMED_NUMBER, "{text}");
            assert_eq!(d.span, Span::new(F, 0, text.len()), "{text}");
            assert_eq!(kinds_of(&tokens), vec![TokenKind::Unknown], "{text}");
            assert_eq!(tokens[0].span, d.span, "{text}");
        }
        // A sign that is not followed by a digit stays outside the run.
        let (tokens, d) = one_error("1_0e+x");
        assert_eq!(d.span, Span::new(F, 0, 4));
        assert_eq!(
            kinds_of(&tokens),
            vec![TokenKind::Unknown, TokenKind::Plus, name("x")]
        );
        // A sign after something other than e or E stays outside the run.
        let (tokens, _) = lex(F, "1_0x+5");
        assert_eq!(
            kinds_of(&tokens),
            vec![TokenKind::Unknown, TokenKind::Plus, n(5.0)]
        );
    }

    #[test]
    fn bare_malformed_number_cases_have_exact_tokens() {
        for text in ["1e2.", "00x10", "0x10", "1.2.3"] {
            let (tokens, d) = one_error(text);
            assert_eq!(d.code, MALFORMED_NUMBER, "{text}");
            assert_eq!(kinds_of(&tokens), vec![TokenKind::Unknown], "{text}");
            assert_eq!(d.span, Span::new(F, 0, text.len()), "{text}");
        }
    }

    #[test]
    fn near_misses_are_not_malformed() {
        use TokenKind::*;
        assert_eq!(kinds("0xy"), vec![n(0.0), name("xy")]);
        assert_eq!(kinds("10x"), vec![n(10.0), name("x")]);
        assert_eq!(kinds("5.x"), vec![n(5.0), name("x")]);
        assert_eq!(kinds("1.5.."), vec![n(1.5), DotDot]);
        // Not a hex form: the integer part has a dot.
        assert_eq!(kinds("0.0x10"), vec![n(0.0), name("x10")]);
        // A number, then `..`, then `.5`.
        assert_eq!(kinds("1...5"), vec![n(1.0), DotDot, n(0.5)]);
    }

    #[test]
    fn number_overflow_is_an_error_but_keeps_the_token() {
        let (tokens, d) = one_error("1e999");
        assert_eq!(d.code, NUMBER_TOO_LARGE);
        assert_eq!(d.severity, Severity::Error);
        assert_eq!(d.span, Span::new(F, 0, 5));
        assert_eq!(tokens[0].kind, n(f64::INFINITY));
        assert_eq!(tokens[0].span, d.span);
        // Underflow is not an error.
        assert_eq!(kinds("1e-999"), vec![n(0.0)]);
        // The largest finite value is fine.
        assert_eq!(kinds("1.7976931348623157e308"), vec![n(f64::MAX)]);
    }

    #[test]
    fn stray_characters() {
        for (text, bad) in [
            ("$", "$"),
            ("a ? b", "?"),
            (".", "."),
            ("é", "é"),
            ("x = Ω", "Ω"),
            (":", ":"),
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
    fn leading_byte_order_mark_is_skipped() {
        let text = "\u{feff}x=1";
        let (tokens, diags) = lex(F, text);
        assert!(diags.is_empty());
        assert_eq!(kinds_of(&tokens), vec![name("x"), TokenKind::Eq, n(1.0)]);
        assert_eq!(tokens[0].span, Span::new(F, 3, 4));
        assert!(!tokens[0].preceded_by_blank);
        let (tokens, diags) = lex(F, "\u{feff}");
        assert!(diags.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].span, Span::new(F, 3, 3));
    }

    #[test]
    fn byte_order_mark_elsewhere_is_a_stray_character() {
        let (tokens, d) = one_error("x\u{feff}");
        assert_eq!(d.code, STRAY_CHARACTER);
        assert_eq!(d.span, Span::new(F, 1, 4));
        assert_eq!(kinds_of(&tokens), vec![name("x"), TokenKind::Unknown]);
        // Only the first one at offset 0 is skipped.
        let (tokens, d) = one_error("\u{feff}\u{feff}x");
        assert_eq!(d.span, Span::new(F, 3, 6));
        assert_eq!(kinds_of(&tokens), vec![TokenKind::Unknown, name("x")]);
        let (_, d) = one_error("\nx\n\u{feff}");
        assert_eq!(d.code, STRAY_CHARACTER);
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
        for (code, summary) in [
            (MALFORMED_NUMBER, "malformed number"),
            (
                NUMBER_TOO_LARGE,
                "number too large for a double precision value",
            ),
            (STRAY_CHARACTER, "stray character"),
        ] {
            let entry = CODES.iter().find(|(c, _, _)| *c == code).unwrap();
            assert_eq!(entry.1, Severity::Error);
            assert_eq!(entry.2, summary);
        }
    }

    /// Checks the span rules for every token and diagnostic of `text`: inside
    /// the text, on char boundaries, in order, not overlapping, one final Eof.
    fn check_spans(text: &str) {
        let (tokens, diags) = lex(F, text);
        assert!(!tokens.is_empty(), "{text:?}");
        assert_eq!(tokens.last().unwrap().kind, TokenKind::Eof, "{text:?}");
        let eofs = tokens.iter().filter(|t| t.kind == TokenKind::Eof).count();
        assert_eq!(eofs, 1, "{text:?}");
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
        for pair in tokens.windows(2) {
            assert!(pair[0].span.end <= pair[1].span.start, "{text:?}");
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
            check_spans(text);
        }
    }

    #[test]
    fn lexer_keeps_span_rules_on_every_short_string() {
        // All strings of length <= 4 over a small alphabet that hits each rule.
        let alphabet = [
            'a', 'e', '0', '1', '.', '_', 'x', '+', '-', '*', '=', '<', '!', '\\', '\n', '\r', '#',
            ' ', 'é', '\u{feff}',
        ];
        let mut frontier = vec![String::new()];
        check_spans("");
        for _ in 0..4 {
            let mut next = Vec::new();
            for s in &frontier {
                for c in alphabet {
                    let mut t = s.clone();
                    t.push(c);
                    check_spans(&t);
                    next.push(t);
                }
            }
            frontier = next;
        }
    }
}
