//! Source files and byte-offset spans.

/// Identifies a file inside a [`SourceMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

/// A half-open byte range `start..end` inside one source file.
///
/// Build spans with [`Span::new`] or [`Span::to`], which assert
/// `start <= end`, rather than with a struct literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub file: FileId,
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Creates a span. Panics if `start > end`.
    pub fn new(file: FileId, start: usize, end: usize) -> Span {
        assert!(start <= end, "span start must not exceed end");
        Span { file, start, end }
    }

    /// Length in bytes. A reversed span has length 0.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// True when the span covers no bytes.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// True when `other` is a well-formed span in the same file that lies
    /// inside this span.
    pub fn contains(&self, other: Span) -> bool {
        self.file == other.file
            && other.start <= other.end
            && self.start <= other.start
            && other.end <= self.end
    }

    /// The span from the start of `self` to the end of `other`. Panics if the
    /// files differ or `other` ends before `self` starts.
    pub fn to(self, other: Span) -> Span {
        assert!(self.file == other.file, "spans must be in the same file");
        Span::new(self.file, self.start, other.end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceFile {
    name: String,
    text: String,
    /// Byte offset of the first character of each line.
    line_starts: Vec<usize>,
}

impl SourceFile {
    /// Byte range of line `index` (0-based) without its terminator.
    fn line_range(&self, index: usize) -> (usize, usize) {
        let start = self.line_starts[index];
        let rest = &self.text[start..];
        // The line ends at the first LF or CR: a CR is a terminator whether an
        // LF follows it or not.
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        (start, start + end)
    }
}

/// Holds the text of every file seen by one parse, addressed by [`FileId`].
///
/// A line ends at `\n`, `\r\n` or a `\r` that no `\n` follows. Columns count
/// Unicode scalar values and start at 1.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceMap {
    files: Vec<SourceFile>,
}

impl SourceMap {
    pub fn new() -> SourceMap {
        SourceMap::default()
    }

    /// Stores a file and returns its id.
    pub fn add(&mut self, name: impl Into<String>, text: impl Into<String>) -> FileId {
        let text = text.into();
        let mut line_starts = vec![0];
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            let ends_line = b == b'\n' || (b == b'\r' && bytes.get(i + 1) != Some(&b'\n'));
            if ends_line {
                line_starts.push(i + 1);
            }
        }
        // Invariant: the map holds fewer than 2^32 files; a load adds at most
        // MAX_INCLUDES + 1 files, so the count fits in a u32.
        let id = FileId(u32::try_from(self.files.len()).expect("too many source files"));
        self.files.push(SourceFile {
            name: name.into(),
            text,
            line_starts,
        });
        id
    }

    fn file(&self, id: FileId) -> &SourceFile {
        self.files
            // Invariant: a FileId only comes from `add` on this map, so it is in range.
            // A foreign id is a caller bug; pc-ffi catches the panic.
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("unknown file id {}", id.0))
    }

    /// The name given to [`SourceMap::add`]. Panics on an unknown id.
    pub fn name(&self, file: FileId) -> &str {
        &self.file(file).name
    }

    /// The full text of a file. Panics on an unknown id.
    pub fn text(&self, file: FileId) -> &str {
        &self.file(file).text
    }

    /// Line and column (both 1-based) of the start of `span`.
    ///
    /// An offset equal to the text length (end of file) is valid. An offset
    /// inside a line terminator maps to the column just past the line text.
    /// Panics if the file is unknown, the offset is past the end, or it is
    /// not on a char boundary.
    pub fn line_col(&self, span: Span) -> (usize, usize) {
        let file = self.file(span.file);
        let offset = span.start;
        // Invariant: spans come from the lexer, which cuts at char boundaries inside
        // the file text; a violation is a caller bug that pc-ffi catches.
        assert!(
            offset <= file.text.len(),
            "offset {offset} is past the end of {}",
            file.name
        );
        assert!(
            file.text.is_char_boundary(offset),
            "offset {offset} is not on a char boundary in {}",
            file.name
        );
        let line = file.line_starts.partition_point(|&s| s <= offset);
        let (line_start, line_end) = file.line_range(line - 1);
        let col = file.text[line_start..offset.min(line_end)].chars().count() + 1;
        (line, col)
    }

    /// Text of a 1-based line without its `\n`, `\r\n` or `\r` terminator.
    /// Returns `None` when the line does not exist. Panics on an unknown
    /// file id.
    pub fn line_text(&self, file: FileId, line: usize) -> Option<&str> {
        let file = self.file(file);
        let index = line.checked_sub(1)?;
        if index >= file.line_starts.len() {
            return None;
        }
        let (start, end) = file.line_range(index);
        Some(&file.text[start..end])
    }

    /// The number of files stored so far.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_with(text: &str) -> (SourceMap, FileId) {
        let mut map = SourceMap::new();
        let id = map.add("t.ode", text);
        (map, id)
    }

    fn at(map: &SourceMap, file: FileId, offset: usize) -> (usize, usize) {
        map.line_col(Span::new(file, offset, offset))
    }

    #[test]
    fn position_at_file_start() {
        let (map, f) = map_with("abc\ndef");
        assert_eq!(at(&map, f, 0), (1, 1));
        assert_eq!(at(&map, f, 2), (1, 3));
    }

    #[test]
    fn position_after_newline() {
        let (map, f) = map_with("abc\ndef\n");
        assert_eq!(at(&map, f, 4), (2, 1));
        assert_eq!(at(&map, f, 6), (2, 3));
        assert_eq!(at(&map, f, 8), (3, 1));
    }

    #[test]
    fn position_after_crlf() {
        let (map, f) = map_with("ab\r\ncd\r\n");
        assert_eq!(at(&map, f, 2), (1, 3));
        assert_eq!(at(&map, f, 4), (2, 1));
        assert_eq!(at(&map, f, 5), (2, 2));
        assert_eq!(at(&map, f, 8), (3, 1));
    }

    #[test]
    fn columns_count_chars_not_bytes() {
        // 'é' is 2 bytes, '漢' is 3 bytes, '😀' is 4 bytes.
        let (map, f) = map_with("é漢😀x\ny");
        assert_eq!(at(&map, f, 0), (1, 1));
        assert_eq!(at(&map, f, 2), (1, 2));
        assert_eq!(at(&map, f, 5), (1, 3));
        assert_eq!(at(&map, f, 9), (1, 4));
        assert_eq!(at(&map, f, 11), (2, 1));
    }

    #[test]
    fn position_at_end_of_file() {
        let (map, f) = map_with("abc");
        assert_eq!(at(&map, f, 3), (1, 4));
        let (map, f) = map_with("abc\n");
        assert_eq!(at(&map, f, 4), (2, 1));
        let (map, f) = map_with("");
        assert_eq!(at(&map, f, 0), (1, 1));
    }

    #[test]
    fn offsets_inside_a_terminator_map_to_end_of_line() {
        let (map, f) = map_with("ab\r\ncd");
        assert_eq!(at(&map, f, 2), (1, 3));
        assert_eq!(at(&map, f, 3), (1, 3));
        assert_eq!(at(&map, f, 4), (2, 1));
        let (lf, g) = map_with("ab\ncd");
        assert_eq!(at(&lf, g, 2), (1, 3));
        assert_eq!(at(&lf, g, 3), (2, 1));
    }

    #[test]
    fn lone_cr_is_a_line_end() {
        // Lines: `a` (ended by a lone CR), an empty line (ended by CRLF), `b`.
        let (map, f) = map_with("a\r\r\nb");
        assert_eq!(map.line_text(f, 1), Some("a"));
        assert_eq!(map.line_text(f, 2), Some(""));
        assert_eq!(map.line_text(f, 3), Some("b"));
        assert_eq!(map.line_text(f, 4), None);
        assert_eq!(at(&map, f, 0), (1, 1));
        assert_eq!(at(&map, f, 1), (1, 2));
        assert_eq!(at(&map, f, 2), (2, 1));
        assert_eq!(at(&map, f, 3), (2, 1));
        assert_eq!(at(&map, f, 4), (3, 1));
        // CR only: three lines, and a CR at the end of the text gives an
        // empty last line.
        let (map, f) = map_with("ab\rcd\ref\r");
        assert_eq!(map.line_text(f, 1), Some("ab"));
        assert_eq!(map.line_text(f, 2), Some("cd"));
        assert_eq!(map.line_text(f, 3), Some("ef"));
        assert_eq!(map.line_text(f, 4), Some(""));
        assert_eq!(at(&map, f, 3), (2, 1));
        assert_eq!(at(&map, f, 4), (2, 2));
        assert_eq!(at(&map, f, 9), (4, 1));
    }

    #[test]
    fn line_col_uses_span_start() {
        let (map, f) = map_with("ab\ncd");
        assert_eq!(map.line_col(Span::new(f, 4, 5)), (2, 2));
    }

    #[test]
    #[should_panic(expected = "char boundary")]
    fn offset_inside_a_char_panics() {
        let (map, f) = map_with("é");
        at(&map, f, 1);
    }

    #[test]
    #[should_panic(expected = "past the end")]
    fn offset_past_end_panics() {
        let (map, f) = map_with("abc");
        at(&map, f, 4);
    }

    #[test]
    fn line_text_strips_terminators() {
        let (map, f) = map_with("one\r\ntwo\nthree");
        assert_eq!(map.line_text(f, 1), Some("one"));
        assert_eq!(map.line_text(f, 2), Some("two"));
        assert_eq!(map.line_text(f, 3), Some("three"));
        assert_eq!(map.line_text(f, 4), None);
        assert_eq!(map.line_text(f, 0), None);
    }

    #[test]
    fn line_text_edge_cases() {
        // Empty last line after a final newline.
        let (map, f) = map_with("a\n");
        assert_eq!(map.line_text(f, 2), Some(""));
        // A lone '\r' that is not before '\n' ends the line.
        let (map, f) = map_with("a\rb\nc\r");
        assert_eq!(map.line_text(f, 1), Some("a"));
        assert_eq!(map.line_text(f, 2), Some("b"));
        assert_eq!(map.line_text(f, 3), Some("c"));
        assert_eq!(map.line_text(f, 4), Some(""));
        // Empty file has one empty line.
        let (map, f) = map_with("");
        assert_eq!(map.line_text(f, 1), Some(""));
    }

    #[test]
    fn files_are_kept_by_id() {
        let mut map = SourceMap::new();
        assert_eq!(map.file_count(), 0);
        let a = map.add("a.ode", "x");
        let b = map.add(String::from("b.ode"), String::from("yy"));
        assert_ne!(a, b);
        assert_eq!(map.name(a), "a.ode");
        assert_eq!(map.text(b), "yy");
        assert_eq!(map.line_col(Span::new(b, 1, 2)), (1, 2));
        assert_eq!(map.file_count(), 2);
    }

    #[test]
    fn span_containment() {
        let f = FileId(0);
        let outer = Span::new(f, 2, 10);
        assert!(outer.contains(outer));
        assert!(outer.contains(Span::new(f, 2, 5)));
        assert!(outer.contains(Span::new(f, 5, 10)));
        assert!(outer.contains(Span::new(f, 4, 4)));
        assert!(!outer.contains(Span::new(f, 1, 5)));
        assert!(!outer.contains(Span::new(f, 5, 11)));
        assert!(!outer.contains(Span::new(FileId(1), 3, 4)));
    }

    #[test]
    fn contains_rejects_reversed_other() {
        let f = FileId(0);
        let outer = Span::new(f, 2, 10);
        let reversed = Span {
            file: f,
            start: 6,
            end: 4,
        };
        assert!(!outer.contains(reversed));
    }

    #[test]
    fn len_of_reversed_span_is_zero() {
        let reversed = Span {
            file: FileId(0),
            start: 6,
            end: 4,
        };
        assert_eq!(reversed.len(), 0);
    }

    #[test]
    fn to_merges_spans() {
        let f = FileId(0);
        assert_eq!(
            Span::new(f, 2, 4).to(Span::new(f, 6, 9)),
            Span::new(f, 2, 9)
        );
        assert_eq!(
            Span::new(f, 2, 7).to(Span::new(f, 3, 5)),
            Span::new(f, 2, 5)
        );
        assert_eq!(
            Span::new(f, 2, 4).to(Span::new(f, 4, 4)),
            Span::new(f, 2, 4)
        );
    }

    #[test]
    #[should_panic(expected = "same file")]
    fn to_rejects_other_file() {
        let _ = Span::new(FileId(0), 0, 1).to(Span::new(FileId(1), 2, 3));
    }

    #[test]
    #[should_panic(expected = "start must not exceed end")]
    fn to_rejects_end_before_start() {
        let _ = Span::new(FileId(0), 5, 6).to(Span::new(FileId(0), 1, 2));
    }

    #[test]
    fn span_len() {
        let s = Span::new(FileId(0), 3, 7);
        assert_eq!(s.len(), 4);
        assert!(!s.is_empty());
        assert!(Span::new(FileId(0), 3, 3).is_empty());
    }

    #[test]
    #[should_panic(expected = "start must not exceed end")]
    fn reversed_span_panics() {
        let _ = Span::new(FileId(0), 5, 4);
    }
}
