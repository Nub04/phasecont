//! Positions: byte offsets of `pc-ode` against LSP lines and UTF-16 columns.
//!
//! `pc-ode` spans are byte offsets into the text. LSP positions are a
//! 0-based line and a 0-based column counted in UTF-16 code units. A line ends
//! at `\n`, `\r\n` or a lone `\r`, which is what both sides use.
//!
//! Neither direction panics. An offset past the end of the text, or one inside
//! a multi-byte character, is moved to the nearest valid place before the
//! conversion. A position past the end of its line, or past the last line,
//! maps to the end of that line or of the text.

use lsp_types::{Position, Range};

/// Line starts of one text, for converting offsets and positions.
///
/// Every conversion takes O(log n) time, however long the lines are. A model
/// can be one line of a megabyte, with thousands of diagnostics on it.
#[derive(Debug, Clone)]
pub struct LineIndex<'t> {
    text: &'t str,
    /// Byte offset of the first character of each line.
    starts: Vec<usize>,
    /// One entry per character outside ASCII: its byte offset and the number
    /// of bytes beyond one per UTF-16 unit that all such characters up to and
    /// including it take.
    wide: Vec<(usize, usize)>,
}

impl<'t> LineIndex<'t> {
    pub fn new(text: &'t str) -> LineIndex<'t> {
        let bytes = text.as_bytes();
        let mut starts = vec![0];
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\n' || (b == b'\r' && bytes.get(i + 1) != Some(&b'\n')) {
                starts.push(i + 1);
            }
        }
        let mut wide = Vec::new();
        let mut extra = 0;
        for (i, c) in text.char_indices().filter(|(_, c)| !c.is_ascii()) {
            extra += c.len_utf8() - c.len_utf16();
            wide.push((i, extra));
        }
        LineIndex { text, starts, wide }
    }

    /// The number of lines. An empty text has one.
    pub fn line_count(&self) -> usize {
        self.starts.len()
    }

    /// Byte range of line `line` (0-based) without its terminator.
    fn line_range(&self, line: usize) -> (usize, usize) {
        let start = self.starts[line];
        let Some(&next) = self.starts.get(line + 1) else {
            return (start, self.text.len());
        };
        let bytes = self.text.as_bytes();
        let crlf = bytes[next - 1] == b'\n' && next >= start + 2 && bytes[next - 2] == b'\r';
        (start, if crlf { next - 2 } else { next - 1 })
    }

    /// `offset` clamped to the text and moved back to a character start.
    fn clamp(&self, offset: usize) -> usize {
        let mut offset = offset.min(self.text.len());
        while !self.text.is_char_boundary(offset) {
            offset -= 1;
        }
        offset
    }

    /// Bytes beyond one per UTF-16 unit in the characters before `offset`.
    fn extra_before(&self, offset: usize) -> usize {
        let n = self.wide.partition_point(|&(o, _)| o < offset);
        if n == 0 { 0 } else { self.wide[n - 1].1 }
    }

    /// UTF-16 units from `start` to `offset`, both on character starts.
    fn units(&self, start: usize, offset: usize) -> usize {
        (offset - start) - (self.extra_before(offset) - self.extra_before(start))
    }

    /// The LSP position of a byte offset. An offset inside a line terminator
    /// maps to the end of the line text.
    pub fn position(&self, offset: usize) -> Position {
        let offset = self.clamp(offset);
        let line = self.starts.partition_point(|&s| s <= offset) - 1;
        let (start, end) = self.line_range(line);
        Position::new(to_u32(line), to_u32(self.units(start, offset.min(end))))
    }

    /// The LSP range of the bytes `start..end`. A reversed pair is read as an
    /// empty range at `start`.
    pub fn range(&self, start: usize, end: usize) -> Range {
        let from = self.position(start);
        let to = if end < start {
            from
        } else {
            self.position(end)
        };
        Range::new(from, to)
    }

    /// The byte offset of an LSP position. A column in the middle of a
    /// surrogate pair maps to the start of that character.
    pub fn offset(&self, pos: Position) -> usize {
        let Some(line) = usize::try_from(pos.line)
            .ok()
            .filter(|&l| l < self.starts.len())
        else {
            return self.text.len();
        };
        let (start, end) = self.line_range(line);
        let target = pos.character as usize;
        if target >= self.units(start, end) {
            return end;
        }
        // `lo` and `hi` stay on character starts, with `lo` at or before the
        // target column and `hi` after it.
        let (mut lo, mut hi) = (start, end);
        while hi - lo > 1 {
            let mut mid = lo + (hi - lo) / 2;
            while !self.text.is_char_boundary(mid) {
                mid -= 1;
            }
            if mid == lo {
                mid = lo + 1;
                while !self.text.is_char_boundary(mid) {
                    mid += 1;
                }
                if mid >= hi {
                    break;
                }
            }
            if self.units(start, mid) <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

fn to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests;
