//! Reads the imported DLL names of a 64-bit Windows PE file.
//!
//! The reader is pure: it takes the bytes of the file and returns the names
//! from the import directory and from the delay import directory. It follows
//! the Microsoft PE format description. Every read is bounds checked and every
//! loop has a cap, so no input makes it panic or run forever.

#![deny(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::arithmetic_side_effects
)]

use std::fmt;

/// Descriptors read from one directory before the reader gives up.
const MAX_DESCRIPTORS: usize = 4096;
/// Bytes of a DLL name, without the NUL.
const MAX_NAME: usize = 256;
/// Size of an import directory entry.
const IMPORT_DESC: usize = 20;
/// Size of a delay import directory entry.
const DELAY_DESC: usize = 32;
/// Size of a section header.
const SECTION_HDR: usize = 40;
/// Optional header magic of a PE32+ (64-bit) file.
const MAGIC_PE32_PLUS: u16 = 0x20b;
/// Index of the import directory in the data directories.
const DIR_IMPORT: usize = 1;
/// Index of the delay import directory in the data directories.
const DIR_DELAY: usize = 13;

/// The DLLs a file imports, in table order and without repeats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imports {
    pub normal: Vec<String>,
    pub delayed: Vec<String>,
}

/// Why a file is not read as a 64-bit PE image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeError {
    /// The file ends inside the named structure, or an offset overflows.
    Truncated(&'static str),
    /// No `MZ` at the start.
    NotDos,
    /// No `PE\0\0` at `e_lfanew`.
    NotPe,
    /// The optional header magic is not PE32+.
    Not64Bit(u16),
    /// An RVA lies outside every section, or past the raw data of its section.
    RvaUnmapped { what: &'static str, rva: u64 },
    /// A delay import name is a VA below the image base.
    VaBelowBase(&'static str),
    /// A directory has more than the allowed number of descriptors.
    TooManyDescriptors(&'static str),
    /// A DLL name has no NUL within the allowed length or before the end of the file.
    NameNotTerminated,
    /// A DLL name is longer than the allowed length.
    NameTooLong,
    /// A DLL name has a byte that is not printable ASCII.
    NameNotPrintable,
    /// A DLL name is empty.
    EmptyName,
}

impl fmt::Display for PeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeError::Truncated(what) => write!(f, "truncated file: cannot read {what}"),
            PeError::NotDos => write!(f, "not a PE file: no MZ header"),
            PeError::NotPe => write!(f, "not a PE file: no PE signature at e_lfanew"),
            PeError::Not64Bit(magic) => {
                write!(f, "not a 64-bit PE file (optional header magic {magic:#x})")
            }
            PeError::RvaUnmapped { what, rva } => {
                write!(f, "{what} at RVA {rva:#x} is outside every section")
            }
            PeError::VaBelowBase(what) => write!(f, "{what} is a VA below the image base"),
            PeError::TooManyDescriptors(what) => {
                write!(f, "{what} has more than {MAX_DESCRIPTORS} entries")
            }
            PeError::NameNotTerminated => write!(f, "a DLL name has no terminating NUL"),
            PeError::NameTooLong => write!(f, "a DLL name is longer than {MAX_NAME} bytes"),
            PeError::NameNotPrintable => write!(f, "a DLL name is not printable ASCII"),
            PeError::EmptyName => write!(f, "a DLL name is empty"),
        }
    }
}

impl std::error::Error for PeError {}

fn u16_at(b: &[u8], off: usize) -> Option<u16> {
    let end = off.checked_add(2)?;
    Some(u16::from_le_bytes(b.get(off..end)?.try_into().ok()?))
}

fn u32_at(b: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    Some(u32::from_le_bytes(b.get(off..end)?.try_into().ok()?))
}

fn u64_at(b: &[u8], off: usize) -> Option<u64> {
    let end = off.checked_add(8)?;
    Some(u64::from_le_bytes(b.get(off..end)?.try_into().ok()?))
}

/// `base + step * count`, or `None` on overflow.
fn offset(base: usize, step: usize, count: usize) -> Option<usize> {
    base.checked_add(step.checked_mul(count)?)
}

struct Section {
    virtual_size: u32,
    virtual_address: u32,
    raw_size: u32,
    raw_ptr: u32,
}

impl Section {
    fn read(b: &[u8], off: usize) -> Option<Section> {
        Some(Section {
            virtual_size: u32_at(b, off.checked_add(8)?)?,
            virtual_address: u32_at(b, off.checked_add(12)?)?,
            raw_size: u32_at(b, off.checked_add(16)?)?,
            raw_ptr: u32_at(b, off.checked_add(20)?)?,
        })
    }

    /// The file offset of `rva` if this section holds it in its raw data.
    /// `Some(None)` means the section covers `rva` in memory but not in the file.
    fn locate(&self, rva: u32) -> Option<Option<usize>> {
        let span = self.virtual_size.max(self.raw_size);
        let end = self.virtual_address.checked_add(span)?;
        if rva < self.virtual_address || rva >= end {
            return None;
        }
        let delta = rva.checked_sub(self.virtual_address)?;
        if delta >= self.raw_size {
            return Some(None);
        }
        let off = usize::try_from(self.raw_ptr.checked_add(delta)?).ok();
        Some(off)
    }
}

/// The parts of the headers the import readers need.
struct Image<'a> {
    bytes: &'a [u8],
    base: u64,
    sections: Vec<Section>,
}

impl Image<'_> {
    fn file_offset(&self, rva: u64, what: &'static str) -> Result<usize, PeError> {
        let unmapped = PeError::RvaUnmapped { what, rva };
        let rva32 = u32::try_from(rva).map_err(|_| unmapped.clone())?;
        for sec in &self.sections {
            if let Some(found) = sec.locate(rva32) {
                return found.ok_or(unmapped);
            }
        }
        Err(unmapped)
    }

    /// The NUL-terminated name at `rva`.
    fn name_at(&self, rva: u64, what: &'static str) -> Result<String, PeError> {
        let start = self.file_offset(rva, what)?;
        let mut name = String::new();
        for i in 0..=MAX_NAME {
            let at = start.checked_add(i).ok_or(PeError::NameNotTerminated)?;
            let byte = *self.bytes.get(at).ok_or(PeError::NameNotTerminated)?;
            if byte == 0 {
                return if name.is_empty() {
                    Err(PeError::EmptyName)
                } else {
                    Ok(name)
                };
            }
            if !(0x20..=0x7e).contains(&byte) {
                return Err(PeError::NameNotPrintable);
            }
            if i == MAX_NAME {
                return Err(PeError::NameTooLong);
            }
            name.push(char::from(byte));
        }
        Err(PeError::NameTooLong)
    }

    /// Names from a directory of `size`-byte descriptors that ends with an
    /// all-zero one. `name_rva` turns a descriptor into the RVA of its name.
    fn names(
        &self,
        dir_rva: u32,
        size: usize,
        what: &'static str,
        name_rva: impl Fn(&[u8]) -> Result<u64, PeError>,
    ) -> Result<Vec<String>, PeError> {
        let mut out: Vec<String> = Vec::new();
        for i in 0..MAX_DESCRIPTORS {
            let step = u64::try_from(size.checked_mul(i).ok_or(PeError::Truncated(what))?)
                .map_err(|_| PeError::Truncated(what))?;
            let rva = u64::from(dir_rva)
                .checked_add(step)
                .ok_or(PeError::Truncated(what))?;
            let off = self.file_offset(rva, what)?;
            let end = off.checked_add(size).ok_or(PeError::Truncated(what))?;
            let desc = self.bytes.get(off..end).ok_or(PeError::Truncated(what))?;
            if desc.iter().all(|&b| b == 0) {
                return Ok(out);
            }
            let name = self.name_at(name_rva(desc)?, what)?;
            if !out.iter().any(|seen| seen.eq_ignore_ascii_case(&name)) {
                out.push(name);
            }
        }
        Err(PeError::TooManyDescriptors(what))
    }
}

/// Reads the DLL names from the import and delay import directories of a
/// 64-bit PE file.
pub fn imports(bytes: &[u8]) -> Result<Imports, PeError> {
    if bytes.get(0..2) != Some(b"MZ") {
        return Err(PeError::NotDos);
    }
    let lfanew = u32_at(bytes, 0x3c).ok_or(PeError::Truncated("the DOS header"))?;
    let pe = usize::try_from(lfanew).map_err(|_| PeError::NotPe)?;
    let sig_end = pe.checked_add(4).ok_or(PeError::NotPe)?;
    if bytes.get(pe..sig_end) != Some(b"PE\0\0") {
        return Err(PeError::NotPe);
    }
    let truncated = PeError::Truncated("the COFF header");
    let coff = sig_end;
    let n_sections =
        u16_at(bytes, offset(coff, 1, 2).ok_or(truncated.clone())?).ok_or(truncated.clone())?;
    let opt_size =
        u16_at(bytes, offset(coff, 1, 16).ok_or(truncated.clone())?).ok_or(truncated.clone())?;
    let opt = coff.checked_add(20).ok_or(truncated.clone())?;

    let truncated = PeError::Truncated("the optional header");
    let magic = u16_at(bytes, opt).ok_or(truncated.clone())?;
    if magic != MAGIC_PE32_PLUS {
        return Err(PeError::Not64Bit(magic));
    }
    let base =
        u64_at(bytes, offset(opt, 1, 24).ok_or(truncated.clone())?).ok_or(truncated.clone())?;
    let n_dirs =
        u32_at(bytes, offset(opt, 1, 108).ok_or(truncated.clone())?).ok_or(truncated.clone())?;
    let dirs = offset(opt, 1, 112).ok_or(truncated.clone())?;

    // A data directory counts when the header declares it and the optional
    // header is large enough to hold it.
    let dir = |index: usize| -> Result<Option<u32>, PeError> {
        let declared = u32::try_from(index).is_ok_and(|i| i < n_dirs);
        let end = offset(112, 8, index.saturating_add(1)).ok_or(truncated.clone())?;
        if !declared || end > usize::from(opt_size) {
            return Ok(None);
        }
        let at = offset(dirs, 8, index).ok_or(truncated.clone())?;
        let rva = u32_at(bytes, at).ok_or(truncated.clone())?;
        let size = u32_at(bytes, at.saturating_add(4)).ok_or(truncated.clone())?;
        Ok((rva != 0 && size != 0).then_some(rva))
    };
    let import_dir = dir(DIR_IMPORT)?;
    let delay_dir = dir(DIR_DELAY)?;

    let mut sections = Vec::new();
    let table = opt
        .checked_add(usize::from(opt_size))
        .ok_or(PeError::Truncated("the section table"))?;
    for i in 0..usize::from(n_sections) {
        let at = offset(table, SECTION_HDR, i).ok_or(PeError::Truncated("the section table"))?;
        sections.push(Section::read(bytes, at).ok_or(PeError::Truncated("the section table"))?);
    }
    let image = Image {
        bytes,
        base,
        sections,
    };

    let normal = match import_dir {
        Some(rva) => image.names(rva, IMPORT_DESC, "the import directory", |d| {
            u32_at(d, 12)
                .map(u64::from)
                .ok_or(PeError::Truncated("an import descriptor"))
        })?,
        None => Vec::new(),
    };
    let delayed = match delay_dir {
        Some(rva) => image.names(rva, DELAY_DESC, "the delay import directory", |d| {
            let attrs = u32_at(d, 0).ok_or(PeError::Truncated("a delay descriptor"))?;
            let name = u64::from(u32_at(d, 4).ok_or(PeError::Truncated("a delay descriptor"))?);
            if attrs & 1 == 1 {
                Ok(name)
            } else {
                name.checked_sub(image.base)
                    .ok_or(PeError::VaBelowBase("a delay import name"))
            }
        })?,
        None => Vec::new(),
    };
    Ok(Imports { normal, delayed })
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
mod tests {
    use super::*;

    const LFANEW: usize = 0x40;
    const OPT: usize = LFANEW + 24;
    const OPT_SIZE: usize = 240;
    const SECTION_HEADER: usize = OPT + OPT_SIZE;
    const RAW: usize = 0x200;
    const VA: u32 = 0x1000;
    const IMAGE_BASE: u64 = 0x1_8000_0000;

    fn put16(v: &mut [u8], at: usize, x: u16) {
        v[at..at + 2].copy_from_slice(&x.to_le_bytes());
    }
    fn put32(v: &mut [u8], at: usize, x: u32) {
        v[at..at + 4].copy_from_slice(&x.to_le_bytes());
    }

    /// DOS header, PE header, optional header with 16 data directories and
    /// one section whose bytes are `section`. The two directories point at
    /// the given RVAs (0 means none).
    fn assemble(section: &[u8], import_rva: u32, delay_rva: u32) -> Vec<u8> {
        let mut v = vec![0u8; RAW];
        v[0..2].copy_from_slice(b"MZ");
        put32(&mut v, 0x3c, LFANEW as u32);
        v[LFANEW..LFANEW + 4].copy_from_slice(b"PE\0\0");
        put16(&mut v, LFANEW + 4, 0x8664);
        put16(&mut v, LFANEW + 6, 1);
        put16(&mut v, LFANEW + 20, OPT_SIZE as u16);
        put16(&mut v, OPT, 0x20b);
        v[OPT + 24..OPT + 32].copy_from_slice(&IMAGE_BASE.to_le_bytes());
        put32(&mut v, OPT + 108, 16);
        for (index, rva) in [(1usize, import_rva), (13, delay_rva)] {
            put32(&mut v, OPT + 112 + 8 * index, rva);
            put32(
                &mut v,
                OPT + 112 + 8 * index + 4,
                if rva == 0 { 0 } else { 64 },
            );
        }
        v[SECTION_HEADER..SECTION_HEADER + 6].copy_from_slice(b".idata");
        put32(&mut v, SECTION_HEADER + 8, section.len() as u32);
        put32(&mut v, SECTION_HEADER + 12, VA);
        put32(&mut v, SECTION_HEADER + 16, section.len() as u32);
        put32(&mut v, SECTION_HEADER + 20, RAW as u32);
        v.extend_from_slice(section);
        v
    }

    /// An image with the given normal and delay imports.
    fn image(normal: &[&str], delay: &[&str]) -> Vec<u8> {
        let imp_len = (normal.len() + 1) * IMPORT_DESC;
        let del_len = (delay.len() + 1) * DELAY_DESC;
        let mut sec = vec![0u8; imp_len + del_len];
        let mut names = Vec::new();
        let mut name_rva = |name: &str| {
            let rva = VA + (imp_len + del_len + names.len()) as u32;
            names.extend_from_slice(name.as_bytes());
            names.push(0);
            rva
        };
        for (i, n) in normal.iter().enumerate() {
            let rva = name_rva(n);
            put32(&mut sec, i * IMPORT_DESC + 12, rva);
        }
        for (i, n) in delay.iter().enumerate() {
            let rva = name_rva(n);
            put32(&mut sec, imp_len + i * DELAY_DESC, 1);
            put32(&mut sec, imp_len + i * DELAY_DESC + 4, rva);
        }
        sec.extend_from_slice(&names);
        let delay_rva = if delay.is_empty() {
            0
        } else {
            VA + imp_len as u32
        };
        let import_rva = if normal.is_empty() { 0 } else { VA };
        assemble(&sec, import_rva, delay_rva)
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn two_normal_and_one_delay_import_in_order() {
        let img = image(&["b.dll", "a.dll"], &["d.dll"]);
        let got = imports(&img).unwrap();
        assert_eq!(got.normal, names(&["b.dll", "a.dll"]));
        assert_eq!(got.delayed, names(&["d.dll"]));
    }

    #[test]
    fn no_directories_means_no_imports() {
        let got = imports(&image(&[], &[])).unwrap();
        assert!(got.normal.is_empty() && got.delayed.is_empty());
    }

    #[test]
    fn duplicates_differing_in_case_keep_the_first_spelling() {
        let img = image(&["kernel32.dll", "ntdll.dll", "KERNEL32.dll"], &[]);
        assert_eq!(
            imports(&img).unwrap().normal,
            names(&["kernel32.dll", "ntdll.dll"])
        );
    }

    #[test]
    fn delay_import_with_a_va_subtracts_the_image_base() {
        let mut img = image(&[], &["d.dll"]);
        let imp_len = IMPORT_DESC;
        let desc = RAW + imp_len;
        let rva = u32::from_le_bytes(img[desc + 4..desc + 8].try_into().unwrap());
        put32(&mut img, desc, 0);
        // The name field is 32 bits wide, so use an image base below 4 GiB.
        let base = 0x1000_0000u32;
        img[OPT + 24..OPT + 32].copy_from_slice(&u64::from(base).to_le_bytes());
        put32(&mut img, desc + 4, base + rva);
        assert_eq!(imports(&img).unwrap().delayed, names(&["d.dll"]));
        // A VA below the image base is an error.
        put32(&mut img, desc + 4, 0x10);
        assert_eq!(
            imports(&img),
            Err(PeError::VaBelowBase("a delay import name"))
        );
    }

    #[test]
    fn pe32_is_not_a_64_bit_file() {
        let mut img = image(&["a.dll"], &[]);
        put16(&mut img, OPT, 0x10b);
        let err = imports(&img).unwrap_err();
        assert_eq!(err, PeError::Not64Bit(0x10b));
        assert!(err.to_string().contains("not a 64-bit PE file"));
    }

    #[test]
    fn bad_headers_are_errors() {
        let good = image(&["a.dll"], &[]);

        let mut img = good.clone();
        img[0] = b'Z';
        assert_eq!(imports(&img), Err(PeError::NotDos));

        let mut img = good.clone();
        img[LFANEW] = b'X';
        assert_eq!(imports(&img), Err(PeError::NotPe));

        let mut img = good.clone();
        put32(&mut img, 0x3c, u32::MAX - 1);
        assert_eq!(imports(&img), Err(PeError::NotPe));

        assert!(imports(b"").is_err());
        assert!(imports(b"MZ").is_err());
    }

    #[test]
    fn an_rva_outside_every_section_is_an_error() {
        let mut img = image(&["a.dll"], &[]);
        put32(&mut img, OPT + 112 + 8, 0x9000);
        assert!(matches!(
            imports(&img),
            Err(PeError::RvaUnmapped { rva: 0x9000, .. })
        ));
        // A name RVA outside the section.
        let mut img = image(&["a.dll"], &[]);
        put32(&mut img, RAW + 12, 0x9000);
        assert!(matches!(imports(&img), Err(PeError::RvaUnmapped { .. })));
    }

    #[test]
    fn a_name_without_nul_before_the_end_is_an_error() {
        let mut img = image(&["a.dll"], &[]);
        img.pop();
        assert_eq!(imports(&img), Err(PeError::NameNotTerminated));
    }

    #[test]
    fn bad_names_are_errors() {
        let mut img = image(&["a.dll"], &[]);
        let at = img.len() - 3;
        img[at] = 0x01;
        assert_eq!(imports(&img), Err(PeError::NameNotPrintable));

        let mut img = image(&["a.dll"], &[]);
        let at = img.len() - 6;
        img[at] = 0;
        assert_eq!(imports(&img), Err(PeError::EmptyName));

        let long = "x".repeat(300);
        assert_eq!(imports(&image(&[&long], &[])), Err(PeError::NameTooLong));
        let ok = "x".repeat(MAX_NAME);
        assert_eq!(imports(&image(&[&ok], &[])).unwrap().normal, vec![ok]);
    }

    #[test]
    fn a_descriptor_loop_without_a_terminator_is_an_error() {
        // The terminator is overwritten, so the next descriptor reads names
        // from garbage or leaves the section.
        let mut img = image(&["a.dll"], &[]);
        for b in &mut img[RAW + IMPORT_DESC..RAW + 2 * IMPORT_DESC] {
            *b = 0xff;
        }
        assert!(imports(&img).is_err());

        // 5000 non-zero descriptors that all name a valid string.
        let count = MAX_DESCRIPTORS + 904;
        let table = count * IMPORT_DESC;
        let mut sec = vec![0u8; table];
        for i in 0..count {
            put32(&mut sec, i * IMPORT_DESC + 12, VA + table as u32);
        }
        sec.extend_from_slice(b"a.dll\0");
        let img = assemble(&sec, VA, 0);
        assert_eq!(
            imports(&img),
            Err(PeError::TooManyDescriptors("the import directory"))
        );
    }

    #[test]
    fn a_section_without_raw_data_for_an_rva_is_an_error() {
        let mut img = image(&["a.dll"], &[]);
        // Virtual size larger than the raw size: the table is still mapped,
        // an RVA beyond the raw data is not.
        put32(&mut img, SECTION_HEADER + 8, 0x4000);
        put32(&mut img, OPT + 112 + 8, VA + 0x3000);
        assert!(matches!(imports(&img), Err(PeError::RvaUnmapped { .. })));
    }

    #[test]
    fn prefixes_and_single_byte_changes_never_panic() {
        let img = image(&["kernel32.dll", "user32.dll"], &["d.dll"]);
        assert!(imports(&img).is_ok());
        for len in 0..=img.len() {
            let _ = imports(&img[..len]);
        }
        for at in 0..img.len() {
            let mut bad = img.clone();
            bad[at] = 0xff;
            let _ = imports(&bad);
            let mut bad = img.clone();
            bad[at] = 0;
            let _ = imports(&bad);
        }
    }

    #[test]
    fn display_is_one_line() {
        for e in [
            PeError::Truncated("x"),
            PeError::NotDos,
            PeError::NotPe,
            PeError::Not64Bit(0x10b),
            PeError::RvaUnmapped { what: "x", rva: 5 },
            PeError::VaBelowBase("x"),
            PeError::TooManyDescriptors("x"),
            PeError::NameNotTerminated,
            PeError::NameTooLong,
            PeError::NameNotPrintable,
            PeError::EmptyName,
        ] {
            assert!(!e.to_string().contains('\n'), "{e:?}");
        }
    }
}
