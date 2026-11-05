//! One run of `pc_ode::load` over a document.

use pc_ode::{FileId, Loaded, SourceLoader, SourceMap};

/// The result of loading one document, with the source map that the spans in
/// it refer to.
#[derive(Debug)]
pub struct Analysis {
    pub map: SourceMap,
    /// The document itself.
    pub top: FileId,
    pub loaded: Loaded,
}

impl Analysis {
    /// Display names of the files read, the document first.
    pub fn file_names(&self) -> Vec<String> {
        (0..self.map.file_count())
            .map(|i| self.map.name(FileId(i as u32)).to_string())
            .collect()
    }
}

/// Loads `text` as the document called `name`. Included files come from
/// `loader`.
pub fn analyze(name: &str, text: String, loader: &mut dyn SourceLoader) -> Analysis {
    let mut map = SourceMap::new();
    let top = map.add(name, text);
    let loaded = pc_ode::load(top, &mut map, loader);
    Analysis { map, top, loaded }
}
