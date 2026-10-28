//! Loads and lowers an `.ode` text for the integration tests (not a cargo build script).

use pc_ir::IrModel;

pub fn build_model(src: &str) -> IrModel {
    use pc_ode::{MapLoader, Severity, SourceMap, load};
    let mut map = SourceMap::new();
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    assert!(
        l.diagnostics.iter().all(|d| d.severity != Severity::Error),
        "{:?}",
        l.diagnostics
    );
    match pc_ir::lower(&l.expanded, &l.info) {
        Ok(m) => m,
        Err(e) => panic!("lower failed: {e:?}"),
    }
}
