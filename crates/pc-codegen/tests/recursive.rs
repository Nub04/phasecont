//! The C emitter refuses a model that calls a recursive function (SPEC 8af-5, F-5): the
//! AOT path does not build such a model, so `emit_c` names the entry instead of emitting
//! C that would need native recursion.

use pc_codegen::{CNameError, emit_c};
use pc_ir::{EntryKind, lower};
use pc_ode::{MapLoader, SourceMap, load};

#[test]
fn emit_c_refuses_a_recursive_call() {
    let mut map = SourceMap::new();
    let src = "par k=3\npw(x,n)=if(n<1)then(1)else(x*pw(x,n-1))\nx'=pw(x,k)\ndone\n";
    let top = map.add("t.ode", src);
    let l = load(top, &mut map, &mut MapLoader::new());
    let mut ir = lower(&l.expanded, &l.info).expect("lower");
    let f = ir.entry(&EntryKind::F).unwrap();
    match emit_c(&ir, &[("f".to_string(), &f)]) {
        Err(e @ CNameError::Recursive { .. }) => {
            let CNameError::Recursive { name } = &e else {
                unreachable!()
            };
            assert_eq!(name, "f");
            // The text says which evaluator runs the model instead.
            let text = e.to_string();
            assert!(text.contains("the JIT runs"), "{text}");
        }
        other => panic!("expected Recursive, got {:?}", other.map(|s| s.len())),
    }
}
