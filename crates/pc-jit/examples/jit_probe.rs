//! Compiles a small model with the JIT, runs `F` and `JacU` once and compares them with
//! `Fallback`. The macOS CI job runs this binary under the hardened runtime with
//! different entitlements (SPEC section 4).
//!
//! Exit status: 0 when the JIT ran and matched, 2 when compiling or finalizing failed,
//! 3 when the results differ. A signal (SIGBUS or SIGSEGV) means the code page could
//! not be executed.

use pc_ir::{EntryKind, Fallback, lower};
use pc_jit::CompiledModel;
use pc_ode::{MapLoader, SourceMap, load};
use std::process::ExitCode;

const MODEL: &str = "\
par mu=0.1
r2=x^2+y^2
x'=mu*x-y-x*r2
y'=x+mu*y-y*r2
init x=0.1,y=0
done
";

fn main() -> ExitCode {
    let mut map = SourceMap::new();
    let top = map.add("probe.ode", MODEL);
    let mut loader = MapLoader::new();
    let loaded = load(top, &mut map, &mut loader);
    let ir = lower(&loaded.expanded, &loaded.info).expect("lower");
    let fallback = Fallback::new(&loaded.expanded, &loaded.info, &ir);
    let (n, n_p) = (ir.n(), ir.n_p());
    let mut model = CompiledModel::new(ir);

    let x = [0.3, -0.7];
    let p = [0.1, 0.0];
    let mut worst = 0u64;
    for (kind, len) in [(EntryKind::F, n), (EntryKind::JacU, n * n)] {
        let mut got = vec![f64::NAN; len];
        if let Err(e) = model.call(&kind, &x, &p[..n_p + 1], &[], &mut got) {
            println!("jit_probe: compile failed for {kind:?}: {e:?}");
            return ExitCode::from(2);
        }
        let mut want = vec![f64::NAN; len];
        fallback.eval(&kind, &x, &p[..n_p + 1], &[], &mut want);
        for (a, b) in got.iter().zip(&want) {
            worst = worst.max(a.to_bits().abs_diff(b.to_bits()));
        }
        println!("jit_probe: {kind:?} jit {got:?} fallback {want:?}");
    }
    if worst > 4 {
        println!("jit_probe: results differ by {worst} ulp");
        return ExitCode::from(3);
    }
    println!(
        "jit_probe: ok on {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    ExitCode::SUCCESS
}
