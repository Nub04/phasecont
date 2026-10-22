//! Native runner for the spikes, so the same code runs in WSL without MATLAB.
//! Prints the spike's JSON report on stdout.
//!
//!   spikecli s2 [calls=N reps=N samples=N seed=N]
//!   spikecli s3 [n=N m=N N=N T=X seed=N threads=N reps=N]
//!   spikecli s3lu [n=N m=N N=N T=X seed=N reps=N budget_ms=N settings=auto+simplicial+supernodal on=full+reduced]
//!   spikecli s4 [n=N reps=N seed=N]

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((name, rest)) = argv.split_first() else {
        usage()
    };
    let args = rest.join(" ");
    let result = match name.as_str() {
        "s2" => pckern::s2_jit::run(&args),
        "s3" => pckern::s3_linalg::run(&args),
        "s3lu" => pckern::s3_linalg::lu_probe::run(&args),
        "s4" => pckern::s4_cuda::run(&args),
        _ => usage(),
    };
    match result {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("spikecli {name}: {e}");
            std::process::exit(1);
        }
    }
}

fn usage() -> ! {
    eprintln!("usage: spikecli s2|s3|s3lu|s4 [key=value ...]");
    std::process::exit(2);
}
