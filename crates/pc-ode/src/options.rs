//! The `@` option names that XPPAUT documents.
//!
//! The list follows the option summary of the XPPAUT documentation. Names
//! written there with a blank (`JAC EPS`) are spelled with `_` here, as
//! XPPAUT reads them. `epsl`, `epsu` and `epss` come from `docs/language.md`,
//! which lists them with the AUTO options. An `@` name outside this list
//! gives a warning.

/// Option names without a trailing plot or slider number, in lower case.
pub const NAMES: &[&str] = &[
    "quiet",
    "logfile",
    "maxstor",
    "forecolor",
    "backcolor",
    "mwcolor",
    "dwcolor",
    "backimage",
    "smallfont",
    "bigfont",
    "smc",
    "umc",
    "xnc",
    "ync",
    "sec",
    "uec",
    "spc",
    "upc",
    "output",
    "grads",
    "height",
    "width",
    "runnow",
    "but",
    "lt",
    "seed",
    "xp",
    "yp",
    "zp",
    "colormap",
    "nplot",
    "multiwin",
    "simplot",
    "axes",
    "colorize",
    "colorvia",
    "colorlo",
    "colorhi",
    "total",
    "dt",
    "njmp",
    "t0",
    "trans",
    "nmesh",
    "dfgrid",
    "meth",
    "dtmin",
    "dtmax",
    "vmaxpts",
    "jac_eps",
    "newt_tol",
    "newt_iter",
    "atoler",
    "toler",
    "bound",
    "delay",
    "bandup",
    "bandlo",
    "phi",
    "theta",
    "xlo",
    "ylo",
    "xhi",
    "yhi",
    "xmax",
    "xmin",
    "ymax",
    "ymin",
    "zmax",
    "zmin",
    "poimap",
    "poivar",
    "poipln",
    "poisgn",
    "poistop",
    "range",
    "rangeover",
    "rangestep",
    "rangelow",
    "rangehigh",
    "rangereset",
    "rangeoldic",
    "tor_per",
    "fold",
    "ps_font",
    "ps_lw",
    "ps_fsize",
    "ps_color",
    "stoch",
    "postprocess",
    "histhi",
    "histlo",
    "histbins",
    "histcol",
    "speccol",
    "speccol2",
    "specwidth",
    "specwin",
    "ntst",
    "nmax",
    "npr",
    "dsmin",
    "dsmax",
    "ds",
    "parmin",
    "parmax",
    "normmin",
    "normmax",
    "autoxmin",
    "autoxmax",
    "autoymin",
    "autoymax",
    "autovar",
    "epsl",
    "epsu",
    "epss",
    "dll_lib",
    "dll_fun",
    "dfdraw",
    "ncdraw",
];

/// Option names that take a number: plot axes and windows 2 to 8, sliders
/// 1 to 3. Each entry is the name root and the first and last number.
pub const NUMBERED: &[(&str, u8, u8)] = &[
    ("xp", 2, 8),
    ("yp", 2, 8),
    ("zp", 2, 8),
    ("xlo", 2, 8),
    ("xhi", 2, 8),
    ("ylo", 2, 8),
    ("yhi", 2, 8),
    ("s", 1, 3),
    ("slo", 1, 3),
    ("shi", 1, 3),
];

/// True when `key`, a lower-case option name, is a documented option.
pub fn is_known_option(key: &str) -> bool {
    if NAMES.contains(&key) {
        return true;
    }
    NUMBERED.iter().any(|&(root, lo, hi)| {
        key.strip_prefix(root)
            .filter(|d| d.len() == 1)
            .and_then(|d| d.parse::<u8>().ok())
            .is_some_and(|n| (lo..=hi).contains(&n))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documented_names_are_known() {
        for key in [
            "total", "dt", "meth", "jac_eps", "ps_font", "ntst", "dsmax", "but",
        ] {
            assert!(is_known_option(key), "{key}");
        }
    }

    #[test]
    fn numbered_names_are_known_in_their_range() {
        for key in ["xp2", "xp8", "yhi5", "s1", "s3", "slo2", "shi3", "zp4"] {
            assert!(is_known_option(key), "{key}");
        }
        for key in ["xp1", "xp9", "s0", "s4", "slo4", "xp22", "zlo2"] {
            assert!(!is_known_option(key), "{key}");
        }
    }

    #[test]
    fn other_names_are_unknown() {
        for key in ["totl", "method", "jac", "eps", "foo"] {
            assert!(!is_known_option(key), "{key}");
        }
    }
}
