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
    "nout",
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

/// Option names that XPPAUT 8.0 also reads at the start of a longer key.
/// Measured on 2026-11-14 (probes `opt-prefix-*` and `meth-method`): `totalx`
/// sets `total`, `dtx` sets `dt`, `t0x` sets `t0`, `method` sets `meth`, and
/// `transx`, `njmpx`, `boundx`, `tolerx` and `dtminx` act like `trans`,
/// `njmp`, `bound`, `toler` and `dtmin`. `nout` is the old name of `njmp`:
/// `nout` and `noutx` set the same value as `njmp` (probes `opt-nout-*`, on
/// 2026-11-18 with XPPAUT 8.2.2). A shorter key such as `tot` or `met` sets
/// nothing. The other names in [`NAMES`] are matched exactly, since no probe
/// has shown the prefix rule for them.
pub const PREFIX_NAMES: &[&str] = &[
    "bound", "dt", "dtmin", "meth", "njmp", "nout", "t0", "toler", "total", "trans",
];

/// True when `key`, a lower-case option name, is a documented option, or a
/// key that starts with one of [`PREFIX_NAMES`].
pub fn is_known_option(key: &str) -> bool {
    if NAMES.contains(&key) || PREFIX_NAMES.iter().any(|n| key.starts_with(n)) {
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
        for key in [
            "totl", "tot", "tota", "met", "jac", "eps", "foo", "d", "t", "nou", "no",
        ] {
            assert!(!is_known_option(key), "{key}");
        }
    }

    #[test]
    fn keys_that_start_with_a_measured_name_are_known() {
        for key in [
            "totalx", "dtx", "dtt", "dtminx", "t0x", "transx", "njmpx", "boundx", "tolerx", "nout",
            "noutx",
        ] {
            assert!(is_known_option(key), "{key}");
        }
    }

    #[test]
    fn longer_keys_of_names_that_were_not_measured_stay_unknown() {
        for key in ["xlox", "seedx", "ntstx", "atolerx", "maxstorx", "colormapx"] {
            assert!(!is_known_option(key), "{key}");
        }
    }

    #[test]
    fn keys_that_start_with_meth_are_known() {
        for key in ["meth", "method", "methx", "methods"] {
            assert!(is_known_option(key), "{key}");
        }
    }
}
