//! Collocation basis on [0, 1]: Gauss-Legendre quadrature is exact to degree 2m-1, and
//! the Lagrange basis on the m+1 equally spaced base points reproduces polynomials of
//! degree m, their first derivatives and their m-th derivative.

use pc_engine::CollocationBasis;

const MS: std::ops::RangeInclusive<usize> = 2..=7;

/// Values of the test polynomial p(z) = sum_k c_k z^k and its derivative.
fn poly(c: &[f64], z: f64) -> (f64, f64) {
    let (mut p, mut dp) = (0.0, 0.0);
    for (k, &ck) in c.iter().enumerate() {
        p += ck * z.powi(k as i32);
        if k > 0 {
            dp += ck * k as f64 * z.powi(k as i32 - 1);
        }
    }
    (p, dp)
}

/// Coefficients 1, -2, 3, ... of a test polynomial of degree d.
fn coeffs(d: usize) -> Vec<f64> {
    (0..=d)
        .map(|k| (k as f64 + 1.0) * if k % 2 == 0 { 1.0 } else { -2.0 })
        .collect()
}

#[test]
fn quadrature_is_exact_to_degree_2m_minus_1() {
    for m in MS {
        let b = CollocationBasis::new(m).unwrap();
        for k in 0..2 * m {
            let q: f64 = (0..m)
                .map(|i| b.weights()[i] * b.nodes()[i].powi(k as i32))
                .sum();
            let exact = 1.0 / (k as f64 + 1.0);
            assert!(
                (q - exact).abs() < 1e-15,
                "m {m}, degree {k}: {q} vs {exact}"
            );
        }
        let k = 2 * m;
        let q: f64 = (0..m)
            .map(|i| b.weights()[i] * b.nodes()[i].powi(k as i32))
            .sum();
        assert!(
            (q - 1.0 / (k as f64 + 1.0)).abs() > 1e-12,
            "m {m} is exact at 2m"
        );
    }
}

#[test]
fn nodes_are_ascending_symmetric_and_inside() {
    for m in MS {
        let b = CollocationBasis::new(m).unwrap();
        let z = b.nodes();
        assert_eq!(z.len(), m);
        assert!(z[0] > 0.0 && z[m - 1] < 1.0);
        for i in 0..m {
            if i > 0 {
                assert!(z[i] > z[i - 1]);
            }
            assert!((z[i] + z[m - 1 - i] - 1.0).abs() < 1e-15);
            assert!((b.weights()[i] - b.weights()[m - 1 - i]).abs() < 1e-15);
        }
    }
}

#[test]
fn two_point_rule_matches_the_closed_form() {
    let b = CollocationBasis::new(2).unwrap();
    let s = 0.5 / 3f64.sqrt();
    assert!((b.nodes()[0] - (0.5 - s)).abs() < 1e-15);
    assert!((b.nodes()[1] - (0.5 + s)).abs() < 1e-15);
    for w in b.weights() {
        assert!((w - 0.5).abs() < 1e-15);
    }
}

#[test]
fn base_points_are_equally_spaced() {
    for m in MS {
        let b = CollocationBasis::new(m).unwrap();
        let s = b.base();
        assert_eq!(s.len(), m + 1);
        for (j, &sj) in s.iter().enumerate() {
            assert_eq!(sj, j as f64 / m as f64);
        }
    }
}

#[test]
fn interpolation_reproduces_polynomials_of_degree_m() {
    for m in MS {
        let b = CollocationBasis::new(m).unwrap();
        let mut l = vec![0.0; m + 1];
        let mut dl = vec![0.0; m + 1];
        for d in 0..=m {
            let c = coeffs(d);
            let vals: Vec<f64> = b.base().iter().map(|&s| poly(&c, s).0).collect();
            for &z in &[0.0, 0.13, 0.5, 0.77, 1.0] {
                b.values(z, &mut l);
                b.derivatives(z, &mut dl);
                let p: f64 = (0..=m).map(|j| l[j] * vals[j]).sum();
                let dp: f64 = (0..=m).map(|j| dl[j] * vals[j]).sum();
                let (pe, dpe) = poly(&c, z);
                assert!((p - pe).abs() < 1e-12, "m {m}, d {d}, z {z}: {p} vs {pe}");
                assert!(
                    (dp - dpe).abs() < 1e-10,
                    "m {m}, d {d}, z {z}: {dp} vs {dpe}"
                );
            }
        }
    }
}

#[test]
fn matrices_at_the_nodes_match_the_pointwise_basis() {
    for m in MS {
        let b = CollocationBasis::new(m).unwrap();
        let mut l = vec![0.0; m + 1];
        let mut dl = vec![0.0; m + 1];
        for i in 0..m {
            b.values(b.nodes()[i], &mut l);
            b.derivatives(b.nodes()[i], &mut dl);
            for j in 0..=m {
                assert_eq!(b.value_at_node(i, j), l[j]);
                assert_eq!(b.derivative_at_node(i, j), dl[j]);
            }
        }
    }
}

#[test]
fn mth_derivative_weights_give_m_factorial_times_the_leading_coefficient() {
    for m in MS {
        let b = CollocationBasis::new(m).unwrap();
        let c = coeffs(m);
        let vals: Vec<f64> = b.base().iter().map(|&s| poly(&c, s).0).collect();
        let dm: f64 = (0..=m).map(|j| b.mth_derivative()[j] * vals[j]).sum();
        let fact: f64 = (1..=m).map(|k| k as f64).product();
        let exact = fact * c[m];
        assert!(
            (dm - exact).abs() < 1e-9 * exact.abs(),
            "m {m}: {dm} vs {exact}"
        );
    }
}

#[test]
fn degrees_outside_2_to_7_are_rejected() {
    for m in [0, 1, 8] {
        assert!(CollocationBasis::new(m).is_err());
    }
}
