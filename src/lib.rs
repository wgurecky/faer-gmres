// Basic GMRES implementation from the wiki:
// https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
//
// Uses the Faer library for sparse matricies and sparse solver.
use faer::prelude::*;
use faer::sparse::*;
use faer::mat;


/// Calculate the givens rotation matrix
fn givens_rotation(v1: f64, v2: f64) -> (f64, f64) {
    let t = (v1.powi(2) + v2.powi(2)).powf(0.5);
    let cs = v1 / t;
    let sn = v2 / t;

    return (cs, sn);
}

/// Apply givens rotation to H col
///
fn apply_givens_rotation(h: &mut Vec<f64>, cs: &mut Vec<f64>, sn: &mut Vec<f64>, k: usize) {
    for i in 0..k {
        let temp = cs[i] * h[i] + sn[i] * h[i + 1];
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
        h[i] = temp;
    }

    // Update the next sin cos values for rotation
    (cs[k], sn[k]) = givens_rotation(h[k], h[k + 1]);

    // Eliminate H(i+1:i)
    h[k] = cs[k] * h[k] + sn[k] * h[k + 1];
    h[k + 1] = 0.;
}

/// Arnoldi decomposition for sparse matrices
fn arnoldi(a: SparseColMatRef<usize, f64>, q: &Vec<Mat<f64>>, k: usize) -> (Vec<f64>, Mat<f64>) {
    // Krylov vector
    let q_col: MatRef<f64> = q[k].as_ref();
    let mut qv: Mat<f64> = a * q_col;
    let mut h = Vec::with_capacity(k + 2);
    for i in 0..=k {
        let qci: MatRef<f64> = q[i].as_ref();
        let ht = qv.transpose() * &qci;
        h.push( ht.read(0, 0) );
        qv = qv - (qci * faer::scale(h[i]));
    }
    h.push(qv.norm_l2());
    qv = qv * faer::scale(1./h[k + 1]);
    return (h, qv);
}


/// Generalized minimal residual method
pub fn gmres(
    a: SparseColMatRef<usize, f64>,
    b: MatRef<f64>,
    x: MatRef<f64>,
    max_iter: usize,
    threshold: f64,
) -> Result<(Mat<f64>, f64, usize), String> {
    // Use x as the initial vector
    // compute initial residual
    let r = b - a * x.as_ref();

    let b_norm = b.norm_l2();
    let r_norm = r.norm_l2();
    let mut error = r_norm / b_norm;

    // Initialize 1D vectors (Optimizable?)
    let mut sn = vec![0.; max_iter];
    let mut cs = vec![0.; max_iter];
    // let mut e1 = vec![0.; max_iter + 1];
    let mut e1: Mat<f64> = mat::Mat::zeros(max_iter+1, 1);
    e1.write(0, 0, 1.);
    let mut e = vec![error];

    let q = r * faer::scale(1.0/r_norm);
    let mut beta = faer::scale(r_norm) * e1;
    let mut hs = Vec::with_capacity(max_iter); //Store hessemberg vectors
    let mut qs = Vec::with_capacity(max_iter);
    qs.push(q.clone());

    let mut k_iters = 0;
    for k in 0..max_iter {
        let (mut hk, qk) = arnoldi(a, &qs, k);
        apply_givens_rotation(&mut hk, &mut cs, &mut sn, k);
        hs.push(hk);
        qs.push(qk);

        // Update the residual vector
        beta.write(k+1, 0, -sn[k] * beta.read(k, 0));
        beta.write(k, 0, cs[k] * beta.read(k, 0));
        error = f64::abs(beta.read(k + 1, 0)) / b_norm;

        // Save the error
        e.push(error);
        k_iters += 1;
        if error <= threshold {
            break;
        }
    }

    // build full sparse H matrix from column vecs
    // create sparse matrix from triplets
    let mut h_triplets = Vec::new();
    let mut h_len = 0;
    for (c, hvec) in (&hs).into_iter().enumerate() {
        h_len = hvec.len();
        for h_i in 0..h_len {
            h_triplets.push((h_i, c, hvec[h_i]));
        }
    }
    let h_sprs = SparseColMat::<usize, f64>::try_new_from_triplets(
        h_len, (&hs).len(), &h_triplets).unwrap();

    // build full sparse Q matrix
    let mut q_triplets = Vec::new();
    let mut q_len = 0;
    for (c, qvec) in (&qs).into_iter().enumerate() {
        q_len = qvec.nrows();
        for q_i in 0..q_len {
            q_triplets.push((q_i, c, qvec.read(q_i, 0)));
        }
    }
    let q_sprs = SparseColMat::<usize, f64>::try_new_from_triplets
        (q_len, (&qs).len(), &q_triplets).unwrap();

    // compute solution
    let h_qr = h_sprs.sp_qr().unwrap();
    let y = h_qr.solve(&beta.get(0..k_iters+1, 0..1));

    if error <= threshold {
        Ok((x.as_ref() + q_sprs * y, error, k_iters))
    } else {
        Err(format!(
            "GMRES did not converge. Error: {}. Threshold: {}",
            error, threshold
        ))
    }
}


#[cfg(test)]
mod test_faer_gmres {
    use assert_approx_eq::assert_approx_eq;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_gmres() {
    let a_test_triplets = vec![
        (0, 0, 1.0),
        (1, 1, 2.0),
        (2, 2, 3.0),
        ];
    let a_test = SparseColMat::<usize, f64>::try_new_from_triplets(
        3, 3,
        &a_test_triplets).unwrap();

    // rhs
    let b = faer::mat![
        [2.0],
        [2.0],
        [2.0],
        ];

    // initia sol guess
    let x0 = faer::mat![
        [0.0],
        [0.0],
        [0.0],
        ];

    let (res_x, err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_ref(), 10, 1e-8).unwrap();
    println!("Result x: {:?}", res_x);
    println!("Error x: {:?}", err);
    println!("Iters : {:?}", iters);
    assert!(err < 1e-4);
    assert!(iters < 10);

    // expect result for x to be [2,1,2/3]
    assert_approx_eq!(res_x.read(0, 0), 2.0, 1e-12);
    assert_approx_eq!(res_x.read(1, 0), 1.0, 1e-12);
    assert_approx_eq!(res_x.read(2, 0), 2.0/3.0, 1e-12);

    }

    #[test]
    fn test_arnoldi() {
    }
}
