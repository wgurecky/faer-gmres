// Basic GMRES implementation from the wiki:
// https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
//
// Includes restarted GMRES implementation for reduced memory requirements.
//
// Uses the Faer library for sparse matricies and sparse solver.
//
// Specifically the givens_rotation, apply_givens_rotation and part of the
// arnoldi implementation is:
// MIT License
//
// Copyright (c) 2023 Ricard Lado
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// https://crates.io/crates/gmres
//
use faer::prelude::*;
use faer::sparse::*;
use faer::mat;
use num_traits::Float;
use thiserror::Error;

#[derive(Error, Debug)]
pub struct GmresError<T>
    where
    T: faer::RealField + Float
{
    cur_x: Mat<T>,
    error: T,
    tol: T,
    msg: String,
}

pub trait LinOp<T>
    where
    T: faer::RealField + Float
{
    fn apply_linop_to_vec(&self, target: MatMut<T>);
}

#[derive(Clone)]
pub struct JacobiPreconLinOp<'a, T>
    where
    T: faer::RealField + Float
{
    m: SparseColMatRef<'a, usize, T>,
}
impl <'a, T> LinOp<T> for JacobiPreconLinOp<'a, T>
    where
    T: faer::RealField + Float + faer::SimpleEntity
{
    fn apply_linop_to_vec(&self, mut target: MatMut<T>) {
        let eps = T::from(1e-20).unwrap();
        for i in 0..target.nrows()
        {
            let v = target.read(i, 0);
            target.write(i, 0,
                 v * (T::from(1.0).unwrap() / (self.m[(i, i)] + eps) ));
        }
    }
}
impl <'a, T> JacobiPreconLinOp <'a, T>
    where
    T: faer::RealField + Float
{
    pub fn new(m_in: SparseColMatRef<'a, usize, T>) -> Self {
        Self {
            m: m_in,
        }
    }
}


/// Calculate the givens rotation matrix
fn givens_rotation<T>(v1: T, v2: T) -> (T, T)
    where
    T: faer::RealField + Float
{
    let t = (v1.powi(2) + v2.powi(2)).powf(T::from(0.5).unwrap());
    let cs = v1 / t;
    let sn = v2 / t;

    return (cs, sn);
}

/// Apply givens rotation to H col
fn apply_givens_rotation<T>(h: &mut Vec<T>, cs: &mut Vec<T>, sn: &mut Vec<T>, k: usize)
    where
    T: faer::RealField + Float
{
    for i in 0..k {
        let temp = cs[i] * h[i] + sn[i] * h[i + 1];
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
        h[i] = temp;
    }

    // Update the next sin cos values for rotation
    (cs[k], sn[k]) = givens_rotation(h[k], h[k + 1]);

    // Eliminate H(i+1:i)
    h[k] = cs[k] * h[k] + sn[k] * h[k + 1];
    h[k + 1] = T::from(0.).unwrap();
}

/// Arnoldi decomposition for sparse matrices
///
/// # Arguments
/// * `a`- The sparse matrix used to build the krylov subspace by forming [k, Ak, A^2k, A^3k...]
/// * `q`- Vector of all prior krylov column vecs
/// * `k`- Current iteration
/// * `m`- An optional preconditioner that is applied to the original system such that
///        the new krylov subspace built is [M^{-1}k, M^{-1}Ak, M^{-1}A^2k, ...].
///        If None, no preconditioner is applied.
fn arnoldi<'a, T>(
    a: SparseColMatRef<'a, usize, T>,
    q: &Vec<Mat<T>>,
    k: usize,
    m: Option<&dyn LinOp<T>>
) -> (Vec<T>, Mat<T>)
    where
    T: faer::RealField + Float
{
    // Krylov vector
    let q_col: MatRef<T> = q[k].as_ref();

    // let mut qv: Mat<f64> = a * q_col;
    // parallel version of above
    let mut qv: Mat<T> = faer::Mat::zeros(q_col.nrows(), 1);
    linalg::matmul::sparse_dense_matmul(
        qv.as_mut(), a.as_ref(), q_col.as_ref(), None, T::from(1.0).unwrap(), faer::get_global_parallelism());

    // Apply left preconditioner if supplied
    match m {
        Some(m) => m.apply_linop_to_vec(qv.as_mut()),
        _ => {}
    }

    let mut h = Vec::with_capacity(k + 2);
    for i in 0..=k {
        let qci: MatRef<T> = q[i].as_ref();
        let ht = qv.transpose() * &qci;
        h.push( ht.read(0, 0) );
        qv = qv - (qci * faer::scale(h[i]));
    }

    h.push(qv.norm_l2());
    qv = qv * faer::scale(T::from(1.).unwrap()/h[k + 1]);
    return (h, qv);
}


/// Generalized minimal residual method
pub fn gmres<'a, T>(
    a: SparseColMatRef<'a, usize, T>,
    b: MatRef<T>,
    x: MatRef<T>,
    max_iter: usize,
    threshold: T,
    m: Option<&dyn LinOp<T>>
) -> Result<(Mat<T>, T, usize), GmresError<T>>
    where
    T: faer::RealField + Float
{
    // compute initial residual
    let mut r = b - a * x.as_ref();
    match &m {
        Some(m) => (&m).apply_linop_to_vec(r.as_mut()),
        _ => {}
    }

    let b_norm = b.norm_l2();
    let r_norm = r.norm_l2();
    let mut error = r_norm / b_norm;

    // Initialize 1D vectors
    let mut sn: Vec<T> = vec![T::from(0.).unwrap(); max_iter];
    let mut cs: Vec<T> = vec![T::from(0.).unwrap(); max_iter];
    // let mut e1 = vec![0.; max_iter + 1];
    let mut e1: Mat<T> = mat::Mat::zeros(max_iter+1, 1);
    e1.write(0, 0, T::from(1.).unwrap());
    let mut e = vec![error];

    let mut beta = faer::scale(r_norm) * e1;
    let mut hs = Vec::with_capacity(max_iter); //Store hessemberg vectors
    let mut qs = Vec::with_capacity(max_iter);
    let q = r * faer::scale(T::from(1.0).unwrap()/r_norm);
    qs.push(q);

    let mut k_iters = 0;
    for k in 0..max_iter {
        let (mut hk, qk) = arnoldi(a, &qs, k, m);
        apply_givens_rotation(&mut hk, &mut cs, &mut sn, k);
        hs.push(hk);
        qs.push(qk);

        // Update the residual vector
        beta.write(k+1, 0, -sn[k] * beta.read(k, 0));
        beta.write(k, 0, cs[k] * beta.read(k, 0));
        error = (beta.read(k + 1, 0)).abs() / b_norm;

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
    let h_sprs = SparseColMat::<usize, T>::try_new_from_triplets(
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
    let q_sprs = SparseColMat::<usize, T>::try_new_from_triplets
        (q_len, (&qs).len(), &q_triplets).unwrap();

    // compute solution
    let h_qr = h_sprs.sp_qr().unwrap();
    let y = h_qr.solve(&beta.get(0..k_iters+1, 0..1));

    let sol = x.as_ref() + q_sprs * y;
    if error <= threshold {
        Ok((sol, error, k_iters))
    } else {
        Err(GmresError{
            cur_x: sol,
            error: error,
            tol: threshold,
            msg: "GMRES did not converge. Error: {:?}. Threshold: {:?}".to_string()}
        )
    }
}

/// Restarted Generalized minimal residual method
pub fn restarted_gmres<'a, T>(
    a: SparseColMatRef<'a, usize, T>,
    b: MatRef<T>,
    x: MatRef<T>,
    max_iter_inner: usize,
    max_iter_outer: usize,
    threshold: T,
    m: Option<&dyn LinOp<T>>
) -> Result<(Mat<T>, T, usize), String>
    where
    T: faer::RealField + Float
{
    let mut res_x = x.to_owned();
    let mut error = T::from(1e20).unwrap();
    let mut tot_iters = 0;
    let mut iters = 0;
    for _ko in 0..max_iter_outer {
        let res = gmres(
            a.as_ref(), b.as_ref(), res_x.as_ref(),
            max_iter_inner, threshold, m);
        match res {
            // done
            Ok(res) => {
                (res_x, error, iters) = res;
                tot_iters += iters;
                break;
            }
            // failed to converge move to next outer iter
            // store current solution for next outer iter
            Err(res) => {
                res_x = res.cur_x;
                error = res.error;
                tot_iters += max_iter_inner;
            }
        }
        if error <= threshold {
            break;
        }
    }
    if error <= threshold {
        Ok((res_x, error, tot_iters))
    } else {
        Err(format!(
            "GMRES did not converge. Error: {:?}. Threshold: {:?}",
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
    fn test_gmres_1() {
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

        let (res_x, err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_ref(), 10, 1e-8, None).unwrap();
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
    fn test_gmres_1b() {
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

        // preconditioner
        let jacobi_pre = JacobiPreconLinOp::new(a_test.as_ref());

        let (res_x, err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_ref(), 10, 1e-8, 
                                        Some(&jacobi_pre)).unwrap();
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
    fn test_gmres_2() {
        let a = faer::mat![
            [0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
            [0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
            [0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
            [0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
            [0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
            ];

        let mut a_test_triplets = vec![];
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                a_test_triplets.push((i, j, a.read(i, j)));
            }
        }
        let a_test = SparseColMat::<usize, f64>::try_new_from_triplets(
            5, 5,
            &a_test_triplets).unwrap();

        // rhs
        let b = faer::mat![
            [0.104594],
            [0.437549],
            [0.040264],
            [0.298842],
            [0.254451]
            ];

        // initia sol guess
        let x0 = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        let (res_x, err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_ref(), 100, 1e-6, None).unwrap();
        println!("Result x: {:?}", res_x);
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 100);

        // expect result for x to be [0.037919, 0.888551, -0.657575, -0.181680, 0.292447]
        assert_approx_eq!(res_x.read(0, 0), 0.037919, 1e-4);
        assert_approx_eq!(res_x.read(1, 0), 0.888551, 1e-4);
        assert_approx_eq!(res_x.read(2, 0), -0.657575, 1e-4);
        assert_approx_eq!(res_x.read(3, 0), -0.181680, 1e-4);
        assert_approx_eq!(res_x.read(4, 0), 0.292447, 1e-4);
    }

    #[test]
    fn test_gmres_3() {
        let a: Mat<f32> = faer::mat![
            [0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
            [0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
            [0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
            [0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
            [0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
            ];

        let mut a_test_triplets = vec![];
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                a_test_triplets.push((i, j, a.read(i, j)));
            }
        }
        let a_test = SparseColMat::<usize, f32>::try_new_from_triplets(
            5, 5,
            &a_test_triplets).unwrap();

        // rhs
        let b: Mat<f32> = faer::mat![
            [0.104594],
            [0.437549],
            [0.040264],
            [0.298842],
            [0.254451]
            ];

        // initia sol guess
        let x0: Mat<f32> = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        let (res_x, err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_ref(), 100, 1e-6, None).unwrap();
        println!("Result x: {:?}", res_x);
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 100);

        // expect result for x to be [0.037919, 0.888551, -0.657575, -0.181680, 0.292447]
        assert_approx_eq!(res_x.read(0, 0), 0.037919, 1e-4);
        assert_approx_eq!(res_x.read(1, 0), 0.888551, 1e-4);
        assert_approx_eq!(res_x.read(2, 0), -0.657575, 1e-4);
        assert_approx_eq!(res_x.read(3, 0), -0.181680, 1e-4);
        assert_approx_eq!(res_x.read(4, 0), 0.292447, 1e-4);
    }


    #[test]
    fn test_restarted_gmres_4() {
        let a: Mat<f32> = faer::mat![
            [0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
            [0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
            [0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
            [0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
            [0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
            ];

        let mut a_test_triplets = vec![];
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                a_test_triplets.push((i, j, a.read(i, j)));
            }
        }
        let a_test = SparseColMat::<usize, f32>::try_new_from_triplets(
            5, 5,
            &a_test_triplets).unwrap();

        // rhs
        let b: Mat<f32> = faer::mat![
            [0.104594],
            [0.437549],
            [0.040264],
            [0.298842],
            [0.254451]
            ];

        // initia sol guess
        let x0: Mat<f32> = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        let (res_x, err, iters) = restarted_gmres(
            a_test.as_ref(), b.as_ref(), x0.as_ref(), 3, 30,
            1e-6, None).unwrap();
        println!("Result x: {:?}", res_x);
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 100);
        assert_approx_eq!(res_x.read(0, 0), 0.037919, 1e-4);
        assert_approx_eq!(res_x.read(1, 0), 0.888551, 1e-4);
        assert_approx_eq!(res_x.read(2, 0), -0.657575, 1e-4);
        assert_approx_eq!(res_x.read(3, 0), -0.181680, 1e-4);
        assert_approx_eq!(res_x.read(4, 0), 0.292447, 1e-4);

        // with preconditioning
        let jacobi_pre = JacobiPreconLinOp::new(a_test.as_ref());
        let (res_x_precon, err_precon, iters_precon) = restarted_gmres(
            a_test.as_ref(), b.as_ref(), x0.as_ref(), 3, 30,
            1e-6, Some(&jacobi_pre)).unwrap();
        assert!(iters_precon < iters);
        assert!(err_precon < 1e-4);
        assert_approx_eq!(res_x_precon.read(0, 0), 0.037919, 1e-4);
        assert_approx_eq!(res_x_precon.read(1, 0), 0.888551, 1e-4);
        assert_approx_eq!(res_x_precon.read(2, 0), -0.657575, 1e-4);
        assert_approx_eq!(res_x_precon.read(3, 0), -0.181680, 1e-4);
        assert_approx_eq!(res_x_precon.read(4, 0), 0.292447, 1e-4);
    }

    #[test]
    fn test_arnoldi() {
    }
}
