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
use faer::linalg::solvers::Solve;
use faer::linalg::temp_mat_scratch;
use faer::sparse::*;
use faer::reborrow::*;
use faer::matrix_free::LinOp;
use faer_traits::math_utils::{one, zero};
use faer_traits::RealField;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use num_traits::Float;
use std::{error::Error, fmt};

#[derive(Debug)]
pub struct GmresError<T>
    where
    T: Float + RealField
{
    error: T,
    tol: T,
    msg: String,
}

impl <T> Error for GmresError <T>
    where
    T: Float + RealField
{}

impl <T> fmt::Display for GmresError<T>
    where
    T: Float + RealField
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GmresError")
    }
}

#[derive(Clone,Debug)]
pub struct JacobiPreconLinOp<'a, T>
    where
    T: Float + RealField
{
    m: SparseColMatRef<'a, usize, T>,
}

impl <'a, T> LinOp<T> for JacobiPreconLinOp<'a, T>
    where
    T: Float + RealField
{
    fn apply_scratch(
            &self,
            rhs_ncols: usize,
            parallelism: Par,
        ) -> StackReq {
        let _ = parallelism;
        let _ = rhs_ncols;
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        self.m.nrows()
    }

    fn ncols(&self) -> usize {
        self.m.ncols()
    }

    fn apply(
        &self,
        out: MatMut<T>,
        rhs: MatRef<T>,
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        // unused
        _ = parallelism;
        _ = stack;

        let mut out = out;
        let eps = T::from(1e-12).unwrap();
        let one_c = one::<T>();
        for i in 0..rhs.nrows()
        {
            let v = rhs[(i, 0)];
            out[(i, 0)] =
                 v * (one_c / (*self.m.get(i, i).unwrap_or(&one_c) + eps) );
        }
    }

    fn conj_apply(
            &self,
            out: MatMut<'_, T>,
            rhs: MatRef<'_, T>,
            parallelism: Par,
            stack: &mut MemStack,
        ) {
        // Not implented error!
        panic!("Not Implemented");
    }
}

impl <'a, T> JacobiPreconLinOp <'a, T>
    where
    T: Float + RealField
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
    T: Float + RealField
{
    let t = (v1.powi(2) + v2.powi(2)).powf(T::from(0.5).unwrap());
    let cs = v1 / t;
    let sn = v2 / t;

    return (cs, sn);
}

/// Apply givens rotation to H col
fn apply_givens_rotation<T>(mut h: ColMut<T>, cs: &mut Vec<T>, sn: &mut Vec<T>, k: usize)
    where
    T: Float + RealField
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
    h[k + 1] = zero();
}


/// Arnoldi decomposition for sparse matrices
///
/// # Arguments
/// * `a`- The sparse matrix used to build the krylov subspace by forming [qk0, A*qk1, A^2*qk2, ...]
/// * `H`- Hessenberg matrix
/// * `q`- Vector of all prior krylov column vecs
/// * `k`- Current iteration
/// * `m`- An optional preconditioner that is applied to the original system such that
///        the new krylov subspace built is [M^{-1}k, M^{-1}Ak, M^{-1}A^2k, ...].
///        If None, no preconditioner is applied.
fn arnoldi<'a, T, Lop: LinOp<T>>(
    a: &Lop,
    mut H: MatMut<T>,
    mut q: MatMut<T>,
    k: usize,
    m: Option<&dyn LinOp<T>>,
    par: Par,
    stack: &mut MemStack
    ) -> bool
    where
    T: Float + RealField
{
    // Krylov vector
    let q_col: MatRef<T> = q.rb().col(k).as_mat();

    let mut qv: Mat<T> = Mat::zeros(q_col.nrows(), 1);
    a.apply(qv.as_mut(), q_col.as_ref(), par, stack);

    // Apply left preconditioner if supplied
    match m {
        Some(m) => {
            let mut lp_out = Mat::zeros(qv.nrows(), qv.ncols());
            m.apply(lp_out.as_mut(), qv.as_ref(), par, stack);
            qv = lp_out;
        },
        _ => {}
    }

    let mut h = H.rb_mut().col_mut(k);
    for i in 0..k+1 {
        let r = qv.col(0).transpose() * q.rb().col(i);
        h[i] = r;
        zip!(qv.col_mut(0), q.rb().col(i)).for_each(|unzip!(mut y, x)| *y = *y - r * *x);
    }

    let norm_v = qv.norm_l2();
    let breakdown_tol = T::from(1.0e-12).unwrap();
    let breakdown_flag: bool = norm_v < breakdown_tol;

    h[k+1] = norm_v;

    if !breakdown_flag {
        qv = qv * faer::Scale(one::<T>()/norm_v);
        q.col_mut(k+1).copy_from(qv.col(0));
    }
    breakdown_flag
}


/// Generalized minimal residual method
pub fn gmres<'a, T, Lop: LinOp<T>>(
    a: Lop,
    b: MatRef<T>,
    mut x: MatMut<T>,
    max_iter: usize,
    threshold: T,
    m: Option<&dyn LinOp<T>>
) -> Result<(T, usize), GmresError<T>>
    where
    T: Float + RealField
{
    let par = faer::get_global_parallelism();
    let n = b.nrows();
    let h_scratch = temp_mat_scratch::<T>(max_iter+1, max_iter);
    let q_scratch = temp_mat_scratch::<T>(n, max_iter+1);
    let a_scratch = a.apply_scratch(1, par).or(StackReq::new::<bool>(max_iter));
    let stackreq = StackReq::all_of(&[h_scratch, q_scratch, a_scratch]);
    let mut buff = MemBuffer::new(stackreq);
    let stack = MemStack::new(&mut buff);

    let mut hs = Mat::zeros(max_iter+1, max_iter);
    let mut qs = Mat::zeros(n, max_iter+1);

    // compute initial residual
    let mut a_x = Mat::zeros(b.nrows(), b.ncols());
    a.apply(a_x.as_mut(), x.as_ref(), par, stack);
    let mut r = b - a_x;

    match &m {
        Some(m) => {
            let mut lp_out = Mat::zeros(r.nrows(), r.ncols());
            (&m).apply(lp_out.as_mut(), r.as_ref(), par, stack);
            r = lp_out;
        },
        _ => {}
    }

    let b_norm = b.norm_l2();
    let r_norm = r.norm_l2();
    let mut error = r_norm / b_norm;

    // Initialize 1D vectors
    let mut sn: Vec<T> = vec![zero(); max_iter];
    let mut cs: Vec<T> = vec![zero(); max_iter];
    // let mut e1 = vec![0.; max_iter + 1];
    let mut e1: Mat<T> = Mat::zeros(max_iter+1, 1);
    e1[(0, 0)] = one::<T>();
    let mut e = vec![error];

    let mut beta = faer::Scale(r_norm) * e1;
    let q = r * faer::Scale(one::<T>()/r_norm);
    qs.col_mut(0).copy_from(q.col(0));

    let mut k_iters = 0;
    for k in 0..max_iter {
        let _brkdwn = arnoldi(&a, hs.as_mut(), qs.as_mut(), k, m, par, stack);
        apply_givens_rotation(hs.col_mut(k), &mut cs, &mut sn, k);

        // Update the residual vector
        beta[(k+1, 0)] = -sn[k] * beta[(k, 0)];
        beta[(k, 0)] = cs[k] * beta[(k, 0)];
        error = (beta[(k + 1, 0)]).abs() / b_norm;

        // Save the error
        e.push(error);
        k_iters += 1;
        if error <= threshold {
            break;
        }
    }

    // trim
    let H = hs.get(0..k_iters, 0..k_iters);
    let V = qs.get(0..n, 0..k_iters);

    // compute solution
    let h_qr = H.col_piv_qr();
    let y2 = h_qr.solve(&beta.get(0..k_iters, 0..1));

    x += V.get(0..V.nrows(), 0..y2.nrows()) * y2;
    if error <= threshold {
        Ok((error, k_iters))
    } else {
        Err(GmresError{
            error,
            tol: threshold,
            msg: format!("GMRES did not converge. Error: {:?}. Threshold: {:?}", error, threshold)}
        )
    }
}

/// Restarted Generalized minimal residual method
pub fn restarted_gmres<'a, T, Lop: LinOp<T>>(
    a: Lop,
    b: MatRef<T>,
    mut x: MatMut<T>,
    max_iter_inner: usize,
    max_iter_outer: usize,
    threshold: T,
    m: Option<&dyn LinOp<T>>
) -> Result<(T, usize), GmresError<T>>
    where
    T: Float + RealField
{
    let mut error = T::from(1e20).unwrap();
    let mut tot_iters = 0;
    let mut iters = 0;
    for _ko in 0..max_iter_outer {
        let res = gmres(
            &a, b.as_ref(), x.rb_mut(),
            max_iter_inner, threshold, m);
        match res {
            // done
            Ok(res) => {
                (error, iters) = res;
                tot_iters += iters;
                break;
            }
            // failed to converge move to next outer iter
            // store current solution for next outer iter
            Err(res) => {
                error = res.error;
                tot_iters += max_iter_inner;
            }
        }
        if error <= threshold {
            break;
        }
    }
    if error <= threshold {
        Ok((error, tot_iters))
    } else {
        Err(GmresError{
            error,
            tol: threshold,
            msg: format!("GMRES did not converge. Error: {:?}. Threshold: {:?}", error, threshold)}
        )
    }
}


#[cfg(test)]
mod test_faer_gmres {
    use assert_approx_eq::assert_approx_eq;
    use faer::sparse::SparseColMat;

    // bring everything from above (parent) module into scope
    use super::*;

    #[test]
    fn test_gmres_1() {
        let a_test_triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
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
        let mut x0 = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            ];

        let (err, iters) = gmres(&a_test, b.as_ref(), x0.as_mut(), 10, 1e-8, None).unwrap();
        println!("Result x: {:?}", x0.as_ref());
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 10);

        // expect result for x to be [2,1,2/3]
        assert_approx_eq!(x0[(0, 0)], 2.0, 1e-10);
        assert_approx_eq!(x0[(1, 0)], 1.0, 1e-10);
        assert_approx_eq!(x0[(2, 0)], 2.0/3.0, 1e-10);
    }

    #[test]
    fn test_gmres_1b() {
        let a_test_triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
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
        let mut x0 = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            ];

        // preconditioner
        let jacobi_pre = JacobiPreconLinOp::new(a_test.as_ref());

        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_mut(), 10, 1e-8,
                                        Some(&jacobi_pre)).unwrap();
        println!("Result x: {:?}", x0.as_ref());
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 10);

        // expect result for x to be [2,1,2/3]
        assert_approx_eq!(x0[(0, 0)], 2.0, 1e-10);
        assert_approx_eq!(x0[(1, 0)], 1.0, 1e-10);
        assert_approx_eq!(x0[(2, 0)], 2.0/3.0, 1e-10);
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
                a_test_triplets.push(Triplet::new(i, j, a[(i, j)]));
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
        let mut x0 = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_mut(), 100, 1e-6, None).unwrap();
        println!("Result x: {:?}", x0.as_ref());
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 100);

        // expect result for x to be [0.037919, 0.888551, -0.657575, -0.181680, 0.292447]
        assert_approx_eq!(x0[(0, 0)], 0.037919, 1e-4);
        assert_approx_eq!(x0[(1, 0)], 0.888551, 1e-4);
        assert_approx_eq!(x0[(2, 0)], -0.657575, 1e-4);
        assert_approx_eq!(x0[(3, 0)], -0.181680, 1e-4);
        assert_approx_eq!(x0[(4, 0)], 0.292447, 1e-4);
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
                a_test_triplets.push(Triplet::new(i, j, a[(i, j)]));
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
        let mut x0: Mat<f32> = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_mut(), 100, 1e-6, None).unwrap();
        println!("Result x: {:?}", x0.as_ref());
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 100);

        // expect result for x to be [0.037919, 0.888551, -0.657575, -0.181680, 0.292447]
        assert_approx_eq!(x0[(0, 0)], 0.037919, 1e-4);
        assert_approx_eq!(x0[(1, 0)], 0.888551, 1e-4);
        assert_approx_eq!(x0[(2, 0)], -0.657575, 1e-4);
        assert_approx_eq!(x0[(3, 0)], -0.181680, 1e-4);
        assert_approx_eq!(x0[(4, 0)], 0.292447, 1e-4);
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
                a_test_triplets.push(Triplet::new(i, j, a[(i, j)]));
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
        let mut x0: Mat<f32> = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        let (err, iters) = restarted_gmres(
            a_test.as_ref(), b.as_ref(), x0.as_mut(), 3, 30,
            1e-6, None).unwrap();
        println!("Result x: {:?}", x0);
        println!("Error x: {:?}", err);
        println!("Iters : {:?}", iters);
        assert!(err < 1e-4);
        assert!(iters < 100);
        assert_approx_eq!(x0[(0, 0)], 0.037919, 1e-4);
        assert_approx_eq!(x0[(1, 0)], 0.888551, 1e-4);
        assert_approx_eq!(x0[(2, 0)], -0.657575, 1e-4);
        assert_approx_eq!(x0[(3, 0)], -0.181680, 1e-4);
        assert_approx_eq!(x0[(4, 0)], 0.292447, 1e-4);

        // initia sol guess
        let mut x0: Mat<f32> = faer::mat![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ];

        // with preconditioning
        let jacobi_pre = JacobiPreconLinOp::new(a_test.as_ref());
        let (err_precon, iters_precon) = restarted_gmres(
            a_test.as_ref(), b.as_ref(), x0.as_mut(), 3, 30,
            1e-6, Some(&jacobi_pre)).unwrap();
        assert!(iters_precon < iters);
        assert!(err_precon < 1e-4);
        assert_approx_eq!(x0[(0, 0)], 0.037919, 1e-4);
        assert_approx_eq!(x0[(1, 0)], 0.888551, 1e-4);
        assert_approx_eq!(x0[(2, 0)], -0.657575, 1e-4);
        assert_approx_eq!(x0[(3, 0)], -0.181680, 1e-4);
        assert_approx_eq!(x0[(4, 0)], 0.292447, 1e-4);
    }

    #[test]
    fn test_arnoldi() {
    }
}
