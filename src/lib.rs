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
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::linalg::matmul::matmul;
use faer::Accum;
use faer::matrix_free::{LinOp, Precond};
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
    /// Absolute residual norm reached, `||M^{-1}(b - A x)||`
    error: T,
    rtol: T,
    atol: T,
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
        write!(f, "GmresError: {}", self.msg)
    }
}

/// Jacobi (diagonal) preconditioner. Stores the precomputed inverse of the
/// diagonal of the system matrix.
#[derive(Clone,Debug)]
pub struct JacobiPrecond<T>
    where
    T: Float + RealField
{
    inv_diag: Vec<T>,
}

impl <T> LinOp<T> for JacobiPrecond<T>
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
        self.inv_diag.len()
    }

    fn ncols(&self) -> usize {
        self.inv_diag.len()
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
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                out[(i, j)] = rhs[(i, j)] * self.inv_diag[i];
            }
        }
    }

    fn conj_apply(
            &self,
            out: MatMut<'_, T>,
            rhs: MatRef<'_, T>,
            parallelism: Par,
            stack: &mut MemStack,
        ) {
        // T is real, so the conjugate is the operator itself
        self.apply(out, rhs, parallelism, stack);
    }
}

impl <T> Precond<T> for JacobiPrecond<T>
    where
    T: Float + RealField
{
    fn apply_in_place_scratch(&self, rhs_ncols: usize, parallelism: Par) -> StackReq {
        _ = rhs_ncols;
        _ = parallelism;
        StackReq::empty()
    }

    fn apply_in_place(
        &self,
        rhs: MatMut<'_, T>,
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        // unused
        _ = parallelism;
        _ = stack;

        let mut rhs = rhs;
        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                rhs[(i, j)] = rhs[(i, j)] * self.inv_diag[i];
            }
        }
    }

    fn conj_apply_in_place(
        &self,
        rhs: MatMut<'_, T>,
        parallelism: Par,
        stack: &mut MemStack,
        )
    {
        // T is real, so the conjugate is the operator itself
        self.apply_in_place(rhs, parallelism, stack);
    }
}

impl <T> JacobiPrecond<T>
    where
    T: Float + RealField
{
    /// Build the preconditioner from the diagonal of the square matrix `m_in`.
    ///
    /// Diagonal entries that are missing, zero, or negligible relative to the
    /// largest one (`|d| <= sqrt(eps) * max|d|`) are treated as 1, i.e. those
    /// rows are left unscaled (e.g. the pressure rows of a saddle point system).
    /// Dividing by such an entry would scale the row by more than `1/sqrt(eps)`
    /// relative to the others and amplify its round-off past usefulness.
    pub fn new(m_in: SparseColMatRef<'_, usize, T>) -> Self {
        assert_eq!(m_in.nrows(), m_in.ncols(), "Jacobi preconditioner requires a square matrix");
        let one_c = one::<T>();
        let diag: Vec<T> = (0..m_in.nrows())
            .map(|i| m_in.get(i, i).copied().unwrap_or(zero::<T>()))
            .collect();
        let d_max = diag.iter().fold(zero::<T>(), |m, d| Float::max(m, Float::abs(*d)));
        let cutoff = Float::sqrt(<T as Float>::epsilon()) * d_max;
        // also false for NaN, which is left unscaled
        let inv_diag = diag.iter()
            .map(|d| if Float::abs(*d) > cutoff { one_c / *d } else { one_c })
            .collect();
        Self { inv_diag }
    }
}

/// Calculate the givens rotation matrix
fn givens_rotation<T>(v1: T, v2: T) -> (T, T)
    where
    T: Float + RealField
{
    let t = v1.hypot(v2);
    // both inputs are zero, any rotation is valid: use the identity rather than 0/0
    if t == zero::<T>() {
        return (one::<T>(), zero::<T>());
    }
    let cs = v1 / t;
    let sn = v2 / t;

    return (cs, sn);
}

/// Apply givens rotation to H col
fn apply_givens_rotation<T>(mut h: ColMut<T>, cs: &mut [T], sn: &mut [T], k: usize)
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


/// Number of independent accumulators used by the unrolled kernels below.
/// Splitting a reduction into lanes lets the compiler emit SIMD code (when the
/// build enables it, see the `target-cpu=native` note in the readme) without
/// any `unsafe` or explicit intrinsics.
const LANES: usize = 8;

/// Contiguous view of a column of a column-major matrix
fn col_slice<'a, T>(c: ColRef<'a, T>) -> &'a [T] {
    c.try_as_col_major().expect("krylov basis must be column major").as_slice()
}

/// Mutable contiguous view of a column of a column-major matrix
fn col_slice_mut<'a, T>(c: ColMut<'a, T>) -> &'a mut [T] {
    c.try_as_col_major_mut().expect("krylov basis must be column major").as_slice_mut()
}

/// Dot product: `v . w`
fn dot<T>(v: &[T], w: &[T]) -> T
    where
    T: Float + RealField
{
    debug_assert_eq!(v.len(), w.len());
    let mut acc = [zero::<T>(); LANES];
    let mut v_chunks = v.chunks_exact(LANES);
    let mut w_chunks = w.chunks_exact(LANES);
    for (vc, wc) in (&mut v_chunks).zip(&mut w_chunks) {
        for l in 0..LANES {
            acc[l] = acc[l] + vc[l] * wc[l];
        }
    }
    let tail = v_chunks.remainder().iter().zip(w_chunks.remainder())
        .fold(zero::<T>(), |s, (a, b)| s + *a * *b);
    acc.iter().fold(tail, |s, a| s + *a)
}

/// In place update: `v <- v - r * q`
fn sub_scaled<T>(v: &mut [T], q: &[T], r: T)
    where
    T: Float + RealField
{
    debug_assert_eq!(v.len(), q.len());
    for (v, q) in v.iter_mut().zip(q) {
        *v = *v - r * *q;
    }
}

/// Fused Gram-Schmidt step: performs `v <- v - r * q` and returns the dot
/// product of the updated `v` with `q_next`, in a single pass over `v`.
///
/// Fusing the update with the next projection means each basis vector is used
/// twice back to back, so it is streamed from main memory only once. This is
/// what makes orthogonalization (the dominant cost for long Krylov spaces)
/// fast, and it is mathematically identical to doing the two steps separately.
fn sub_scaled_dot<T>(v: &mut [T], q: &[T], q_next: &[T], r: T) -> T
    where
    T: Float + RealField
{
    debug_assert_eq!(v.len(), q.len());
    debug_assert_eq!(v.len(), q_next.len());
    let mut acc = [zero::<T>(); LANES];
    let mut v_chunks = v.chunks_exact_mut(LANES);
    let mut q_chunks = q.chunks_exact(LANES);
    let mut n_chunks = q_next.chunks_exact(LANES);
    for ((vc, qc), nc) in (&mut v_chunks).zip(&mut q_chunks).zip(&mut n_chunks) {
        for l in 0..LANES {
            vc[l] = vc[l] - r * qc[l];
            acc[l] = acc[l] + vc[l] * nc[l];
        }
    }
    let mut tail = zero::<T>();
    for ((v, q), n) in v_chunks.into_remainder().iter_mut()
        .zip(q_chunks.remainder()).zip(n_chunks.remainder())
    {
        *v = *v - r * *q;
        tail = tail + *v * *n;
    }
    acc.iter().fold(tail, |s, a| s + *a)
}

/// Arnoldi decomposition for sparse matrices
///
/// The new krylov vector is orthogonalized against the basis with a single
/// pass of modified Gram-Schmidt, without reorthogonalization. In single
/// precision, over long krylov spaces, the basis can lose orthogonality and
/// slow convergence.
///
/// Returns `true` on breakdown: the new krylov vector is (numerically) in the
/// span of the prior ones, so no new direction was added to the basis and the
/// vector is left unnormalized.
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
    q: MatMut<T>,
    k: usize,
    m: Option<&dyn Precond<T>>,
    par: Par,
    stack: &mut MemStack
    ) -> bool
    where
    T: Float + RealField
{
    // Split the basis so the new Krylov vector (column k+1) is built in place
    // from the prior vectors (columns 0..=k)
    let (prev, mut next) = q.split_at_col_mut(k+1);
    let mut qv = next.rb_mut().col_mut(0);

    a.apply(qv.rb_mut().as_mat_mut(), prev.rb().col(k).as_mat(), par, stack);

    // Apply left preconditioner in place if supplied
    if let Some(m) = m {
        m.apply_in_place(qv.rb_mut().as_mat_mut(), par, stack);
    }

    // Modified Gram-Schmidt. The update with q_i is fused with the projection
    // onto q_{i+1} so each prior krylov vector is read from memory once.
    let mut h = H.rb_mut().col_mut(k);
    {
        let v = col_slice_mut(qv.rb_mut());
        let mut r = dot(v, col_slice(prev.rb().col(0)));
        for i in 0..k {
            h[i] = r;
            r = sub_scaled_dot(
                v, col_slice(prev.rb().col(i)), col_slice(prev.rb().col(i+1)), r);
        }
        h[k] = r;
        sub_scaled(v, col_slice(prev.rb().col(k)), r);
    }

    // Breakdown is judged relative to the size of the vector before it was
    // orthogonalized, ||M^-1 A q_k||, so that it does not depend on the scale
    // of the operator. For an orthonormal basis this is
    // sqrt(sum_i h_i^2 + ||v||^2), so it is recovered from the projection
    // coefficients without another pass over the vector. hypot avoids
    // overflow/underflow for very large/small operators.
    let norm_v = qv.rb().norm_l2();
    let w_norm = (0..=k).fold(norm_v, |w, i| w.hypot(h[i]));
    let breakdown_tol = <T as Float>::epsilon() * T::from(10.0).unwrap();
    // `<=` so that A q_k == 0 (w_norm == 0) is also flagged
    let breakdown_flag: bool = norm_v <= breakdown_tol * w_norm;

    h[k+1] = norm_v;

    if !breakdown_flag {
        let inv_norm = one::<T>()/norm_v;
        zip!(qv.rb_mut()).for_each(|unzip!(mut y)| *y = *y * inv_norm);
    }
    breakdown_flag
}


/// Panics unless both tolerances are non-negative (this also rejects NaN)
fn check_tolerances<T>(rtol: T, atol: T)
    where
    T: Float + RealField
{
    assert!(
        rtol >= zero::<T>() && atol >= zero::<T>(),
        "rtol and atol must be non-negative");
}

/// Generalized minimal residual method
///
/// Solves `A x = b`. `x` holds the initial guess on entry and the solution on
/// exit. An optional left preconditioner `M` turns the system into
/// `M^{-1} A x = M^{-1} b`.
///
/// Iteration stops once the residual norm satisfies
///
/// `||M^{-1}(b - A x)|| <= max(atol, rtol * ||M^{-1} b||)`
///
/// i.e. `rtol` is a tolerance relative to the size of the right hand side and
/// `atol` is an absolute floor. Pass `atol = 0` for a purely relative test, or
/// `rtol = 0` for a purely absolute one. Both must be non-negative (a panic
/// otherwise). The norms are those of the preconditioned residual, the
/// quantity GMRES minimizes and tracks, so with a preconditioner `atol` is in
/// units of `M^{-1}(b - A x)`, not of the true residual `b - A x`. Without a
/// preconditioner the two are the same.
///
/// On success returns the achieved absolute residual norm
/// `||M^{-1}(b - A x)||` and the iteration count. Divide by `||M^{-1} b||` for
/// the relative residual.
///
/// Returns an error if the residual norm is above the tolerance after
/// `max_iter` iterations, or if the Krylov subspace was exhausted (breakdown)
/// before reaching it, which for a singular system means no further progress
/// is possible.
pub fn gmres<'a, T, Lop: LinOp<T>>(
    a: Lop,
    b: MatRef<T>,
    mut x: MatMut<T>,
    max_iter: usize,
    rtol: T,
    atol: T,
    m: Option<&dyn Precond<T>>,
    par: Par
) -> Result<(T, usize), GmresError<T>>
    where
    T: Float + RealField
{
    check_tolerances(rtol, atol);
    let n = b.nrows();
    let m_scratch = m.map_or(StackReq::empty(), |m| m.apply_in_place_scratch(1, par));
    let stackreq = a.apply_scratch(1, par).or(m_scratch);
    let mut buff = MemBuffer::new(stackreq);
    let stack = MemStack::new(&mut buff);

    let mut hs = Mat::zeros(max_iter+1, max_iter);
    let mut qs = Mat::zeros(n, max_iter+1);
    let breakdown_tol = <T as Float>::epsilon() * T::from(10.0).unwrap();

    // The residual tracked by GMRES is the preconditioned one, M^{-1}(b - A x),
    // so the relative tolerance is taken against ||M^{-1} b|| (which is just
    // ||b|| when there is no preconditioner).
    let b_norm = match m {
        Some(m) => {
            let mut pb = b.to_owned();
            m.apply_in_place(pb.as_mut(), par, stack);
            pb.norm_l2()
        }
        None => b.norm_l2(),
    };
    // converged once the residual norm is at or below this
    let target = Float::max(atol, rtol * b_norm);

    // compute the (preconditioned) initial residual directly in the first
    // column of the krylov basis and normalize it in place
    let r_norm = {
        let mut r = qs.col_mut(0).as_mat_mut();
        a.apply(r.rb_mut(), x.as_ref(), par, stack);
        zip!(r.rb_mut(), b).for_each(|unzip!(mut r, b)| *r = *b - *r);

        // apply left preconditioner in place if supplied
        if let Some(m) = m {
            m.apply_in_place(r.rb_mut(), par, stack);
        }

        let r_norm = r.rb().norm_l2();
        if r_norm != zero::<T>() {
            let inv_r_norm = one::<T>() / r_norm;
            zip!(r.rb_mut()).for_each(|unzip!(mut r)| *r = *r * inv_r_norm);
        }
        r_norm
    };

    // the initial guess already meets the tolerance (this includes an exact
    // solution), there is no krylov space to build
    if r_norm <= target {
        return Ok((r_norm, 0));
    }
    let mut error = r_norm;

    // Initialize 1D vectors
    let mut sn: Vec<T> = vec![zero(); max_iter];
    let mut cs: Vec<T> = vec![zero(); max_iter];

    // residual vector: beta = r_norm * e1
    let mut beta: Mat<T> = Mat::zeros(max_iter+1, 1);
    beta[(0, 0)] = r_norm;

    let mut k_iters = 0;
    let mut broke_down = false;
    for k in 0..max_iter {
        let brkdwn = arnoldi(&a, hs.as_mut(), qs.as_mut(), k, m, par, stack);

        // scale of the new Hessenberg column, taken before the rotations are applied
        let col_norm = hs.col(k).subrows(0, k + 2).norm_l2();
        apply_givens_rotation(hs.col_mut(k), &mut cs, &mut sn, k);

        // Hard breakdown: no new direction and the rotated diagonal entry is
        // round-off, so H is singular. The column is unusable (solving with it
        // would divide by ~0), so drop it and stop with the previous iterate.
        if brkdwn && hs[(k, k)].abs() <= breakdown_tol * col_norm {
            broke_down = true;
            break;
        }

        // Update the residual vector
        beta[(k+1, 0)] = -sn[k] * beta[(k, 0)];
        beta[(k, 0)] = cs[k] * beta[(k, 0)];
        error = (beta[(k + 1, 0)]).abs();

        k_iters += 1;
        if error <= target {
            break;
        }
        // breakdown (no new Krylov direction left) without reaching the tolerance
        if brkdwn {
            broke_down = true;
            break;
        }
    }

    // compute solution: H is upper triangular after the givens rotations
    if k_iters > 0 {
        let H = hs.get(0..k_iters, 0..k_iters);
        let V = qs.get(0..n, 0..k_iters);

        let mut y = beta.get(0..k_iters, 0..1).to_owned();
        solve_upper_triangular_in_place(H, y.as_mut(), Par::Seq);

        matmul(x.rb_mut(), Accum::Add, V, y.as_ref(), one::<T>(), Par::Seq);
    }
    if error <= target {
        Ok((error, k_iters))
    } else {
        let msg = if broke_down {
            format!(
                "GMRES broke down (Krylov subspace exhausted, system may be singular) \
                 before reaching the tolerance. Residual norm: {:?}. Target: {:?} \
                 (rtol: {:?}, atol: {:?})", error, target, rtol, atol)
        } else {
            format!(
                "GMRES did not converge. Residual norm: {:?}. Target: {:?} \
                 (rtol: {:?}, atol: {:?})", error, target, rtol, atol)
        };
        Err(GmresError{ error, rtol, atol, msg })
    }
}

/// Restarted Generalized minimal residual method
///
/// Runs [`gmres`] for up to `max_iter_inner` iterations at a time, restarting
/// from the current iterate for at most `max_iter_outer` cycles. `x` holds the
/// initial guess on entry and the solution on exit.
///
/// `rtol` and `atol` have the same meaning as in [`gmres`]: a cycle converges
/// once `||M^{-1}(b - A x)|| <= max(atol, rtol * ||M^{-1} b||)`. Both must be
/// non-negative (a panic otherwise).
///
/// On success returns the achieved absolute residual norm and the total
/// iteration count over all cycles.
pub fn restarted_gmres<'a, T, Lop: LinOp<T>>(
    a: Lop,
    b: MatRef<T>,
    mut x: MatMut<T>,
    max_iter_inner: usize,
    max_iter_outer: usize,
    rtol: T,
    atol: T,
    m: Option<&dyn Precond<T>>,
    par: Par
) -> Result<(T, usize), GmresError<T>>
    where
    T: Float + RealField
{
    check_tolerances(rtol, atol);
    let mut error = T::from(1e20).unwrap();
    let mut tot_iters = 0;
    let mut converged = false;
    for _ko in 0..max_iter_outer {
        let res = gmres(
            &a, b.as_ref(), x.rb_mut(),
            max_iter_inner, rtol, atol, m, par);
        match res {
            // done
            Ok((res_norm, iters)) => {
                error = res_norm;
                tot_iters += iters;
                converged = true;
                break;
            }
            // failed to converge move to next outer iter
            // store current solution for next outer iter
            Err(res) => {
                error = res.error;
                tot_iters += max_iter_inner;
            }
        }
    }
    if converged {
        Ok((error, tot_iters))
    } else {
        Err(GmresError{
            error,
            rtol,
            atol,
            msg: format!(
                "GMRES did not converge. Residual norm: {:?}. rtol: {:?}, atol: {:?}",
                error, rtol, atol)}
        )
    }
}


#[cfg(test)]
mod test_faer_gmres {
    use assert_approx_eq::assert_approx_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use std::panic::AssertUnwindSafe;

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

        let par = faer::get_global_parallelism();
        let (err, iters) = gmres(&a_test, b.as_ref(), x0.as_mut(), 10, 1e-8, 0.0, None, par).unwrap();
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
        let jacobi_pre = JacobiPrecond::new(a_test.as_ref());

        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_mut(), 10, 1e-8, 0.0,
            Some(&jacobi_pre), Par::Seq).unwrap();
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

        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_mut(), 100, 1e-6, 0.0, None, Par::Seq).unwrap();
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

        let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x0.as_mut(), 100, 1e-6, 0.0, None, Par::Seq).unwrap();
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
            1e-6, 0.0, None, Par::Seq).unwrap();
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
        let jacobi_pre = JacobiPrecond::new(a_test.as_ref());
        let (err_precon, iters_precon) = restarted_gmres(
            a_test.as_ref(), b.as_ref(), x0.as_mut(), 3, 30,
            1e-6, 0.0, Some(&jacobi_pre), Par::Seq).unwrap();
        assert!(iters_precon < iters);
        assert!(err_precon < 1e-4);
        assert_approx_eq!(x0[(0, 0)], 0.037919, 1e-4);
        assert_approx_eq!(x0[(1, 0)], 0.888551, 1e-4);
        assert_approx_eq!(x0[(2, 0)], -0.657575, 1e-4);
        assert_approx_eq!(x0[(3, 0)], -0.181680, 1e-4);
        assert_approx_eq!(x0[(4, 0)], 0.292447, 1e-4);
    }

    #[test]
    fn test_jacobi_apply_in_place() {
        let a_test_triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 4.0),
            Triplet::new(0, 2, 5.0),
            ];
        let a_test = SparseColMat::<usize, f64>::try_new_from_triplets(
            3, 3,
            &a_test_triplets).unwrap();
        let pre = JacobiPrecond::new(a_test.as_ref());

        let rhs = faer::mat![
            [1.0, 2.0],
            [4.0, 8.0],
            [8.0, 16.0],
            ];
        let par = Par::Seq;
        let mut out = Mat::<f64>::zeros(3, 2);
        pre.apply(out.as_mut(), rhs.as_ref(), par, MemStack::new(&mut []));

        let mut in_place = rhs.clone();
        pre.apply_in_place(in_place.as_mut(), par, MemStack::new(&mut []));

        for j in 0..2 {
            for i in 0..3 {
                assert_approx_eq!(in_place[(i, j)], out[(i, j)], 1e-12);
            }
        }
        assert_approx_eq!(in_place[(0, 0)], 1.0, 1e-9);
        assert_approx_eq!(in_place[(1, 1)], 4.0, 1e-9);
        assert_approx_eq!(in_place[(2, 1)], 4.0, 1e-9);
    }

    #[test]
    fn test_jacobi_zero_and_missing_diagonal() {
        // row 1 has an explicitly stored zero diagonal, row 3 has none at all:
        // both must be left unscaled rather than blown up by 1/0
        let triplets = vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 1, 0.0),
            Triplet::new(1, 0, 5.0),
            Triplet::new(2, 2, 4.0),
            Triplet::new(3, 2, 7.0),
            ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(4, 4, &triplets).unwrap();
        let pre = JacobiPrecond::new(a.as_ref());

        let mut rhs = Mat::<f64>::from_fn(4, 1, |_, _| 1.0);
        pre.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_eq!(
            [rhs[(0, 0)], rhs[(1, 0)], rhs[(2, 0)], rhs[(3, 0)]],
            [0.5, 1.0, 0.25, 1.0]);
    }

    #[test]
    fn test_jacobi_negligible_diagonal() {
        // cutoff is sqrt(eps) * max|d|: ~1.5e-8 * 4 = 6e-8 for f64
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0e-12), // negligible: left unscaled
            Triplet::new(2, 2, -4.0),    // largest magnitude, negative sign kept
            Triplet::new(3, 3, 1.0e-6),  // small but above the cutoff: scaled
            ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(4, 4, &triplets).unwrap();
        let pre = JacobiPrecond::new(a.as_ref());
        let mut rhs = Mat::<f64>::from_fn(4, 1, |_, _| 1.0);
        pre.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_approx_eq!(rhs[(0, 0)], 1.0, 1e-12);
        assert_approx_eq!(rhs[(1, 0)], 1.0, 1e-12);
        assert_approx_eq!(rhs[(2, 0)], -0.25, 1e-12);
        assert_approx_eq!(rhs[(3, 0)], 1.0e6, 1e-3);

        // the cutoff follows the precision: sqrt(eps) is ~3.4e-4 for f32
        let triplets = vec![
            Triplet::new(0, 0, 1.0f32),
            Triplet::new(1, 1, 1.0e-5), // below cutoff
            Triplet::new(2, 2, 1.0e-2), // above cutoff
            ];
        let a = SparseColMat::<usize, f32>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let pre = JacobiPrecond::new(a.as_ref());
        let mut rhs = Mat::<f32>::from_fn(3, 1, |_, _| 1.0);
        pre.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_approx_eq!(rhs[(0, 0)], 1.0, 1e-6);
        assert_approx_eq!(rhs[(1, 0)], 1.0, 1e-6);
        assert_approx_eq!(rhs[(2, 0)], 100.0, 1e-3);

        // nothing usable on the diagonal: identity preconditioner
        let triplets = vec![Triplet::new(0, 1, 3.0), Triplet::new(1, 0, 2.0)];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &triplets).unwrap();
        let pre = JacobiPrecond::new(a.as_ref());
        let mut rhs = Mat::<f64>::from_fn(2, 1, |i, _| 1.0 + i as f64);
        pre.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_eq!([rhs[(0, 0)], rhs[(1, 0)]], [1.0, 2.0]);
    }

    #[test]
    fn test_orthogonalization_kernels() {
        // lengths cover empty, shorter than a lane group, exact multiples and tails
        for n in [0usize, 1, 5, 8, 9, 16, 37, 100] {
            let v0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();
            let q: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).cos()).collect();
            let qn: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();
            let r = 0.75;

            let naive_dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
            assert_approx_eq!(dot(&v0, &q), naive_dot(&v0, &q), 1e-12);

            let expect: Vec<f64> = v0.iter().zip(&q).map(|(v, q)| v - r * q).collect();

            let mut v = v0.clone();
            sub_scaled(&mut v, &q, r);
            for i in 0..n { assert_approx_eq!(v[i], expect[i], 1e-14); }

            let mut v = v0.clone();
            let d = sub_scaled_dot(&mut v, &q, &qn, r);
            for i in 0..n { assert_approx_eq!(v[i], expect[i], 1e-14); }
            assert_approx_eq!(d, naive_dot(&expect, &qn), 1e-12);
        }
    }

    #[test]
    fn test_gmres_large_nonsymmetric() {
        // 1D convection-diffusion style tridiagonal system, large enough that the
        // orthogonalization loops run over many krylov vectors
        let n = 300;
        let mut triplets = vec![];
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 2.0));
            if i > 0 { triplets.push(Triplet::new(i, i - 1, -1.2)); }
            if i + 1 < n { triplets.push(Triplet::new(i, i + 1, -0.8)); }
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| 1.0 + (i % 7) as f64);

        for precond in [false, true] {
            let mut x = Mat::<f64>::zeros(n, 1);
            let jacobi = JacobiPrecond::new(a.as_ref());
            let m: Option<&dyn Precond<f64>> = if precond { Some(&jacobi) } else { None };
            let (err, iters) = gmres(
                a.as_ref(), b.as_ref(), x.as_mut(), n, 1e-10, 0.0, m, Par::Seq).unwrap();
            assert!(err <= 1e-10 * b.norm_l2());
            assert!(iters <= n);

            // true (unpreconditioned) residual: ||Ax - b|| / ||b||
            let mut res = b.clone();
            for t in &triplets {
                res[(t.row, 0)] -= t.val * x[(t.col, 0)];
            }
            assert!(res.norm_l2() / b.norm_l2() < 1e-8);
        }
    }

    /// Run `arnoldi` on diag(1, 2, 3) * scale starting from q0 = [1, 1, 1] / sqrt(3)
    /// and return the breakdown flag of each of the 3 steps.
    fn arnoldi_breakdown_flags<T>(scale: T) -> Vec<bool>
        where
        T: Float + RealField
    {
        let n = 3;
        let triplets: Vec<Triplet<usize, usize, T>> = (0..n)
            .map(|i| Triplet::new(i, i, scale * T::from(i + 1).unwrap()))
            .collect();
        let a = SparseColMat::<usize, T>::try_new_from_triplets(n, n, &triplets).unwrap();
        let a = a.as_ref();

        let par = Par::Seq;
        let mut buff = MemBuffer::new(a.apply_scratch(1, par));
        let stack = MemStack::new(&mut buff);

        let mut h = Mat::<T>::zeros(n + 1, n);
        let mut q = Mat::<T>::zeros(n, n + 1);
        let q0 = one::<T>() / Float::sqrt(T::from(3.0).unwrap());
        for i in 0..n {
            q[(i, 0)] = q0;
        }

        let mut flags = vec![];
        for k in 0..n {
            flags.push(arnoldi(&a, h.as_mut(), q.as_mut(), k, None, par, stack));
        }
        flags
    }

    #[test]
    fn test_arnoldi() {
        // Three distinct eigenvalues and a start vector touching all of them:
        // the krylov space is the whole space, so the third step (and only the
        // third) must break down. The result must not depend on the scale of
        // the operator or on the working precision.
        for scale in [1.0e-14, 1.0, 1.0e10] {
            assert_eq!(arnoldi_breakdown_flags::<f64>(scale),
                vec![false, false, true], "f64, scale {:e}", scale);
            assert_eq!(arnoldi_breakdown_flags::<f32>(scale as f32),
                vec![false, false, true], "f32, scale {:e}", scale);
        }
    }

    #[test]
    fn test_givens_rotation_zero() {
        let (cs, sn) = givens_rotation(0.0_f64, 0.0);
        assert_eq!((cs, sn), (1.0, 0.0));
        let (cs, sn) = givens_rotation(3.0_f64, 4.0);
        assert_approx_eq!(cs, 0.6, 1e-14);
        assert_approx_eq!(sn, 0.8, 1e-14);
    }

    #[test]
    fn test_gmres_scaled_operator() {
        // Breakdown is judged relative to the size of the krylov vector, so the
        // solver must behave the same however the operator is scaled. With an
        // absolute tolerance the 1e-14 case is flagged as breakdown immediately.
        let n = 50;
        for scale in [1.0e-14, 1.0, 1.0e10] {
            let mut triplets = vec![];
            for i in 0..n {
                triplets.push(Triplet::new(i, i, 2.0 * scale));
                if i > 0 { triplets.push(Triplet::new(i, i - 1, -1.2 * scale)); }
                if i + 1 < n { triplets.push(Triplet::new(i, i + 1, -0.8 * scale)); }
            }
            let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
            let b = Mat::<f64>::from_fn(n, 1, |i, _| 1.0 + (i % 7) as f64);

            let mut x = Mat::<f64>::zeros(n, 1);
            let (err, iters) = gmres(
                a.as_ref(), b.as_ref(), x.as_mut(), n, 1e-10, 0.0, None, Par::Seq)
                .unwrap_or_else(|e| panic!("scale {:e}: {}", scale, e));
            assert!(err <= 1e-10 * b.norm_l2());
            assert!(iters <= n);

            let mut res = b.clone();
            for t in &triplets {
                res[(t.row, 0)] -= t.val * x[(t.col, 0)];
            }
            assert!(res.norm_l2() / b.norm_l2() < 1e-8, "scale {:e}", scale);
        }
    }

    #[test]
    fn test_gmres_singular_system() {
        // diag(1, 0, 3) with a rhs that has a component in the null space: no
        // solution exists. The krylov space still fills R^3, so the third step
        // breaks down with a singular Hessenberg matrix. This must be reported
        // as an error with a finite iterate, not NaN/inf from dividing by ~0.
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(2, 2, 3.0),
            ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let b = faer::mat![[1.0], [1.0], [1.0]];
        let mut x = Mat::<f64>::zeros(3, 1);

        let err = gmres(a.as_ref(), b.as_ref(), x.as_mut(), 10, 1e-10, 0.0, None, Par::Seq)
            .unwrap_err();
        assert!(err.error.is_finite());
        assert!(err.error > 1e-10);
        assert!(err.msg.contains("broke down"), "{}", err);
        for i in 0..3 {
            assert!(x[(i, 0)].is_finite(), "x = {:?}", x);
        }
        // dividing by the round-off sized diagonal would blow the iterate up to ~1e15
        assert!(x.norm_l2() < 1e3, "x = {:?}", x);
    }

    #[test]
    fn test_gmres_exact_initial_guess() {
        // a zero initial residual used to divide by zero and fill the basis with NaN
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
            ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let b = faer::mat![[1.0], [2.0], [3.0]];
        let jacobi = JacobiPrecond::new(a.as_ref());

        for m in [None, Some(&jacobi as &dyn Precond<f64>)] {
            let mut x = faer::mat![[1.0], [1.0], [1.0]];
            let (err, iters) = gmres(a.as_ref(), b.as_ref(), x.as_mut(), 10, 1e-10, 0.0, m, Par::Seq)
                .unwrap();
            assert_eq!(err, 0.0);
            assert_eq!(iters, 0);
            for i in 0..3 {
                assert_eq!(x[(i, 0)], 1.0);
            }

            // zero rhs and zero guess
            let mut x = Mat::<f64>::zeros(3, 1);
            let b0 = Mat::<f64>::zeros(3, 1);
            let (err, iters) = gmres(a.as_ref(), b0.as_ref(), x.as_mut(), 10, 1e-10, 0.0, m, Par::Seq)
                .unwrap();
            assert_eq!((err, iters), (0.0, 0));
        }
    }

    #[test]
    fn test_gmres_preconditioned_residual() {
        // Rows with very different scales so that ||M^-1 b|| and ||b|| are far
        // apart. The reported error must be the absolute *preconditioned*
        // residual ||M^-1 (b - A x)||, i.e. what GMRES tracks, and rtol must
        // be taken relative to ||M^-1 b||.
        let n = 20;
        let row_scale = |i: usize| 10f64.powi((i % 5) as i32);
        let mut triplets = vec![];
        for i in 0..n {
            let s = row_scale(i);
            triplets.push(Triplet::new(i, i, 2.0 * s));
            if i > 0 { triplets.push(Triplet::new(i, i - 1, -1.2 * s)); }
            if i + 1 < n { triplets.push(Triplet::new(i, i + 1, -0.8 * s)); }
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| row_scale(i) * (1.0 + (i % 3) as f64));
        let jacobi = JacobiPrecond::new(a.as_ref());

        // ||M^-1 b|| differs a lot from ||b||
        let mut pb = b.clone();
        jacobi.apply_in_place(pb.as_mut(), Par::Seq, MemStack::new(&mut []));
        let pb_norm = pb.norm_l2();
        assert!(pb_norm < 1e-2 * b.norm_l2());

        // absolute preconditioned residual of an iterate, computed independently
        let precond_res = |x: &Mat<f64>| {
            let mut res = b.clone();
            for t in &triplets {
                res[(t.row, 0)] -= t.val * x[(t.col, 0)];
            }
            jacobi.apply_in_place(res.as_mut(), Par::Seq, MemStack::new(&mut []));
            res.norm_l2()
        };

        // too few iterations to converge: error is the Givens estimate at the last step
        let mut x = Mat::<f64>::zeros(n, 1);
        let err = gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), 4, 1e-12, 0.0, Some(&jacobi), Par::Seq)
            .unwrap_err();
        let expect = precond_res(&x);
        assert!(expect > 1e-6 * pb_norm);
        assert!((err.error - expect).abs() <= 1e-6 * expect,
            "reported {:e}, actual {:e}", err.error, expect);

        // converged: the reported error meets the relative tolerance (against
        // ||M^-1 b||, not ||b||) and so does the true value
        let mut x = Mat::<f64>::zeros(n, 1);
        let (err, _) = gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), n, 1e-10, 0.0, Some(&jacobi), Par::Seq).unwrap();
        assert!(err <= 1e-10 * pb_norm);
        assert!(precond_res(&x) < 1e-8 * pb_norm);

        // absolute tolerance is also in units of the preconditioned residual
        let atol = 1e-9;
        let mut x = Mat::<f64>::zeros(n, 1);
        let (err, _) = gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), n, 0.0, atol, Some(&jacobi), Par::Seq).unwrap();
        assert!(err <= atol);
        assert!(precond_res(&x) <= 2.0 * atol, "residual {:e}", precond_res(&x));
    }

    /// Nonsymmetric tridiagonal test system and its triplets
    fn tridiagonal(n: usize) -> (SparseColMat<usize, f64>, Vec<Triplet<usize, usize, f64>>) {
        let mut triplets = vec![];
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 2.0));
            if i > 0 { triplets.push(Triplet::new(i, i - 1, -1.2)); }
            if i + 1 < n { triplets.push(Triplet::new(i, i + 1, -0.8)); }
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();
        (a, triplets)
    }

    /// ||b - A x||, computed independently of the solver
    fn true_residual_norm(
        triplets: &[Triplet<usize, usize, f64>], b: &Mat<f64>, x: &Mat<f64>) -> f64
    {
        let mut res = b.clone();
        for t in triplets {
            res[(t.row, 0)] -= t.val * x[(t.col, 0)];
        }
        res.norm_l2()
    }

    #[test]
    fn test_gmres_atol_only() {
        let n = 100;
        let (a, triplets) = tridiagonal(n);
        let b = Mat::<f64>::from_fn(n, 1, |i, _| 1.0 + (i % 7) as f64);

        let solve = |atol: f64| {
            let mut x = Mat::<f64>::zeros(n, 1);
            let (err, iters) = gmres(
                a.as_ref(), b.as_ref(), x.as_mut(), n, 0.0, atol, None, Par::Seq).unwrap();
            assert!(err <= atol);
            // no preconditioner: the tracked residual is the true residual
            let res = true_residual_norm(&triplets, &b, &x);
            assert!(res <= 1.01 * atol, "atol {:e}, residual {:e}", atol, res);
            iters
        };
        let loose = solve(1e-2);
        let tight = solve(1e-8);
        assert!(loose < tight, "loose {}, tight {}", loose, tight);
    }

    #[test]
    fn test_gmres_rtol_atol_max() {
        // stops at max(atol, rtol * ||b||): whichever is looser decides
        let n = 100;
        let (a, _) = tridiagonal(n);
        let b = Mat::<f64>::from_fn(n, 1, |i, _| 1.0 + (i % 7) as f64);
        let b_norm = b.norm_l2();

        let solve = |rtol: f64, atol: f64| {
            let mut x = Mat::<f64>::zeros(n, 1);
            let (err, iters) = gmres(
                a.as_ref(), b.as_ref(), x.as_mut(), n, rtol, atol, None, Par::Seq).unwrap();
            assert!(err <= f64::max(atol, rtol * b_norm));
            iters
        };

        // rtol dominates: a negligible atol changes nothing
        assert_eq!(solve(1e-4, 1e-10), solve(1e-4, 0.0));
        // atol dominates: a negligible rtol changes nothing
        assert_eq!(solve(1e-10, 1e-4), solve(0.0, 1e-4));
        // and the looser of the two is what is used
        let both = solve(1e-10, 1e-4);
        assert!(both < solve(1e-10, 0.0), "atol did not loosen the test");
        let both = solve(1e-4, 1e-10);
        assert!(both < solve(0.0, 1e-10), "rtol did not loosen the test");
    }

    #[test]
    fn test_gmres_atol_initial_guess_within_tolerance() {
        // ||b|| ~ 4e-8: the zero initial guess is already within atol, so no
        // iterations are done and x is untouched. A relative test must still iterate.
        let n = 100;
        let (a, _) = tridiagonal(n);
        let b = Mat::<f64>::from_fn(n, 1, |i, _| 1e-9 * (1.0 + (i % 7) as f64));

        let mut x = Mat::<f64>::zeros(n, 1);
        let (err, iters) = gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), n, 1e-6, 1e-6, None, Par::Seq).unwrap();
        assert_eq!(iters, 0);
        assert_approx_eq!(err, b.norm_l2(), 1e-15);
        assert_eq!(x.norm_l2(), 0.0);

        let mut x = Mat::<f64>::zeros(n, 1);
        let (_, iters) = gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), n, 1e-6, 0.0, None, Par::Seq).unwrap();
        assert!(iters > 0);
    }

    #[test]
    fn test_gmres_zero_rhs_nonzero_guess() {
        // b = 0 makes any relative tolerance vanish, only atol can be met. The
        // solution is x = 0.
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
            ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let b = Mat::<f64>::zeros(3, 1);
        let mut x = faer::mat![[1.0], [1.0], [1.0]];

        let (err, iters) = gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), 10, 1e-8, 1e-8, None, Par::Seq).unwrap();
        assert!(err <= 1e-8);
        assert!(iters > 0);
        assert!(x.norm_l2() < 1e-7, "x = {:?}", x);
    }

    #[test]
    fn test_restarted_gmres_atol() {
        let a = faer::mat![
            [0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
            [0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
            [0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
            [0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
            [0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
            ];
        let mut triplets = vec![];
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                triplets.push(Triplet::new(i, j, a[(i, j)]));
            }
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(5, 5, &triplets).unwrap();
        let b = faer::mat![[0.104594], [0.437549], [0.040264], [0.298842], [0.254451]];

        // absolute only, then both
        for (rtol, atol) in [(0.0, 1e-8), (1e-8, 1e-8)] {
            let mut x = Mat::<f64>::zeros(5, 1);
            let (err, iters) = restarted_gmres(
                a.as_ref(), b.as_ref(), x.as_mut(), 3, 100, rtol, atol, None, Par::Seq).unwrap();
            assert!(err <= 1e-8);
            assert!(iters > 0);
            assert!(true_residual_norm(&triplets, &b, &x) <= 1.01e-8);
            assert_approx_eq!(x[(0, 0)], 0.037919, 1e-4);
            assert_approx_eq!(x[(1, 0)], 0.888551, 1e-4);
            assert_approx_eq!(x[(2, 0)], -0.657575, 1e-4);
            assert_approx_eq!(x[(3, 0)], -0.181680, 1e-4);
            assert_approx_eq!(x[(4, 0)], 0.292447, 1e-4);
        }
    }

    #[test]
    fn test_gmres_error_reports_tolerances() {
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
            ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let b = faer::mat![[1.0], [1.0], [1.0]];

        // a single iteration cannot reach this
        let mut x = Mat::<f64>::zeros(3, 1);
        let err = gmres(a.as_ref(), b.as_ref(), x.as_mut(), 1, 1e-10, 1e-12, None, Par::Seq)
            .unwrap_err();
        assert_eq!((err.rtol, err.atol), (1e-10, 1e-12));
        assert!(err.error > 1e-10 * b.norm_l2());
        assert!(err.msg.contains("rtol") && err.msg.contains("atol"), "{}", err);

        let mut x = Mat::<f64>::zeros(3, 1);
        let err = restarted_gmres(
            a.as_ref(), b.as_ref(), x.as_mut(), 1, 1, 1e-10, 1e-12, None, Par::Seq)
            .unwrap_err();
        assert_eq!((err.rtol, err.atol), (1e-10, 1e-12));
        assert!(err.msg.contains("rtol") && err.msg.contains("atol"), "{}", err);
    }

    #[test]
    fn test_gmres_invalid_tolerances() {
        let triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 2.0)];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &triplets).unwrap();
        let b = faer::mat![[1.0], [1.0]];

        // negative and NaN tolerances are rejected by both entry points
        for (rtol, atol) in [(-1e-8, 0.0), (0.0, -1e-8), (f64::NAN, 0.0), (0.0, f64::NAN)] {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut x = Mat::<f64>::zeros(2, 1);
                let _ = gmres(a.as_ref(), b.as_ref(), x.as_mut(), 5, rtol, atol, None, Par::Seq);
            }));
            assert!(r.is_err(), "gmres accepted rtol {}, atol {}", rtol, atol);

            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut x = Mat::<f64>::zeros(2, 1);
                let _ = restarted_gmres(
                    a.as_ref(), b.as_ref(), x.as_mut(), 5, 5, rtol, atol, None, Par::Seq);
            }));
            assert!(r.is_err(), "restarted_gmres accepted rtol {}, atol {}", rtol, atol);
        }
    }
}
