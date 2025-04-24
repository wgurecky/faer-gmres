About
=====

[![Crates.io](https://img.shields.io/crates/v/faer_gmres)](https://crates.io/crates/faer_gmres)
![status](https://github.com/wgurecky/faer-gmres/actions/workflows/rust.yml/badge.svg)

GMRES in rust using [faer](https://github.com/sarah-ek/faer-rs).

Solves linear systems of the form: Ax=b, where A is sparse.  Depends on faer for sparse matrix implementation.

Use
===

Example use:

    use faer_gmres::gmres;
    use faer::prelude::*;
    use faer::sparse::*;
    use faer::reborrow::*;
    use faer::mat;

    // create faer sparse mat from triplets
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

    // init sol guess
    // Note: x is modified in-place, the result is stored in x
    let mut x = faer::mat![
        [0.0],
        [0.0],
        [0.0],
        ];

    // the final None arg means do not apply left preconditioning
    let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x.as_mut(), 10, 1e-8, None).unwrap();
    println!("Result x: {:?}", x);
    println!("Error x: {:?}", err);
    println!("Iters : {:?}", iters);

## Preconditioned GMRES:

A preconditioner can be supplied:

    // continued from above...
    use faer_gmres::{JacobiPreconLinOp};
    let jacobi_pre = JacobiPreconLinOp::new(a_test.as_ref());
    let (err, iters) = gmres(a_test.as_ref(), b.as_ref(), x.as_mut(), 10, 1e-8, Some(&jacobi_pre)).unwrap();

## Restarted GMRES:

A restarted GMRES routine is provided:

    use faer_gmres::restarted_gmres;
    let max_inner = 30;
    let max_outer = 50;
    let (err, iters) = restarted_gmres(
        a_test.as_ref(), b.as_ref(), x.as_mut(), max_inner, max_outer, 1e-8, None).unwrap();

This will repeatedly call the inner GMRES routine, using the previous outer iteration's solution as the inital guess for the next outer solve.  The current implementation of restarted GMRES in this package can reduce the memory requirements needed, but slow convergence.

TODO
====

- Python bindings
- Additional tests
- Benchmarks
- Performance improvements


References and Credits
=======================

This package is an adaptation of GMRES implementation by RLando:

- https://github.com/RLado/GMRES
- https://crates.io/crates/gmres

License
=======

MIT
