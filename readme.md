About
=====

GMRES in rust using [faer](https://github.com/sarah-ek/faer-rs).

Solves linear systems of the form: Ax=b, where A is sparse.  Depends on faer for sparse matrix implementation and sparse QR solver.

Use
===

Example use:

    // create faer sparse mat from triplets
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
