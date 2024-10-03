use faer_gmres::{gmres, restarted_gmres, JacobiPreconLinOp};
use faer::prelude::*;
use faer::sparse::*;
use std::fs::read_to_string;


fn main() {
    let mut a_triplets = vec![];
    let mut b_rhs = vec![];

    // read A matrix from file
    for line in read_to_string("./examples/data/fidap001.txt").unwrap().lines() {
        let mut tmp_vec_line = vec![];
        let iter = line.split_whitespace();
        for word in iter {
            tmp_vec_line.push(word);
        }
        let idx_i = tmp_vec_line[0].parse::<usize>().unwrap()-1;
        let idx_j = tmp_vec_line[1].parse::<usize>().unwrap()-1;
        let val = tmp_vec_line[tmp_vec_line.len()-1].parse::<f64>().unwrap();
        a_triplets.push((idx_i, idx_j, val));
    }

    // read rhs from file
    for line in read_to_string("./examples/data/fidap001_rhs1.txt").unwrap().lines() {
        let mut tmp_vec_line = vec![];
        let iter = line.split_whitespace();
        for word in iter {
            tmp_vec_line.push(word);
        }
        let val = tmp_vec_line[tmp_vec_line.len()-1].parse::<f64>().unwrap();
        b_rhs.push(val);
    }

    // create sparse mat
    let a_test = SparseColMat::<usize, f64>::try_new_from_triplets(
        216, 216,
        &a_triplets).unwrap();

    // create rhs
    let mut rhs = faer::Mat::zeros(216, 1);
    for (i, rhs_val) in b_rhs.into_iter().enumerate() {
        rhs.write(i, 0, rhs_val);
    }

    // init guess
    let mut x0 = faer::Mat::zeros(216, 1);

    // solve the system
    let jacobi_pre = JacobiPreconLinOp::new(a_test.as_ref());
    let (err, iters) = gmres(&a_test.as_ref(), rhs.as_ref(), x0.as_mut(), 500, 1e-8, Some(&jacobi_pre)).unwrap();
    println!("Result x: {:?}", x0);
    println!("Error x: {:?}", err);
    println!("Iters : {:?}", iters);

}
