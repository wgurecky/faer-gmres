use faer_gmres::{gmres, restarted_gmres, JacobiPreconLinOp};
use faer::prelude::*;
use faer::sparse::*;
use faer::mat;
use std::error::Error;
use csv::Reader;


fn main() {
    // read the sparse matrix from file
    let mut a_triplets = vec![];
    let file = std::fs::File::open("./examples/data/fidap001.txt").unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .flexible(true)
        .from_reader(file);
    for result in rdr.records() {
        let record = result.unwrap();
        let idx_i = record[0].parse::<usize>().unwrap() - 1;
        let idx_j = record[1].parse::<usize>().unwrap() - 1;
        let val = record[record.len()-1].parse::<f64>().unwrap();
        a_triplets.push((idx_i, idx_j, val));
    }
    // read the rhs from file
    let mut b_rhs = vec![];
    let file = std::fs::File::open("./examples/data/fidap001_rhs1.txt").unwrap();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b' ')
        .from_reader(file);
    for result in rdr.records() {
        let record = result.unwrap();
        let val = record[record.len()-1].parse::<f64>().unwrap();
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
    let x0 = faer::Mat::zeros(216, 1);

    // solve the system
    let (res_x, err, iters) = gmres(a_test.as_ref(), rhs.as_ref(), x0.as_ref(), 500, 1e-8, None).unwrap();
    println!("Result x: {:?}", res_x);
    println!("Error x: {:?}", err);
    println!("Iters : {:?}", iters);

}
