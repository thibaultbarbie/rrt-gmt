#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate rand;
extern crate nalgebra as na;
extern crate rusty_machine;
extern crate kdtree;
#[macro_use]
extern crate log;

use na::{DVector,DMatrix};
use rusty_machine::learning::gmm::GaussianMixtureModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::prelude::*;

use rand::distributions::{IndependentSample,Range};
use dataset::is_colliding;

#[derive(Serialize, Deserialize, Debug)]
pub struct ProbSol {
    x:  Vec<f64>,
    xi: Vec<f64>,
}

#[derive(Debug)]
pub struct Ellipsoid {
    mu : DVector<f64>,
    phi : DMatrix<f64>,
    beta : f64,
}

mod dataset; 
mod gmt;
mod rrt;

fn main() {
    let generate = false;
    let training_gmm = false;
    
    let n_obs=10;
    let collision_limit=0.05;
    let n_data = 10_000;
    let n_gaussians =2;
    let total_dim = 2*(2+n_obs)+2*5;
    let n_test=100_000;
    let epsilon = 0.01;
    
    // Dataset generation
    if generate {dataset::dataset_generation(n_data,n_obs,collision_limit);}

    
    // Training
    if training_gmm {
        let dataset = dataset::load_dataset();
        println!("Dataset loaded");
    
        let dataset_as_vector : Vec<f64> = dataset.iter().
            fold(Vec::new(),|mut v, d| {v.extend(&d.x);
                                        v.extend(&d.xi);
                                        v });    

        println!("Beginning of GMT training");
        let mut gmm = GaussianMixtureModel::new(n_gaussians);
        match gmm.train(&Matrix::new(dataset.len(), total_dim, dataset_as_vector)) {
            Err(e) => println!("{}",e),
            Ok(_) => {gmt::save_gmm(&gmm,n_gaussians)}
        }
    }


    // Testing
    let mut rng = rand::thread_rng();
    let x_range = Range::new(0.001, 0.999);

    let mut list_x = Vec::new();
    for _ in 0..n_test {
        let mut x: Vec<f64> = vec![0.0;2*(2+n_obs)];
        for j in 0..(2*(2+n_obs) as usize) {
            x[j]=x_range.ind_sample(&mut rng);
        }
        list_x.push(x);
    }

    
    let gmm = gmt::load_gmm(n_gaussians, total_dim);
    
    let mut rrt_gmt_iter : Vec<u64> = Vec::new();
    let mut rrt_iter : Vec<u64> = Vec::new();
    for x in list_x {
        // Simple RRT-connect
        let result = rrt::dual_rrt_connect(&vec![x[0],x[1]],
                                           &vec![x[2],x[3]],
                                           |p: &[f64]| !is_colliding(&x, &p.to_vec(),
                                                                     2, n_obs, collision_limit),
                                           || {
                                               let between = Range::new(0.0, 1.0);
                                               let mut rng = rand::thread_rng();
                                               vec![between.ind_sample(&mut rng),
                                                    between.ind_sample(&mut rng)]
                                           },
                                           epsilon,
                                           1000);
        match result {
            Err(_) => {},
            Ok(sol) => {rrt_iter.push(sol.iterations as u64)},
        }

        // GMT RRT-connect
        let prediction = gmt::gmt(x.clone(), n_gaussians, 2*(2+n_obs), 2*5, &gmm);
        let result = rrt::rrt_connect_gmt(&vec![x[0],x[1]],
                                          &vec![x[2],x[3]],
                                          |p: &[f64]| !is_colliding(&x, &p.to_vec(),
                                                                    2, n_obs, collision_limit),
                                           || {
                                               let between = Range::new(0.0, 1.0);
                                               let mut rng = rand::thread_rng();
                                               vec![between.ind_sample(&mut rng),
                                                    between.ind_sample(&mut rng)]
                                           },
                                          epsilon,
                                          1000,
                                          prediction);
        match result {
            Err(_) => {},
            Ok(sol) => {rrt_gmt_iter.push(sol.iterations as u64)},
        }

    }
    let mean_rrt_iter = rrt_iter.iter().fold(0, |sum, i| sum+i) as f64 / n_test as f64;
    let mean_rrt_gmt_iter = rrt_gmt_iter.iter().fold(0, |sum, i| sum+i) as f64 / n_test as f64;
    
    //let result = gmt::gmt(list_x,n_gaussians,2*(2+n_obs),10,gmm);
    println!("RRT-Connect :");
    println!("     Iterations {}  Success {} %",mean_rrt_iter,
             rrt_iter.len() as f64 *100.0 / n_test as f64);
    println!();
    println!("RRT-Connect Gmt :");
    println!("     Iterations {}  Success {} %",mean_rrt_iter,
             rrt_gmt_iter.len() as f64 *100.0 / n_test as f64);
}
