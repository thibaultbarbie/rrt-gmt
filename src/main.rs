#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;
extern crate rand;
extern crate nalgebra as na;
extern crate rusty_machine;

use na::{DVector,DMatrix};
use rusty_machine::learning::gmm::GaussianMixtureModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::prelude::*;


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


fn main() {
    let n_obs=0;
    let collision_limit=0.05;
    let n_data = 1_000;
    let generate = true;
    let training_gmm = true;
    
    let n_gaussians =2;
    let total_dim = 2*(2+n_obs)+2*5;
    
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

    let gmm = gmt::load_gmm(n_gaussians, total_dim);
    
}
