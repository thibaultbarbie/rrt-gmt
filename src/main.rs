#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

extern crate rand;

extern crate rusty_machine;
use rusty_machine::learning::gmm::GaussianMixtureModel;
use rusty_machine::linalg::Matrix;
use rusty_machine::prelude::*;
use std::fs::File;
use std::io::prelude::*;


#[derive(Serialize, Deserialize, Debug)]
pub struct ProbSol {
    x:  Vec<f64>,
    xi: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Ellipsoid {
    mean: Vec<f64>,
    covariance: Vec<f64>,
    weight: f64,
}

mod dataset; 

fn save_gmm(gmm : &GaussianMixtureModel, n_gaussians: usize) {
    println!("Saving the Gaussian Mixture Model parameters");

    let weights = gmm.mixture_weights().data().to_vec();

    for i in 0..n_gaussians {
        let e = Ellipsoid{mean: gmm.means().unwrap().select_rows(&[i]).clone().into_vec(),
                          covariance: gmm.covariances().unwrap().clone()[0].clone().into_vec(),
                          weight: weights[i]};
        let serialized = serde_json::to_string(&e).unwrap();
        let filename = "approximations/ellipsoid_".to_string()+&i.to_string()+".dat"; 
        let mut file = File::create(filename).unwrap();
        file.write_all(serialized.into_bytes().as_slice()).unwrap();
    }
}

fn main() {
    let n_obs=0;
    let collision_limit=0.05;
    let n_data = 1_000;
    let generate = true;
    let n_gaussians =2;
    
    // Dataset generation
    if generate {dataset::dataset_generation(n_data,n_obs,collision_limit);}

    
    // Dataset loading
    let dataset = dataset::load_dataset();
    println!("Dataset loaded");
    
    // Training

    let dataset_as_vector : Vec<f64> = dataset.iter().
        fold(Vec::new(),|mut v, d| {v.extend(&d.x);
                                    v.extend(&d.xi);
                                    v });    

    println!("Beginning of GMT training");
    let mut gmm = GaussianMixtureModel::new(n_gaussians);
    match gmm.train(&Matrix::new(dataset.len(), 2*(2+n_obs)+2*5, dataset_as_vector)) {
        Err(e) => println!("{}",e),
        Ok(_) => {save_gmm(&gmm,n_gaussians)}
    }

}
