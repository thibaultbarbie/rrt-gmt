#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

extern crate rand;

extern crate rusty_machine;
use rusty_machine::learning::gmm::GaussianMixtureModel;
use rusty_machine::prelude::*;


#[derive(Serialize, Deserialize, Debug)]
pub struct ProbSol {
    x:  Vec<f64>,
    xi: Vec<f64>,
}

mod dataset; 


fn main() {
    let n_obs=4;
    let collision_limit=0.05;
    let n_data = 100_000;
    let generate = true;

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
    let mut gmm = GaussianMixtureModel::new(2);
    match gmm.train(&Matrix::new(dataset.len(), 2*(2+n_obs)+2*5, dataset_as_vector)) {
        Err(e) => println!("{}",e),
        Ok(_) => {
            println!("Means = {:?}", gmm.means());
            println!("Covs = {:?}", gmm.covariances());
            println!("Mix Weights = {:?}", gmm.mixture_weights());
        }
    }

}
