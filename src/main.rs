#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

extern crate rand;


#[derive(Serialize, Deserialize, Debug)]
pub struct ProbSol {
    x:  Vec<f64>,
    xi: Vec<f64>,
}

mod dataset; 


fn main() {
    let n_obs=10;
    let collision_limit=0.05;
    let n_data = 10000;
    let generate = false;

    // Dataset generation
    if generate {dataset::dataset_generation(n_data,n_obs,collision_limit);}

    // Dataset loading
    let dataset = dataset::load_dataset();

    // Training
    
}
