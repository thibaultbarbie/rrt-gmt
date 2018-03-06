extern crate serde;
extern crate serde_json;
extern crate rand;

use super::ProbSol;
use rand::distributions::{IndependentSample,Range};
use std::fs::File;
use std::io::prelude::*;

    
pub fn dataset_generation(n_data: usize, n_obs: usize, collision_limit: f64){
    let mut rng = rand::thread_rng();
    let mut dataset : Vec<ProbSol> = Vec::with_capacity(n_data); 

    while dataset.len() < n_data {
        // x
        let x_range = Range::new(0.001, 0.999);
        let mut feasible_problem = false;
        let mut x: Vec<f64> = vec![0.0;2*(2+n_obs)];
        while !feasible_problem {
            for i in 0..(2*(2+n_obs) as usize) {
                x[i]=x_range.ind_sample(&mut rng);
            }
            
            feasible_problem = !(is_colliding(&x,&vec![x[0],x[1]],2,n_obs,collision_limit)||
                                 is_colliding(&x,&vec![x[2],x[3]],2,n_obs,collision_limit));
        }

        // xi
        let result = super::rrt::dual_rrt_connect(&vec![x[0],x[1]],
                                                  &vec![x[2],x[3]],
                                                  |p: &[f64]| !is_colliding(&x, &p.to_vec(),
                                                                            2, n_obs,
                                                                            collision_limit),
                                                  || {
                                                      let between = Range::new(0.0, 1.0);
                                                      let mut rng = rand::thread_rng();
                                                      vec![between.ind_sample(&mut rng),
                                                           between.ind_sample(&mut rng)]
                                                  },
                                                  0.05,
                                                  1000);
        match result {
            Err(_) => {},
            Ok(sol) => { let n=sol.path.len() as u32;
                         let traj=sol.path;
                         if n>7 {
                             let xi=vec![traj[(n/6) as usize][0],traj[(n/6) as usize][1],
                                         traj[(2*n/6)as usize][0],traj[(2*n/6)as usize][1],
                                         traj[(3*n/6)as usize][0],traj[(3*n/6)as usize][1],
                                         traj[(4*n/6)as usize][0],traj[(4*n/6)as usize][1],
                                         traj[(5*n/6)as usize][0],traj[(5*n/6)as usize][1]];
                             dataset.push(ProbSol{x: x,xi :xi});
                         }
            },
        }
    }
    println!("Dataset created, beginning of serialization");
    let serialized = serde_json::to_string(&dataset).unwrap();
    let mut file = File::create("dataset/dataset.dat").unwrap();
    file.write_all(serialized.into_bytes().as_slice()).unwrap();
    println!("Dataset written as a file");
}

pub fn load_dataset() -> Vec<ProbSol> {
    let mut file = File::open("dataset/dataset.dat").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let dataset: Vec<ProbSol> = serde_json::from_str(&contents).unwrap();
    dataset
    
}

pub fn is_colliding(x: &Vec<f64>, p: &Vec<f64>, n_dim: usize,
                n_obs: usize, collision_limit: f64 ) -> bool {

    let squared_collision_limit = collision_limit.powf(2.0);
    
    for i in 0..n_obs as usize{
        let mut tmp=0.0;
        for j in (2+i)*n_dim..(3+i)*n_dim as usize{
            tmp+=(x[j]-p[j-(2+i)*n_dim as usize]).powf(2.0);
        }
        if tmp<squared_collision_limit{
            return true
        }
    }
    
    false
}
