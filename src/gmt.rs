extern crate serde;
extern crate serde_json;
extern crate nalgebra as na;
extern crate rusty_machine;

use na::{DVector,DMatrix};
use std::fs::File;
use std::io::prelude::*;
use rusty_machine::learning::gmm::GaussianMixtureModel;
use rusty_machine::prelude::*;

use super::Ellipsoid;

#[derive(Serialize, Deserialize, Debug)]
struct SerdeEllipsoid {
    mu : Vec<f64>,
    phi : Vec<f64>,
    beta : f64,
}

pub fn save_gmm(gmm : &GaussianMixtureModel, n_gaussians: usize) {
    println!("Saving the Gaussian Mixture Model parameters");
    
    // useful for the determinant computation
    let dim = gmm.means().unwrap().select_rows(&[0]).into_vec().len();

    let weights = gmm.mixture_weights().data().to_vec();

    for i in 0..n_gaussians {
        let phi = gmm.covariances().unwrap()[i].inverse().unwrap().into_vec();
        let beta = 2.0*weights[i].ln()-
            (6.283185 as f64).powi(dim as i32)*gmm.covariances().unwrap()[i].det();
        
        let e = SerdeEllipsoid{mu: gmm.means().unwrap().select_rows(&[i]).clone().into_vec(),
                          phi: phi,
                          beta: beta};
        let serialized = serde_json::to_string(&e).unwrap();
        let filename = "approximations/ellipsoid_".to_string()+&i.to_string()+".dat"; 
        let mut file = File::create(filename).unwrap();
        file.write_all(serialized.into_bytes().as_slice()).unwrap();
    }
}


pub fn load_gmm(n_gaussians: usize, dim: usize) -> Vec<Ellipsoid>{
    println!("Loading the GMT parameters");
    let mut ellipsoids : Vec<Ellipsoid> = Vec::new();

    for i in 0..n_gaussians {
        let filename = "approximations/ellipsoid_".to_string()+&i.to_string()+".dat"; 
        let mut file = File::open(filename).unwrap();
        let mut json_ellipsoid = String::new();
        file.read_to_string(&mut json_ellipsoid).unwrap();
        let e : SerdeEllipsoid = serde_json::from_str(&json_ellipsoid).unwrap();
        let mu = DVector::from_iterator(dim,e.mu.iter().cloned());
        let phi = DMatrix::from_iterator(dim,dim, e.phi.iter().cloned());
        ellipsoids.push(Ellipsoid{mu: mu, phi: phi, beta: e.beta});
    }
    ellipsoids
}

fn best_xi_in_ellipsoid(e: &Ellipsoid, x: &Vec<f64>,
                        x_dim: usize, xi_dim: usize) -> DVector<f64> {
    
    let inv_phi_xi_xi = e.phi.slice((x_dim,x_dim),(xi_dim,xi_dim)).pseudo_inverse(0.00001);

    let mut mu = DVector::<f64>::identity(x_dim+xi_dim);
    mu.copy_from(&e.mu);

    let x_vec = DVector::<f64>::from_iterator(x_dim,x.iter().cloned());
    let lambda = e.phi.rows(x_dim,xi_dim)*mu-e.phi.slice((x_dim,0),(xi_dim,x_dim))*x_vec;

    return inv_phi_xi_xi * lambda
}

fn pdf_normal(e: &Ellipsoid, d: &DVector<f64>,x_dim: usize, xi_dim: usize) -> f64{

    let mu1 = DVector::from_row_slice(x_dim+xi_dim,&(e.mu.as_slice()));
    let mu2 = DVector::from_row_slice(x_dim+xi_dim,&(e.mu.as_slice()));
    
    let mut phi = DMatrix::<f64>::identity(x_dim+xi_dim,x_dim+xi_dim);
    phi.copy_from(&e.phi);
    
    e.beta-((d-mu1).transpose()*phi*(d-mu2)).as_slice()[0]
}

pub fn gmt(x: Vec<f64>, n_gaussians: usize,
       x_dim: usize, xi_dim: usize, ellipsoids: &Vec<Ellipsoid>) -> Vec<Vec<f64>>{

    let list_x=vec![x];
    let mut list_xi : Vec<Vec<f64>> = Vec::new();
    for x in list_x {

        // For each gaussian we compute the best possible trajectory xi
        let best_xi_gaussians : Vec<DVector<f64>> =
            (0..n_gaussians).map(|p| best_xi_in_ellipsoid(&ellipsoids[p],&x, x_dim, xi_dim))
            .collect();

        // We find the highest likelihood
        let best_xi_proba : Vec<f64> = (0..n_gaussians).map(|p| {
            let mut tmp_d = x.to_vec();
            tmp_d.extend_from_slice(&best_xi_gaussians[p].as_slice());
            let d = DVector::<f64>::from_iterator(x_dim+xi_dim,tmp_d.iter().cloned());
            pdf_normal(&ellipsoids[p], &d, x_dim, xi_dim)
        }).collect();

        // We select the best xi based on the highest likelihood
        let mut best_xi_index=0;
        for (i,&value) in best_xi_proba.iter().enumerate() {
            if value>best_xi_proba[best_xi_index] {best_xi_index=i;}
        }

        let best_xi = best_xi_gaussians[best_xi_index].as_slice().to_vec();
        list_xi.push(best_xi);
    }
    list_xi
}
