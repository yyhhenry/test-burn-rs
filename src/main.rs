use burn::backend::{WgpuAutodiffBackend, WgpuBackend};
use config::PathConfig;

mod config;
mod data;
mod infer;
mod model;
mod train;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_config = PathConfig::new(String::from("./tmp"));
    if args.contains(&String::from("--train")) {
        train::train::<WgpuAutodiffBackend>(&path_config);
    }
    let output = infer::infer::<WgpuBackend>(&path_config, &[0.0; 28 * 28]);
    println!("Output: {:?}", output);
}
