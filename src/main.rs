use burn::backend::WgpuAutodiffBackend;

mod data;
mod model;
mod train;

fn main() {
    train::run::<WgpuAutodiffBackend>();
}
