use burn::backend::WgpuBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Shape, Tensor};

fn run<B: Backend>()
where
    B::FloatElem: From<f32> + Into<f32>,
{
    B::seed(4703);
    let a = Tensor::<B, 2>::random([3, 3], Distribution::Uniform((0.0).into(), (1.0).into()));
    let b = Tensor::<B, 2>::ones_like(&a);

    println!("a: {}", a);
    println!("b: {}", b);
    println!("a * b: {}", a.clone() * b.clone());
    println!("a.matmul(b): {}", a.matmul(b));
}

fn main() {
    run::<WgpuBackend>();
}
