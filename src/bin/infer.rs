use burn::{backend::WgpuBackend, record::FullPrecisionSettings};
use image::imageops;
use image::imageops::FilterType::Gaussian;
use test_burn_rs::model::Model;

use burn::record::Recorder;
use burn::{
    module::Module,
    record::BinBytesRecorder,
    tensor::{backend::Backend, Tensor},
};

pub fn infer<B: Backend>(model: &Model<B>, data: &[u8; 28 * 28]) -> [f32; 10] {
    let data = data
        .iter()
        .map(|&x| 1.0 - x as f32 / 255.0)
        .collect::<Vec<_>>();
    let input = Tensor::<B, 1>::from_floats(data.as_slice()).reshape([1, 28, 28]);

    let output = model.forward(input).reshape([10]);
    let output = output.into_data().convert::<f32>().value;

    output.try_into().unwrap()
}
fn build_model<B: Backend>() -> Model<B> {
    // After training, we can load the model and use it to make predictions.
    let model_bytes = include_bytes!("../../tmp/model.bin");

    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(model_bytes.to_vec())
        .expect("Failed to load model");

    Model::<B>::new().load_record(record)
}
fn get_max_probability(output: &[f32; 10]) -> (usize, f32) {
    let (number, probability) = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (number, *probability)
}
fn read_grayscale_image(path: &str) -> [u8; 28 * 28] {
    let image = image::open(path).expect("Failed to open image").to_luma8();
    let image = imageops::resize(&image, 28, 28, Gaussian);
    image.to_vec().try_into().unwrap()
}
fn infer_image_and_print<B: Backend>(model: &Model<B>, image_path: &str, detailed: bool) {
    println!("Inferring {}", image_path);
    let image = read_grayscale_image(image_path);
    let output = infer::<B>(&model, &image);
    let (number, probability) = get_max_probability(&output);
    println!("number = {}", number);
    println!("probability = {:.3}", probability);
    if detailed {
        println!(
            "Details: {}",
            output
                .into_iter()
                .enumerate()
                .map(|(i, x)| format!("{}: {:.3}", i, x))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!();
}
struct Args {
    image_path: String,
    detailed: bool,
}
fn get_args() -> Args {
    let mut args = pico_args::Arguments::from_env();
    if args.contains("--help") {
        println!("Usage: infer [OPTIONS] <image_path>");
        println!();
        println!("Arguments:");
        println!("    image_path    Path to an image or the directory of images");
        println!();
        println!("Options:");
        println!("    --detailed    Print detailed output");
        println!("    --help        Print this help message");
        std::process::exit(0);
    }
    let detailed = args.contains("--detailed");
    let image_path = args
        .free_from_str()
        .expect("Failed to get image path from arguments, Try --help");
    Args {
        image_path,
        detailed,
    }
}

fn main() {
    let args = get_args();
    let model = build_model::<WgpuBackend>();

    let accept_extensions = ["png", "jpg", "jpeg"];

    let path = std::path::Path::new(&args.image_path);
    if !path.exists() {
        println!("{} does not exist", args.image_path);
        std::process::exit(1);
    }
    if path.is_dir() {
        println!("{} is a directory", args.image_path);
        let paths = std::fs::read_dir(path).unwrap();
        for path in paths {
            let path = path.unwrap().path();
            let ext = path.extension().unwrap().to_str().unwrap();
            if path.is_file() && accept_extensions.contains(&ext) {
                let path = path.to_str().unwrap();
                infer_image_and_print(&model, path, args.detailed);
            }
        }
    } else {
        infer_image_and_print(&model, &args.image_path, args.detailed);
    }
}
