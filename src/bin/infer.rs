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
fn get_image_path() -> String {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        println!("Usage: infer <image_path>");
        std::process::exit(1);
    }
    args[1].to_owned()
}
fn build_model() -> Model<WgpuBackend> {
    // After training, we can load the model and use it to make predictions.
    let model_bytes = include_bytes!("../../tmp/model.bin");

    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(model_bytes.to_vec())
        .expect("Failed to load model");

    Model::<WgpuBackend>::new().load_record(record)
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

fn main() {
    let image_path = get_image_path();
    let model = build_model();

    let image = read_grayscale_image(&image_path);

    let output = infer::<WgpuBackend>(&model, &image);
    println!("Output: {:?}", output);

    let (number, probability) = get_max_probability(&output);
    println!("(number, probability) = ({number}, {probability})");
}
