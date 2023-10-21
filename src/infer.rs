use burn::{
    module::Module,
    record::NoStdTrainingRecorder,
    tensor::{backend::Backend, Tensor},
};

use crate::{config::PathConfig, model::Model};

pub fn infer<B: Backend>(path_config: &PathConfig, data: &[f32; 28 * 28]) -> [f32; 10] {
    let model = Model::<B>::new();
    let recorder = NoStdTrainingRecorder::new();
    let model = model
        .load_file(path_config.get_model_path(), &recorder)
        .expect("Failed to load model");
    let input = Tensor::<B, 1>::from_floats(*data).reshape([1, 28, 28]);
    model
        .forward(input)
        .reshape([10])
        .into_data()
        .convert::<f32>()
        .value
        .as_slice()
        .try_into()
        .expect(
            "Failed to convert output tensor to array. \
            This is likely a bug in the inference code.",
        )
}
