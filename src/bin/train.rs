use burn::backend::WgpuAutodiffBackend;
use test_burn_rs::config::MnistTrainingConfig;
use test_burn_rs::data::{MNISTBatcher, NewSubDataset, SubDataset};
use test_burn_rs::model::Model;

use burn::module::Module;
use burn::record::NoStdTrainingRecorder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    tensor::backend::ADBackend,
    train::LearnerBuilder,
};

pub fn train<B: ADBackend>(directory: &str) {
    let directory_path = std::path::Path::new(directory);
    let config_path = directory_path.join("config.json");
    let config = MnistTrainingConfig::load(&config_path).unwrap_or(MnistTrainingConfig::default());
    config.save(&config_path).expect("Failed to save config");

    B::seed(config.seed);

    println!("Config: {:?}", config);
    // Data
    let batcher_train = MNISTBatcher::<B>::new();
    let batcher_test = MNISTBatcher::<B::InnerBackend>::new();

    let dataset_train = SubDataset::new(MNISTDataset::train(), config.subset_size);
    let dataset_valid = SubDataset::new(MNISTDataset::test(), config.subset_size);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_valid);

    println!("Dataset loaded");

    // Model
    let learner = LearnerBuilder::new(directory)
        .metric_train(AccuracyMetric::new())
        .metric_valid(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        // .with_file_checkpointer(2, CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .build(Model::new(), config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(
            directory_path.join("model.bin"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}

fn main() {
    train::<WgpuAutodiffBackend>("./tmp");
}
