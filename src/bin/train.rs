use burn::backend::{Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::{MetricEarlyStoppingStrategy, StoppingCondition};
use test_burn_rs::config::MnistTrainingConfig;
use test_burn_rs::data::{MNISTBatcher, NewSubDataset, SubDataset};
use test_burn_rs::model::Model;

use burn::module::Module;
use burn::record::{CompactRecorder, NoStdTrainingRecorder};
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    tensor::backend::AutodiffBackend,
    train::LearnerBuilder,
};
use rand::SeedableRng;

pub fn train<B: AutodiffBackend>(directory: &str) {
    let directory_path = std::path::Path::new(directory);
    let config_path = directory_path.join("config.json");
    let config = MnistTrainingConfig::load(&config_path).unwrap_or(MnistTrainingConfig::default());
    println!("Config: {:?}", config);
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    B::seed(config.seed);

    // Data
    let batcher_train = MNISTBatcher::<B>::new();
    let batcher_test = MNISTBatcher::<B::InnerBackend>::new();

    let dataset_train = SubDataset::new(MNISTDataset::train(), config.subset_size, &mut rng);
    let dataset_valid = SubDataset::new(MNISTDataset::test(), config.subset_size / 4, &mut rng);
    println!(
        "Train size: {}, Valid size: {}",
        dataset_train.len(),
        dataset_valid.len()
    );

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
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(config.num_epochs)
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 3 },
        ))
        .build(Model::new(), config.optimizer.init(), 1e-4);

    config.save(&config_path).expect("Failed to save config");
    println!("Config saved");

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(
            directory_path.join("model.bin"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}

fn main() {
    train::<Autodiff<Wgpu>>("./model");
}
