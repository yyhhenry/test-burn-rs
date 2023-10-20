use crate::data::MNISTBatcher;
use crate::model::Model;

use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::record::NoStdTrainingRecorder;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    tensor::backend::ADBackend,
    train::LearnerBuilder,
};

static ARTIFACT_DIR: &str = "./tmp/burn-example-mnist";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

struct TinySizeDataset<I> {
    size: usize,
    dataset: Box<dyn Dataset<I>>,
}
impl<I> TinySizeDataset<I> {
    pub fn new(dataset: Box<dyn Dataset<I>>, size: usize) -> Self {
        Self { size, dataset }
    }
}
impl<I> Dataset<I> for TinySizeDataset<I> {
    fn len(&self) -> usize {
        self.size.min(self.dataset.len())
    }
    fn get(&self, index: usize) -> Option<I> {
        self.dataset.get(index)
    }
    fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

pub fn train<B: ADBackend>() {
    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = MnistTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    println!(
        "Config: {:?}",
        (
            "num_epochs",
            config.num_epochs,
            "batch_size",
            config.batch_size,
            "num_workers",
            config.num_workers,
            "seed",
            config.seed,
        )
    );
    // Data
    let batcher_train = MNISTBatcher::<B>::new();
    let batcher_test = MNISTBatcher::<B::InnerBackend>::new();

    let tiny_size = 50;
    let tiny_train = TinySizeDataset::new(Box::new(MNISTDataset::train()), tiny_size);
    let tiny_valid = TinySizeDataset::new(Box::new(MNISTDataset::test()), tiny_size);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(tiny_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(tiny_valid);

    println!("Dataset loaded");

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .num_epochs(config.num_epochs)
        .build(Model::new(), config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
