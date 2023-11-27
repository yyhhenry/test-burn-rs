use std::fmt::Debug;

use burn::{
    config::Config,
    optim::{decay::WeightDecayConfig, AdamConfig},
};

#[derive(Config)]
pub struct MnistTrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub subset_size: usize,
    pub optimizer: AdamConfig,
}
impl Default for MnistTrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_size: 128,
            num_workers: 8,
            seed: 42,
            subset_size: 4096,
            optimizer: AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
        }
    }
}
impl Debug for MnistTrainingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MnistTrainingConfig")
            .field("num_epochs", &self.num_epochs)
            .field("batch_size", &self.batch_size)
            .field("num_workers", &self.num_workers)
            .field("seed", &self.seed)
            .field("subset_size", &self.subset_size)
            .finish()
    }
}
