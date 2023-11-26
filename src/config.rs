use std::fmt::Debug;

use burn::{
    config::Config,
    optim::{decay::WeightDecayConfig, AdamConfig},
};

pub struct PathConfig {
    artifact_dir: std::path::PathBuf,
}
impl PathConfig {
    pub fn new<T: Into<std::path::PathBuf>>(artifact_dir: T) -> Self {
        Self {
            artifact_dir: artifact_dir.into(),
        }
    }
    pub fn get_artifact_dir(&self) -> &std::path::PathBuf {
        &self.artifact_dir
    }
    pub fn get_config_path(&self) -> std::path::PathBuf {
        self.artifact_dir.join("config.json")
    }
    pub fn get_model_path(&self) -> std::path::PathBuf {
        self.artifact_dir.join("model.pt")
    }
}

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
            batch_size: 64,
            num_workers: 4,
            seed: 42,
            subset_size: usize::MAX,
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
            .finish()
    }
}
