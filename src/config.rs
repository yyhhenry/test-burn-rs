use burn::{config::Config, optim::AdamConfig};

pub struct PathConfig {
    artifact_dir: String,
}
impl PathConfig {
    pub fn new(artifact_dir: String) -> Self {
        Self { artifact_dir }
    }
    pub fn default() -> Self {
        Self::new(String::from("./tmp"))
    }
    pub fn get_artifact_dir(&self) -> &str {
        &self.artifact_dir
    }
    pub fn get_config_path(&self) -> String {
        format!("{}/config.json", self.artifact_dir)
    }
    pub fn get_model_path(&self) -> String {
        format!("{}/model", self.artifact_dir)
    }
}

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
