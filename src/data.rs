use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{source::huggingface::MNISTItem, Dataset},
    },
    tensor::{backend::Backend, Int, Tensor},
};
use rand::seq::SliceRandom;

pub struct MNISTBatcher<B: Backend> {
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new() -> Self {
        Self {
            _backend: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Tensor::<B, 2>::from_floats(item.image))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Make values in [0., 1.]
            .map(|tensor| (tensor / 255))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([item.label as i32]))
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MNISTBatch { images, targets }
    }
}
pub struct CroppedDataset<I> {
    size: usize,
    dataset: Box<dyn Dataset<I>>,
    random_indices: Vec<usize>,
}
impl<I> CroppedDataset<I> {
    pub fn new(dataset: Box<dyn Dataset<I>>, crop_size: usize) -> Self {
        let size = if crop_size > dataset.len() {
            dataset.len()
        } else {
            crop_size
        };
        let random_indices = {
            let mut indices = (0..dataset.len()).collect::<Vec<_>>();
            indices.shuffle(&mut rand::thread_rng());
            indices[..size].to_vec()
        };
        Self {
            size,
            dataset,
            random_indices,
        }
    }
}
impl<I> Dataset<I> for CroppedDataset<I> {
    fn len(&self) -> usize {
        self.size
    }
    fn get(&self, index: usize) -> Option<I> {
        if index >= self.len() {
            return None;
        }
        let index = self.random_indices[index];
        self.dataset.get(index)
    }
}
