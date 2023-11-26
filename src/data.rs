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
fn generate_random_indices(subset_size: usize, dataset_size: usize) -> Vec<usize> {
    let mut indices = (0..dataset_size).collect::<Vec<_>>();
    indices.shuffle(&mut rand::thread_rng());
    indices[..subset_size].to_vec()
}
pub struct SubDataset<D> {
    subset_size: usize,
    dataset: D,
    random_indices: Vec<usize>,
}
pub trait NewSubDataset<D, I>
where
    D: Dataset<I>,
{
    fn new(dataset: D, subset_size: usize) -> Self;
}
impl<D, I> NewSubDataset<D, I> for SubDataset<D>
where
    D: Dataset<I>,
{
    fn new(dataset: D, subset_size: usize) -> Self {
        let dataset_size = dataset.len();
        let subset_size = subset_size.min(dataset_size);
        let random_indices = generate_random_indices(subset_size, dataset_size);
        Self {
            subset_size,
            dataset,
            random_indices,
        }
    }
}
impl<D, I> Dataset<I> for SubDataset<D>
where
    D: Dataset<I>,
{
    fn len(&self) -> usize {
        self.subset_size
    }
    fn get(&self, index: usize) -> Option<I> {
        self.dataset.get(*self.random_indices.get(index)?)
    }
}
