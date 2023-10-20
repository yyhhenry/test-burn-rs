use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Int, Tensor},
};

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
