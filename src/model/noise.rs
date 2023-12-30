use crate::utils::TrainParameters;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::{Cuda, Device, Kind, Tensor};

pub struct Noise {
    mode: (Kind, Device),
    state: Tensor,
    theta: f64,
    sigma: f64,
    mu: f64,
}
impl Noise {
    pub fn new(train_parameters: &TrainParameters, action_space: i64) -> Self {
        let mode = if Cuda::is_available() {
            FLOAT_CUDA
        } else {
            FLOAT_CPU
        };
        let state = Tensor::ones([action_space], mode.clone());
        Self {
            mode,
            state,
            theta: train_parameters.theta,
            sigma: train_parameters.sigma,
            mu: train_parameters.mu,
        }
    }

    pub fn sample(&mut self) -> &Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(self.state.size(), self.mode.clone());
        self.state += dx;
        &self.state
    }
}
