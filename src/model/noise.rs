use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::{Cuda, Tensor};

struct Noise {
    state: Tensor,
    theta: f64,
    sigma: f64,
    mu: f64,
}
impl Noise {
    fn new(theta: f64, sigma: f64, mu: f64, action_space: usize) -> Self {
        let mode = if Cuda::is_available() {
            FLOAT_CUDA
        } else {
            FLOAT_CPU
        };
        let state = Tensor::ones([action_space], mode);
        Self {
            state,
            theta,
            sigma,
            mu,
        }
    }

    fn sample(&mut self) -> &Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(self.state.size(), FLOAT_CPU);
        self.state += dx;
        &self.state
    }
}
