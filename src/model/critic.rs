use crate::utils::consts::{HD_DIM, HD_DIM_2};
use tch::nn::{linear, seq, Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{Device, Tensor};

pub struct Critic {
    vs: VarStore,
    network: Sequential,
    device: Device,
    obs_space: usize,
    action_space: usize,
    optimizer: Optimizer,
    lr: f64,
}

impl Clone for Critic {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.obs_space, self.action_space, self.lr);
        new.vs.copy(&self.vs).unwrap();
        new
    }
}

impl Critic {
    fn new(obs_space: usize, action_space: usize, lr: f64) -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device);
        let optimizer = Adam::default().build(&vs, lr).unwrap();
        let p = &vs.root();
        Self {
            network: seq()
                .add(linear(
                    p / "in",
                    (obs_space + action_space) as i64,
                    HD_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(linear(p / "hd", HD_DIM, HD_DIM_2, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(linear(p / "out", HD_DIM_2, 1, Default::default())),
            device: p.device(),
            vs,
            obs_space,
            action_space,
            optimizer,
            lr,
        }
    }

    fn forward(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[actions.copy(), obs.copy()], 1);
        xs.to_device(self.device).apply(&self.network)
    }
}
