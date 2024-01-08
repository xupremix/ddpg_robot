use crate::utils::functions::create_network;
use tch::nn::{Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{Device, Tensor};

pub struct Critic {
    vs: VarStore,
    network: Sequential,
    device: Device,
    optimizer: Optimizer,
}

impl Critic {
    pub fn new(
        observation_space: usize,
        action_space: usize,
        lr: f64,
        hidden_layers: &[i64],
    ) -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device);
        let optimizer = Adam::default().build(&vs, lr).unwrap();
        let p = &vs.root();
        let network = create_network(
            p,
            observation_space as i64,
            action_space as i64,
            hidden_layers,
            true,
        );
        Self {
            network,
            device: p.device(),
            vs,
            optimizer,
        }
    }

    pub fn forward(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[actions.copy(), obs.copy()], 1);
        xs.to_device(self.device).apply(&self.network)
    }

    pub fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }
    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }
    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.vs
    }
    pub fn import(&mut self, other: &Self) {
        self.vs.copy(&other.vs).unwrap();
    }
}
