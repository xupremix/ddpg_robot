use crate::utils::functions::create_network;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::nn::{Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{CModule, Cuda, Device, Tensor};

pub struct Actor {
    save_path: String,
    vs: VarStore,
    network: Sequential,
    device: Device,
    observation_space: usize,
    action_space: usize,
    optimizer: Optimizer,
}

impl Actor {
    pub fn new(
        observation_space: usize,
        action_space: usize,
        lr: f64,
        hidden_layers: &[i64],
        save_path: String,
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
        );
        Self {
            save_path,
            device: p.device(),
            network,
            observation_space,
            action_space,
            vs,
            optimizer,
        }
    }

    pub fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }

    pub fn save(&mut self) {
        // save via tracing

        // disable gradient tracking
        self.vs.freeze();
        let mut forward_fn = |x: &[Tensor]| vec![self.forward(&x[0])];
        let mode = if Cuda::is_available() {
            FLOAT_CUDA
        } else {
            FLOAT_CPU
        };
        // trace the module with a dummy input
        let cmodule = CModule::create_by_tracing(
            "Actor",
            "forward",
            &[Tensor::zeros([self.observation_space as i64], mode)],
            &mut forward_fn,
        )
        .unwrap();
        // save the module
        cmodule.save(&self.save_path).unwrap();
        self.vs.unfreeze();
    }

    pub fn observation_space(&self) -> usize {
        self.observation_space
    }
    pub fn action_space(&self) -> usize {
        self.action_space
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
