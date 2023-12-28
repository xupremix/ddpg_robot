use crate::utils::consts::{HD_DIM, HD_DIM_2, MODEL_PATH};
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::nn::{linear, seq, Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{CModule, Cuda, Device, Tensor};

pub struct Actor {
    vs: VarStore,
    network: Sequential,
    device: Device,
    observation_space: usize,
    action_space: usize,
    optimizer: Optimizer,
    lr: f64,
}

impl Clone for Actor {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.observation_space, self.action_space, self.lr);
        new.vs.copy(&self.vs).unwrap();
        new
    }
}

impl Actor {
    pub fn new(observation_space: usize, action_space: usize, lr: f64) -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device);
        let optimizer = Adam::default().build(&vs, lr).unwrap();
        let p = &vs.root();
        Self {
            network: seq()
                .add(linear(
                    p / "in",
                    observation_space as i64,
                    HD_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(linear(p / "hd", HD_DIM, HD_DIM_2, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(linear(
                    p / "out",
                    HD_DIM_2,
                    action_space as i64,
                    Default::default(),
                )),
            device: p.device(),
            observation_space,
            action_space,
            vs,
            optimizer,
            lr,
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
        cmodule.save(MODEL_PATH).unwrap();
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
    pub fn optimizer(&self) -> &Optimizer {
        &self.optimizer
    }
    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }
    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.vs
    }
}
