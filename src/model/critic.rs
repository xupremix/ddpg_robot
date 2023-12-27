use crate::utils::consts::{ACTOR_MODEL_PATH, CRITIC_MODEL_PATH, HD_DIM, HD_DIM_2};
use tch::nn::{linear, seq, Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{Device, Tensor};
use crate::model::actor::Actor;

pub struct Critic {
    vs: VarStore,
    network: Sequential,
    device: Device,
    observation_space: usize,
    action_space: usize,
    optimizer: Optimizer,
    lr: f64,
}

impl Clone for Critic {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.observation_space, self.action_space, self.lr);
        new.vs.copy(&self.vs).unwrap();
        new
    }
}

impl Critic {
    pub fn new(observation_space: usize, action_space: usize, lr: f64) -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device);
        let optimizer = Adam::default().build(&vs, lr).unwrap();
        let p = &vs.root();
        Self {
            network: seq()
                .add(linear(
                    p / "in",
                    (observation_space + action_space) as i64,
                    HD_DIM,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(linear(p / "hd", HD_DIM, HD_DIM_2, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(linear(p / "out", HD_DIM_2, 1, Default::default())),
            device: p.device(),
            vs,
            observation_space,
            action_space,
            optimizer,
            lr,
        }
    }

    pub fn load(observation_space: usize, action_space: usize, lr: f64) -> Self {
        let mut critic = Critic::new(observation_space: usize, action_space: usize, lr: f64);
        critic.vs.load(CRITIC_MODEL_PATH).unwrap();
        critic
    }

    pub fn forward(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[actions.copy(), obs.copy()], 1);
        xs.to_device(self.device).apply(&self.network)
    }

    pub fn save(&mut self) {
        self.vs.freeze();
        self.vs.save(CRITIC_MODEL_PATH).unwrap();
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
