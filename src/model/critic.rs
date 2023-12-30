use tch::nn::{linear, seq, Adam, Optimizer, OptimizerConfig, Sequential, VarStore};
use tch::{Device, Tensor};

use crate::utils::TrainParameters;

pub struct Critic {
    vs: VarStore,
    network: Sequential,
    device: Device,
    observation_space: usize,
    action_space: usize,
    optimizer: Optimizer,
    lr: f64,
}

impl Critic {
    pub fn new(
        observation_space: usize,
        action_space: usize,
        train_parameters: &TrainParameters,
    ) -> Self {
        let device = Device::cuda_if_available();
        let vs = VarStore::new(device);
        let optimizer = Adam::default()
            .build(&vs, train_parameters.lr_critic)
            .unwrap();
        let p = &vs.root();
        let mut network = seq()
            .add(linear(
                p / "in",
                (observation_space + action_space) as i64,
                train_parameters.critic_hidden_layers[0],
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());
        for (i, (&x, &y)) in train_parameters
            .critic_hidden_layers
            .iter()
            .zip(train_parameters.critic_hidden_layers.iter().skip(1))
            .enumerate()
        {
            network = network
                .add(linear(p / format!("hd{}", i), x, y, Default::default()))
                .add_fn(|xs| xs.relu());
        }
        network = network.add(linear(
            p / "out",
            *train_parameters.critic_hidden_layers.last().unwrap(),
            1,
            Default::default(),
        ));
        Self {
            network,
            device: p.device(),
            vs,
            observation_space,
            action_space,
            optimizer,
            lr: train_parameters.lr_critic,
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
