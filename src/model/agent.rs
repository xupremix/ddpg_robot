use tch::Kind::Float;
use tch::{no_grad, Reduction, Tensor};

use crate::model::actor::Actor;
use crate::model::critic::Critic;
use crate::model::memory::ReplayMemory;
use crate::model::noise::Noise;
use crate::utils::functions::update_vs;

pub struct Agent {
    actor: Actor,
    actor_target: Actor,
    critic: Critic,
    critic_target: Critic,
    replay_memory: ReplayMemory,
    noise: Noise,
    train: bool,
    gamma: f64,
    tau: f64,
}

impl Agent {
    pub fn new(
        actor: Actor,
        critic: Critic,
        noise: Noise,
        mem_dim: usize,
        train: bool,
        gamma: f64,
        tau: f64,
    ) -> Self {
        let actor_target = actor.clone();
        let critic_target = critic.clone();
        let replay_memory = ReplayMemory::new(
            mem_dim as i64,
            actor.observation_space() as i64,
            actor.action_space() as i64,
        );
        Self {
            actor,
            actor_target,
            critic,
            critic_target,
            replay_memory,
            noise,
            train,
            gamma,
            tau,
        }
    }

    pub fn actions(&mut self, obs: &Tensor) -> Tensor {
        let mut actions = no_grad(|| self.actor.forward(obs));
        if self.train {
            actions += self.noise.sample();
        }
        actions
    }

    pub fn remember(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor) {
        self.replay_memory.push(obs, actions, reward, next_obs);
    }

    pub fn train(&mut self, batch_size: usize) {
        let (states, actions, rewards, next_states) =
            match self.replay_memory.random_batch(batch_size) {
                Some(v) => v,
                _ => return, // Not enough samples for training yet.
            };

        let mut q_target = self
            .critic_target
            .forward(&next_states, &self.actor_target.forward(&next_states));
        q_target = rewards + (self.gamma * q_target).detach();

        let q = self.critic.forward(&states, &actions);

        let celoss = q_target.cross_entropy_loss(&q, None, Reduction::Mean, -1, 0.);
        // let diff = q_target - q;
        // let critic_loss = (&diff * &diff).mean(Float);

        self.critic.optimizer_mut().zero_grad();
        // critic_loss.backward();
        celoss.backward();
        self.critic.optimizer_mut().step();

        let actor_loss = -self
            .critic
            .forward(&states, &self.actor.forward(&states))
            .mean(Float);

        self.actor.optimizer_mut().zero_grad();
        actor_loss.backward();
        self.actor.optimizer_mut().step();

        update_vs(
            self.critic_target.var_store_mut(),
            self.critic.var_store(),
            self.tau,
        );
        update_vs(
            self.actor_target.var_store_mut(),
            self.actor.var_store(),
            self.tau,
        );
    }

    pub fn save(&mut self) {
        self.actor.save();
    }
}
