use tch::kind::{FLOAT_CPU, FLOAT_CUDA, INT64_CPU, INT64_CUDA};
use tch::{Cuda, Tensor};

/// # Replay memory
///
/// - `obs`: observations
/// - `next_obs`: next observations
/// - `rewards`: rewards
/// - `actions`: actions
/// - `capacity`: maximum number of elements
/// - `len`: number of elements
/// - `i`: index of the current element
pub struct ReplayMemory {
    obs: Tensor,
    next_obs: Tensor,
    rewards: Tensor,
    actions: Tensor,
    max_dim: usize,
    len: usize,
    i: usize,
}

impl ReplayMemory {
    pub fn new(max_dim: usize, obs_space: usize, action_space: usize) -> Self {
        let mode = if Cuda::is_available() {
            FLOAT_CUDA
        } else {
            FLOAT_CPU
        };
        Self {
            obs: Tensor::zeros([max_dim, obs_space], mode.clone()),
            next_obs: Tensor::zeros([max_dim, obs_space], mode.clone()),
            rewards: Tensor::zeros([max_dim, 1], mode.clone()),
            actions: Tensor::zeros([max_dim, action_space], mode),
            max_dim,
            len: 0,
            i: 0,
        }
    }
    fn push(&mut self, obs: &Tensor, actions: &Tensor, reward: &Tensor, next_obs: &Tensor) {
        let i = (self.i % self.max_dim) as i64;
        self.obs.get(i).copy_(obs);
        self.rewards.get(i).copy_(reward);
        self.actions.get(i).copy_(actions);
        self.next_obs.get(i).copy_(next_obs);
        self.i += 1;
        if self.len < self.max_dim {
            self.len += 1;
        }
    }

    fn random_batch(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1) as i64;
        let mode = if Cuda::is_available() {
            INT64_CUDA
        } else {
            INT64_CPU
        };
        let batch_indexes = Tensor::randint((self.len - 2) as i64, [batch_size], mode);

        let states = self.obs.index_select(0, &batch_indexes);
        let next_states = self.next_obs.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);

        Some((states, actions, rewards, next_states))
    }
}
