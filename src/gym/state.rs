use tch::{Device, Kind, Tensor};

pub struct State {
    pub action: i64,
    pub reward: f64,
    pub done: bool,
    pub danger: [f64; 4],
    pub coin_dir: [f64; 4],
    pub bank_dir: [f64; 4],
    pub coin_adj: [f64; 4],
    pub bank_adj: [f64; 4],
}

impl Default for State {
    fn default() -> Self {
        Self {
            action: -1,
            reward: 0.0,
            done: false,
            danger: [0.0; 4],
            coin_dir: [0.0; 4],
            bank_dir: [0.0; 4],
            coin_adj: [0.0; 4],
            bank_adj: [0.0; 4],
        }
    }
}

impl State {
    pub fn build(&self) -> Tensor {
        Tensor::from_slice(
            &self
                .danger
                .iter()
                .chain(&self.coin_dir)
                .chain(&self.bank_dir)
                .chain(&self.coin_adj)
                .chain(&self.bank_adj)
                .copied()
                .collect::<Vec<f64>>(),
        )
        .to_kind(Kind::Float)
        .to(Device::cuda_if_available())
    }

    pub fn reset(&mut self) {
        *self = Self {
            action: self.action,
            reward: self.reward,
            done: self.done,
            ..Default::default()
        };
    }
}
