use std::cell::{Ref, RefCell};
use std::rc::Rc;

use robotics_lib::runner::Runner;
use tch::Tensor;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

use crate::utils::consts::{N_ACTIONS, N_OBSERVATIONS};
pub mod robot;
pub mod state;
// Implementation following the OpenAI Gym standard

use robot::GymRobot;
use state::State;

/// # Gym environment
///
/// - `actions_space`: shape of the output layer
/// - `observation_space`: shape of the input layer
pub struct GymEnv {
    action_space: i64,
    observation_space: Vec<i64>,
    generator: WorldgeneratorUnwrap,
    state: Rc<RefCell<State>>,
    runner: Runner,
}

pub struct Step {
    pub obs: Tensor,
    pub action: i64,
    pub reward: f64,
    pub done: bool,
}

impl GymEnv {
    pub fn new(mut generator: WorldgeneratorUnwrap) -> Self {
        let state = Rc::new(RefCell::new(State::default()));
        let mut runner =
            Runner::new(Box::new(GymRobot::new(state.clone())), &mut generator).unwrap();
        // let a tick pass to get the near data and init the danger map
        runner.game_tick().unwrap();
        Self {
            action_space: N_ACTIONS,
            observation_space: vec![N_OBSERVATIONS],
            generator,
            runner,
            state,
        }
    }
    pub fn action_space(&self) -> i64 {
        self.action_space
    }
    pub fn observation_space(&self) -> &[i64] {
        &self.observation_space
    }
    pub fn reset(&mut self) -> Tensor {
        *self.state.borrow_mut() = State::default();
        self.runner = Runner::new(
            Box::new(GymRobot::new(self.state.clone())),
            &mut self.generator,
        )
        .unwrap();
        // let a tick pass to get the near data and init the danger map
        self.runner.game_tick().unwrap();
        self.state.borrow().build()
    }
    pub fn step(&mut self, action: i64) -> Step {
        // update logic
        self.state.borrow_mut().action = action;
        self.runner.game_tick().unwrap();
        Step {
            obs: self.state.borrow().build(),
            action,
            reward: self.state.borrow().reward,
            done: self.state.borrow().done,
        }
    }
    pub fn state(&self) -> Ref<State> {
        self.state.borrow()
    }
}
