use crate::gym::state::State;
use crate::utils::consts::{
    BASE_GO_REWARD, COEFFICIENT_X_COINS, LIM_F_COINS,
    LOG_BASE_COINS, PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING, REWARD_FOR_ILLEGAL_ACTION,
};
use crate::utils::functions::{reward_fn, scan_reward, update_closest, update_danger};
use robotics_lib::energy::Energy;
use robotics_lib::event::events::Event;
use robotics_lib::interface::{destroy, go, one_direction_view, put, Direction};
use robotics_lib::runner::backpack::BackPack;
use robotics_lib::runner::{Robot, Runnable};
use robotics_lib::world::coordinates::Coordinate;
use robotics_lib::world::tile::Content;
use robotics_lib::world::World;
use std::cell::RefCell;
use std::rc::Rc;

pub struct GymRobot {
    pub state: Rc<RefCell<State>>,
    pub robot: Robot,
    pub closest_coin: Option<(usize, usize)>,
    pub closest_bank: Option<(usize, usize)>,
    pub coins_destroyed: usize,
    pub coins_stored: usize,
    pub coins_destroyed_goal: usize,
    pub coins_stored_goal: usize,
    setup: bool,
}

impl GymRobot {
    pub fn new(
        state: Rc<RefCell<State>>,
        coins_destroyed_goal: usize,
        coins_stored_goal: usize,
    ) -> Self {
        Self {
            robot: Robot::new(),
            closest_coin: None,
            closest_bank: None,
            coins_destroyed: 0,
            coins_stored: 0,
            coins_destroyed_goal,
            coins_stored_goal,
            setup: true,
            state,
        }
    }

    fn can_destroy_or_put(&self) -> bool {
        // can destroy since it has a coin adj
        self.state.borrow().coin_adj.contains(&1.)
            // can put since it has a bank adj and at least a coin in the backpack
            || self.state.borrow().bank_adj.contains(&1.)
                && *self
                    .get_backpack()
                    .get_contents()
                    .get(&Content::Coin(0))
                    .unwrap()
                    != 0
    }

    fn wrong_dir(&self, i: usize) -> bool {
        self.state.borrow().coin_dir[i] != 1. && self.state.borrow().bank_dir[i] != 1.
    }

    pub fn step(&mut self, world: &mut World) -> f64 {
        let action = self.state.borrow().action;
        let dir = match action % 4 {
            0 => Direction::Up,
            1 => Direction::Right,
            2 => Direction::Down,
            _ => Direction::Left,
        };
        match action / 4 {
            0 => {
                // move
                if self.can_destroy_or_put() || self.wrong_dir((action % 4) as usize) {
                    return REWARD_FOR_ILLEGAL_ACTION;
                }
                if let Ok((surroundings, _)) = go(self, world, dir) {
                    update_danger(self, world);
                    update_closest(self, world);
                    BASE_GO_REWARD
                        - surroundings[1][1]
                            .as_ref()
                            .unwrap()
                            .tile_type
                            .properties()
                            .cost() as f64
                } else {
                    REWARD_FOR_ILLEGAL_ACTION
                }
            }
            1 => {
                // destroy
                if let Ok(amt_dst) = destroy(self, world, dir) {
                    update_closest(self, world);
                    self.coins_destroyed += amt_dst;
                    if self.coins_destroyed >= self.coins_destroyed_goal {
                        println!("Completed the coins destroyed task");
                        self.state.borrow_mut().done = true;
                        0.
                    } else if amt_dst == 0 {
                        REWARD_FOR_ILLEGAL_ACTION
                    } else {
                        reward_fn(
                            amt_dst as f64,
                            COEFFICIENT_X_COINS,
                            LOG_BASE_COINS,
                            LIM_F_COINS,
                        )
                    }
                } else {
                    REWARD_FOR_ILLEGAL_ACTION
                }
            }
            2 => {
                // put
                let amount = *self
                    .get_backpack()
                    .get_contents()
                    .get(&Content::Coin(0))
                    .unwrap();
                if let Ok(amt_put) = put(self, world, Content::Coin(0), amount, dir) {
                    update_closest(self, world);
                    self.coins_stored += amt_put;
                    if self.coins_stored >= self.coins_stored_goal {
                        println!("Completed the coins stored task");
                        self.state.borrow_mut().done = true;
                        0.
                    } else if amt_put == 0 {
                        REWARD_FOR_ILLEGAL_ACTION
                    } else {
                        reward_fn(
                            amt_put as f64,
                            COEFFICIENT_X_COINS,
                            LOG_BASE_COINS,
                            LIM_F_COINS,
                        )
                    }
                } else {
                    REWARD_FOR_ILLEGAL_ACTION
                }
            }
            _ => {
                // scan
                if self.can_destroy_or_put() {
                    return REWARD_FOR_ILLEGAL_ACTION;
                }
                let distance = (self.get_energy().get_energy_level() as f64 / 3.
                    * PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING)
                    .floor() as usize;
                if distance < 2 {
                    REWARD_FOR_ILLEGAL_ACTION
                } else {
                    let rect = one_direction_view(self, world, dir.clone(), distance).unwrap();
                    update_closest(self, world);
                    scan_reward(self, rect, dir, world)
                }
            }
        }
    }
}

impl Runnable for GymRobot {
    fn process_tick(&mut self, world: &mut World) {
        // setup the state if it's the first game tick
        if self.setup {
            update_danger(self, world);
            update_closest(self, world);
            self.setup = false;
            return;
        }
        let reward = self.step(world);
        self.state.borrow_mut().reward = reward;
    }

    fn handle_event(&mut self, _event: Event) {
        // TODO
    }

    fn get_energy(&self) -> &Energy {
        &self.robot.energy
    }

    fn get_energy_mut(&mut self) -> &mut Energy {
        &mut self.robot.energy
    }

    fn get_coordinate(&self) -> &Coordinate {
        &self.robot.coordinate
    }

    fn get_coordinate_mut(&mut self) -> &mut Coordinate {
        &mut self.robot.coordinate
    }

    fn get_backpack(&self) -> &BackPack {
        &self.robot.backpack
    }

    fn get_backpack_mut(&mut self) -> &mut BackPack {
        &mut self.robot.backpack
    }
}
