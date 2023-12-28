use std::cell::RefCell;
use std::rc::Rc;

use robotics_lib::energy::Energy;
use robotics_lib::event::events::Event;
use robotics_lib::interface::{destroy, go, one_direction_view, put, Direction};
use robotics_lib::runner::backpack::BackPack;
use robotics_lib::runner::{Robot, Runnable};
use robotics_lib::world::coordinates::Coordinate;
use robotics_lib::world::tile::Content;
use robotics_lib::world::World;

use crate::gym::state::State;
use crate::utils::consts::{
    BASE_GO_REWARD, COINS_DESTROYED_GOAL, COINS_STORED_GOAL,
    PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING, REWARD_FOR_ILLEGAL_ACTION,
};
use crate::utils::functions::{scan_reward, update_closest, update_danger};

pub struct GymRobot {
    pub state: Rc<RefCell<State>>,
    pub robot: Robot,
    pub closest_coin: Option<(usize, usize)>,
    pub closest_bank: Option<(usize, usize)>,
    pub coins_destroyed: usize,
    pub coins_stored: usize,
    setup: bool,
}

impl GymRobot {
    pub fn new(state: Rc<RefCell<State>>) -> Self {
        Self {
            robot: Robot::new(),
            closest_coin: None,
            closest_bank: None,
            coins_destroyed: 0,
            coins_stored: 0,
            setup: true,
            state,
        }
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
                if let Ok((surroundings, _)) = go(self, world, dir) {
                    update_danger(self, world);
                    update_closest(self, world);
                    BASE_GO_REWARD - surroundings[1][1].unwrap().tile_type.properties().cost()
                } else {
                    REWARD_FOR_ILLEGAL_ACTION
                }
            }
            1 => {
                // destroy
                if let Ok(amount) = destroy(self, world, dir) {
                    update_closest(self, world);
                    self.coins_destroyed += amount;
                    if self.coins_destroyed >= COINS_DESTROYED_GOAL {
                        self.state.borrow_mut().done = true;
                        0.
                    } else {
                        destroy_reward(amount)
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
                if let Ok(amount) = put(self, world, Content::Coin(0), amount, dir) {
                    update_closest(self, world);
                    self.coins_stored += amount;
                    if self.coins_stored >= COINS_STORED_GOAL {
                        self.state.borrow_mut().done = true;
                        0.
                    } else {
                        put_reward(amount)
                    }
                } else {
                    REWARD_FOR_ILLEGAL_ACTION
                }
            }
            _ => {
                // scan
                let distance = (self.get_energy().get_energy_level() as f64 / 3.
                    * PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING)
                    .floor() as usize;
                if distance < 2 {
                    REWARD_FOR_ILLEGAL_ACTION
                } else {
                    let Ok(rect) = one_direction_view(self, world, dir.clone(), distance);
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
        self.state.borrow_mut().reward = self.step(world);
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
