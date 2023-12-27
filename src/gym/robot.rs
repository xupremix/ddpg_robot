use crate::gym::state::State;
use rand::{thread_rng, Rng};
use robotics_lib::energy::Energy;
use robotics_lib::event::events::Event;
use robotics_lib::runner::backpack::BackPack;
use robotics_lib::runner::{Robot, Runnable};
use robotics_lib::world::coordinates::Coordinate;
use robotics_lib::world::World;
use std::cell::RefCell;
use std::rc::Rc;

pub struct GymRobot {
    state: Rc<RefCell<State>>,
    robot: Robot,
}

impl GymRobot {
    pub fn new(state: Rc<RefCell<State>>) -> Self {
        Self {
            robot: Robot::new(),
            state,
        }
    }
}

impl Runnable for GymRobot {
    fn process_tick(&mut self, world: &mut World) {
        let reward = match self.state.borrow().action {
            0 => -2.,
            1 => 1.,
            2 => -1.,
            _ => -1.,
        };
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
