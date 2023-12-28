use crate::gym::state::State;
use crate::utils::functions::update_with_surroundings;
use robotics_lib::energy::Energy;
use robotics_lib::event::events::Event;
use robotics_lib::interface::robot_view;
use robotics_lib::runner::backpack::BackPack;
use robotics_lib::runner::{Robot, Runnable};
use robotics_lib::world::coordinates::Coordinate;
use robotics_lib::world::World;
use std::cell::RefCell;
use std::rc::Rc;

pub struct GymRobot {
    pub state: Rc<RefCell<State>>,
    pub robot: Robot,
    setup: bool,
}

impl GymRobot {
    pub fn new(state: Rc<RefCell<State>>) -> Self {
        Self {
            robot: Robot::new(),
            setup: true,
            state,
        }
    }

    pub fn setup(&mut self, world: &mut World) {
        // update the near danger + coin + bank on 3x3 surroundings
        let surr = robot_view(self, world);
        update_with_surroundings(self, surr, world);
    }
}

impl Runnable for GymRobot {
    fn process_tick(&mut self, world: &mut World) {
        // setup the state if it's the first game tick
        if self.setup {
            self.setup(world);
            self.setup = false;
            return;
        }
        // otherwise alternate between robot movement and manual action
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
