use crate::gym::GymEnv;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

pub fn init() {
    GymEnv::new(WorldgeneratorUnwrap::init(true, None), 0, 0);
}
