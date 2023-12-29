use clap::Parser;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

use eval::eval;
use gym::GymEnv;
use train::train;
use utils::args::{Args, Mode};

mod eval;
mod gym;
mod model;
mod train;
mod utils;

fn main() {
    let args = Args::parse();
    if let Some(mode) = args.mode {
        match mode {
            Mode::Init => init(),
            Mode::Train {
                episodes,
                max_ep_len,
            } => train(&mode, episodes, max_ep_len),
            Mode::Eval { max_ep_len } => eval(&mode, max_ep_len),
        }
    } else {
        println!("No mode specified");
    }
}

fn init() {
    println!("Entering init mode");
    println!("Initializing the Gym environment...");
    let _ = GymEnv::new(WorldgeneratorUnwrap::init(true, None));
    println!("Done");
}
