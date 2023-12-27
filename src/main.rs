mod eval;
mod gym;
mod model;
mod train;
mod utils;

use clap::Parser;
use eval::eval;
use gym::GymEnv;
use train::train;
use utils::args::{Args, Mode};
use worldgen_unwrap::public::WorldgeneratorUnwrap;

fn main() {
    let args = Args::parse();
    if let Some(mode) = args.mode {
        match mode {
            Mode::Init => init(),
            Mode::Train { episodes } => train(&mode, episodes),
            Mode::Eval => eval(&mode),
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
