use crate::gym::GymEnv;
use crate::utils::args::{Args, Mode};
use crate::utils::eval::eval;
use crate::utils::train::train;
use clap::Parser;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

mod gym;
mod model;
mod utils;

fn main() {
    let args = Args::parse();
    if let Some(mode) = args.mode {
        match mode {
            Mode::Init => init(),
            Mode::Train { .. } => train(mode, args.i),
            Mode::Eval { .. } => eval(mode, args.i),
        }
    } else {
        println!("No mode specified");
    }
}

fn init() {
    println!("Entering init mode");
    println!("Initializing the Gym environment...");
    let _ = GymEnv::new(WorldgeneratorUnwrap::init(true, None), 0, 0);
    println!("Done");
}
