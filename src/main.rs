use crate::gym::Gym;
use crate::model::actor::Actor;
use crate::model::agent::Agent;
use crate::model::critic::Critic;
use crate::model::noise::Noise;
use crate::utils::args::{Args, Mode};
use crate::utils::consts::{
    GAMMA, LR_A, LR_C, MAP_PATH, MEM_DIM, MODEL_PATH, MU, SIGMA, TAU, THETA,
};
use clap::Parser;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

mod gym;
mod model;
mod utils;

fn main() {
    // consider whether loading the model via the vs' or via tracing
    let args = Args::parse();
    if let Some(mode) = args.mode {
        match mode {
            Mode::Init => init(),
            Mode::Train => train(),
            Mode::Eval => eval(),
        }
    } else {
        println!("No mode specified");
    }
}

fn init() {
    println!("Initializing the generator...");
    let generator = WorldgeneratorUnwrap::init(true, None);
    println!("Initializing the Gym environment...");
    let gym = Gym::new(generator);
    let observation_space = gym.observation_space().iter().product::<i64>() as usize;
    let action_space = gym.action_space() as usize;
    println!("Initializing the actor...");
    let actor = Actor::new(observation_space, action_space, LR_A);
    println!("Initializing the critic...");
    let critic = Critic::new(observation_space, action_space, LR_C);
    println!("Initializing the noise...");
    let noise = Noise::new(THETA, SIGMA, MU, action_space as i64);
    println!("Initializing the agent...");
    let mut agent = Agent::new(actor, critic, noise, MEM_DIM, false, GAMMA, TAU);
    println!("Saving the agent...");
    agent.save();
    println!("Done");
}

fn train() {
    println!("Train");
}

fn eval() {
    println!("Eval");
}
