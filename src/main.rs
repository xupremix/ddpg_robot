use crate::run::{eval, init, load, train};
use crate::utils::Mode;

mod gym;
mod model;
mod run;
mod utils;

fn main() {
    let mode = Mode::Train;
    match mode {
        Mode::Init => init(),
        Mode::Train => train(),
        Mode::TrainLoad => load(),
        Mode::Eval => eval(),
    }
}
