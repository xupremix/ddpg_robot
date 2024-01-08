use crate::run::{eval, init, load, train};
use std::env;

mod gym;
mod model;
mod run;
mod utils;

fn main() {
    match env::args().nth(1) {
        None => println!("Missing mode"),
        Some(mode) => match mode.as_str() {
            "init" => init(),
            "train" => train(),
            "load" => load(),
            "eval" => eval(),
            _ => println!("Wrong mode provided: {}", mode),
        },
    }
}
