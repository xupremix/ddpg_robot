use tch::{CModule, Device};
use worldgen_unwrap::public::WorldgeneratorUnwrap;

use crate::gym::GymEnv;
use crate::utils::args::Mode;
use crate::utils::consts::{MAP_PATH, MAX_EPISODE_LEN, MODEL_PATH};
use crate::utils::functions::plot;

pub fn eval(mode: &Mode) {
    let generator = WorldgeneratorUnwrap::init(false, Some(MAP_PATH.into()));
    let mut env = GymEnv::new(generator);
    let mut model = CModule::load_on_device(MODEL_PATH, Device::cuda_if_available()).unwrap();
    model.set_eval();
    let mut memory = vec![];
    let mut min_rw = f64::MAX;
    let mut max_rw = f64::MIN;
    let mut acc_rw = 0.;
    let mut i = 0;

    let mut obs = env.reset();
    loop {
        let actions = obs.apply(&model);
        let action = actions
            .softmax(-1, tch::Kind::Float)
            .argmax(-1, true)
            .int64_value(&[]);
        let step = env.step(action);
        acc_rw += step.reward;
        memory.push(acc_rw);
        if acc_rw < min_rw {
            min_rw = acc_rw;
        }
        if acc_rw > max_rw {
            max_rw = acc_rw;
        }
        if i >= MAX_EPISODE_LEN || step.done {
            break;
        }
        obs = step.obs;
        i += 1;
    }
    println!("Evaluation: total reward of {acc_rw:.4}");
    plot(mode, memory, min_rw, max_rw);
}
