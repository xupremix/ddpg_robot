use std::fs::File;
use std::io::Write;
use tch::{CModule, Device};
use worldgen_unwrap::public::WorldgeneratorUnwrap;

use crate::gym::GymEnv;
use crate::utils::args::Mode;
use crate::utils::consts::{EVAL_LOG_PATH, EVAL_STATE_LOG_PATH, MAP_PATH, MODEL_PATH};
use crate::utils::functions::plot;

pub fn eval(mode: &Mode, max_ep_len: usize) {
    let generator = WorldgeneratorUnwrap::init(false, Some(MAP_PATH.into()));
    let mut env = GymEnv::new(generator);
    let mut model = CModule::load_on_device(MODEL_PATH, Device::cuda_if_available()).unwrap();
    model.set_eval();
    let mut log_file = File::create(EVAL_LOG_PATH).unwrap();
    let mut state_log_file = File::create(EVAL_STATE_LOG_PATH).unwrap();
    log_file
        .write_all(b"Iter| Action |\tReward\t|\tDone\t|\tAcc_rw\n")
        .unwrap();
    state_log_file
        .write_all(b"Danger | CoinDir | BankDir | CoinAdj | BankAdj\n")
        .unwrap();
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

        // log to file
        let log = format!(
            "{i}\t|\t{action}\t |\t{reward:.4}\t|\t{done}\t|\t {acc_rw:.4}\n",
            i = i,
            action = action,
            reward = step.reward,
            done = step.done,
            acc_rw = acc_rw
        );
        log_file.write_all(log.as_bytes()).unwrap();
        let state = env.state();
        let state_log = format!(
            "{:?} | {:?} | {:?} | {:?} | {:?}\n",
            state.danger, state.coin_dir, state.bank_dir, state.coin_adj, state.bank_adj,
        );
        state_log_file.write_all(state_log.as_bytes()).unwrap();

        acc_rw += step.reward;
        memory.push(acc_rw);
        if acc_rw < min_rw {
            min_rw = acc_rw;
        }
        if acc_rw > max_rw {
            max_rw = acc_rw;
        }
        if i >= max_ep_len || step.done {
            break;
        }
        obs = step.obs;
        i += 1;
    }
    println!("Evaluation: {acc_rw:.4}");
    plot(mode, memory, min_rw, max_rw);
}
