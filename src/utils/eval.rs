use std::fs::File;
use std::io::Write;
use tch::{CModule, Device};
use worldgen_unwrap::public::WorldgeneratorUnwrap;

use crate::gym::GymEnv;
use crate::utils::args::Mode;
use crate::utils::functions::{create_eval_params, plot};

pub fn eval(mode: Mode, thread_i: usize) {
    let eval_parameters = create_eval_params(mode.clone()).unwrap();
    let generator =
        WorldgeneratorUnwrap::init(false, Some(eval_parameters.save_map_path.clone().into()));
    let mut env = GymEnv::new(
        generator,
        eval_parameters.coins_destroyed_target,
        eval_parameters.coins_stored_target,
    );
    let mut model =
        CModule::load_on_device(&eval_parameters.path_model, Device::cuda_if_available()).unwrap();
    model.set_eval();
    let mut log_file = File::create(&eval_parameters.eval_log_path).unwrap();
    let mut state_log_file = File::create(&eval_parameters.eval_state_path).unwrap();
    log_file
        .write_all(b"Iter| Action |\tReward\t|\tDone\t|\tAcc_rw\n")
        .unwrap();
    state_log_file
        .write_all(b"Danger | CoinDir | CoinAdj | BankDir | BankAdj\n")
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

        acc_rw += step.reward;
        memory.push(acc_rw);
        if acc_rw < min_rw {
            min_rw = acc_rw;
        }
        if acc_rw > max_rw {
            max_rw = acc_rw;
        }

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
            state.danger, state.coin_dir, state.coin_adj, state.bank_dir, state.bank_adj,
        );
        state_log_file.write_all(state_log.as_bytes()).unwrap();

        if i >= eval_parameters.max_ep_len || step.done {
            break;
        }
        obs = step.obs;
        i += 1;
    }
    println!("T: {thread_i}, evaluation: {acc_rw:.4}");
    plot(&eval_parameters.eval_plot_path, memory, min_rw, max_rw);
}
