use crate::gym::GymEnv;
use crate::utils::consts::{
    COINS_DESTROYED_TARGET, COINS_STORED_TARGET, EVAL_LOG, EVAL_PLOT, EVAL_STATE, MAPS, MAP_BASE,
    MAX_EP, MODEL_BASE, N_WORKERS,
};
use crate::utils::functions::plot;
use std::fs::File;
use std::io::Write;
use std::thread::spawn;
use tch::{CModule, Device};
use worldgen_unwrap::public::WorldgeneratorUnwrap;

pub fn eval() {
    let mut handles = vec![];
    (0..N_WORKERS).for_each(|worker| {
        handles.push(spawn(move || {
            let generator = WorldgeneratorUnwrap::init(
                false,
                Some(format!("{}/{}", MAP_BASE, MAPS[worker]).into()),
            );
            let mut env = GymEnv::new(generator, COINS_DESTROYED_TARGET, COINS_STORED_TARGET);
            let mut model = CModule::load_on_device(
                format!("{}_{}.pt", MODEL_BASE, worker),
                Device::cuda_if_available(),
            )
            .unwrap();
            model.set_eval();
            let mut log_file = File::create(format!("{}_{}.log", EVAL_LOG, worker)).unwrap();
            let mut state_log_file =
                File::create(format!("{}_{}.log", EVAL_STATE, worker)).unwrap();
            log_file
                .write_all(
                    format!(
                        "|{:_^11}|{:_^8}|{:_^12}|{:_^6}|{:_^12}|\n",
                        "Iteration", "Action", "Reward", "Done", "Acc. Reward"
                    )
                    .as_bytes(),
                )
                .unwrap();
            state_log_file
                .write_all(
                    format!(
                        "|{:_^22}|{:_^22}|{:_^22}|{:_^22}|{:_^22}|\n",
                        "Danger",
                        "Coin Direction",
                        "Coin Adjacency",
                        "Bank Direction",
                        "Bank Adjacency"
                    )
                    .as_bytes(),
                )
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
                    "|{:^11}|{:^8}|{:^12.3}|{:^6}|{:^12.3}|\n",
                    i, action, step.reward, step.done, acc_rw
                );
                log_file.write_all(log.as_bytes()).unwrap();
                let state = env.state();
                let state_log = format!(
                    "| {:?} | {:?} | {:?} | {:?} | {:?} |\n",
                    state.danger, state.coin_dir, state.coin_adj, state.bank_dir, state.bank_adj,
                );
                state_log_file.write_all(state_log.as_bytes()).unwrap();

                if i >= MAX_EP || step.done {
                    break;
                }
                obs = step.obs;
                i += 1;
            }
            println!("T: {worker}, evaluation: {acc_rw:.4}");
            plot(
                format!("{}_{}.png", EVAL_PLOT, worker),
                memory,
                min_rw,
                max_rw,
            );
        }));
    });
    for handle in handles {
        handle.join().unwrap();
    }
}
