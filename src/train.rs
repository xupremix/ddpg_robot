use std::fs::File;
use std::io::Write;
use tch::Kind::Float;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

use crate::gym::GymEnv;
use crate::model::actor::Actor;
use crate::model::agent::Agent;
use crate::model::critic::Critic;
use crate::model::noise::Noise;
use crate::utils::args::Mode;
use crate::utils::args::Mode::Train;
use crate::utils::consts::{
    BATCH_SIZE, EVAL_LOG_PATH, EVAL_STATE_LOG_PATH, GAMMA, LR_A, LR_C, MAP_PATH, MEM_DIM, MU,
    SIGMA, TAU, THETA, TRAIN_LOG_PATH, TRAIN_STATE_LOG_PATH,
};
use crate::utils::functions::plot;

pub fn train(mode: &Mode, episodes: usize, max_ep_len: usize) {
    println!("Entering training mode");
    let generator = WorldgeneratorUnwrap::init(false, Some(MAP_PATH.into()));
    let mut env = GymEnv::new(generator);
    let observation_space = env.observation_space().iter().product::<i64>() as usize;
    let action_space = env.action_space() as usize;
    let actor = Actor::new(observation_space, action_space, LR_A);
    let critic = Critic::new(observation_space, action_space, LR_C);
    let noise = Noise::new(THETA, SIGMA, MU, action_space as i64);
    let mut agent = Agent::new(actor, critic, noise, MEM_DIM, true, GAMMA, TAU);

    // data for plotting and saving
    let mut log_file = File::create(TRAIN_LOG_PATH).unwrap();
    let mut state_log_file = File::create(TRAIN_STATE_LOG_PATH).unwrap();
    log_file
        .write_all(b"Iter| Action |\tReward\t|\tDone\t|\tAcc_rw\n")
        .unwrap();
    state_log_file
        .write_all(b"Danger | CoinDir | BankDir | CoinAdj | BankAdj\n")
        .unwrap();
    let mut log_data = vec![];
    let mut state_log_data = vec![];
    let mut best_acc_rw = f64::MIN;
    let mut memory = vec![];
    let mut min_rw = f64::MAX;
    let mut max_rw = f64::MIN;

    for episode in 0..episodes {
        let mut ep_log_data = vec![];
        let mut ep_state_log_data = vec![];
        let mut obs = env.reset();
        let mut acc_rw = 0.;
        let mut ep_min_rw = f64::MAX;
        let mut ep_max_rw = f64::MIN;
        let mut ep_memory = vec![];

        for i in 0..max_ep_len {
            // get an action given an observation
            let actions = agent.actions(&obs);
            // get the max action
            let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
            // perform an action in the environment
            let step = env.step(action);
            acc_rw += step.reward;
            // remember the reward, min and max for plotting

            let log = format!(
                "{i}\t|\t{action}\t |\t{reward:.4}\t|\t{done}\t|\t {acc_rw:.4}\n",
                i = i,
                action = action,
                reward = step.reward,
                done = step.done,
                acc_rw = acc_rw
            );
            ep_log_data.push(log);
            let state = env.state();
            let state_log = format!(
                "{:?} | {:?} | {:?} | {:?} | {:?}\n",
                state.danger, state.coin_dir, state.bank_dir, state.coin_adj, state.bank_adj,
            );
            ep_state_log_data.push(state_log);

            ep_memory.push(acc_rw);
            if acc_rw < ep_min_rw {
                ep_min_rw = acc_rw;
            }
            if acc_rw > ep_max_rw {
                ep_max_rw = acc_rw;
            }
            // store the transition into the replay memory
            agent.remember(&obs, &actions, &step.reward.into(), &step.obs);
            if i > BATCH_SIZE {
                // perform backpropagation
                agent.train(BATCH_SIZE);
            }
            if step.done {
                break;
            }
            // update the observation
            obs = step.obs;
        }

        println!("Episode: {episode} with a total reward of {acc_rw:.4}");

        // save if the episode is better than the previous best
        if acc_rw > best_acc_rw {
            log_data = ep_log_data;
            state_log_data = ep_state_log_data;
            best_acc_rw = acc_rw;
            min_rw = ep_min_rw;
            max_rw = ep_max_rw;
            memory = ep_memory;
            // save the actor model
            println!("Found new best");
            agent.save()
        }
    }
    // log the data
    for (log, state_log) in log_data.iter().zip(state_log_data.iter()) {
        log_file.write_all(log.as_bytes()).unwrap();
        state_log_file.write_all(state_log.as_bytes()).unwrap();
    }

    // plot the best episode
    plot(mode, memory, min_rw, max_rw);
}
