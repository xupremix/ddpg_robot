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
use crate::utils::consts::MEM_DIM;
use crate::utils::functions::{create_train_params, plot};

pub fn train(mode: Mode) {
    println!("Entering training mode");
    let train_parameters = create_train_params(mode.clone()).unwrap();
    let generator =
        WorldgeneratorUnwrap::init(false, Some(train_parameters.save_map_path.clone().into()));
    let mut env = GymEnv::new(generator);
    let observation_space = env.observation_space().iter().product::<i64>() as usize;
    let action_space = env.action_space() as usize;
    let actor = Actor::new(observation_space, action_space, &train_parameters);
    let mut actor_target = Actor::new(observation_space, action_space, &train_parameters);
    actor_target.import(&actor);
    let critic = Critic::new(observation_space, action_space, &train_parameters);
    let mut critic_target = Critic::new(observation_space, action_space, &train_parameters);
    critic_target.import(&critic);
    let noise = Noise::new(&train_parameters, action_space as i64);
    let mut agent = Agent::new(
        actor,
        actor_target,
        critic,
        critic_target,
        noise,
        MEM_DIM,
        true,
        train_parameters.gamma,
        train_parameters.tau,
    );

    // data for plotting and saving
    let mut log_file = File::create(&train_parameters.train_log_path).unwrap();
    let mut state_log_file = File::create(&train_parameters.train_state_path).unwrap();
    log_file
        .write_all(b"Iter| Action |\tReward\t|\tDone\t|\tAcc_rw\n")
        .unwrap();
    state_log_file
        .write_all(b"Danger | CoinDir | CoinAdj | BankDir | BankAdj\n")
        .unwrap();

    let mut log_data = vec![];
    let mut state_log_data = vec![];
    let mut best_acc_rw = f64::MIN;
    let mut memory = vec![];
    let mut min_rw = f64::MAX;
    let mut max_rw = f64::MIN;

    for episode in 0..train_parameters.episodes {
        let mut ep_log_data = vec![];
        let mut ep_state_log_data = vec![];
        let mut obs = env.reset();
        let mut acc_rw = 0.;
        let mut ep_min_rw = f64::MAX;
        let mut ep_max_rw = f64::MIN;
        let mut ep_memory = vec![];

        for i in 0..train_parameters.max_ep_len {
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
                state.danger, state.coin_dir, state.coin_adj, state.bank_dir, state.bank_adj,
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

        for _ in 0..train_parameters.train_iterations {
            agent.train(train_parameters.batch_size);
        }
    }
    // log the data
    for (log, state_log) in log_data.iter().zip(state_log_data.iter()) {
        log_file.write_all(log.as_bytes()).unwrap();
        state_log_file.write_all(state_log.as_bytes()).unwrap();
    }

    // plot the best episode
    plot(&train_parameters.train_plot_path, memory, min_rw, max_rw);
}
