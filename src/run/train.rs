use crate::gym::GymEnv;
use crate::model::{Actor, Agent, Critic, Noise};
use crate::utils::consts::{
    ACTOR_LAYERS, BATCH, COINS_DESTROYED_TARGET, COINS_STORED_TARGET, CRITIC_LAYERS, EP, GAMMA,
    LR_A, LR_C, MAPS, MAP_BASE, MAX_EP, MEM_DIM, MODEL_BASE, MU, N_WORKERS, SIGMA, TAU, THETA,
    TRAIN_ITERATIONS, TRAIN_LOG, TRAIN_PLOT, TRAIN_STATE,
};
use crate::utils::functions::plot;
use std::fs::File;
use std::io::Write;
use std::thread::spawn;
use tch::Kind::Float;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

pub fn train() {
    let mut handles = vec![];
    for worker in 0..N_WORKERS {
        handles.push(spawn(move || {
            let generator = WorldgeneratorUnwrap::init(
                false,
                Some(format!("{}/{}", MAP_BASE, MAPS[worker]).into()),
            );
            let mut env = GymEnv::new(generator, COINS_DESTROYED_TARGET, COINS_STORED_TARGET);
            let observation_space = env.observation_space().iter().product::<i64>() as usize;
            let action_space = env.action_space() as usize;
            let actor = Actor::new(
                observation_space,
                action_space,
                LR_A,
                &ACTOR_LAYERS,
                format!("{}_{}.pt", MODEL_BASE, worker),
            );
            let mut actor_target = Actor::new(
                observation_space,
                action_space,
                LR_A,
                &ACTOR_LAYERS,
                format!("{}_{}.pt", MODEL_BASE, worker),
            );
            actor_target.import(&actor);
            let critic = Critic::new(observation_space, action_space, LR_C, &CRITIC_LAYERS);
            let mut critic_target =
                Critic::new(observation_space, action_space, LR_C, &CRITIC_LAYERS);
            critic_target.import(&critic);
            let noise = Noise::new(THETA, SIGMA, MU, action_space as i64);
            let mut agent = Agent::new(
                actor,
                actor_target,
                critic,
                critic_target,
                noise,
                MEM_DIM,
                true,
                GAMMA,
                TAU,
            );

            // data for plotting and saving
            let mut log_file = File::create(format!("{}_{}.log", TRAIN_LOG, worker)).unwrap();
            let mut state_log_file =
                File::create(format!("{}_{}.log", TRAIN_STATE, worker)).unwrap();
            log_file
                .write_all(b"Iter | Action | Reward | Done | Acc_r\n")
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

            for episode in 0..EP {
                let mut ep_log_data = vec![];
                let mut ep_state_log_data = vec![];
                let mut obs = env.reset();
                let mut acc_rw = 0.;
                let mut ep_min_rw = f64::MAX;
                let mut ep_max_rw = f64::MIN;
                let mut ep_memory = vec![];

                for i in 0..MAX_EP {
                    // get an action given an observation
                    let actions = agent.actions(&obs);
                    // get the max action
                    let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
                    // perform an action in the environment
                    let step = env.step(action);
                    acc_rw += step.reward;
                    // remember the reward, min and max for plotting

                    let log = format!(
                        "{} | {} | {:.4} | {} | {:.4}\n",
                        i, action, step.reward, step.done, acc_rw
                    );
                    ep_log_data.push(log);
                    let state = env.state();
                    let state_log = format!(
                        "{:?} | {:?} | {:?} | {:?} | {:?}\n",
                        state.danger,
                        state.coin_dir,
                        state.coin_adj,
                        state.bank_dir,
                        state.bank_adj,
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

                println!("T: {worker}, episode: {episode} with a total reward of {acc_rw:.4}");

                // save if the episode is better than the previous best
                if acc_rw > best_acc_rw {
                    log_data = ep_log_data;
                    state_log_data = ep_state_log_data;
                    best_acc_rw = acc_rw;
                    min_rw = ep_min_rw;
                    max_rw = ep_max_rw;
                    memory = ep_memory;
                    // save the actor model
                    println!("T: {worker}, found new best");
                    agent.save()
                }

                for _ in 0..TRAIN_ITERATIONS {
                    agent.train(BATCH);
                }
            }
            // log the data
            for (log, state_log) in log_data.iter().zip(state_log_data.iter()) {
                log_file.write_all(log.as_bytes()).unwrap();
                state_log_file.write_all(state_log.as_bytes()).unwrap();
            }

            // plot the best episode
            plot(
                format!("{}_{}.png", TRAIN_PLOT, worker),
                memory,
                min_rw,
                max_rw,
            );
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
}
