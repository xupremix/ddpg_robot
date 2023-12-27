use crate::gym::GymEnv;
use crate::model::actor::Actor;
use crate::model::agent::Agent;
use crate::model::critic::Critic;
use crate::model::noise::Noise;
use crate::utils::args::Mode;
use crate::utils::consts::{
    BATCH_SIZE, GAMMA, LR_A, LR_C, MAP_PATH, MAX_EPISODE_LEN, MEM_DIM, MU, SIGMA, TAU, THETA,
    TRAINING_ITERATIONS,
};
use crate::utils::functions::plot;
use tch::Kind::Float;
use worldgen_unwrap::public::WorldgeneratorUnwrap;

pub fn train(mode: &Mode, episodes: usize) {
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
    let mut best_acc_rw = f64::MIN;
    let mut memory = vec![];
    let mut min_rw = f64::MAX;
    let mut max_rw = f64::MIN;

    for episode in 0..episodes {
        let mut obs = env.reset();
        let mut acc_rw = 0.;
        let mut ep_min_rw = f64::MAX;
        let mut ep_max_rw = f64::MIN;
        let mut ep_memory = vec![];

        for i in 0..MAX_EPISODE_LEN {
            // get an action given an observation
            let actions = agent.actions(&obs);
            // get the max action
            let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
            // perform an action in the environment
            let step = env.step(action);
            // acc_rw += step.reward;
            // remember the reward, min and max for plotting
            // ep_memory.push(acc_rw);
            // if acc_rw < ep_min_rw {
            //     ep_min_rw = acc_rw;
            // }
            // if acc_rw > ep_max_rw {
            //     ep_max_rw = acc_rw;
            // }
            // store the transition into the replay memory
            agent.remember(&obs, &actions, &step.reward.into(), &step.obs);
            if step.done {
                break;
            }
            // update the observation
            obs = step.obs;
        }

        // perform backpropagation
        for _ in 0..TRAINING_ITERATIONS {
            agent.train(BATCH_SIZE);
        }

        obs = env.reset();
        // perform an episode with the rollout data
        for _ in 0..MAX_EPISODE_LEN {
            // get an action given an observation
            let actions = agent.actions(&obs);
            // get the max action
            let action = actions.softmax(-1, Float).argmax(-1, true).int64_value(&[]);
            // perform an action in the environment
            let step = env.step(action);
            acc_rw += step.reward;
            // remember the reward, min and max for plotting
            ep_memory.push(acc_rw);
            if acc_rw < ep_min_rw {
                ep_min_rw = acc_rw;
            }
            if acc_rw > ep_max_rw {
                ep_max_rw = acc_rw;
            }
            // store the transition into the replay memory
            // agent.remember(&obs, &actions, &step.reward.into(), &step.obs);
            if step.done {
                break;
            }
            // update the observation
            obs = step.obs;
        }

        println!("Rollout Episode: {episode} with a total reward of {acc_rw:.4}");

        // save if the episode is better than the previous best
        if acc_rw > best_acc_rw {
            best_acc_rw = acc_rw;
            min_rw = ep_min_rw;
            max_rw = ep_max_rw;
            memory = ep_memory;
            // save the actor model
            agent.save()
        }
    }
    // plot the best episode
    plot(mode, memory, min_rw, max_rw);
}
