pub mod args;
pub mod consts;
pub mod functions;
mod macros;

pub struct TrainParameters {
    pub episodes: usize,
    /// Maximum episode length
    pub max_ep_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// Map path
    pub save_map_path: String,
    /// Training iterations
    pub train_iterations: usize,
    /// Model save path
    pub path_model: String,
    /// Actor hidden layers
    pub actor_hidden_layers: Vec<usize>,
    /// Critic hidden layers
    pub critic_hidden_layers: Vec<usize>,
    /// Actor learning rate
    pub lr_actor: f64,
    /// Critic learning rate
    pub lr_critic: f64,
    /// Gamma hyperparameter
    pub gamma: f64,
    /// Tau hyperparameter
    pub tau: f64,
    /// Sigma hyperparameter
    pub sigma: f64,
    /// Theta hyperparameter
    pub theta: f64,
    /// Mu hyperparameter
    pub mu: f64,
    /// How many coins the robot has to destroy to mark an episode as done
    pub coins_destroyed_target: usize,
    /// How many coins the robot has to store to mark an episode as done
    pub coin_stored_target: usize,
    /// Train plot path
    pub train_plot_path: String,
    /// Train log path
    pub train_log_path: String,
    /// Train state path
    pub train_state_path: String,
}

pub struct EvalParameters {
    /// Map path
    pub save_map_path: String,
    /// Model load path
    pub path_model: String,
    /// Maximum episode length
    pub max_ep_len: usize,
    /// Eval plot path
    pub eval_plot_path: String,
    /// Eval log path
    pub eval_log_path: String,
    /// Eval state path
    pub eval_state_path: String,
}
