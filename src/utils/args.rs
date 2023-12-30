use clap::{Parser, Subcommand};

// call parameters
// base: cargo run --
//      init: [create a custom map]
//      train: [start the training]
//          -e         usize       [episodes]
//          -m         usize       [max episode length]
//          -b         usize       [batch size]
//          -s         String      [map path]
//          -t         usize       [training iterations]
//          -p         String      [model save path]
//          -a         vec[nk]     [k hidden layers (with sizes n0..nk)] for the actor (+ relu)
//          -c         vec[nk]     [k hidden layers (with sizes n0..nk)] for the critic (+ relu)
//          --lra      f64         [lr actor]
//          --lrc      f64         [lr critic]
//          --gamma    f64         [gamma]
//          --tau      f64         [tau]
//          --sigma    f64         [sigma]
//          --theta    f64         [theta]
//          --mu       f64         [mu]
//          --cdt      usize       [coin destroyed target]
//          --cpt      usize       [coin placed target]
//      eval: [start the evaluation]
//          -s         String      [map path]
//          -p         String      [model load path]
//          -m         usize       [max episode length]

#[derive(Subcommand, Clone)]
pub enum Mode {
    Init,
    Train {
        /// Number of episodes
        #[arg(short, long, default_value = "100")]
        episodes: usize,
        /// Maximum episode length
        #[arg(short, long, default_value = "200")]
        max_ep_len: usize,
        /// Batch size
        #[arg(short, long, default_value = "10")]
        batch_size: usize,
        /// Map path
        #[arg(short, long, default_value = "src/save/map.bin")]
        save_map_path: String,
        /// Training iterations
        #[arg(short, long, default_value = "100")]
        train_iterations: usize,
        /// Model save path
        #[arg(short, long, default_value = "src/save/model.pt")]
        path_model: String,
        /// Actor hidden layers
        #[arg(short, long, value_delimiter = ' ', num_args = 0.., default_values_t = [256])]
        actor_hidden_layers: Vec<i64>,
        /// Critic hidden layers
        #[arg(short, long, value_delimiter = ' ', num_args = 0.., default_values_t = [256])]
        critic_hidden_layers: Vec<i64>,
        /// Actor learning rate
        #[arg(long, alias = "lra", default_value = "0.0001")]
        lr_actor: f64,
        /// Critic learning rate
        #[arg(long, alias = "lrc", default_value = "0.001")]
        lr_critic: f64,
        /// Gamma hyperparameter
        #[arg(long, default_value = "0.99")]
        gamma: f64,
        /// Tau hyperparameter
        #[arg(long, default_value = "0.005")]
        tau: f64,
        /// Sigma hyperparameter
        #[arg(long, default_value = "0.1")]
        sigma: f64,
        /// Theta hyperparameter
        #[arg(long, default_value = "0.15")]
        theta: f64,
        /// Mu hyperparameter
        #[arg(long, default_value = "0.0")]
        mu: f64,
        /// How many coins the robot has to destroy to mark an episode as done
        #[arg(long, alias = "ctd", default_value = "60")]
        coins_destroyed_target: usize,
        /// How many coins the robot has to store to mark an episode as done
        #[arg(long, alias = "csd", default_value = "40")]
        coin_stored_target: usize,
        /// Train plot path
        #[arg(long, alias = "tpp", default_value = "src/save/train_plot.png")]
        train_plot_path: String,
        /// Eval plot path
        #[arg(long, alias = "tlp", default_value = "src/save/train_data.log")]
        train_log_path: String,
        /// Train state path
        #[arg(long, alias = "tsp", default_value = "src/save/train_state.log")]
        train_state_path: String,
    },
    Eval {
        /// Map path
        #[arg(short, long, default_value = "src/save/map.bin")]
        save_map_path: String,
        /// Model load path
        #[arg(short, long, default_value = "src/save/model.pt")]
        path_model: String,
        /// Maximum episode length
        #[arg(short, long, default_value = "200")]
        max_ep_len: usize,
        /// Eval plot path
        #[arg(long, alias = "epp", default_value = "src/save/eval_plot.png")]
        eval_plot_path: String,
        /// Eval log path
        #[arg(long, alias = "elp", default_value = "src/save/eval_data.log")]
        eval_log_path: String,
        /// Eval state path
        #[arg(long, alias = "esp", default_value = "src/save/eval_state.log")]
        eval_state_path: String,
    },
}

#[derive(Parser)]
#[command(author = "Filippo Lollato")]
pub struct Args {
    /// mode
    #[command(subcommand)]
    pub mode: Option<Mode>,
}
