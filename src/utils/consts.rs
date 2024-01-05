use robotics_lib::world::tile::Content;

pub const N_ACTIONS: i64 = 16;
pub const N_OBSERVATIONS: i64 = 20;
pub const MEM_DIM: usize = 100_000;
pub const PLOT_WIDTH: u32 = 1024;
pub const PLOT_HEIGHT: u32 = 768;
pub const PLOT_FONT: &str = "times-new-roman";
pub const FONT_SIZE: u32 = 20;
pub const LABEL_AREA_SIZE: u32 = 40;
pub const X_LABELS: usize = 20;
pub const Y_LABELS: usize = 30;
pub const CONTENT_TARGETS: [Content; 2] = [Content::Coin(0), Content::Bank(0..0)];
// Rewards fn
pub const REWARD_FOR_ILLEGAL_ACTION: f64 = -1000.;
pub const RW_NO_SCAN: f64 = -900.;
pub const PERCENTAGE_ENERGY_RESERVED_FOR_SCANNING: f64 = 0.04;
pub const LIM_F_COINS: f64 = 2.0;
pub const BASE_GO_REWARD: f64 = -10.;
pub const COEFFICIENT_X_COINS: f64 = 8.0;
pub const LOG_BASE_COINS: f64 = 2.0;
pub const LIM_F_SCAN: f64 = 3.0;
pub const COEFFICIENT_X_SCAN: f64 = 4.0;
pub const LOG_BASE_SCAN: f64 = 1.5;
pub const MAPS: [&str; 4] = [
    "adj_danger_map.bin",
    "coin_bank_1_away_map.bin",
    "coin_bank_adj_map.bin",
    "test_normal_map.bin",
];
pub const TRAIN_BASE: &str = "src/save/train/train";
pub const EVAL_BASE: &str = "src/save/eval/eval";
pub const MODEL_BASE: &str = "src/save/models/model";
pub const MAP_BASE: &str = "src/save/maps";
pub const EVAL_LOG: &str = "src/save/eval/log";
pub const EVAL_STATE: &str = "src/save/eval/state";
pub const TRAIN_LOG: &str = "src/save/train/log";
pub const TRAIN_STATE: &str = "src/save/train/state";
pub const EVAL_PLOT: &str = "src/save/eval/plot";
pub const TRAIN_PLOT: &str = "src/save/train/plot";
pub const EP: usize = 200;
pub const MAX_EP: usize = 100;
pub const BATCH: usize = 30;
pub const TRAIN_N: usize = 100;
pub const ACTOR_LAYERS: [i64; 2] = [1000, 600];
pub const CRITIC_LAYERS: [i64; 2] = [1000, 600];
pub const LR_A: f64 = 0.0001;
pub const LR_C: f64 = 0.0004;
pub const COINS_STORED_TARGET: usize = 50;
pub const COINS_DESTROYED_TARGET: usize = 60;
pub const N_WORKERS: usize = 4;
pub const THETA: f64 = 0.15;
pub const SIGMA: f64 = 0.2;
pub const MU: f64 = 0.0;
pub const GAMMA: f64 = 0.99;
pub const TAU: f64 = 0.001;
pub const TRAIN_ITERATIONS: usize = 100;
