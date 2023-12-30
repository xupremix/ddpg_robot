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
pub const COINS_DESTROYED_GOAL: usize = 40;
pub const COINS_STORED_GOAL: usize = 52;
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
