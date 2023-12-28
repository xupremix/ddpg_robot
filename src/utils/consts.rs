use robotics_lib::world::tile::Content;

pub const N_ACTIONS: i64 = 16; // 17? stay still?
pub const N_OBSERVATIONS: i64 = 20;
pub const HD_DIM: i64 = 256;
pub const HD_DIM_2: i64 = 128;
pub const LR_A: f64 = 0.001;
pub const LR_C: f64 = 0.002;
pub const TRAIN_PLOT_PATH: &str = "src/save/train_plot.png";
pub const EVAL_PLOT_PATH: &str = "src/save/eval_plot.png";
pub const MODEL_PATH: &str = "src/save/model.pt";
pub const MAP_PATH: &str = "src/save/map.bin";
pub const MU: f64 = 0.0;
pub const THETA: f64 = 0.15;
pub const SIGMA: f64 = 0.1;
pub const TAU: f64 = 0.005;
pub const GAMMA: f64 = 0.99;
pub const MEM_DIM: usize = 100000;
pub const BATCH_SIZE: usize = 100;
pub const MAX_EPISODE_LEN: usize = 200;
pub const PLOT_WIDTH: u32 = 1024;
pub const PLOT_HEIGHT: u32 = 768;
pub const PLOT_FONT: &str = "times-new-roman";
pub const FONT_SIZE: u32 = 20;
pub const LABEL_AREA_SIZE: u32 = 40;
pub const X_LABELS: usize = 20;
pub const Y_LABELS: usize = 30;
pub const CONTENT_TARGETS: [Content; 2] = [Content::Coin(0), Content::Bank(0..0)];
