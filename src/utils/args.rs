use clap::{Parser, Subcommand};

#[derive(Subcommand, Clone)]
pub enum Mode {
    Init,
    Train {
        #[arg(short, long, default_value = "100")]
        episodes: usize,
        #[arg(short, long, default_value = "200")]
        max_ep_len: usize,
    },
    Eval {
        #[arg(short, long, default_value = "200")]
        max_ep_len: usize,
    },
}

#[derive(Parser)]
#[command(author = "Filippo Lollato")]
pub struct Args {
    /// mode
    #[command(subcommand)]
    pub mode: Option<Mode>,
}
