use clap::{Parser, Subcommand};

#[derive(Subcommand, Clone)]
pub enum Mode {
    Init,
    Train,
    Eval,
}

#[derive(Parser)]
#[command(author = "Filippo Lollato")]
pub struct Args {
    /// mode
    #[command(subcommand)]
    pub mode: Option<Mode>,
}
