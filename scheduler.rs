const N_WORKERS: usize = 4;
const MAPS: [&str; 4] = [
    "adj_danger_map.bin",
    "coin_bank_1_away_map.bin",
    "coin_bank_adj_map.bin",
    "test_normal_map.bin",
];
const TRAIN_BASE: &str = "src/save/train/train";
const EVAL_BASE: &str = "src/save/eval/eval";
const MODEL_BASE: &str = "src/save/models/model";
const MAP_BASE: &str = "src/save/maps";
const EP: usize = 1000;
const MAX_EP: usize = 100;
const BATCH: usize = 30;
const TRAIN_N: usize = 100;
const ACTOR_LAYERS: &str = "1000 600";
const CRITIC_LAYERS: &str = "1000 600";
const LR_A: f64 = 0.0001;
const LR_C: f64 = 0.0004;
const COINS_STORED_TARGET: usize = 50;
const COINS_DESTROYED_TARGET: usize = 60;

fn create_eval_arg(i: usize) -> String {
    format!(
        r#"
            cargo run --
                -i {i}
                eval
                    -m {}
                    -s {}/{}
                    -p {}_{i}.pt
                    --cst {}
                    --cdt {}
                    --epp {t}_plot_{i}.png
                    --elp {t}_log_{i}.log
                    --esp {t}_state_{i}.log
        "#,
        MAX_EP,
        MAP_BASE,
        MAPS[i],
        MODEL_BASE,
        COINS_STORED_TARGET,
        COINS_DESTROYED_TARGET,
        t = EVAL_BASE,
    )
    .replace("\n", " ")
    .replace("\t", " ")
}

fn create_train_arg(i: usize) -> String {
    format!(
        r#"
            cargo run --
                -i {i}
                train
                    -e {}
                    -m {}
                    -b {}
                    -t {}
                    -s {}/{}
                    -p {}_{i}.pt
                    -a {}
                    -c {}
                    --lra {}
                    --lrc {}
                    --cst {}
                    --cdt {}
                    --tpp {t}_plot_{i}.png
                    --tlp {t}_log_{i}.log
                    --tsp {t}_state_{i}.log
        "#,
        EP,
        MAX_EP,
        BATCH,
        TRAIN_N,
        MAP_BASE,
        MAPS[i],
        MODEL_BASE,
        ACTOR_LAYERS,
        CRITIC_LAYERS,
        LR_A,
        LR_C,
        COINS_STORED_TARGET,
        COINS_DESTROYED_TARGET,
        t = TRAIN_BASE,
    )
    .replace("\n", " ")
    .replace("\t", " ")
}

fn main() {
    let mut t = vec![];
    let mut e = vec![];
    for i in 0..N_WORKERS {
        let train_cmd = create_train_arg(i);
        let ts = std::process::Command::new("sh")
            .arg("-c")
            .arg(train_cmd)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        t.push(ts);
    }
    for mut ts in t {
        let _ = ts.wait();
    }
    for i in 0..N_WORKERS {
        let eval_cmd = create_eval_arg(i);
        let es = std::process::Command::new("sh")
            .arg("-c")
            .arg(eval_cmd)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        e.push(es);
    }
    for mut es in e {
        let _ = es.wait();
    }
}
