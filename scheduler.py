from subprocess import run
from concurrent.futures import ThreadPoolExecutor

n_workers = 4
base = "cargo run --"
ep = 50
max_ep = 150
batch = 52
save_base = "src/save"
maps = [
    "adj_danger_map.bin",
    "coin_bank_1_away_map.bin",
    "coin_bank_adj_map.bin",
    "test_normal_map.bin",
]
actor_layers = "400 300 100"
critic_layers = "256 128 64"


def gen_eval_cmd(i):
    return f"""
        {base} eval
            -i {i}
            
    """


def gen_train_cmd(i):
    return f"""
        {base} train
            -i {i} 
            -e {ep} 
            -m {max_ep} 
            -b {batch} 
            -s {save_base}/maps/{maps[i]} 
            -p {save_base}/models/model_{i}.pt 
            -a {actor_layers}
            -c {critic_layers}
            --cst 80
            --cdt 100
            --tpp {save_base}/train/train_plot_{i}.png
            --tlp {save_base}/train/train_log_{i}.log
            --tsp {save_base}/train/train_state_{i}.log
    """


def run_scheduler(i):
    train_cmd = gen_train_cmd(i)
    run(train_cmd, shell=True)

    eval_cmd = gen_eval_cmd(i)
    run(eval_cmd, shell=True)


def main():
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pool.map(run_scheduler, range(4))
    print("Done")


if __name__ == "__main__":
    main()
