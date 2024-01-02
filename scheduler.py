from subprocess import run
from concurrent.futures import ThreadPoolExecutor

n_workers = 4
base = "cargo run -- "
ep = 1000
max_ep = 60
batch = 30
save_base = "src/save"
maps = [
    "adj_danger_map.bin",
    "coin_bank_1_away_map.bin",
    "coin_bank_adj_map.bin",
    "test_normal_map.bin",
]
train_n = 20
actor_layers = "200 100"
critic_layers = "100 50"
coins_stored_target = 50
coins_destroyed_target = 70


def gen_eval_cmd(i):
    return f"""
        {base} -i {i}
            eval
            -s {save_base}/maps/{maps[i]}
            -p {save_base}/models/model_{i}.pt
            -m {max_ep}
            --cst {coins_stored_target}
            --cdt {coins_destroyed_target}
            --epp {save_base}/eval/eval_plot_{i}.png
            --elp {save_base}/eval/eval_log_{i}.log
            --esp {save_base}/eval/eval_state_{i}.log
    """.strip("\t").replace("\n", " ")


def gen_train_cmd(i):
    return f"""
        {base} -i {i}
            train
            -e {ep}
            -m {max_ep}
            -b {batch}
            -t {train_n}
            -s {save_base}/maps/{maps[i]}
            -p {save_base}/models/model_{i}.pt
            -a {actor_layers}
            -c {critic_layers}
            --cst {coins_stored_target}
            --cdt {coins_destroyed_target}
            --tpp {save_base}/train/train_plot_{i}.png
            --tlp {save_base}/train/train_log_{i}.log
            --tsp {save_base}/train/train_state_{i}.log
    """.strip("\t").replace("\n", " ")


def run_scheduler(i):
    train_cmd = gen_train_cmd(i)
    run(train_cmd, shell=True)

    eval_cmd = gen_eval_cmd(i)
    run(eval_cmd, shell=True)


def main():
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pool.map(run_scheduler, range(n_workers))
    print("Done")


if __name__ == "__main__":
    main()
