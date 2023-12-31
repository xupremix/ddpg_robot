import subprocess
import threading


base = "cargo run --"
ep = 100
max_ep = 200
batch = 100
save_base = "src/save"
maps = [
    "adj_danger_map.bin",
    "coin_bank_1_away_map.bin",
    "coin_bank_adj_map.bin",
    "test_normal_map.bin",
]


def run_scheduler(i):
    cmd = f'{base} train -e {ep} -m {max_ep} -b {batch} -s {save_base}/maps/{maps[i]} -p {save_base}/models/model_{i}.pt -a 400 300 -c 256 128 --cst 100 --cdt 170 --tpp {save_base}/train/train_plot_{i}.png --tlp {save_base}/train/train_log_{i}.log --tsp {save_base}/train/train_state_{i}.log'
    subprocess.run(cmd, shell=True)
    cmd = f'{base} eval -s {save_base}/maps/{maps[i]} -p {save_base}/models/model_{i}.pt --epp {save_base}/eval/eval_plot_{i}.png --elp {save_base}/eval/eval_log_{i}.log --esp {save_base}/eval/eval_state_{i}.log'
    subprocess.run(cmd, shell=True)


def main():
    threads = []
    for i in range(0, 4):
        threads.append(threading.Thread(target=run_scheduler, args=(i,)))
        threads[i].start()

    for thread in threads:
        thread.join()
    print("Done")


if __name__ == "__main__":
    main()
