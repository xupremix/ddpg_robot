import subprocess
import threading


base = "cargo run -- "


def run_scheduler(i):
    match i:
        case 0:
            print("case 0")
        case 1:
            print("case 1")
        case 2:
            print("case 2")
        case 3:
            print("case 3")
        case _:
            print("Invalid scheduler number")
            return


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
