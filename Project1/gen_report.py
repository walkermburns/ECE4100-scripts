import itertools
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ----------------------------
# User configuration
# ----------------------------

EXECUTABLE = "./cachesim"
TRACE_DIR = Path("traces")
OUTPUT_CSV = "cache_results.csv"

TRACE_FILES = [
    "gcc.trace",
    "leela.trace",
    "linpack.trace",
    "matmul_naive.trace",
    "matmul_tiled.trace",
    "mcf.trace",
]

B_RANGE = range(5, 8)          # 5..7
C1_RANGE = range(14, 16)       # 14..15
C2_RANGE = range(16, 18)       # 16..17
REPLACEMENT_POLICIES = ["mip", "lip"]
PREFETCHERS = ["none", "plus1"]
L2_ENABLED = [0, 1]

MAX_WORKERS = 10   # <-- configurable parallelism

# ----------------------------
# Helpers
# ----------------------------

def parse_l1_aat(stdout):
    """Extract L1 AAT from simulator output."""
    for line in stdout.splitlines():
        if "L1 average access time" in line:
            return float(line.split(":")[1].strip())
    raise ValueError("L1 AAT not found in output")

def run_sim(config, trace):
    """Run cache simulator and return L1 AAT."""
    cmd = [
        EXECUTABLE,
        "-b", str(config["B"]),
        "-c", str(config["C1"]),
        "-s", str(config["S1"]),
    ]

    if config["L2_en"]:
        cmd += [
            "-C", str(config["C2"]),
            "-S", str(config["S2"]),
            "-P", config["Rep"],
            "-F", config["Pref"],
        ]
    else:
        cmd += ["-D"]

    with open(TRACE_DIR / trace, "r") as f:
        result = subprocess.run(
            cmd,
            stdin=f,
            text=True,
            capture_output=True,
            check=True,
        )

    return parse_l1_aat(result.stdout)

# ----------------------------
# Configuration generation
# ----------------------------

def generate_configs():
    configs = []

    for B, C1 in itertools.product(B_RANGE, C1_RANGE):
        max_S1 = C1 - B
        for S1 in range(max_S1 + 1):

            for l2_en in L2_ENABLED:
                if not l2_en:
                    configs.append({
                        "B": B, "C1": C1, "S1": S1,
                        "L2_en": 0,
                        "C2": None, "S2": None,
                        "Rep": None, "Pref": None
                    })
                    continue

                for C2 in C2_RANGE:
                    if C2 <= C1:
                        continue

                    max_S2 = C2 - B
                    for S2 in range(max_S2 + 1):
                        if S2 <= S1:
                            continue

                        for Rep, Pref in itertools.product(
                            REPLACEMENT_POLICIES, PREFETCHERS
                        ):
                            configs.append({
                                "B": B, "C1": C1, "S1": S1,
                                "L2_en": 1,
                                "C2": C2, "S2": S2,
                                "Rep": Rep, "Pref": Pref
                            })
    return configs

# ----------------------------
# Parallel worker
# ----------------------------

def run_one(task):
    trace, config, idx, total = task
    print(f"[{idx}/{total}] {trace} {config}")

    try:
        aat = run_sim(config, trace)
        return {
            "trace": trace,
            **config,
            "L1_AAT": aat
        }
    except Exception as e:
        print("ERROR:", e)
        return None

# ----------------------------
# Main experiment loop
# ----------------------------

def main():
    configs = generate_configs()
    print(f"Total configurations: {len(configs)}")

    # print([
    #     c for c in configs
    #     if c["B"] == 5 and c["C1"] == 15 and c["L2_en"] == 0
    # ][:5])
    
    # return

    tasks = []
    for trace in TRACE_FILES:
        for idx, config in enumerate(configs, 1):
            tasks.append((trace, config, idx, len(configs)))

    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]

        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    df = pd.DataFrame(results)
    df = df.sort_values(
    by=["trace", "B", "C1", "S1", "L2_en", "C2", "S2", "Rep", "Pref"],
    na_position="last"
)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved results to {OUTPUT_CSV}")

# def main():
#     config = {
#         "B": 5, "C1": 15, "S1": 10,
#         "L2_en": 0,
#         "C2": None, "S2": None,
#         "Rep": None, "Pref": None
#     }

#     print(run_sim(config, "gcc.trace"))

if __name__ == "__main__":
    main()
