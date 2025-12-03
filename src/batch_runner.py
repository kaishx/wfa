import subprocess
import datetime
import os
import sys
from tqdm import tqdm

configs = [
    # BELOW IS ONLY A TEMPLATE, please please please change it
{"ASSET_A": "XXX", "ASSET_B": "YYY", "START_DATE": "2021-11-01", "TIME_CONFIG": "15m", "ADF_P_VALUE_THRESHOLD": 0.99},
]

wfa_script = "cpp_wfa.py" # direct it to the wfa in the same root

LOG_DIR = "wfa_logs"
OUTPUT_DIR = "wfa_outputs" # equity curve outputs , but actually, this is overwritten by the wfa code itself but not gonna change it just yet. if you do want to make it dynamic tho, the variable is called the same name in cpp_wfa
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for cfg in configs:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = f"{cfg['ASSET_A']}_{cfg['ASSET_B']}_{datetime.datetime.strptime(cfg['START_DATE'], '%Y-%m-%d').strftime('%d%m%y')}_{cfg['TIME_CONFIG']}_{cfg['ADF_P_VALUE_THRESHOLD']}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    csv_filename = f"oos_equity_curve_{cfg['ASSET_A']}_{cfg['ASSET_B']}_{cfg['TIME_CONFIG']}_{timestamp}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    print(f"\nRunning config: {cfg['ASSET_A']} vs {cfg['ASSET_B']} | {cfg['TIME_CONFIG']} | {cfg['START_DATE']}")
    print(f"Log file: {log_path}")
    print(f"CSV output: {csv_path}")

    env = os.environ.copy()
    env["ASSET_A"] = cfg["ASSET_A"]
    env["ASSET_B"] = cfg["ASSET_B"]
    env["START_DATE"] = cfg["START_DATE"]
    env["TIME_CONFIG"] = cfg["TIME_CONFIG"]
    env["ADF_P_VALUE_THRESHOLD"] = str(cfg["ADF_P_VALUE_THRESHOLD"])
    env["OUTPUT_FILE"] = csv_path

    with open(log_path, "w") as log_file:
        with tqdm(total=1, desc=f"{cfg['ASSET_A']}/{cfg['ASSET_B']} {cfg['TIME_CONFIG']} {cfg['ADF_P_VALUE_THRESHOLD']} WFA", unit="run") as pbar: # on my personal usage, this pbar is useless. it only shows 0% or 100%...
            process = subprocess.Popen(
                [sys.executable, wfa_script],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env
            )
            process.wait()
            pbar.update(1)

    print(f"finished {cfg['ASSET_A']}/{cfg['ASSET_B']} ({cfg['TIME_CONFIG']})")
    print(f"Logs saved at: {log_path}")
    print(f"CSV saved at: {csv_path}\n")

