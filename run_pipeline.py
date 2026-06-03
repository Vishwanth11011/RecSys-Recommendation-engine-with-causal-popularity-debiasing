import os
import subprocess
import sys
import time

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    start = time.time()
    # Use the local virtual environment's python
    venv_python = ".venv/bin/python"
    
    # Prefix command with virtualenv python if it is a python script
    if cmd.endswith(".py") or " " in cmd and cmd.split(" ")[0].endswith(".py"):
        full_cmd = f"{venv_python} {cmd}"
    else:
        # If running a script directly or custom command, parse it
        parts = cmd.split(" ")
        if parts[0] == "python":
            parts[0] = venv_python
        full_cmd = " ".join(parts)
        
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {full_cmd}. Return code: {result.returncode}")
        sys.exit(result.returncode)
    print(f">>> Finished in {time.time() - start:.1f}s\n")

def main():
    print("================ RECSYS MASTER PIPELINE ================")
    start_all = time.time()
    
    # 1. Train Standard NCF (this will also preprocess data, build encoders, and save parquet splits)
    run_cmd("src/training/train_ncf.py --epochs 2 --batch_size 1024")
    
    # 2. Train Debiased NCF with IPS
    run_cmd("src/training/train_ncf.py --epochs 2 --batch_size 1024 --debias")
    
    # 3. Train SVD Baseline
    run_cmd("src/training/train_svd.py")
    
    # 4. Train Two-Tower Retrieval Model
    run_cmd("src/training/train_two_tower.py --epochs 2 --batch_size 1024")
    
    # 5. Evaluate all models and create metrics.json
    run_cmd("evaluate_all.py")
    
    print("========================================================")
    print(f"All models trained and evaluated in {time.time() - start_all:.1f}s.")
    print("Metrics saved to saved_models/metrics.json.")

if __name__ == "__main__":
    main()
