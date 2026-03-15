#!/usr/bin/env python3
"""Deploy and run 5 plug-in module ablation experiments on remote server.

Launches 5 parallel training jobs (screen sessions) on uhost:
  1. duel+CA       (Coordinate Attention)
  2. duel+IQN      (Implicit Quantile Networks)
  3. duel+FADC     (Frequency-Adaptive Dilated Conv)
  4. duel+Deform   (Deformable Conv v2)
  5. duel+Noisy    (NoisyNet)

All based on CNN-DDQN+Duel, 2ch (drop EDT), diag collision, 10000 episodes.
"""

import paramiko
import time
import sys

HOST = "117.50.216.203"
USER = "ubuntu"
PASSWD = "g7TXK26Q85Jp493f"
REMOTE_PROJ = "$HOME/DQN9"
CONDA = "$HOME/miniconda3/bin/conda"
ENV = "ros2py310"

PROFILES = [
    "ablation_20260314_duel_ca",
    "ablation_20260314_duel_iqn",
    "ablation_20260314_duel_fadc",
    "ablation_20260314_duel_deform",
    "ablation_20260314_duel_noisy",
]

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {HOST}...")
    ssh.connect(HOST, username=USER, password=PASSWD, timeout=30)

    def run(cmd: str, timeout: int = 120) -> str:
        print(f"  $ {cmd}")
        _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out.strip():
            print(f"    stdout: {out.strip()[:200]}")
        if err.strip():
            print(f"    stderr: {err.strip()[:200]}")
        return out

    # Step 1: git pull
    print("\n=== Step 1: git pull ===")
    run(f"cd {REMOTE_PROJ} && git pull origin main", timeout=120)

    # Step 2: Check torchvision for deform_conv2d
    print("\n=== Step 2: Check torchvision ===")
    run(f"{CONDA} run --cwd {REMOTE_PROJ} -n {ENV} python -c \"from torchvision.ops import deform_conv2d; print('deform_conv2d OK')\"")

    # Step 3: Self-check
    print("\n=== Step 3: Self-check ===")
    run(f"{CONDA} run --cwd {REMOTE_PROJ} -n {ENV} python train.py --self-check")

    # Step 4: Quick module import test
    print("\n=== Step 4: Module import test ===")
    run(f"{CONDA} run --cwd {REMOTE_PROJ} -n {ENV} python -c \"from ugv_dqn.modules import CoordAttention, NoisyLinear, FADC, DeformConv2dBlock, IQNHead; print('All modules imported OK')\"")

    # Step 5: Launch 5 screen sessions
    print("\n=== Step 5: Launching 5 training jobs ===")
    for profile in PROFILES:
        screen_name = profile.replace("ablation_20260314_", "abl_")
        log_file = f"{REMOTE_PROJ}/runs/{profile}.log"
        train_cmd = (
            f"{CONDA} run --cwd {REMOTE_PROJ} -n {ENV} "
            f"python train.py --profile {profile}"
        )
        screen_cmd = f"screen -dmS {screen_name} bash -c '{train_cmd} > {log_file} 2>&1'"
        run(screen_cmd)
        print(f"  Launched: {screen_name}")
        time.sleep(2)

    # Step 6: Verify
    print("\n=== Step 6: Verify running sessions ===")
    run("screen -ls")

    print("\n=== Done! Monitor with: ===")
    for profile in PROFILES:
        screen_name = profile.replace("ablation_20260314_", "abl_")
        print(f"  screen -r {screen_name}")

    ssh.close()

if __name__ == "__main__":
    main()
