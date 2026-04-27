"""
CUDA와 PyTorch GPU 인식 상태를 확인하는 스크립트입니다.

예시:
    python scripts/check_cuda.py
    CUDA_VISIBLE_DEVICES=2 python scripts/check_cuda.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch


def run_nvidia_smi() -> None:
    print("\n[nvidia-smi -L]")
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        print("nvidia-smi 명령을 찾지 못했습니다.")
        return
    except subprocess.TimeoutExpired:
        print("nvidia-smi 실행 시간이 초과되었습니다.")
        return

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"exit code: {result.returncode}")


def main() -> None:
    print("[Python / PyTorch]")
    print(f"python executable: {sys.executable}")
    print(f"torch version: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
    else:
        print(
            "\nPyTorch가 CUDA를 감지하지 못했습니다. "
            "CUDA_VISIBLE_DEVICES를 비우고 실행해 보거나, nvidia-smi에서 보이는 GPU 번호를 확인하세요."
        )

    run_nvidia_smi()


if __name__ == "__main__":
    main()
