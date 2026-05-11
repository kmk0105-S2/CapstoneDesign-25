import os
import sys
import subprocess
from pathlib import Path

PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
    "protobuf",
]

MODEL_ID = "Qwen/Qwen3.5-0.8B"
SAVE_DIR = Path("E:/project/Ewha_graduate_project/src/RAG/models/Qwen3.5-0.8B")


def install_packages():
    print("[STEP 1] Installing packages...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
    ])

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        *PACKAGES,
    ])

    print("[DONE] Package installation complete.")


def download_model():
    print("[STEP 2] Downloading model...")

    from huggingface_hub import snapshot_download

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(SAVE_DIR),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )

    print("[DONE] Model download complete.")
    print(f"[MODEL PATH] {SAVE_DIR.resolve()}")


def main():
    install_packages()
    download_model()


if __name__ == "__main__":
    main()