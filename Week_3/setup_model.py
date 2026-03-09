from huggingface_hub import snapshot_download
import os

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

print(f"Downloading model '{MODEL_NAME}'...")
print("This file is ~2.2GB. Please wait while the progress bar completes.")

# This function shows a progress bar by default
snapshot_download(repo_id=MODEL_NAME)

print("\n\nDownload complete! You can now run 'python main.py' and it will start instantly.")
