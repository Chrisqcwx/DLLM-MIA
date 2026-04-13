import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

for i in range(10):
    try:
        snapshot_download(repo_id="GSAI-ML/LLaDA-1.5")
        break
    except Exception as e:
        print(f"Attempt {i+1} failed: {e}")
