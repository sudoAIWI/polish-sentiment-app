from huggingface_hub import snapshot_download

def download_model():
    model_id = "bardsai/twitter-sentiment-pl-base"
    snapshot_download(
        repo_id=model_id,
        local_dir="models/original",
        ignore_patterns=["*.bin", "*.h5"]
    )

if __name__ == "__main__":
    download_model()