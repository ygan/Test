from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = "/project/models/"

local_path = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_dir=f"{cache_dir.rstrip('/')}/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"Model downloaded to: {local_path}")
