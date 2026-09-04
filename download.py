# from pathlib import Path
# from huggingface_hub import snapshot_download

# def download_model_all(
#     model_id: str,
#     base_dir: str = "models/",
#     revision: str = "main",
#     token: bool = True,
#     force_download: bool = False,
# ):
#     model_name = model_id.split("/")[-1]
#     local_dir = Path(base_dir).expanduser() / model_name

#     local_path = snapshot_download(
#         repo_id=model_id,
#         revision=revision,
#         local_dir=str(local_dir),
#         token=token,
#         force_download=force_download,
#         max_workers=8,
#     )

#     print(f"Model downloaded to: {local_path}")
#     return local_path

# download_model_all("deepseek-ai/DeepSeek-V4-Flash-0731")


import os

# 必须尽量放在 import huggingface_hub 之前
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

from pathlib import Path
from huggingface_hub import snapshot_download


def download_model_all(
    model_id: str,
    base_dir: str = "models/",
    revision: str = "main",
    token: bool = True,
    force_download: bool = False,
):
    model_name = model_id.split("/")[-1]
    local_dir = Path(base_dir).expanduser() / model_name

    local_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
        force_download=force_download,

        # TRE / proxy 环境先用 1
        max_workers=1,
    )

    print(f"Model downloaded to: {local_path}")
    return local_path


download_model_all("deepseek-ai/DeepSeek-V4-Flash-0731")