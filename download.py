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
import ssl
import httpx

# --------------------------------------------------
# TRE settings
# --------------------------------------------------

CERT_PATH = "/etc/ssl/certs/ca-certificates.crt"

os.environ["SSL_CERT_FILE"] = CERT_PATH
os.environ["REQUESTS_CA_BUNDLE"] = CERT_PATH
os.environ["CURL_CA_BUNDLE"] = CERT_PATH

# Keep using the TRE proxy from your environment.
# Do NOT unset http_proxy / https_proxy.

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"


# --------------------------------------------------
# SSL workaround for TRE proxy certificate
# --------------------------------------------------

ssl_context = ssl.create_default_context(cafile=CERT_PATH)

# Still require a trusted certificate chain
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Work around the TRE proxy's *.hf.co certificate,
# which does not match us.aws.cdn.hf.co
ssl_context.check_hostname = False


# --------------------------------------------------
# Hugging Face HTTP client
# --------------------------------------------------

from huggingface_hub import (
    snapshot_download,
    set_client_factory,
)


def make_client():
    return httpx.Client(
        verify=ssl_context,
        trust_env=True,          # use tre-proxy.er.kcl.ac.uk:3128
        follow_redirects=True,
        timeout=httpx.Timeout(
            300.0,
            connect=60.0,
        ),
    )


set_client_factory(make_client)


# --------------------------------------------------
# Download
# --------------------------------------------------

local_path = snapshot_download(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    local_dir="/dataset/models/DeepSeek-V4-Flash-0731",
    max_workers=1,
)

print("Model downloaded to:", local_path)