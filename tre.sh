set -euo pipefail

source ~/venvs/dsv4-vllm/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/dataset/models/DeepSeek-V4-Flash
export PORT=8000

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=600000
export TILELANG_CLEANUP_TEMP_FILES=1
export VLLM_DISABLE_COMPILE_CACHE=1
echo "Running on node: $(hostname)"
nvidia-smi

echo "Starting vLLM for DeepSeek v4 flash..."

export VLLM_USE_DEEP_GEMM=0
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

PORT="${PORT:-8000}"

vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --enforce-eager \
  --no-disable-hybrid-kv-cache-manager \
  --host 127.0.0.1 \
  --port "$PORT" &

SERVER_PID=$!
echo "vLLM PID: $SERVER_PID"

cleanup() {
  echo "Cleaning up vLLM server..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

AUTH_ARGS=()
if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

echo "Waiting for vLLM to become ready..."

READY=0
for i in $(seq 1 180); do
  HTTP_CODE=$(curl --noproxy "*" -sS -m 10 \
    -o /tmp/vllm_models.json \
    -w "%{http_code}" \
    "${AUTH_ARGS[@]}" \
    "http://127.0.0.1:${PORT}/v1/models" || echo "000")

  if [ "$HTTP_CODE" = "200" ]; then
    echo "vLLM is ready."
    cat /tmp/vllm_models.json
    READY=1
    break
  fi

  echo "Still waiting... attempt $i, HTTP_CODE=$HTTP_CODE"
  cat /tmp/vllm_models.json 2>/dev/null || true
  echo

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM process exited before becoming ready."
    wait "$SERVER_PID" || true
    exit 1
  fi

  sleep 20
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM did not become ready in time."
  echo "Check the HPC job output/error file for vLLM logs."
  exit 1
fi

echo
echo "Sending test request..."

curl --noproxy "*" -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 17*19? Return only the integer."}
    ],
    "temperature": 0.0,
    "max_tokens": 64
  }' | tee /tmp/deepseek_v4_flash_test_response.json

echo
echo "Test response saved to /tmp/deepseek_v4_flash_test_response.json"
echo "Done."



















#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_PATH="${MODEL_PATH:-/dataset/models/DeepSeek-V4-Flash}"
export PORT="${PORT:-8000}"

# 默认用 TP，避免 DP Coordinator / ZMQ startup timeout。
# 如果之后想回到官方推荐 DP+EP，可以这样跑：
#   PARALLEL_MODE=dp bash run_deepseek_v4_flash.sh
export PARALLEL_MODE="${PARALLEL_MODE:-tp}"

# 启动先保守一点，起来后再加大。
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# HPC 上不要依赖共享 /tmp；把 IPC/cache/log 都放到本地 scratch。
SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
RUN_ID="${SLURM_JOB_ID:-manual}_$$"
RUN_DIR="${SCRATCH_BASE}/${USER:-user}_vllm_dsv4_${RUN_ID}"

mkdir -p "$RUN_DIR"/{tmp,rpc,cache,triton,torchinductor,deep_gemm}
export TMPDIR="$RUN_DIR/tmp"
export VLLM_RPC_BASE_PATH="$RUN_DIR/rpc"
export VLLM_CACHE_ROOT="$RUN_DIR/cache"
export TRITON_CACHE_DIR="$RUN_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor"
export DG_JIT_CACHE_DIR="$RUN_DIR/deep_gemm"

# 超时和日志。
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-10}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1

# 避免代理影响 localhost。
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

# 不要把 API server 的 host/port 和 vLLM 内部 host/port 混起来。
unset VLLM_HOST_IP || true
unset VLLM_PORT || true

# 先不要禁用 compile cache，否则每次启动都更慢。
unset VLLM_DISABLE_COMPILE_CACHE || true

# 你没有系统 CUDA13 / nvcc 时，先关闭 MoE DeepGEMM 路径，减少 JIT/warmup 变量。
# 这个变量是 vLLM DeepSeek 文档里建议过的绕过方式之一。
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export TILELANG_CLEANUP_TEMP_FILES=1

echo "Running on node: $(hostname)"
echo "RUN_DIR=$RUN_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "PARALLEL_MODE=$PARALLEL_MODE"
echo

nvidia-smi
echo

python - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError

def pkgver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"

print("python:", sys.version.replace("\n", " "))
print("vllm:", pkgver("vllm"))
print("torch:", pkgver("torch"))
print("transformers:", pkgver("transformers"))
print("pyzmq:", pkgver("pyzmq"))

import torch
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.device_count() < 4:
    raise SystemExit("ERROR: need 4 visible GPUs for this script.")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"gpu {i}: {name}, capability={cap}, memory={mem:.1f} GiB")
    if cap < (8, 9):
        print(f"WARNING: GPU {i} has capability {cap}; FP8/DeepSeek-V4 path may not be supported.")
PY

IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${#GPU_LIST[@]}"

COMMON_ARGS=(
  serve "$MODEL_PATH"
  --served-model-name deepseek-v4-flash
  --trust-remote-code
  --kv-cache-dtype fp8
  --block-size 256
  --tokenizer-mode deepseek_v4
  --reasoning-parser deepseek_v4
  --load-format safetensors
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enforce-eager
  --no-disable-hybrid-kv-cache-manager
  --disable-uvicorn-access-log
  --host 127.0.0.1
  --port "$PORT"
)

if [[ "$PARALLEL_MODE" == "dp" ]]; then
  echo "Using DP+EP mode. This can still hit DP Coordinator startup issues."
  COMMON_ARGS+=(
    --data-parallel-size "$GPU_COUNT"
    --data-parallel-address 127.0.0.1
    --data-parallel-rpc-port "${DP_RPC_PORT:-13345}"
    --enable-expert-parallel
  )
elif [[ "$PARALLEL_MODE" == "tp" ]]; then
  echo "Using TP+EP mode to avoid DP Coordinator."
  COMMON_ARGS+=(
    --tensor-parallel-size "$GPU_COUNT"
    --enable-expert-parallel
    --disable-custom-all-reduce
  )
else
  echo "ERROR: PARALLEL_MODE must be tp or dp, got: $PARALLEL_MODE" >&2
  exit 2
fi

LOG_FILE="$RUN_DIR/vllm_server.log"
MODELS_JSON="$RUN_DIR/vllm_models.json"
TEST_JSON="$RUN_DIR/deepseek_v4_flash_test_response.json"

echo
echo "Starting vLLM..."
printf 'Command: vllm'
printf ' %q' "${COMMON_ARGS[@]}"
echo
echo "Log file: $LOG_FILE"
echo

setsid vllm "${COMMON_ARGS[@]}" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "vLLM PID: $SERVER_PID"

cleanup() {
  echo
  echo "Cleaning up vLLM server..."
  kill -TERM "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
  sleep 3
  kill -KILL "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

AUTH_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

echo "Waiting for vLLM to become ready..."

READY=0
for i in $(seq 1 360); do
  HTTP_CODE="$(curl --noproxy "*" -sS -m 10 \
    -o "$MODELS_JSON" \
    -w "%{http_code}" \
    "${AUTH_ARGS[@]}" \
    "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo "000")"

  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "vLLM is ready."
    cat "$MODELS_JSON"
    echo
    READY=1
    break
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM process exited before becoming ready."
    echo
    echo "===== last 200 lines of vLLM log ====="
    tail -n 200 "$LOG_FILE" || true
    wait "$SERVER_PID" || true
    exit 1
  fi

  if (( i % 12 == 0 )); then
    echo "Still waiting... attempt $i, HTTP_CODE=$HTTP_CODE"
    echo "vLLM is still alive. Log file: $LOG_FILE"
    echo "GPU snapshot:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  else
    echo "Still waiting... attempt $i, HTTP_CODE=$HTTP_CODE"
  fi

  sleep 10
done

if [[ "$READY" -ne 1 ]]; then
  echo "vLLM did not become ready in time."
  echo
  echo "===== last 300 lines of vLLM log ====="
  tail -n 300 "$LOG_FILE" || true
  exit 1
fi

echo
echo "Sending test request..."

curl --noproxy "*" -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 17*19? Return only the integer."}
    ],
    "temperature": 0.0,
    "max_tokens": 64
  }' | tee "$TEST_JSON"

echo
echo "Test response saved to $TEST_JSON"
echo "vLLM log saved to $LOG_FILE"
echo "Done."













echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

echo "Starting vLLM for Qwen2.5..."

export VLLM_USE_DEEP_GEMM=0
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

PORT="${PORT:-8000}"

vllm serve "$MODEL_PATH" \
  --served-model-name qwen2.5 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --host 127.0.0.1 \
  --port "$PORT" &

SERVER_PID=$!
echo "vLLM PID: $SERVER_PID"

cleanup() {
  echo "Cleaning up vLLM server..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

AUTH_ARGS=()
if [ -n "${VLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

echo "Waiting for vLLM to become ready..."

READY=0
for i in $(seq 1 120); do
  HTTP_CODE=$(curl --noproxy "*" -sS -m 10 \
    -o /tmp/vllm_models.json \
    -w "%{http_code}" \
    "${AUTH_ARGS[@]}" \
    "http://127.0.0.1:${PORT}/v1/models" || echo "000")

  if [ "$HTTP_CODE" = "200" ]; then
    echo "vLLM is ready."
    cat /tmp/vllm_models.json
    READY=1
    break
  fi

  echo "Still waiting... attempt $i, HTTP_CODE=$HTTP_CODE"
  cat /tmp/vllm_models.json 2>/dev/null || true
  echo

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM process exited before becoming ready."
    wait "$SERVER_PID" || true
    exit 1
  fi

  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM did not become ready in time."
  exit 1
fi

echo
echo "Sending test request..."

curl --noproxy "*" -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 17*19? Return only the integer."}
    ],
    "temperature": 0.0,
    "top_p": 0.8,
    "repetition_penalty": 1.05,
    "max_tokens": 64
  }' | tee /tmp/qwen25_test_response.json

echo
echo "Done."















#!/usr/bin/env bash
set -euo pipefail

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true


MODEL_PATH="${MODEL_PATH:-~/DeepSeek-V4-Flash-Int4}"


PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
MODEL_NAME="${MODEL_NAME:-ds}"
DEVICE="${DEVICE:-cuda}"
MOE_DEVICE="${MOE_DEVICE:-cuda}"

# 对原始 FP16/BF16 模型可设 DTYPE=int4/int8/fp8/float16；
# 如果 MODEL_PATH 已经是 ftllm export 出来的量化模型，可以设 DTYPE=""。
DTYPE="${DTYPE:-int4}"

# 可选：CPU 线程数。空则不传 -t。
THREADS="${THREADS:-}"

export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

echo "Starting FastLLM server..."
echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "HOST=${HOST}"
echo "PORT=${PORT}"
echo "DEVICE=${DEVICE}"
echo "DTYPE=${DTYPE:-<not set>}"
echo "THREADS=${THREADS:-<not set>}"

SERVER_ARGS=(
  ftllm server "$MODEL_PATH"
  --model_name "$MODEL_NAME"
  --port "$PORT"
  --device "$DEVICE"
  --moe_device "$MOE_DEVICE"
)

if [ -n "$DTYPE" ]; then
  SERVER_ARGS+=(--dtype "$DTYPE")
fi

if [ -n "$THREADS" ]; then
  SERVER_ARGS+=(-t "$THREADS")
fi

if [ -n "${FTLLM_API_KEY:-}" ]; then
  SERVER_ARGS+=(--api_key "$FTLLM_API_KEY")
fi

# 也可以外部追加参数，例如:
# export FTLLM_EXTRA_ARGS="--moe_device cpu --kv_cache_limit 20G"
if [ -n "${FTLLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=($FTLLM_EXTRA_ARGS)
  SERVER_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Command: ${SERVER_ARGS[*]}"
"${SERVER_ARGS[@]}" &

SERVER_PID=$!
echo "FastLLM PID: $SERVER_PID"

cleanup() {
  echo "Cleaning up FastLLM server..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

AUTH_ARGS=()
if [ -n "${FTLLM_API_KEY:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${FTLLM_API_KEY}")
fi

echo "Waiting for FastLLM to become ready..."

READY=0
for i in $(seq 1 120); do
  HTTP_CODE=$(curl --noproxy "*" -sS -m 10 \
    -o /tmp/ftllm_models.json \
    -w "%{http_code}" \
    "${AUTH_ARGS[@]}" \
    "http://${HOST}:${PORT}/v1/models" || echo "000")

  if [ "$HTTP_CODE" = "200" ]; then
    echo "FastLLM is ready."
    cat /tmp/ftllm_models.json
    READY=1
    break
  fi

  echo "Still waiting... attempt $i, HTTP_CODE=$HTTP_CODE"
  cat /tmp/ftllm_models.json 2>/dev/null || true
  echo

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FastLLM process exited before becoming ready."
    wait "$SERVER_PID" || true
    exit 1
  fi

  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "FastLLM did not become ready in time."
  exit 1
fi

echo
echo "Sending test request..."

HTTP_CODE=$(curl --noproxy "*" -sS -m 120 \
  -o /tmp/qwen25_ftllm_test_response.json \
  -w "%{http_code}" \
  "http://${HOST}:${PORT}/v1/chat/completions" \
  "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
      {\"role\": \"user\", \"content\": \"What is 17*19? Return only the integer.\"}
    ],
    \"temperature\": 0.0,
    \"top_p\": 0.8,
    \"repetition_penalty\": 1.05,
    \"max_tokens\": 64
  }" || echo "000")

cat /tmp/qwen25_ftllm_test_response.json
echo

if [ "$HTTP_CODE" != "200" ]; then
  echo "Test request failed, HTTP_CODE=$HTTP_CODE"
  exit 1
fi

echo
echo "Done."







curl https://api.ai.create.kcl.ac.uk/v1/chat/completions \
  -H "Authorization: sk-XQ3u4h1pl7AT_ozxescSYA" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arc:lite",
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ]
  }'

singularity shell --nv \
  --bind /dataset \
  --bind ~/envs \
  --bind ~/python \
  --bind usr/bin \
  --bind /etc/ssl/certs \
  /software/containers/singularity/epile/epile.sif

singularity exec --nv \
  --bind /dataset \
  --bind ~/envs \
  --bind ~/python \
  /software/containers/singularity/epile/epile.sif \
  bash -c "
source ~/envs/vllm312/bin/activate
cd ~/python/
python your_script.py
"

singularity exec /software/containers/singularity/epile/epile.sif python --version
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 vllm serve /home/3059733@eeecs.qub.ac.uk/models/Qwen3.6-27B-FP8   --served-model-name qwen3.6   --trust-remote-code   --dtype auto   --max-model-len 256   --gpu-memory-utilization 0.95   --kv-cache-dtype fp8   --max-num-seqs 1   --max-num-batched-tokens 256   --limit-mm-per-prompt '{"image":0,"video":0}'   --language-model-only   --reasoning-parser qwen3   --default-chat-template-kwargs '{"enable_thinking": false}'   --attention-backend TRITON_ATTN    --host 127.0.0.1   --port 8000   2>&1 | tee vllm-test-startup.log







set -Eeuo pipefail

SCRATCH_BASE="${SLURM_TMPDIR:-${TMPDIR:-$HOME/job_scratch}}"
RUN_ID="${SLURM_JOB_ID:-manual}_$$"
RUN_DIR="${SCRATCH_BASE}/${USER:-user}_vllm_dsv4_${RUN_ID}"

mkdir -p "$RUN_DIR"/{tmp,rpc,cache,triton,torchinductor,cuda_cache,deep_gemm}

export RUN_DIR
export TMPDIR="$RUN_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

export VLLM_RPC_BASE_PATH="$RUN_DIR/rpc"
export VLLM_CACHE_ROOT="$RUN_DIR/cache"
export XDG_CACHE_HOME="$RUN_DIR/cache"
export TRITON_CACHE_DIR="$RUN_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor"
export CUDA_CACHE_PATH="$RUN_DIR/cuda_cache"
export DG_JIT_CACHE_DIR="$RUN_DIR/deep_gemm"

export LD_LIBRARY_PATH="/.singularity.d/libs:${LD_LIBRARY_PATH:-}"

echo "RUN_DIR=$RUN_DIR"
echo "TMPDIR=$TMPDIR"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"


cat > "$RUN_DIR/check_libcuda.c" <<'C'
#include <stdio.h>

extern int cuInit(unsigned int Flags);

int main() {
    int r = cuInit(0);
    printf("cuInit returned %d\n", r);
    return 0;
}
C

ls -l "$RUN_DIR/check_libcuda.c"

gcc "$RUN_DIR/check_libcuda.c" \
  -L/.singularity.d/libs \
  -Wl,-rpath,/.singularity.d/libs \
  -l:libcuda.so.1 \
  -o "$RUN_DIR/check_libcuda" \
  -v

"$RUN_DIR/check_libcuda"
echo "libcuda link test OK"




echo "which nvcc: $(which nvcc)"
nvcc -V || true

echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_PATH=$CUDA_PATH"
echo "PATH=$PATH" | tr ':' '\n' | grep -i cuda || true
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i cuda || true

python - <<'PY'
import os, torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch cpp_extension CUDA_HOME:", CUDA_HOME)
print("env CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("env CUDA_PATH:", os.environ.get("CUDA_PATH"))
PY


export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL,SHM,P2P
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
monitor() {
  while true; do
    echo "===== $(date) ====="
    hostname
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi
    df -h /dev/shm
    free -h
    ps -o pid,ppid,stat,etime,pcpu,pmem,cmd -fu "$USER" | grep -E "vllm|python|singularity|apptainer" | grep -v grep || true
    sleep 60
  done
}

monitor > job_scratch/monitor_uu.log 2>&1 &