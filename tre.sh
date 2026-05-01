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
echo "Starting vLLM..."

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

echo "Waiting for vLLM to become ready..."

READY=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /tmp/vllm_models.json; then
    echo "vLLM is ready."
    cat /tmp/vllm_models.json
    READY=1
    break
  fi

  echo "Still waiting... attempt $i"
  sleep 20
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM did not become ready in time."
  echo "Check the HPC job output/error file for vLLM logs."
  exit 1
fi

echo
echo "Sending test request..."

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "What is 17*19? Return only the integer."}
    ],
    "temperature": 0.0,
    "max_tokens": 64
  }' | tee /tmp/vllm_test_response.json

echo
echo "Test response saved to /tmp/vllm_test_response.json"
echo "Done."















echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

echo "Starting vLLM for Qwen2.5..."

export VLLM_USE_DEEP_GEMM=0
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
  HTTP_CODE=$(curl -sS -m 10 \
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

curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
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