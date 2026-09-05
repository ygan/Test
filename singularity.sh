
mkdir -p /datasets/container_test
cd /datasets/container_test

singularity pull alpine.sif docker://alpine:latest
singularity exec alpine.sif cat /etc/os-release


echo $TMPDIR
echo $SINGULARITY_TMPDIR
echo $SINGULARITY_CACHEDIR

mkdir -p /dataset/singularity/tmp
mkdir -p /dataset/singularity/cache

export SINGULARITY_TMPDIR=/dataset/singularity/tmp
export SINGULARITY_CACHEDIR=/dataset/singularity/cache

echo $TMPDIR
echo $SINGULARITY_TMPDIR
echo $SINGULARITY_CACHEDIR

singularity pull /dataset/singularity/vllm-openai-v0280.sif \
  docker://vllm/vllm-openai:v0.28.0-x86_64-cu129