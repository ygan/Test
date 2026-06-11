
mkdir -p /datasets/container_test
cd /datasets/container_test

singularity pull alpine.sif docker://alpine:latest
singularity exec alpine.sif cat /etc/os-release