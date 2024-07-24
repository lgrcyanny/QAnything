#!/bin/bash
source /home/blackhole/miniforge3/etc/profile.d/conda.sh
conda activate qanything-python
pushd /home/blackhole/project/QAnything
bash scripts/run_for_openai_api_with_gpu_in_Linux_or_WSL.sh
popd
