#!/bin/bash
BASEDIR=${BASEDIR:-$(dirname $(cd $(dirname ${BASH_SOURCE[0]}) && pwd))}
set -a  # mark all variables below as exported (environment) variables

# Indentify this script as source of job configuration
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_NUM_GPU=1  # max of 1 (contact ETS to raise limit)
K8S_NUM_CPU=8  # max of 8 ("")
K8S_GB_MEM=64 # max of 64 ("")

# Controls whether an interactive Bash shell is started
SPAWN_INTERACTIVE_SHELL=NO

# Increase maximum execution time to 12h
K8S_TIMEOUT_SECONDS=43200

# Sets up proxy URL for Jupyter notebook inside
PROXY_ENABLED=NO

#K8S_ENTRYPOINT="/bin/sleep"
#K8S_ENTRYPOINT_ARGS_EXPANDED='"8640000"'
K8S_DOCKER_IMAGE="ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable"


exec launch.sh "$@" -- python -u diabetes_scGPT_mlm_objective/src/mlm_finetune_adipocyte_claude.py
