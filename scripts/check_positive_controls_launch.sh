#!/bin/bash
BASEDIR=${BASEDIR:-$(dirname $(cd $(dirname ${BASH_SOURCE[0]}) && pwd))}
set -a
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}
K8S_NUM_GPU=0
K8S_NUM_CPU=4
K8S_GB_MEM=32
SPAWN_INTERACTIVE_SHELL=NO
K8S_TIMEOUT_SECONDS=600
PROXY_ENABLED=NO
K8S_DOCKER_IMAGE="ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable"
exec launch.sh "$@" -- python -u diabetes_scGPT_mlm_objective/src/check_positive_controls.py
