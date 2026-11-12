#!/bin/bash
BASEDIR=${BASEDIR:-$(dirname $(cd $(dirname ${BASH_SOURCE[0]}) && pwd))}
set -a

K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_NUM_GPU=1
K8S_NUM_CPU=8
K8S_GB_MEM=64

SPAWN_INTERACTIVE_SHELL=NO
K8S_TIMEOUT_SECONDS=43200
PROXY_ENABLED=NO
K8S_DOCKER_IMAGE="ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable"

exec launch.sh "$@" -- python -u diabetes_scGPT_mlm_objective/src/profile_candidate_tfs.py
