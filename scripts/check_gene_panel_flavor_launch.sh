#!/bin/bash
BASEDIR=${BASEDIR:-$(dirname $(cd $(dirname ${BASH_SOURCE[0]}) && pwd))}
set -a  # mark all variables below as exported (environment) variables

# Indentify this script as source of job configuration
K8S_CONFIG_SOURCE=${BASH_SOURCE[0]}

K8S_NUM_GPU=0  # diagnostic only, no training -- no GPU needed
K8S_NUM_CPU=8
K8S_GB_MEM=64  # matches mlm_obj_launch.sh/grn_launch.sh -- this h5ad needs it

# Controls whether an interactive Bash shell is started
SPAWN_INTERACTIVE_SHELL=NO

K8S_TIMEOUT_SECONDS=900

# Sets up proxy URL for Jupyter notebook inside
PROXY_ENABLED=NO

K8S_DOCKER_IMAGE="ghcr.io/ucsd-ets/scipy-ml-notebook:2025.1-stable"


exec launch.sh "$@" -- python -u diabetes_scGPT_mlm_objective/src/check_gene_panel_flavor.py
