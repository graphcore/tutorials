#!/usr/bin/env bash
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Run IPU program and capture reports.
# NOTE: Existing reports will be removed first.
#
if [ "${1}" == "" ]; then
 echo "Specify which program to run (e.g. step1_single_ipu.py, step4_configurable_stages.py)"
 echo "$ scripts/profile.sh step4_configurable_stages.py"
 echo "Reports will be generated at ./profile_step4_configurable_stages/..."
 return
fi

PROGBASE="${1%%.*}"
PROG="${PROGBASE}.py"
ARGS="${@:2}"
PROFILE_DIR="profile_${PROGBASE}"
ARGS_OVERRIDE="--batches-to-accumulate 16 --repeat-count 1 --steps 1"

echo "PROG: ${PROG}"
echo "ARGS: ${ARGS}"
echo "ARGS_OVERRIDE: ${ARGS_OVERRIDE}"
echo "PROFILE_DIR: ${PROFILE_DIR}"

echo "Removing existing reports from ${PROFILE_DIR}"
rm ${PROFILE_DIR}/* 2>/dev/null

echo "Running ${PROG}"
echo "=================================================================="
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"'${PROFILE_DIR}'"}' \
 PVTI_OPTIONS='{"enable":"true", "directory":"'${PROFILE_DIR}'"}' \
 TF_POPLAR_FLAGS='--use_synthetic_data' \
 python3 ${PROG} ${ARGS} ${ARGS_OVERRIDE}
