#!/bin/bash
# ---------------------------------------------------------------------------
# Docker wrapper for FastSurfer (SurfPrep)
#
# Mounts your data directory and FreeSurfer license into the FastSurfer
# container, then forwards all arguments to it. Set the two env vars below
# (or edit the defaults) to match your machine.
#
#   SURFPREP_DATA   host directory that contains BOTH your raw datasets and
#                   the FastSurfer output (subjects) dir — mounted as-is so
#                   absolute paths inside match the host.
#   FS_LICENSE      path to your FreeSurfer license.txt (free, see README).
#
# Example:
#   export SURFPREP_DATA=$HOME/data
#   export FS_LICENSE=$HOME/freesurfer/license.txt
#   ./run_fastsurfer.sh --sid fcdbonn__sub-00055 --sd /data/fastsurfer_subjects \
#       --t1 /data/FCDBONN/sub-00055/anat/sub-00055_T1w.nii.gz --seg_only
# ---------------------------------------------------------------------------
set -euo pipefail

SURFPREP_DATA="${SURFPREP_DATA:-$HOME/data}"
FS_LICENSE="${FS_LICENSE:-$HOME/freesurfer/license.txt}"
# Default to :latest; pin a specific image with FASTSURFER_IMAGE,
# e.g. export FASTSURFER_IMAGE=deepmi/fastsurfer:gpu-v2.4.2
FASTSURFER_IMAGE="${FASTSURFER_IMAGE:-deepmi/fastsurfer:latest}"
# GPU is needed for segmentation (seg_only/full). For surface-only runs on a
# CPU-only machine, disable it with: export DOCKER_GPU_FLAG=""
DOCKER_GPU_FLAG="${DOCKER_GPU_FLAG:---gpus all}"

docker run --rm $DOCKER_GPU_FLAG \
  -v "${SURFPREP_DATA}:${SURFPREP_DATA}" \
  -v "${FS_LICENSE}:/fs_license/license.txt:ro" \
  --user "$(id -u):$(id -g)" \
  "${FASTSURFER_IMAGE}" \
  "$@"