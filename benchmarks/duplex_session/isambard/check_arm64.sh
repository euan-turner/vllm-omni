#!/usr/bin/env bash
# Report whether the vLLM base image and the Python wheels this repo needs
# exist for linux/arm64 (Isambard-AI GH200). Run on a login node; downloads
# wheel metadata only (--no-deps) into a temporary directory.
#
# Prototype (branch long-context-prototype). External dependencies:
# podman-hpc or skopeo (image check), pip.
set -uo pipefail

IMAGE=${IMAGE:-docker.io/vllm/vllm-openai:v0.32.0}
PYVER=${PYVER:-3.12}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== base image $IMAGE"
if command -v skopeo >/dev/null; then
  skopeo inspect --raw "docker://$IMAGE" | grep -o '"architecture": *"[a-z0-9]*"' | sort -u
elif command -v podman-hpc >/dev/null; then
  podman-hpc manifest inspect "$IMAGE" 2>/dev/null | grep -o '"architecture": *"[a-z0-9]*"' | sort -u \
    || echo "manifest inspect failed; try: podman-hpc pull --arch arm64 $IMAGE"
else
  echo "no skopeo/podman-hpc; skipping"
fi

echo "== aarch64 wheels (python $PYVER)"
for req in vllm==0.32.0 step-audio2==1.0.0 onnxruntime av datasets==3.6.0; do
  if pip download --quiet --no-deps --only-binary=:all: --python-version "$PYVER" \
      --platform manylinux_2_28_aarch64 --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64 \
      --platform any -d "$TMP" "$req" >/dev/null 2>&1; then
    echo "ok       $req"
  else
    echo "MISSING  $req (no aarch64 or pure-python wheel; needs a source build or an alternative)"
  fi
done
