#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_NAME="${ROBOCASA365_ENV_NAME:-onewm-robocasa365}"

clone_pinned() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination" ]]; then
    git clone --filter=blob:none "$url" "$destination"
    git -C "$destination" checkout "$revision"
  fi
  local actual
  actual="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$actual" != "$revision" ]]; then
    echo "Expected $destination at $revision; found $actual. Use a separate checkout." >&2
    exit 1
  fi
}

mkdir -p third_party
clone_pinned https://github.com/robocasa/robocasa.git third_party/RoboCasa365 b4684e6ee37d377cc392e98302a6b916d588b415
clone_pinned https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
if ! conda env list --json | python -c 'import json,sys; from pathlib import Path; raise SystemExit(not any(Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.11 -y
fi
conda run -n "$ENV_NAME" python -m pip install \
  numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 \
  pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml \
  hidapi gymnasium websockets==15.0.1
conda run -n "$ENV_NAME" python -m pip install -e third_party/robosuite
conda run -n "$ENV_NAME" python -m pip install --no-deps -e third_party/RoboCasa365
if [[ ! -f third_party/robosuite/robosuite/macros_private.py ]]; then
  conda run -n "$ENV_NAME" python -m robosuite.scripts.setup_macros
fi
if [[ ! -f third_party/RoboCasa365/robocasa/macros_private.py ]]; then
  conda run -n "$ENV_NAME" python -m robocasa.scripts.setup_macros
fi
echo "Installed $ENV_NAME. Download kitchen assets separately before simulation."
